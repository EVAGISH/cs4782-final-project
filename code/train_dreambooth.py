import json
import random
from collections import deque
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    TextToVideoSDPipeline,
    UNet2DConditionModel,
    UNet3DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm

from class_priors import load_class_prior_images
from lora import get_lora_state_dict, patch_unet_with_lora, summarize_lora_state_dict

try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False


INSTANCE_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
INSTANCE_VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def _list_instance_image_paths(directory: Path) -> list[Path]:
    paths = [
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in INSTANCE_IMAGE_SUFFIXES
    ]
    return sorted(paths)


def _list_instance_video_paths(directory: Path) -> list[Path]:
    paths = [
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in INSTANCE_VIDEO_SUFFIXES
    ]
    return sorted(paths)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def validate_config(cfg: DictConfig):
    if cfg.task not in {"image", "video"}:
        raise ValueError(f"task must be 'image' or 'video', got {cfg.task!r}")
    if cfg.training.max_train_steps < 1:
        raise ValueError("training.max_train_steps must be at least 1")
    if cfg.training.log_every_steps < 1:
        raise ValueError("training.log_every_steps must be at least 1")
    if cfg.task == "video":
        if cfg.data.num_frames < 1:
            raise ValueError("data.num_frames must be at least 1")
        if cfg.data.frame_stride < 1:
            raise ValueError("data.frame_stride must be at least 1")


def resolve_config_paths(cfg: DictConfig):
    if cfg.task == "image":
        cfg.data.instance_data_dir = to_absolute_path(cfg.data.instance_data_dir)
        cfg.data.class_images_npz = to_absolute_path(cfg.data.class_images_npz)
    else:
        cfg.data.instance_video_dir = to_absolute_path(cfg.data.instance_video_dir)
    cfg.training.output_dir = to_absolute_path(cfg.training.output_dir)


class DreamBoothDataset(Dataset):
    def __init__(self, instance_data_dir, class_images_npz, instance_prompt, class_prompt, tokenizer, resolution=512):
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.instance_prompt = instance_prompt
        self.class_prompt = class_prompt

        self.instance_images = _list_instance_image_paths(Path(instance_data_dir))
        self._class_stack = load_class_prior_images(class_images_npz)

        self.num_instance_images = len(self.instance_images)
        self.num_class_images = self._class_stack.shape[0]
        if self.num_instance_images == 0:
            raise ValueError(f"No images found in {instance_data_dir}")
        if self.num_class_images == 0:
            raise ValueError(f"Empty class priors in {class_images_npz}")
        self._length = max(self.num_class_images, self.num_instance_images)

        self.image_transforms = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        instance_image = Image.open(self.instance_images[index % self.num_instance_images]).convert("RGB")
        instance_image = self.image_transforms(instance_image)
        instance_tokens = self.tokenizer(
            self.instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        class_image = Image.fromarray(self._class_stack[index % self.num_class_images], mode="RGB")
        class_image = self.image_transforms(class_image)
        class_tokens = self.tokenizer(
            self.class_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        return {
            "instance_images": instance_image,
            "instance_prompt_ids": instance_tokens,
            "class_images": class_image,
            "class_prompt_ids": class_tokens,
        }


class VideoDreamBoothDataset(Dataset):
    """Returns a clip per index of shape [3, F, H, W] in [-1, 1].

    No prior preservation in v1: only instance videos are loaded; the training
    loop skips the class loss when task == video.
    """

    def __init__(
        self,
        instance_video_dir,
        instance_prompt,
        tokenizer,
        resolution: int = 256,
        num_frames: int = 16,
        frame_stride: int = 1,
    ):
        self.tokenizer = tokenizer
        self.instance_prompt = instance_prompt
        self.resolution = resolution
        self.num_frames = num_frames
        self.frame_stride = frame_stride

        self.instance_videos = _list_instance_video_paths(Path(instance_video_dir))
        if not self.instance_videos:
            raise ValueError(f"No video files found in {instance_video_dir}")

        self.frame_transforms = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return max(len(self.instance_videos), 100)

    def _load_clip(self, path: Path) -> torch.Tensor:
        # Lazy import so image-only runs don't require torchcodec at module load.
        from torchcodec.decoders import VideoDecoder

        decoder = VideoDecoder(str(path))
        total = decoder.metadata.num_frames
        if total is None:
            raise RuntimeError(f"Could not determine frame count for {path}")

        span = (self.num_frames - 1) * self.frame_stride + 1
        if total < span:
            indices = list(range(total))
            while len(indices) < self.num_frames:
                indices.append(indices[-1])
            indices = indices[: self.num_frames]
        else:
            start = random.randint(0, total - span)
            indices = list(range(start, start + span, self.frame_stride))

        frames = decoder.get_frames_at(indices=indices).data  # uint8 [F, 3, H, W]
        clip = frames.float() / 255.0
        clip = self.frame_transforms(clip)

        if random.random() < 0.5:
            clip = torch.flip(clip, dims=[-1])

        # [F, 3, H, W] -> [3, F, H, W] for UNet3D convention.
        return clip.permute(1, 0, 2, 3).contiguous()

    def __getitem__(self, index):
        video_path = self.instance_videos[index % len(self.instance_videos)]
        instance_video = self._load_clip(video_path)
        instance_tokens = self.tokenizer(
            self.instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]
        return {
            "instance_video": instance_video,
            "instance_prompt_ids": instance_tokens,
        }


def _video_vae_encode(vae, video: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Encode a [B, 3, F, H, W] pixel video to [B, 4, F, H/8, W/8] latents.

    The 2D VAE is reused frame-by-frame; we reshape to (B*F, 3, H, W),
    optionally chunk to bound VAE activations, then reshape back.
    """
    B, C, F_, H, W = video.shape
    flat = video.permute(0, 2, 1, 3, 4).reshape(B * F_, C, H, W)

    if chunk_size and chunk_size < flat.shape[0]:
        chunks = [
            vae.encode(flat[i : i + chunk_size]).latent_dist.sample()
            for i in range(0, flat.shape[0], chunk_size)
        ]
        flat_latents = torch.cat(chunks, dim=0) * vae.config.scaling_factor
    else:
        flat_latents = vae.encode(flat).latent_dist.sample() * vae.config.scaling_factor

    latent_h, latent_w = flat_latents.shape[-2], flat_latents.shape[-1]
    latents = flat_latents.reshape(B, F_, -1, latent_h, latent_w).permute(0, 2, 1, 3, 4)
    return latents.contiguous()


@hydra.main(version_base=None, config_path="conf", config_name="train_dreambooth")
def main(cfg: DictConfig):
    validate_config(cfg)
    resolve_config_paths(cfg)
    print(OmegaConf.to_yaml(cfg))

    device = get_device()
    print(f"Using device: {device}")
    print(f"Using dtype: {torch.float32}")

    torch.manual_seed(cfg.training.seed)
    random.seed(cfg.training.seed)

    unet_cls = UNet3DConditionModel if cfg.task == "video" else UNet2DConditionModel

    tokenizer = CLIPTokenizer.from_pretrained(cfg.model.pretrained_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(cfg.model.pretrained_model, subfolder="text_encoder", torch_dtype=torch.float32)
    vae = AutoencoderKL.from_pretrained(cfg.model.pretrained_model, subfolder="vae", torch_dtype=torch.float32)
    unet = unet_cls.from_pretrained(cfg.model.pretrained_model, subfolder="unet", torch_dtype=torch.float32)
    noise_scheduler = DDPMScheduler.from_pretrained(cfg.model.pretrained_model, subfolder="scheduler")

    vae.requires_grad_(False)
    vae.to(device)
    text_encoder.requires_grad_(False)
    text_encoder.to(device)
    text_encoder.eval()
    unet.to(device)
    unet.train()
    if cfg.task == "video":
        # diffusers <= 0.38 ships UNet3DConditionModel with
        # _supports_gradient_checkpointing=False (the 3D blocks have no
        # checkpoint plumbing). Try to enable it for newer versions; ignore
        # the failure on older ones — at F=16, 256, fp32 with LoRA we still
        # fit comfortably in 24 GB+ without checkpointing.
        try:
            unet.enable_gradient_checkpointing()
            print("Enabled gradient checkpointing on UNet3D.")
        except (ValueError, AttributeError) as e:
            print(f"Gradient checkpointing unavailable on this UNet3D: {e}. Continuing without it.")

    if cfg.lora.enabled:
        unet.requires_grad_(False)
        lora_params = patch_unet_with_lora(unet, rank=cfg.lora.rank, alpha=cfg.lora.alpha)
        unet.to(device)
        trainable_params = lora_params
        print(f"LoRA enabled: rank={cfg.lora.rank}, alpha={cfg.lora.alpha}")
    else:
        trainable_params = list(unet.parameters())

    trainable_param_count = sum(p.numel() for p in trainable_params if p.requires_grad)
    unet_total_param_count = sum(p.numel() for p in unet.parameters())
    unet_trainable_param_count = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    print(
        f"Trainable optimizer params: {trainable_param_count:,} "
        f"({trainable_param_count / unet_total_param_count:.2%} of UNet params)"
    )
    print(
        f"UNet params: total={unet_total_param_count:,}, "
        f"trainable={unet_trainable_param_count:,}"
    )

    optimizer_cls = bnb.optim.AdamW8bit if HAS_BNB else torch.optim.AdamW
    optimizer = optimizer_cls(trainable_params, lr=cfg.training.learning_rate)

    if cfg.task == "image":
        dataset = DreamBoothDataset(
            instance_data_dir=cfg.data.instance_data_dir,
            class_images_npz=cfg.data.class_images_npz,
            instance_prompt=cfg.data.instance_prompt,
            class_prompt=cfg.data.class_prompt,
            tokenizer=tokenizer,
            resolution=cfg.data.resolution,
        )
    else:
        dataset = VideoDreamBoothDataset(
            instance_video_dir=cfg.data.instance_video_dir,
            instance_prompt=cfg.data.instance_prompt,
            tokenizer=tokenizer,
            resolution=cfg.data.resolution,
            num_frames=cfg.data.num_frames,
            frame_stride=cfg.data.frame_stride,
        )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.train_batch_size,
        shuffle=True,
        num_workers=0,
    )

    progress_bar = tqdm(range(cfg.training.max_train_steps), desc="Training")
    global_step = 0
    recent_losses = deque(maxlen=cfg.training.log_every_steps)

    while global_step < cfg.training.max_train_steps:
        for batch in dataloader:
            if global_step >= cfg.training.max_train_steps:
                break

            instance_prompt_ids = batch["instance_prompt_ids"].to(device)
            encoder_hidden_states_instance = text_encoder(instance_prompt_ids)[0]

            if cfg.task == "image":
                instance_images = batch["instance_images"].to(device, dtype=torch.float32)
                class_images = batch["class_images"].to(device, dtype=torch.float32)
                class_prompt_ids = batch["class_prompt_ids"].to(device)

                with torch.no_grad():
                    latents_instance = vae.encode(instance_images).latent_dist.sample() * vae.config.scaling_factor
                    latents_class = vae.encode(class_images).latent_dist.sample() * vae.config.scaling_factor

                noise_instance = torch.randn_like(latents_instance)
                noise_class = torch.randn_like(latents_class)

                timesteps_instance = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents_instance.shape[0],), device=device).long()
                timesteps_class = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents_class.shape[0],), device=device).long()

                noisy_latents_instance = noise_scheduler.add_noise(latents_instance, noise_instance, timesteps_instance)
                noisy_latents_class = noise_scheduler.add_noise(latents_class, noise_class, timesteps_class)

                encoder_hidden_states_class = text_encoder(class_prompt_ids)[0]

                noise_pred_instance = unet(noisy_latents_instance, timesteps_instance, encoder_hidden_states_instance).sample
                noise_pred_class = unet(noisy_latents_class, timesteps_class, encoder_hidden_states_class).sample

                loss_instance = F.mse_loss(noise_pred_instance, noise_instance, reduction="mean")
                loss_class = F.mse_loss(noise_pred_class, noise_class, reduction="mean")

                loss = loss_instance + cfg.training.prior_loss_weight * loss_class
                step_losses = {
                    "loss": loss.item(),
                    "loss_inst": loss_instance.item(),
                    "loss_cls": loss_class.item(),
                }
            else:
                instance_video = batch["instance_video"].to(device, dtype=torch.float32)

                with torch.no_grad():
                    latents_instance = _video_vae_encode(
                        vae, instance_video, cfg.training.vae_encode_chunk_size
                    )

                noise_instance = torch.randn_like(latents_instance)
                # One timestep per batch element (NOT per frame): the temporal
                # blocks expect a coherent denoising step across the clip.
                timesteps_instance = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (latents_instance.shape[0],),
                    device=device,
                ).long()
                noisy_latents_instance = noise_scheduler.add_noise(
                    latents_instance, noise_instance, timesteps_instance
                )
                noise_pred_instance = unet(
                    noisy_latents_instance, timesteps_instance, encoder_hidden_states_instance
                ).sample
                loss_instance = F.mse_loss(noise_pred_instance, noise_instance, reduction="mean")
                loss = loss_instance
                step_losses = {
                    "loss": loss.item(),
                    "loss_inst": loss_instance.item(),
                    "loss_cls": 0.0,
                }

            loss.backward()
            recent_losses.append(step_losses)

            if (global_step + 1) % cfg.training.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            global_step += 1
            avg_loss = sum(item["loss"] for item in recent_losses) / len(recent_losses)
            avg_inst = sum(item["loss_inst"] for item in recent_losses) / len(recent_losses)
            avg_cls = sum(item["loss_cls"] for item in recent_losses) / len(recent_losses)
            progress_bar.set_postfix(
                loss=step_losses["loss"],
                avg_loss=avg_loss,
                avg_inst=avg_inst,
                avg_cls=avg_cls,
            )
            if global_step % cfg.training.log_every_steps == 0 or global_step == cfg.training.max_train_steps:
                tqdm.write(
                    f"step {global_step:05d}/{cfg.training.max_train_steps} "
                    f"loss={step_losses['loss']:.4f} "
                    f"avg_loss={avg_loss:.4f} "
                    f"avg_inst={avg_inst:.4f} "
                    f"avg_cls={avg_cls:.4f}"
                )

    output_dir = Path(cfg.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if cfg.lora.enabled:
        lora_state = get_lora_state_dict(unet)
        lora_summary = summarize_lora_state_dict(lora_state)
        torch.save(lora_state, output_dir / "lora_weights.pt")
        with open(output_dir / "lora_config.json", "w") as f:
            json.dump({
                "task": cfg.task,
                "rank": cfg.lora.rank,
                "alpha": cfg.lora.alpha,
                "pretrained_model": cfg.model.pretrained_model,
            }, f, indent=2)
        print(
            f"LoRA weights saved to {output_dir} "
            f"(tensors={lora_summary['num_tensors']}, "
            f"mean_abs={lora_summary['mean_abs']:.6f}, "
            f"max_abs={lora_summary['max_abs']:.6f}, "
            f"sha256={lora_summary['sha256']})"
        )
    else:
        if cfg.task == "image":
            pipeline = StableDiffusionPipeline.from_pretrained(
                cfg.model.pretrained_model,
                unet=unet,
                text_encoder=text_encoder,
                safety_checker=None,
            )
        else:
            pipeline = TextToVideoSDPipeline.from_pretrained(
                cfg.model.pretrained_model,
                unet=unet,
                text_encoder=text_encoder,
            )
        pipeline.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    main()
