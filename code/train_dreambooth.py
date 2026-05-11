import json
import time
from collections import deque
from datetime import datetime, timezone
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
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm

from class_priors import load_class_prior_images
from hydra_compat import patch_argparse_help_for_hydra
from lora import get_lora_state_dict, patch_unet_with_lora, summarize_lora_state_dict


patch_argparse_help_for_hydra()

try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False


INSTANCE_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _list_instance_image_paths(directory: Path) -> list[Path]:
    paths = [
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in INSTANCE_IMAGE_SUFFIXES
    ]
    return sorted(paths)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def validate_config(cfg: DictConfig):
    if cfg.task != "image":
        raise ValueError(f"Only image DreamBooth training is supported, got task={cfg.task!r}")
    if cfg.training.max_train_steps < 1:
        raise ValueError("training.max_train_steps must be at least 1")
    if cfg.training.train_batch_size < 1:
        raise ValueError("training.train_batch_size must be at least 1")
    if cfg.training.gradient_accumulation_steps < 1:
        raise ValueError("training.gradient_accumulation_steps must be at least 1")
    if cfg.training.log_every_steps < 1:
        raise ValueError("training.log_every_steps must be at least 1")


def resolve_config_paths(cfg: DictConfig):
    cfg.data.instance_data_dir = to_absolute_path(cfg.data.instance_data_dir)
    cfg.data.class_images_npz = to_absolute_path(cfg.data.class_images_npz)
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


@hydra.main(version_base=None, config_path="conf", config_name="train_dreambooth")
def main(cfg: DictConfig):
    validate_config(cfg)
    resolve_config_paths(cfg)
    resolved_cfg_yaml = OmegaConf.to_yaml(cfg, resolve=True)
    print(resolved_cfg_yaml)

    output_dir = Path(cfg.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "train_config.yaml", "w") as f:
        f.write(resolved_cfg_yaml)

    device = get_device()
    print(f"Using device: {device}")
    print(f"Using dtype: {torch.float32}")
    started_at = datetime.now(timezone.utc)
    start_time = time.time()

    torch.manual_seed(cfg.training.seed)

    tokenizer = CLIPTokenizer.from_pretrained(cfg.model.pretrained_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(cfg.model.pretrained_model, subfolder="text_encoder", torch_dtype=torch.float32)
    vae = AutoencoderKL.from_pretrained(cfg.model.pretrained_model, subfolder="vae", torch_dtype=torch.float32)
    unet = UNet2DConditionModel.from_pretrained(cfg.model.pretrained_model, subfolder="unet", torch_dtype=torch.float32)
    noise_scheduler = DDPMScheduler.from_pretrained(cfg.model.pretrained_model, subfolder="scheduler")

    vae.requires_grad_(False)
    vae.to(device)
    text_encoder.requires_grad_(False)
    text_encoder.to(device)
    text_encoder.eval()
    unet.to(device)
    unet.train()

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

    dataset = DreamBoothDataset(
        instance_data_dir=cfg.data.instance_data_dir,
        class_images_npz=cfg.data.class_images_npz,
        instance_prompt=cfg.data.instance_prompt,
        class_prompt=cfg.data.class_prompt,
        tokenizer=tokenizer,
        resolution=cfg.data.resolution,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.train_batch_size,
        shuffle=True,
        num_workers=0,
    )

    progress_bar = tqdm(range(cfg.training.max_train_steps), desc="Training")
    global_step = 0
    optimizer_steps = 0
    pending_accumulation_steps = 0
    recent_losses = deque(maxlen=cfg.training.log_every_steps)
    loss_history = []

    while global_step < cfg.training.max_train_steps:
        for batch in dataloader:
            if global_step >= cfg.training.max_train_steps:
                break

            instance_prompt_ids = batch["instance_prompt_ids"].to(device)
            encoder_hidden_states_instance = text_encoder(instance_prompt_ids)[0]

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

            (loss / cfg.training.gradient_accumulation_steps).backward()
            pending_accumulation_steps += 1
            recent_losses.append(step_losses)

            if pending_accumulation_steps == cfg.training.gradient_accumulation_steps:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                optimizer_steps += 1
                pending_accumulation_steps = 0

            loss_history.append({
                "step": global_step,
                "loss": loss.item(),
                "loss_inst": loss_instance.item(),
                "loss_cls": loss_class.item(),
            })

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
            loss_history.append({
                "step": global_step,
                "loss": step_losses["loss"],
                "loss_inst": step_losses["loss_inst"],
                "loss_cls": step_losses["loss_cls"],
                "avg_loss": avg_loss,
                "avg_inst": avg_inst,
                "avg_cls": avg_cls,
            })
            if global_step % cfg.training.log_every_steps == 0 or global_step == cfg.training.max_train_steps:
                tqdm.write(
                    f"step {global_step:05d}/{cfg.training.max_train_steps} "
                    f"loss={step_losses['loss']:.4f} "
                    f"avg_loss={avg_loss:.4f} "
                    f"avg_inst={avg_inst:.4f} "
                    f"avg_cls={avg_cls:.4f}"
                )

    if pending_accumulation_steps:
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        optimizer_steps += 1

    train_summary = {
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": time.time() - start_time,
        "task": cfg.task,
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "global_step": global_step,
        "optimizer_steps": optimizer_steps,
        "gradient_accumulation_steps": cfg.training.gradient_accumulation_steps,
        "max_train_steps": cfg.training.max_train_steps,
        "trainable_optimizer_params": trainable_param_count,
        "unet_total_params": unet_total_param_count,
        "unet_trainable_params": unet_trainable_param_count,
        "final_loss": loss_history[-1]["loss"] if loss_history else None,
        "final_avg_loss": loss_history[-1]["avg_loss"] if loss_history else None,
    }

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
        train_summary["lora"] = lora_summary
    else:
        pipeline = StableDiffusionPipeline.from_pretrained(
            cfg.model.pretrained_model,
            unet=unet,
            text_encoder=text_encoder,
            safety_checker=None,
        )
        pipeline.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")

    with open(output_dir / "loss_history.jsonl", "w") as f:
        for row in loss_history:
            f.write(json.dumps(row) + "\n")
    with open(output_dir / "train_summary.json", "w") as f:
        json.dump(train_summary, f, indent=2)


if __name__ == "__main__":
    main()
