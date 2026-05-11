import json
import random
from collections import deque
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from diffusers import (
    AutoencoderKLFlux2,
    Flux2KleinPipeline,
    Flux2Transformer2DModel,
    FlowMatchEulerDiscreteScheduler,
)
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM

from class_priors import load_class_prior_images
from lora_flux2 import (
    get_lora_state_dict,
    patch_flux2_transformer_with_lora,
    summarize_lora_state_dict,
)

try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False


WEIGHT_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
WEIGHTING_SCHEMES = {"uniform", "logit_normal"}
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
    if cfg.training.max_train_steps < 1:
        raise ValueError("training.max_train_steps must be at least 1")
    if cfg.training.log_every_steps < 1:
        raise ValueError("training.log_every_steps must be at least 1")
    if cfg.data.resolution % 16 != 0:
        raise ValueError("data.resolution must be divisible by 16 (FLUX2 vae_scale_factor * 2)")
    if cfg.training.weighting_scheme not in WEIGHTING_SCHEMES:
        raise ValueError(f"training.weighting_scheme must be one of {sorted(WEIGHTING_SCHEMES)}")
    if cfg.training.weight_dtype not in WEIGHT_DTYPES:
        raise ValueError(f"training.weight_dtype must be one of {sorted(WEIGHT_DTYPES)}")


def resolve_config_paths(cfg: DictConfig):
    cfg.data.instance_data_dir = to_absolute_path(cfg.data.instance_data_dir)
    cfg.data.class_images_npz = to_absolute_path(cfg.data.class_images_npz)
    cfg.training.output_dir = to_absolute_path(cfg.training.output_dir)


class Flux2DreamBoothDataset(Dataset):
    """Returns instance + class image tensors in [-1, 1]. Text is encoded once outside the loop."""

    def __init__(self, instance_data_dir, class_images_npz, resolution=512):
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
        class_image = Image.fromarray(self._class_stack[index % self.num_class_images], mode="RGB")
        return {
            "instance_images": self.image_transforms(instance_image),
            "class_images": self.image_transforms(class_image),
        }


@torch.no_grad()
def _encode_text_qwen3(text_encoder, tokenizer, prompts, device, dtype, max_seq_len):
    embeds = Flux2KleinPipeline._get_qwen3_prompt_embeds(
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        prompt=prompts,
        dtype=dtype,
        device=device,
        max_sequence_length=max_seq_len,
        hidden_states_layers=(9, 18, 27),
    )
    text_ids = Flux2KleinPipeline._prepare_text_ids(embeds).to(device)
    return embeds, text_ids


def _encode_vae_train(vae, image):
    """Replicates Flux2KleinPipeline._encode_vae_image but uses latent_dist.sample() for training."""
    z = vae.encode(image).latent_dist.sample()
    z = Flux2KleinPipeline._patchify_latents(z)  # (B, C*4, H/2, W/2)
    bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(z.device, z.dtype)
    bn_std = torch.sqrt(
        vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps
    ).to(z.device, z.dtype)
    return (z - bn_mean) / bn_std


def _sample_continuous_timestep(batch_size, scheme, device, dtype):
    if scheme == "uniform":
        return torch.rand(batch_size, device=device, dtype=dtype)
    if scheme == "logit_normal":
        return torch.sigmoid(torch.randn(batch_size, device=device, dtype=dtype))
    raise ValueError(f"unknown weighting_scheme: {scheme!r}")


@hydra.main(version_base=None, config_path="conf/flux2", config_name="train_dreambooth")
def main(cfg: DictConfig):
    validate_config(cfg)
    resolve_config_paths(cfg)
    print(OmegaConf.to_yaml(cfg))

    device = get_device()
    weight_dtype = WEIGHT_DTYPES[cfg.training.weight_dtype]
    print(f"Using device: {device}")
    print(f"Using dtype: {weight_dtype}")

    torch.manual_seed(cfg.training.seed)
    random.seed(cfg.training.seed)

    model = cfg.model.pretrained_model

    # Load components separately to keep refs and avoid full pipeline construction cost.
    tokenizer = Qwen2TokenizerFast.from_pretrained(model, subfolder="tokenizer")
    text_encoder = Qwen3ForCausalLM.from_pretrained(
        model, subfolder="text_encoder", torch_dtype=weight_dtype
    )
    vae = AutoencoderKLFlux2.from_pretrained(model, subfolder="vae", torch_dtype=weight_dtype)
    transformer = Flux2Transformer2DModel.from_pretrained(
        model, subfolder="transformer", torch_dtype=weight_dtype
    )
    # Scheduler is not used in the training loop (we compute sigma manually) but loaded
    # for completeness / future use.
    _ = FlowMatchEulerDiscreteScheduler.from_pretrained(model, subfolder="scheduler")

    vae.requires_grad_(False)
    vae.eval()
    vae.to(device)
    text_encoder.requires_grad_(False)
    text_encoder.eval()
    text_encoder.to(device)

    # DreamBooth uses a single fixed instance prompt and class prompt. Encode each ONCE,
    # then drop the text encoder to free ~3-4 GB of VRAM.
    instance_embeds, instance_text_ids = _encode_text_qwen3(
        text_encoder, tokenizer,
        [cfg.data.instance_prompt], device, weight_dtype, cfg.data.max_sequence_length,
    )
    class_embeds, class_text_ids = _encode_text_qwen3(
        text_encoder, tokenizer,
        [cfg.data.class_prompt], device, weight_dtype, cfg.data.max_sequence_length,
    )
    print(
        f"Cached text embeddings: instance={tuple(instance_embeds.shape)}, "
        f"class={tuple(class_embeds.shape)}. Dropping text encoder to CPU."
    )
    text_encoder.to("cpu")
    del text_encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()

    transformer.requires_grad_(False)

    if cfg.lora.enabled:
        lora_params = patch_flux2_transformer_with_lora(
            transformer, rank=cfg.lora.rank, alpha=cfg.lora.alpha
        )
        # Move base + freshly-added LoRA params to device first, then upcast LoRA to fp32
        # for optimizer stability (base stays in bf16).
        transformer.to(device)
        for p in lora_params:
            p.data = p.data.to(device=device, dtype=torch.float32)
            p.requires_grad_(True)
        trainable_params = lora_params
        print(f"LoRA enabled: rank={cfg.lora.rank}, alpha={cfg.lora.alpha}")
    else:
        # Full finetune: rare for FLUX2 4B at this scale, but supported.
        transformer.to(device)
        transformer.requires_grad_(True)
        trainable_params = list(transformer.parameters())

    transformer.train()
    if cfg.training.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        print("Enabled gradient checkpointing on Flux2Transformer2DModel.")

    trainable_count = sum(p.numel() for p in trainable_params if p.requires_grad)
    transformer_total = sum(p.numel() for p in transformer.parameters())
    print(
        f"Trainable optimizer params: {trainable_count:,} "
        f"({trainable_count / transformer_total:.4%} of transformer params)"
    )

    optimizer_cls = bnb.optim.AdamW8bit if HAS_BNB else torch.optim.AdamW
    optimizer = optimizer_cls(trainable_params, lr=cfg.training.learning_rate)

    dataset = Flux2DreamBoothDataset(
        instance_data_dir=cfg.data.instance_data_dir,
        class_images_npz=cfg.data.class_images_npz,
        resolution=cfg.data.resolution,
    )
    dataloader = DataLoader(
        dataset, batch_size=cfg.training.train_batch_size, shuffle=True, num_workers=0,
    )

    progress_bar = tqdm(range(cfg.training.max_train_steps), desc="Training")
    global_step = 0
    recent_losses = deque(maxlen=cfg.training.log_every_steps)

    while global_step < cfg.training.max_train_steps:
        for batch in dataloader:
            if global_step >= cfg.training.max_train_steps:
                break

            instance_images = batch["instance_images"].to(device, dtype=weight_dtype)
            class_images = batch["class_images"].to(device, dtype=weight_dtype)
            B = instance_images.shape[0]

            with torch.no_grad():
                z_inst = _encode_vae_train(vae, instance_images)  # (B, C, H/2, W/2)
                z_cls = _encode_vae_train(vae, class_images)

            # Latent IDs (T,H,W,L position coords, shape (1, H*W, 4)). The transformer
            # drops the batch dim internally (txt_ids[0]/img_ids[0]), so size-1 is fine.
            latent_ids_inst = Flux2KleinPipeline._prepare_latent_ids(z_inst).to(device)
            latent_ids_cls = Flux2KleinPipeline._prepare_latent_ids(z_cls).to(device)
            z_inst_packed = Flux2KleinPipeline._pack_latents(z_inst)  # (B, H*W, C)
            z_cls_packed = Flux2KleinPipeline._pack_latents(z_cls)

            t_inst = _sample_continuous_timestep(B, cfg.training.weighting_scheme, device, weight_dtype)
            t_cls = _sample_continuous_timestep(B, cfg.training.weighting_scheme, device, weight_dtype)

            noise_inst = torch.randn_like(z_inst_packed)
            noise_cls = torch.randn_like(z_cls_packed)

            # Rectified flow forward process: noisy = sigma * noise + (1 - sigma) * clean,
            # with sigma = t. Equivalent to FlowMatchEulerDiscreteScheduler.scale_noise but
            # without the inference-time set_timesteps/step_indices machinery.
            sigma_inst = t_inst.view(B, 1, 1)
            sigma_cls = t_cls.view(B, 1, 1)
            noisy_inst = sigma_inst * noise_inst + (1.0 - sigma_inst) * z_inst_packed
            noisy_cls = sigma_cls * noise_cls + (1.0 - sigma_cls) * z_cls_packed

            enc_inst = instance_embeds.expand(B, -1, -1)
            enc_cls = class_embeds.expand(B, -1, -1)

            # Klein is distilled (is_distilled=True); always pass guidance=None at training
            # and inference time.
            pred_inst = transformer(
                hidden_states=noisy_inst,
                timestep=t_inst,
                guidance=None,
                encoder_hidden_states=enc_inst,
                txt_ids=instance_text_ids,
                img_ids=latent_ids_inst,
                return_dict=False,
            )[0]
            pred_cls = transformer(
                hidden_states=noisy_cls,
                timestep=t_cls,
                guidance=None,
                encoder_hidden_states=enc_cls,
                txt_ids=class_text_ids,
                img_ids=latent_ids_cls,
                return_dict=False,
            )[0]

            # Rectified-flow target: velocity = noise - clean.
            target_inst = noise_inst - z_inst_packed
            target_cls = noise_cls - z_cls_packed

            loss_inst = F.mse_loss(pred_inst.float(), target_inst.float(), reduction="mean")
            loss_cls = F.mse_loss(pred_cls.float(), target_cls.float(), reduction="mean")
            loss = loss_inst + cfg.training.prior_loss_weight * loss_cls

            loss.backward()
            step_losses = {
                "loss": loss.item(),
                "loss_inst": loss_inst.item(),
                "loss_cls": loss_cls.item(),
            }
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
        lora_state = get_lora_state_dict(transformer)
        lora_summary = summarize_lora_state_dict(lora_state)
        torch.save(lora_state, output_dir / "lora_weights.pt")
        with open(output_dir / "lora_config.json", "w") as f:
            json.dump({
                "task": "image",
                "model_kind": "flux2_klein",
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
        # Full finetune: save the transformer subfolder. Pipeline can be reassembled later.
        transformer.save_pretrained(output_dir / "transformer")
        with open(output_dir / "lora_config.json", "w") as f:
            json.dump({
                "task": "image",
                "model_kind": "flux2_klein_full_finetune",
                "pretrained_model": cfg.model.pretrained_model,
            }, f, indent=2)
        print(f"Full transformer saved to {output_dir / 'transformer'}")


if __name__ == "__main__":
    main()
