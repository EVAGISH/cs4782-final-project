"""Classifier-guidance-style biasing toward a VAE latent of the subject images.

Image-only inference-time experiment. After each scheduler step:
    latents <- (1 - w_t) * latents + w_t * noised_anchor
where the anchor is selected per `mode` and w_t is computed per `schedule`.

Activated by setting `anchor.enabled: true` in the generate config. When
disabled, none of this code runs and the standard SD inference path is used.
"""
from pathlib import Path

import torch
from hydra.utils import to_absolute_path


MODES = {"pooled", "random_per_sample", "random_per_step"}
SCHEDULES = {"constant", "linear_decay", "early_only"}


def validate(cfg):
    if cfg.anchor.mode not in MODES:
        raise ValueError(f"anchor.mode must be one of {sorted(MODES)}, got {cfg.anchor.mode!r}")
    if cfg.anchor.schedule not in SCHEDULES:
        raise ValueError(f"anchor.schedule must be one of {sorted(SCHEDULES)}, got {cfg.anchor.schedule!r}")


def resolve_paths(cfg):
    cfg.anchor.subject_dir = to_absolute_path(cfg.anchor.subject_dir)


def _load_subject_latents(
    subject_dir: str,
    pipeline,
    device: torch.device,
    weight_dtype: torch.dtype,
) -> torch.Tensor:
    """Encode every image in subject_dir through the VAE. Returns [N, 4, H/8, W/8]."""
    import numpy as np
    from PIL import Image

    subject_path = Path(subject_dir)
    if not subject_path.is_dir():
        raise FileNotFoundError(f"anchor.subject_dir does not exist: {subject_dir}")
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    image_paths = sorted(p for p in subject_path.iterdir() if p.suffix.lower() in exts)
    if not image_paths:
        raise ValueError(f"No images with extensions {exts} in {subject_dir}")

    scaling_factor = pipeline.vae.config.scaling_factor
    latents = []
    with torch.no_grad():
        for path in image_paths:
            img = Image.open(path).convert("RGB")
            w, h = img.size
            side = min(w, h)
            left = (w - side) // 2
            top = (h - side) // 2
            img = img.crop((left, top, left + side, top + side)).resize((512, 512), Image.LANCZOS)
            arr = torch.from_numpy(np.array(img)).float() / 127.5 - 1.0  # HxWx3 in [-1, 1]
            tensor = arr.permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=weight_dtype)
            latent = pipeline.vae.encode(tensor).latent_dist.mean * scaling_factor
            latents.append(latent)
    stacked = torch.cat(latents, dim=0)  # [N, 4, H/8, W/8]
    print(
        f"Loaded {stacked.shape[0]} subject latents from {subject_dir} "
        f"(per-latent shape={tuple(stacked.shape[1:])})"
    )
    return stacked


def _select_anchor_for_sample(all_latents, mode, anchor_generator):
    """Returns the anchor for modes that fix it per-sample, or None for per-step modes."""
    if mode == "pooled":
        return all_latents.mean(dim=0, keepdim=True)
    if mode == "random_per_sample":
        idx = torch.randint(
            0, all_latents.shape[0], (1,),
            device=all_latents.device,
            generator=anchor_generator,
        ).item()
        return all_latents[idx:idx + 1]
    if mode == "random_per_step":
        return None
    raise ValueError(f"unknown anchor mode: {mode!r}")


def _resolve_step_weight(base_weight: float, schedule: str, step_idx: int, total_steps: int) -> float:
    if schedule == "constant":
        return base_weight
    if schedule == "linear_decay":
        if total_steps <= 1:
            return base_weight
        return base_weight * (1.0 - step_idx / (total_steps - 1))
    if schedule == "early_only":
        return base_weight if step_idx < total_steps / 2 else 0.0
    raise ValueError(f"unknown anchor schedule: {schedule!r}")


def _make_anchor_callback(
    all_latents,
    fixed_anchor,
    mode,
    base_weight,
    schedule,
    total_steps,
    scheduler,
    anchor_generator,
):
    """Returns a diffusers callback_on_step_end that nudges latents toward a noised subject latent."""
    def _callback(pipe, step_idx, timestep, callback_kwargs):
        latents = callback_kwargs["latents"]
        w = _resolve_step_weight(base_weight, schedule, step_idx, total_steps)
        if w == 0.0:
            return {"latents": latents}

        if mode == "random_per_step":
            idx = torch.randint(
                0, all_latents.shape[0], (1,),
                device=all_latents.device,
                generator=anchor_generator,
            ).item()
            anchor = all_latents[idx:idx + 1]
        else:
            anchor = fixed_anchor

        anchor = anchor.to(device=latents.device, dtype=latents.dtype)
        if anchor.shape[0] != latents.shape[0]:
            anchor = anchor.expand(latents.shape[0], *anchor.shape[1:])
        noise = torch.randn(
            latents.shape,
            device=latents.device,
            dtype=latents.dtype,
            generator=anchor_generator,
        )
        noised_anchor = scheduler.add_noise(anchor, noise, timestep)
        return {"latents": (1.0 - w) * latents + w * noised_anchor}
    return _callback


def prepare(cfg, pipeline, device, weight_dtype):
    """Encode subject images and log the active anchor settings."""
    subject_latents = _load_subject_latents(cfg.anchor.subject_dir, pipeline, device, weight_dtype)
    print(
        f"Anchor settings: mode={cfg.anchor.mode}, weight={cfg.anchor.weight}, "
        f"schedule={cfg.anchor.schedule}"
    )
    return subject_latents


def build_callback_kwargs(cfg, subject_latents, scheduler, num_inference_steps, device, seed):
    """Build per-sample diffusers callback kwargs that splat into pipeline(...)."""
    anchor_gen = torch.Generator(device).manual_seed(seed + 9000)
    fixed_anchor = _select_anchor_for_sample(subject_latents, cfg.anchor.mode, anchor_gen)
    callback = _make_anchor_callback(
        subject_latents,
        fixed_anchor,
        cfg.anchor.mode,
        float(cfg.anchor.weight),
        cfg.anchor.schedule,
        num_inference_steps,
        scheduler,
        anchor_gen,
    )
    return {
        "callback_on_step_end": callback,
        "callback_on_step_end_tensor_inputs": ["latents"],
    }
