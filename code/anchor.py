"""Classifier-guidance-style biasing toward a VAE latent of the subject images.

Image-only inference-time experiment. After each scheduler step:
    latents <- (1 - w_t) * latents + w_t * noised_anchor
where the anchor is selected per `mode` and w_t is computed per `schedule`.
Adaptive modes can instead choose a nearest or soft-nearest target from all
subject latents at the current timestep.

Activated by setting `anchor.enabled: true` in the generate config. When
disabled, none of this code runs and the standard SD inference path is used.
"""
from pathlib import Path

import torch
from hydra.utils import to_absolute_path


MODES = {"pooled", "random_per_sample", "random_per_step", "nearest", "soft_nearest"}
SCHEDULES = {"constant", "linear_decay", "early_only"}


def validate(cfg):
    if cfg.anchor.mode not in MODES:
        raise ValueError(f"anchor.mode must be one of {sorted(MODES)}, got {cfg.anchor.mode!r}")
    if cfg.anchor.schedule not in SCHEDULES:
        raise ValueError(f"anchor.schedule must be one of {sorted(SCHEDULES)}, got {cfg.anchor.schedule!r}")
    if cfg.anchor.weight < 0:
        raise ValueError("anchor.weight must be non-negative")
    if cfg.anchor.mode == "soft_nearest" and cfg.anchor.temperature <= 0:
        raise ValueError("anchor.temperature must be positive when anchor.mode='soft_nearest'")


def resolve_paths(cfg):
    cfg.anchor.subject_dir = to_absolute_path(cfg.anchor.subject_dir)


def _load_subject_latents(
    subject_dir: str,
    pipeline,
    device: torch.device,
    weight_dtype: torch.dtype,
    height: int,
    width: int,
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
            img = img.crop((left, top, left + side, top + side)).resize((width, height), Image.LANCZOS)
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
    """Returns the anchor for legacy modes that fix it before the first step."""
    if mode == "pooled":
        return all_latents.mean(dim=0, keepdim=True)
    if mode == "random_per_sample":
        idx = torch.randint(
            0, all_latents.shape[0], (1,),
            device=all_latents.device,
            generator=anchor_generator,
        ).item()
        return all_latents[idx:idx + 1]
    if mode in {"random_per_step", "nearest", "soft_nearest"}:
        return None
    raise ValueError(f"unknown anchor mode: {mode!r}")


def _expand_timestep(timestep, batch_size: int, device: torch.device) -> torch.Tensor:
    timestep = torch.as_tensor(timestep, device=device)
    if timestep.ndim == 0:
        return timestep.expand(batch_size)
    if timestep.shape[0] == batch_size:
        return timestep
    if timestep.shape[0] == 1:
        return timestep.expand(batch_size)
    raise ValueError(f"Cannot expand timestep with shape {tuple(timestep.shape)} to batch={batch_size}")


def _noise_all_anchors(all_latents, scheduler, timestep, noise):
    """Forward-noise every subject anchor at `timestep`.

    Supports `noise` with shape [N, C, H, W] or [B, N, C, H, W]. The latter
    gives each generated sample its own coherent noise trajectory per anchor.
    """
    if noise.shape == all_latents.shape:
        timesteps = _expand_timestep(timestep, all_latents.shape[0], all_latents.device)
        return scheduler.add_noise(all_latents, noise, timesteps)

    expected_tail = all_latents.shape
    if noise.ndim != all_latents.ndim + 1 or noise.shape[1:] != expected_tail:
        raise ValueError(
            f"noise must have shape {tuple(all_latents.shape)} or "
            f"(B, {', '.join(str(x) for x in all_latents.shape)}), got {tuple(noise.shape)}"
        )

    batch_size, num_anchors = noise.shape[:2]
    anchors = all_latents.unsqueeze(0).expand(batch_size, *all_latents.shape)
    flat_anchors = anchors.reshape(batch_size * num_anchors, *all_latents.shape[1:])
    flat_noise = noise.reshape(batch_size * num_anchors, *all_latents.shape[1:])
    timesteps = _expand_timestep(timestep, batch_size * num_anchors, all_latents.device)
    noised = scheduler.add_noise(flat_anchors, flat_noise, timesteps)
    return noised.reshape(batch_size, num_anchors, *all_latents.shape[1:])


def _squared_anchor_distances(latents, noised_anchors) -> torch.Tensor:
    """Mean squared latent distances.

    Nearest selection is unchanged by using mean instead of sum, while
    soft-nearest temperatures remain meaningful across latent resolutions.
    """
    latents_f = latents.float()
    anchors_f = noised_anchors.float()
    if noised_anchors.ndim == latents.ndim:
        diff = latents_f[:, None] - anchors_f[None]
    elif noised_anchors.ndim == latents.ndim + 1:
        if noised_anchors.shape[0] != latents.shape[0]:
            raise ValueError(
                f"Batched anchors have batch={noised_anchors.shape[0]} but latents have batch={latents.shape[0]}"
            )
        diff = latents_f[:, None] - anchors_f
    else:
        raise ValueError(f"Unsupported noised anchor shape: {tuple(noised_anchors.shape)}")
    return diff.flatten(start_dim=2).pow(2).mean(dim=-1)


def _select_nearest_anchor(latents, noised_anchors):
    """Return the closest noised anchor for each sample as [B, C, H, W]."""
    distances = _squared_anchor_distances(latents, noised_anchors)
    nearest_idx = distances.argmin(dim=1)
    if noised_anchors.ndim == latents.ndim:
        return noised_anchors[nearest_idx]
    gather_idx = nearest_idx.view(-1, 1, 1, 1, 1).expand(-1, 1, *noised_anchors.shape[2:])
    return noised_anchors.gather(dim=1, index=gather_idx).squeeze(1)


def _select_soft_nearest_anchor(latents, noised_anchors, temperature: float):
    """Return a softmax-weighted nearest anchor target as [B, C, H, W]."""
    distances = _squared_anchor_distances(latents, noised_anchors)
    weights = torch.softmax(-distances / temperature, dim=1).to(dtype=noised_anchors.dtype)
    if noised_anchors.ndim == latents.ndim:
        return torch.einsum("bn,nchw->bchw", weights, noised_anchors)
    return (weights[:, :, None, None, None] * noised_anchors).sum(dim=1)


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
    temperature,
    consistent_noise,
    total_steps,
    scheduler,
    anchor_generator,
):
    """Returns a diffusers callback_on_step_end that nudges latents toward a noised subject latent."""
    state = {
        "anchor_noise": None,
    }

    def _get_all_anchor_noise(latents):
        noise_shape = (latents.shape[0], *all_latents.shape)
        if consistent_noise:
            cached = state.get("anchor_noise")
            if cached is None or tuple(cached.shape) != tuple(noise_shape):
                state["anchor_noise"] = torch.randn(
                    noise_shape,
                    device=latents.device,
                    dtype=latents.dtype,
                    generator=anchor_generator,
                )
            return state["anchor_noise"]
        return torch.randn(
            noise_shape,
            device=latents.device,
            dtype=latents.dtype,
            generator=anchor_generator,
        )

    def _noise_selected_anchor(anchor, latents, timestep):
        if anchor.shape[0] != latents.shape[0]:
            anchor = anchor.expand(latents.shape[0], *anchor.shape[1:])
        noise = torch.randn(
            latents.shape,
            device=latents.device,
            dtype=latents.dtype,
            generator=anchor_generator,
        )
        timesteps = _expand_timestep(timestep, latents.shape[0], latents.device)
        return scheduler.add_noise(anchor, noise, timesteps)

    def _callback(pipe, step_idx, timestep, callback_kwargs):
        latents = callback_kwargs["latents"]
        w = _resolve_step_weight(base_weight, schedule, step_idx, total_steps)
        if w == 0.0:
            return {"latents": latents}

        subject_latents = all_latents.to(device=latents.device, dtype=latents.dtype)

        if mode in {"nearest", "soft_nearest"} or consistent_noise:
            anchor_noise = _get_all_anchor_noise(latents)
            noised_anchors = _noise_all_anchors(subject_latents, scheduler, timestep, anchor_noise)

            if mode == "nearest":
                noised_anchor = _select_nearest_anchor(latents, noised_anchors)
            elif mode == "soft_nearest":
                noised_anchor = _select_soft_nearest_anchor(latents, noised_anchors, temperature)
            elif mode == "pooled":
                noised_anchor = noised_anchors.mean(dim=1)
            elif mode == "random_per_sample":
                if fixed_anchor is None:
                    raise RuntimeError("random_per_sample requires a fixed anchor")
                distances = _squared_anchor_distances(fixed_anchor.to(latents.device, latents.dtype), subject_latents)
                idx = distances.argmin(dim=1).expand(latents.shape[0])
                gather_idx = idx.view(-1, 1, 1, 1, 1).expand(-1, 1, *noised_anchors.shape[2:])
                noised_anchor = noised_anchors.gather(dim=1, index=gather_idx).squeeze(1)
            elif mode == "random_per_step":
                idx = torch.randint(
                    0,
                    subject_latents.shape[0],
                    (latents.shape[0],),
                    device=latents.device,
                    generator=anchor_generator,
                )
                gather_idx = idx.view(-1, 1, 1, 1, 1).expand(-1, 1, *noised_anchors.shape[2:])
                noised_anchor = noised_anchors.gather(dim=1, index=gather_idx).squeeze(1)
            else:
                raise ValueError(f"unknown anchor mode: {mode!r}")
            return {"latents": (1.0 - w) * latents + w * noised_anchor}

        if mode == "random_per_step":
            idx = torch.randint(
                0, subject_latents.shape[0], (1,),
                device=latents.device,
                generator=anchor_generator,
            ).item()
            anchor = subject_latents[idx:idx + 1]
        else:
            anchor = fixed_anchor

        anchor = anchor.to(device=latents.device, dtype=latents.dtype)
        noised_anchor = _noise_selected_anchor(anchor, latents, timestep)
        return {"latents": (1.0 - w) * latents + w * noised_anchor}
    return _callback


def prepare(cfg, pipeline, device, weight_dtype):
    """Encode subject images and log the active anchor settings."""
    subject_latents = _load_subject_latents(
        cfg.anchor.subject_dir,
        pipeline,
        device,
        weight_dtype,
        int(cfg.inference.height),
        int(cfg.inference.width),
    )
    print(
        f"Anchor settings: mode={cfg.anchor.mode}, weight={cfg.anchor.weight}, "
        f"schedule={cfg.anchor.schedule}, temperature={cfg.anchor.temperature}, "
        f"consistent_noise={cfg.anchor.consistent_noise}"
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
        float(cfg.anchor.temperature),
        bool(cfg.anchor.consistent_noise),
        num_inference_steps,
        scheduler,
        anchor_gen,
    )
    return {
        "callback_on_step_end": callback,
        "callback_on_step_end_tensor_inputs": ["latents"],
    }
