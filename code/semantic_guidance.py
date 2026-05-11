"""Optional inference-time semantic guidance for image generation.

This module implements a lightweight Universal-Guidance-style callback:
it decodes a predicted clean latent, compares DINO features against subject
image features, and nudges the current latent by the feature-loss gradient.
It is image-only and intentionally separate from `anchor.py`.
"""
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import to_absolute_path
from PIL import Image


SUPPORTED_MODELS = {"dino"}
SUPPORTED_TARGETS = {"mean"}
SCHEDULES = {"constant_window", "linear_decay_window", "warmup_hold_decay"}
GRADIENT_NORMALIZATIONS = {"none", "l2", "rms"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

DINO_IMAGE_SIZE = 224
DINO_MEAN = (0.485, 0.456, 0.406)
DINO_STD = (0.229, 0.224, 0.225)


@dataclass
class SemanticGuidanceState:
    model: torch.nn.Module
    target_embedding: torch.Tensor
    model_name: str


@dataclass(frozen=True)
class GuidanceStepOptions:
    gradient_normalization: str = "none"
    normalization_eps: float = 1e-8
    max_update_norm_ratio: float = 0.0
    diagnostics_recompute_loss_after: bool = True


class GuidanceDiagnosticsWriter:
    FIELDNAMES = [
        "step_idx",
        "timestep",
        "weight",
        "gradient_normalization",
        "loss_before",
        "loss_after",
        "raw_grad_norm",
        "raw_grad_rms",
        "direction_norm",
        "direction_rms",
        "update_norm",
        "update_rms",
        "latent_norm",
        "update_to_latent_norm",
        "clamp_scale_min",
        "clamp_scale_mean",
    ]

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists() or self.path.stat().st_size == 0:
            with open(self.path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.FIELDNAMES).writeheader()

    def record(self, row: dict) -> None:
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            writer.writerow({name: row.get(name) for name in self.FIELDNAMES})


def validate(cfg):
    sg = cfg.semantic_guidance
    if sg.model not in SUPPORTED_MODELS:
        raise ValueError(f"semantic_guidance.model must be one of {sorted(SUPPORTED_MODELS)}, got {sg.model!r}")
    if sg.target not in SUPPORTED_TARGETS:
        raise ValueError(f"semantic_guidance.target must be one of {sorted(SUPPORTED_TARGETS)}, got {sg.target!r}")
    if sg.schedule not in SCHEDULES:
        raise ValueError(f"semantic_guidance.schedule must be one of {sorted(SCHEDULES)}, got {sg.schedule!r}")
    if sg.weight < 0:
        raise ValueError("semantic_guidance.weight must be non-negative")
    if sg.gradient_normalization not in GRADIENT_NORMALIZATIONS:
        raise ValueError(
            "semantic_guidance.gradient_normalization must be one of "
            f"{sorted(GRADIENT_NORMALIZATIONS)}, got {sg.gradient_normalization!r}"
        )
    if sg.normalization_eps <= 0:
        raise ValueError("semantic_guidance.normalization_eps must be positive")
    if sg.max_update_norm_ratio < 0:
        raise ValueError("semantic_guidance.max_update_norm_ratio must be non-negative")
    if sg.every_n_steps < 1:
        raise ValueError("semantic_guidance.every_n_steps must be at least 1")
    if not 0 <= sg.start_step_frac <= 1:
        raise ValueError("semantic_guidance.start_step_frac must be in [0, 1]")
    if not 0 <= sg.end_step_frac <= 1:
        raise ValueError("semantic_guidance.end_step_frac must be in [0, 1]")
    if sg.start_step_frac > sg.end_step_frac:
        raise ValueError("semantic_guidance.start_step_frac must be <= end_step_frac")
    if not sg.start_step_frac <= sg.hold_until_frac <= sg.end_step_frac:
        raise ValueError(
            "semantic_guidance.hold_until_frac must be between start_step_frac and end_step_frac"
        )


def resolve_paths(cfg):
    # Reuse the existing subject-dir setting so anchoring and semantic guidance
    # compare against the same reference images.
    cfg.anchor.subject_dir = to_absolute_path(cfg.anchor.subject_dir)


def _list_subject_images(subject_dir: str) -> list[Path]:
    subject_path = Path(subject_dir)
    if not subject_path.is_dir():
        raise FileNotFoundError(f"subject image directory does not exist: {subject_dir}")
    paths = sorted(p for p in subject_path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise ValueError(f"No subject images with extensions {IMAGE_SUFFIXES} in {subject_dir}")
    return paths


def _center_crop_square(img: Image.Image) -> Image.Image:
    width, height = img.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return img.crop((left, top, left + side, top + side))


def _pil_to_unit_tensor(img: Image.Image, size: int) -> torch.Tensor:
    img = _center_crop_square(img.convert("RGB")).resize((size, size), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def _normalize_dino_input(images: torch.Tensor) -> torch.Tensor:
    images = F.interpolate(
        images.float(),
        size=(DINO_IMAGE_SIZE, DINO_IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
    )
    mean = torch.tensor(DINO_MEAN, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    std = torch.tensor(DINO_STD, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    return (images - mean) / std


def _load_subject_tensor(subject_dir: str, device: torch.device) -> torch.Tensor:
    tensors = []
    for path in _list_subject_images(subject_dir):
        with Image.open(path) as img:
            tensors.append(_pil_to_unit_tensor(img, DINO_IMAGE_SIZE))
    return torch.stack(tensors, dim=0).to(device=device, dtype=torch.float32)


def _load_dino_model(device: torch.device) -> torch.nn.Module:
    model = torch.hub.load("facebookresearch/dino:main", "dino_vits16", trust_repo=True)
    model.eval().to(device)
    model.requires_grad_(False)
    return model


def _compute_target_embedding(model, images: torch.Tensor, target: str) -> torch.Tensor:
    if target != "mean":
        raise ValueError(f"Unsupported semantic target: {target!r}")
    with torch.no_grad():
        embeddings = model(_normalize_dino_input(images))
        embeddings = F.normalize(embeddings, dim=-1)
        target_embedding = F.normalize(embeddings.mean(dim=0, keepdim=True), dim=-1)
    return target_embedding.detach()


def prepare(cfg, pipeline, device, weight_dtype) -> SemanticGuidanceState:
    """Load DINO and precompute the subject feature target."""
    del weight_dtype
    if cfg.semantic_guidance.model != "dino":
        raise ValueError(f"Unsupported semantic guidance model: {cfg.semantic_guidance.model!r}")
    callback_inputs = set(getattr(pipeline, "_callback_tensor_inputs", []))
    missing_inputs = {"latents", "prompt_embeds"} - callback_inputs
    if missing_inputs:
        raise RuntimeError(
            "Semantic guidance requires a Diffusers image pipeline whose "
            f"_callback_tensor_inputs include ['latents', 'prompt_embeds']; missing {sorted(missing_inputs)}. "
            "Use a recent StableDiffusionPipeline-compatible Diffusers version."
        )

    model = _load_dino_model(device)
    subject_images = _load_subject_tensor(cfg.anchor.subject_dir, device)
    target_embedding = _compute_target_embedding(model, subject_images, cfg.semantic_guidance.target)

    pipeline.vae.requires_grad_(False)
    pipeline.vae.eval()
    pipeline.unet.requires_grad_(False)
    pipeline.unet.eval()

    print(
        f"Semantic guidance settings: model={cfg.semantic_guidance.model}, "
        f"target={cfg.semantic_guidance.target}, weight={cfg.semantic_guidance.weight}, "
        f"gradient_normalization={cfg.semantic_guidance.gradient_normalization}, "
        f"max_update_norm_ratio={cfg.semantic_guidance.max_update_norm_ratio}, "
        f"schedule={cfg.semantic_guidance.schedule}, "
        f"every_n_steps={cfg.semantic_guidance.every_n_steps}, "
        f"window=[{cfg.semantic_guidance.start_step_frac:.2f}, "
        f"{cfg.semantic_guidance.hold_until_frac:.2f}, "
        f"{cfg.semantic_guidance.end_step_frac:.2f}], "
        f"subject_images={subject_images.shape[0]}"
    )
    return SemanticGuidanceState(
        model=model,
        target_embedding=target_embedding,
        model_name=cfg.semantic_guidance.model,
    )


def _should_apply(step_idx: int, total_steps: int, every_n_steps: int, start_frac: float, end_frac: float) -> bool:
    start_step = int(total_steps * start_frac)
    end_step = int(total_steps * end_frac)
    return start_step <= step_idx <= end_step and step_idx % every_n_steps == 0


def _resolve_step_weight(
    base_weight: float,
    schedule: str,
    step_idx: int,
    total_steps: int,
    start_frac: float,
    hold_until_frac: float,
    end_frac: float,
) -> float:
    start_step = int(total_steps * start_frac)
    hold_step = int(total_steps * hold_until_frac)
    end_step = int(total_steps * end_frac)

    if step_idx < start_step or step_idx > end_step:
        return 0.0
    if schedule == "constant_window":
        return base_weight
    if schedule == "linear_decay_window":
        if end_step <= start_step:
            return base_weight
        progress = (step_idx - start_step) / (end_step - start_step)
        return base_weight * max(0.0, 1.0 - progress)
    if schedule == "warmup_hold_decay":
        if step_idx <= hold_step:
            return base_weight
        if end_step <= hold_step:
            return base_weight
        progress = (step_idx - hold_step) / (end_step - hold_step)
        return base_weight * max(0.0, 1.0 - progress)
    raise ValueError(f"unknown semantic guidance schedule: {schedule!r}")


def _get_guidance_scale(pipe) -> float:
    for attr in ("guidance_scale", "_guidance_scale"):
        value = getattr(pipe, attr, None)
        if value is not None:
            return float(value)
    return 1.0


def _predict_noise(pipe, latents: torch.Tensor, timestep, callback_kwargs: dict) -> torch.Tensor:
    prompt_embeds = callback_kwargs.get("prompt_embeds")
    if prompt_embeds is None:
        raise RuntimeError(
            "Semantic guidance requires 'prompt_embeds' in callback kwargs. "
            "Keep callback_on_step_end_tensor_inputs wired through generate.py."
        )

    do_classifier_free_guidance = prompt_embeds.shape[0] == latents.shape[0] * 2
    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
    latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, timestep)
    noise_pred = pipe.unet(latent_model_input, timestep, encoder_hidden_states=prompt_embeds).sample

    if do_classifier_free_guidance:
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        guidance_scale = _get_guidance_scale(pipe)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    return noise_pred


def _alpha_prod_for_timestep(scheduler, timestep, latents: torch.Tensor) -> torch.Tensor:
    if not hasattr(scheduler, "alphas_cumprod"):
        raise RuntimeError(
            f"{scheduler.__class__.__name__} does not expose alphas_cumprod; "
            "semantic guidance currently supports DDPM/DDIM/PNDM-style schedulers."
        )
    timestep_tensor = torch.as_tensor(timestep, device=latents.device)
    if timestep_tensor.ndim > 0:
        timestep_tensor = timestep_tensor.flatten()[0]
    timestep_index = int(timestep_tensor.long().item())
    alphas_cumprod = scheduler.alphas_cumprod.to(device=latents.device)
    if timestep_index < 0 or timestep_index >= alphas_cumprod.shape[0]:
        raise RuntimeError(
            f"Semantic guidance timestep {timestep_index} is outside "
            f"scheduler.alphas_cumprod length {alphas_cumprod.shape[0]}"
        )
    alpha = alphas_cumprod[timestep_index].to(dtype=latents.dtype)
    while alpha.ndim < latents.ndim:
        alpha = alpha.view(*alpha.shape, 1)
    return alpha


def _predict_original_sample(scheduler, latents: torch.Tensor, noise_pred: torch.Tensor, timestep) -> torch.Tensor:
    alpha_prod_t = _alpha_prod_for_timestep(scheduler, timestep, latents)
    beta_prod_t = 1 - alpha_prod_t
    scheduler_config = getattr(scheduler, "config", {})
    if isinstance(scheduler_config, dict):
        prediction_type = scheduler_config.get("prediction_type", "epsilon")
    else:
        prediction_type = getattr(scheduler_config, "prediction_type", "epsilon")

    if prediction_type == "epsilon":
        return (latents - beta_prod_t.sqrt() * noise_pred) / alpha_prod_t.sqrt().clamp_min(1e-8)
    if prediction_type == "sample":
        return noise_pred
    if prediction_type == "v_prediction":
        return alpha_prod_t.sqrt() * latents - beta_prod_t.sqrt() * noise_pred
    raise RuntimeError(f"Unsupported scheduler prediction_type for semantic guidance: {prediction_type!r}")


def _decode_latents_to_dino_input(vae, predicted_original_sample: torch.Tensor) -> torch.Tensor:
    scaling_factor = getattr(vae.config, "scaling_factor", 0.18215)
    image = vae.decode(predicted_original_sample / scaling_factor).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    return _normalize_dino_input(image)


def _dino_feature_loss(generated_embedding: torch.Tensor, target_embedding: torch.Tensor) -> torch.Tensor:
    """Cosine-distance objective matching the DINO evaluation score."""
    generated_embedding = F.normalize(generated_embedding, dim=-1)
    target_embedding = F.normalize(target_embedding, dim=-1)
    return 1.0 - (generated_embedding * target_embedding).sum(dim=-1).mean()


def _mean_scalar(value: torch.Tensor) -> float:
    return float(value.detach().float().mean().item())


def _per_sample_l2_norm(value: torch.Tensor) -> torch.Tensor:
    return value.detach().float().flatten(start_dim=1).norm(dim=1)


def _per_sample_rms(value: torch.Tensor, eps: float) -> torch.Tensor:
    return value.detach().float().flatten(start_dim=1).pow(2).mean(dim=1).sqrt().clamp_min(eps)


def _expand_per_sample(value: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return value.to(device=like.device, dtype=like.dtype).view(value.shape[0], *([1] * (like.ndim - 1)))


def _build_guidance_update(
    latents: torch.Tensor,
    grad: torch.Tensor,
    weight: float,
    options: GuidanceStepOptions,
) -> tuple[torch.Tensor, dict]:
    raw_grad_norm = _per_sample_l2_norm(grad)
    raw_grad_rms = _per_sample_rms(grad, options.normalization_eps)

    if options.gradient_normalization == "none":
        direction = grad
    elif options.gradient_normalization == "l2":
        direction = grad / _expand_per_sample(raw_grad_norm.clamp_min(options.normalization_eps), grad)
    elif options.gradient_normalization == "rms":
        direction = grad / _expand_per_sample(raw_grad_rms, grad)
    else:
        raise ValueError(f"unknown gradient_normalization: {options.gradient_normalization!r}")

    direction = torch.nan_to_num(direction)
    update = float(weight) * direction
    update_norm = _per_sample_l2_norm(update)
    latent_norm = _per_sample_l2_norm(latents).clamp_min(options.normalization_eps)

    if options.max_update_norm_ratio > 0:
        max_update_norm = latent_norm * float(options.max_update_norm_ratio)
        clamp_scale = torch.minimum(
            torch.ones_like(update_norm),
            max_update_norm / update_norm.clamp_min(options.normalization_eps),
        )
        update = update * _expand_per_sample(clamp_scale, update)
        update_norm = _per_sample_l2_norm(update)
    else:
        clamp_scale = torch.ones_like(update_norm)

    direction_norm = _per_sample_l2_norm(direction)
    direction_rms = _per_sample_rms(direction, options.normalization_eps)
    update_rms = _per_sample_rms(update, options.normalization_eps)
    stats = {
        "raw_grad_norm": _mean_scalar(raw_grad_norm),
        "raw_grad_rms": _mean_scalar(raw_grad_rms),
        "direction_norm": _mean_scalar(direction_norm),
        "direction_rms": _mean_scalar(direction_rms),
        "update_norm": _mean_scalar(update_norm),
        "update_rms": _mean_scalar(update_rms),
        "latent_norm": _mean_scalar(latent_norm),
        "update_to_latent_norm": _mean_scalar(update_norm / latent_norm),
        "clamp_scale_min": float(clamp_scale.detach().float().min().item()),
        "clamp_scale_mean": _mean_scalar(clamp_scale),
    }
    return torch.nan_to_num(update), stats


def _semantic_loss_for_latents(
    pipe,
    state: SemanticGuidanceState,
    latents: torch.Tensor,
    timestep,
    callback_kwargs: dict,
) -> torch.Tensor:
    noise_pred = _predict_noise(pipe, latents, timestep, callback_kwargs)
    predicted_original_sample = _predict_original_sample(pipe.scheduler, latents, noise_pred.detach(), timestep)
    dino_input = _decode_latents_to_dino_input(pipe.vae, predicted_original_sample)
    generated_embedding = F.normalize(state.model(dino_input), dim=-1)
    target_embedding = state.target_embedding.to(
        device=generated_embedding.device,
        dtype=generated_embedding.dtype,
    ).expand_as(generated_embedding)
    return _dino_feature_loss(generated_embedding, target_embedding)


def _semantic_guidance_step(
    pipe,
    state: SemanticGuidanceState,
    latents: torch.Tensor,
    timestep,
    callback_kwargs: dict,
    weight: float,
    options: GuidanceStepOptions | None = None,
    diagnostics_writer: GuidanceDiagnosticsWriter | None = None,
    step_idx: int | None = None,
) -> torch.Tensor:
    if weight == 0:
        return latents
    if options is None:
        options = GuidanceStepOptions()

    latents_for_grad = latents.detach().clone().requires_grad_(True)
    # Stop-gradient through the denoiser keeps this usable as a lightweight
    # inference-time experiment while still letting DINO/VAE gradients update z_t.
    with torch.no_grad():
        noise_pred = _predict_noise(pipe, latents_for_grad.detach(), timestep, callback_kwargs)

    predicted_original_sample = _predict_original_sample(pipe.scheduler, latents_for_grad, noise_pred.detach(), timestep)
    dino_input = _decode_latents_to_dino_input(pipe.vae, predicted_original_sample)
    generated_embedding = F.normalize(state.model(dino_input), dim=-1)
    target_embedding = state.target_embedding.to(
        device=generated_embedding.device,
        dtype=generated_embedding.dtype,
    ).expand_as(generated_embedding)
    loss = _dino_feature_loss(generated_embedding, target_embedding)
    grad = torch.autograd.grad(loss, latents_for_grad, retain_graph=False, create_graph=False)[0]
    grad = torch.nan_to_num(grad)
    update, update_stats = _build_guidance_update(latents, grad, weight, options)
    guided_latents = (latents - update).detach()

    if diagnostics_writer is not None:
        loss_after = None
        if options.diagnostics_recompute_loss_after:
            with torch.no_grad():
                loss_after = _semantic_loss_for_latents(pipe, state, guided_latents, timestep, callback_kwargs)
        timestep_tensor = torch.as_tensor(timestep).flatten()[0]
        row = {
            "step_idx": step_idx,
            "timestep": int(timestep_tensor.long().item()),
            "weight": float(weight),
            "gradient_normalization": options.gradient_normalization,
            "loss_before": float(loss.detach().float().item()),
            "loss_after": None if loss_after is None else float(loss_after.detach().float().item()),
        }
        row.update(update_stats)
        diagnostics_writer.record(row)

    return guided_latents


def build_callback_kwargs(
    cfg,
    state: SemanticGuidanceState,
    num_inference_steps: int,
    diagnostics_path: str | Path | None = None,
):
    """Build Diffusers callback kwargs for semantic guidance."""
    sg = cfg.semantic_guidance
    weight = float(sg.weight)
    schedule = sg.schedule
    every_n_steps = int(sg.every_n_steps)
    start_frac = float(sg.start_step_frac)
    hold_until_frac = float(sg.hold_until_frac)
    end_frac = float(sg.end_step_frac)
    options = GuidanceStepOptions(
        gradient_normalization=getattr(sg, "gradient_normalization", "none"),
        normalization_eps=float(getattr(sg, "normalization_eps", 1e-8)),
        max_update_norm_ratio=float(getattr(sg, "max_update_norm_ratio", 0.0)),
        diagnostics_recompute_loss_after=bool(getattr(sg, "diagnostics_recompute_loss_after", True)),
    )
    diagnostics_writer = GuidanceDiagnosticsWriter(diagnostics_path) if diagnostics_path is not None else None

    def _callback(pipe, step_idx, timestep, callback_kwargs):
        latents = callback_kwargs["latents"]
        if not _should_apply(step_idx, num_inference_steps, every_n_steps, start_frac, end_frac):
            return {"latents": latents}
        step_weight = _resolve_step_weight(
            weight,
            schedule,
            step_idx,
            num_inference_steps,
            start_frac,
            hold_until_frac,
            end_frac,
        )
        if step_weight == 0.0:
            return {"latents": latents}
        with torch.enable_grad():
            guided_latents = _semantic_guidance_step(
                pipe,
                state,
                latents,
                timestep,
                callback_kwargs,
                step_weight,
                options,
                diagnostics_writer,
                step_idx,
            )
        return {"latents": guided_latents.to(device=latents.device, dtype=latents.dtype)}

    return {
        "callback_on_step_end": _callback,
        "callback_on_step_end_tensor_inputs": ["latents", "prompt_embeds"],
    }
