import json
from pathlib import Path

import hydra
import torch
from diffusers import StableDiffusionPipeline, TextToVideoSDPipeline
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from lora import get_lora_parameter_names, patch_unet_with_lora, summarize_lora_state_dict


VIDEO_PIPELINE_CLASS_NAMES = {"TextToVideoSDPipeline"}


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def validate_config(cfg: DictConfig):
    if cfg.task is not None and cfg.task not in {"image", "video"}:
        raise ValueError(f"task must be 'image', 'video', or null, got {cfg.task!r}")
    if cfg.inference.prompts_file is None and not cfg.inference.prompts:
        raise ValueError("Provide inference.prompts or inference.prompts_file")
    if cfg.inference.num_images_per_prompt < 1:
        raise ValueError("inference.num_images_per_prompt must be at least 1")
    if cfg.inference.num_frames < 1:
        raise ValueError("inference.num_frames must be at least 1")
    if cfg.inference.fps < 1:
        raise ValueError("inference.fps must be at least 1")
    if cfg.anchor.enabled:
        if cfg.anchor.mode not in ANCHOR_MODES:
            raise ValueError(f"anchor.mode must be one of {sorted(ANCHOR_MODES)}, got {cfg.anchor.mode!r}")
        if cfg.anchor.schedule not in ANCHOR_SCHEDULES:
            raise ValueError(f"anchor.schedule must be one of {sorted(ANCHOR_SCHEDULES)}, got {cfg.anchor.schedule!r}")


def resolve_config_paths(cfg: DictConfig):
    cfg.inference.output_dir = to_absolute_path(cfg.inference.output_dir)
    if cfg.inference.prompts_file is not None:
        cfg.inference.prompts_file = to_absolute_path(cfg.inference.prompts_file)
    if cfg.anchor.enabled:
        cfg.anchor.subject_dir = to_absolute_path(cfg.anchor.subject_dir)


ANCHOR_MODES = {"pooled", "random_per_sample", "random_per_step"}
ANCHOR_SCHEDULES = {"constant", "linear_decay", "early_only"}


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


def resolve_model_path(model_path: str) -> str:
    path = Path(model_path)
    if path.is_absolute() or model_path.startswith((".", "..")) or path.exists():
        return to_absolute_path(model_path)
    return model_path


def _detect_task_from_pipeline_dir(model_dir: Path) -> str | None:
    """Read model_index.json from a saved pipeline dir to infer image vs video."""
    index_path = model_dir / "model_index.json"
    if not index_path.exists():
        return None
    with open(index_path, "r") as f:
        index = json.load(f)
    class_name = index.get("_class_name")
    if class_name in VIDEO_PIPELINE_CLASS_NAMES:
        return "video"
    return "image"


def _pipeline_kwargs_for_task(task: str, weight_dtype: torch.dtype) -> dict:
    if task == "image":
        return {"torch_dtype": weight_dtype, "safety_checker": None}
    return {"torch_dtype": weight_dtype}


def _load_base_pipeline(base_model: str, task: str, weight_dtype: torch.dtype):
    cls = TextToVideoSDPipeline if task == "video" else StableDiffusionPipeline
    return cls.from_pretrained(base_model, **_pipeline_kwargs_for_task(task, weight_dtype))


def load_pipeline(
    model_path: str,
    device: torch.device,
    weight_dtype: torch.dtype,
    task_override: str | None = None,
) -> tuple[object, str]:
    """Load a pipeline and return (pipeline, resolved_task).

    Resolution order for `task`:
      1. `task_override` from config if not None.
      2. `task` field of `lora_config.json` (LoRA dir case).
      3. `_class_name` in `model_index.json` (saved-pipeline-dir case).
      4. Default to "image" for plain HF model ids.
    """
    resolved_model_path = resolve_model_path(model_path)
    model_dir = Path(resolved_model_path)
    lora_config_path = model_dir / "lora_config.json"
    lora_weights_path = model_dir / "lora_weights.pt"

    if model_dir.is_dir() and lora_config_path.exists() and lora_weights_path.exists():
        with open(lora_config_path, "r") as f:
            lora_config = json.load(f)
        task = task_override or lora_config.get("task", "image")

        pipeline = _load_base_pipeline(lora_config["pretrained_model"], task, weight_dtype).to(device)

        patch_unet_with_lora(pipeline.unet, rank=lora_config["rank"], alpha=lora_config["alpha"])
        state = torch.load(lora_weights_path, map_location="cpu")
        expected_lora_keys = get_lora_parameter_names(pipeline.unet)
        loaded_lora_keys = set(state)
        missing_lora = sorted(expected_lora_keys - loaded_lora_keys)
        unexpected_lora = sorted(loaded_lora_keys - expected_lora_keys)
        if missing_lora:
            raise RuntimeError(f"Missing LoRA keys when loading weights: {missing_lora}")
        if unexpected_lora:
            raise RuntimeError(f"Unexpected LoRA keys when loading weights: {unexpected_lora}")

        lora_summary = summarize_lora_state_dict(state)
        state = {k: v.to(device=device, dtype=weight_dtype) for k, v in state.items()}
        _, unexpected = pipeline.unet.load_state_dict(state, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected keys when loading LoRA: {unexpected}")
        pipeline.unet.to(device)
        print(
            f"Loaded LoRA model from {model_dir} (task={task}, "
            f"base={lora_config['pretrained_model']}, rank={lora_config['rank']}, "
            f"tensors={lora_summary['num_tensors']}, "
            f"mean_abs={lora_summary['mean_abs']:.6f}, "
            f"max_abs={lora_summary['max_abs']:.6f}, "
            f"sha256={lora_summary['sha256']})"
        )
        return pipeline, task

    if model_dir.is_dir():
        task = task_override or _detect_task_from_pipeline_dir(model_dir) or "image"
    else:
        task = task_override or "image"

    pipeline = _load_base_pipeline(resolved_model_path, task, weight_dtype).to(device)
    if model_dir.is_dir():
        print(f"Loaded full pipeline from {model_dir} (task={task})")
    else:
        print(f"Loaded base model {resolved_model_path} (task={task})")
    return pipeline, task


@hydra.main(version_base=None, config_path="conf", config_name="generate")
def main(cfg: DictConfig):
    validate_config(cfg)
    resolve_config_paths(cfg)
    print(OmegaConf.to_yaml(cfg))

    if cfg.inference.prompts_file:
        with open(cfg.inference.prompts_file, "r") as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        prompts = list(cfg.inference.prompts)

    device = get_device()
    print(f"Using device: {device}")
    weight_dtype = torch.float32
    print(f"Using dtype: {weight_dtype}")

    pipeline, task = load_pipeline(cfg.model.model_path, device, weight_dtype, task_override=cfg.task)

    output_dir = Path(cfg.inference.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subject_latents = None
    if cfg.anchor.enabled:
        if task != "image":
            print(f"anchor.enabled=true but task={task!r}; anchoring is image-only and will be skipped.")
        else:
            subject_latents = _load_subject_latents(cfg.anchor.subject_dir, pipeline, device, weight_dtype)
            print(
                f"Anchor settings: mode={cfg.anchor.mode}, weight={cfg.anchor.weight}, "
                f"schedule={cfg.anchor.schedule}"
            )

    metadata = []
    samples_label = "videos" if task == "video" else "images"

    for prompt_idx, prompt in enumerate(prompts):
        prompt_dir = output_dir / f"prompt_{prompt_idx:02d}"
        prompt_dir.mkdir(exist_ok=True)

        for i in range(cfg.inference.num_images_per_prompt):
            seed = cfg.runtime.seed + i
            generator = torch.Generator(device).manual_seed(seed)

            if task == "video":
                # TextToVideoSDPipeline returns .frames as a list with one
                # ndarray per batch element of shape (F, H, W, 3), uint8.
                result = pipeline(
                    prompt,
                    num_frames=cfg.inference.num_frames,
                    num_inference_steps=cfg.inference.num_inference_steps,
                    guidance_scale=cfg.inference.guidance_scale,
                    generator=generator,
                    output_type="np",
                )
                frames_np = result.frames[0]  # (F, H, W, 3) float in [0,1] or uint8
                if frames_np.dtype != "uint8":
                    import numpy as np
                    frames_np = (np.clip(frames_np, 0.0, 1.0) * 255).astype("uint8")
                # Torchcodec wants (N, 3, H, W) uint8.
                frames_chw = torch.from_numpy(frames_np).permute(0, 3, 1, 2).contiguous()
                filename = f"vid_{i:02d}_seed{seed}.mp4"
                from torchcodec.encoders import VideoEncoder
                VideoEncoder(frames_chw, frame_rate=cfg.inference.fps).to_file(prompt_dir / filename)
            else:
                extra_kwargs = {}
                if subject_latents is not None:
                    anchor_gen = torch.Generator(device).manual_seed(seed + 9000)
                    fixed_anchor = _select_anchor_for_sample(
                        subject_latents, cfg.anchor.mode, anchor_gen,
                    )
                    extra_kwargs["callback_on_step_end"] = _make_anchor_callback(
                        subject_latents,
                        fixed_anchor,
                        cfg.anchor.mode,
                        float(cfg.anchor.weight),
                        cfg.anchor.schedule,
                        cfg.inference.num_inference_steps,
                        pipeline.scheduler,
                        anchor_gen,
                    )
                    extra_kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]
                image = pipeline(
                    prompt,
                    num_inference_steps=cfg.inference.num_inference_steps,
                    guidance_scale=cfg.inference.guidance_scale,
                    generator=generator,
                    **extra_kwargs,
                ).images[0]
                filename = f"img_{i:02d}_seed{seed}.png"
                image.save(prompt_dir / filename)

            entry = {
                "prompt": prompt,
                "prompt_idx": prompt_idx,
                "sample_idx": i,
                "seed": seed,
                "filename": str(prompt_dir / filename),
            }
            if task == "video":
                entry["num_frames"] = cfg.inference.num_frames
                entry["fps"] = cfg.inference.fps
            metadata.append(entry)

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Generated {len(metadata)} {samples_label} in {output_dir}")


if __name__ == "__main__":
    main()
