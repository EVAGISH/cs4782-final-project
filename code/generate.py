import json
import time
from datetime import datetime, timezone
from pathlib import Path

import hydra
import torch
from diffusers import StableDiffusionPipeline
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

import anchor
import semantic_guidance
from hydra_compat import patch_argparse_help_for_hydra
from lora import get_lora_parameter_names, patch_unet_with_lora, summarize_lora_state_dict


patch_argparse_help_for_hydra()


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def validate_config(cfg: DictConfig):
    if cfg.task is not None and cfg.task != "image":
        raise ValueError(f"task must be 'image' or null, got {cfg.task!r}")
    if cfg.inference.prompts_file is None and not cfg.inference.prompts:
        raise ValueError("Provide inference.prompts or inference.prompts_file")
    if cfg.inference.num_images_per_prompt < 1:
        raise ValueError("inference.num_images_per_prompt must be at least 1")
    if cfg.inference.height < 1 or cfg.inference.width < 1:
        raise ValueError("inference.height and inference.width must be positive")
    if cfg.inference.height % 8 != 0 or cfg.inference.width % 8 != 0:
        raise ValueError("inference.height and inference.width must be divisible by 8")
    if cfg.anchor.enabled:
        anchor.validate(cfg)
    if cfg.semantic_guidance.enabled:
        semantic_guidance.validate(cfg)


def resolve_config_paths(cfg: DictConfig):
    cfg.inference.output_dir = to_absolute_path(cfg.inference.output_dir)
    if cfg.inference.prompts_file is not None:
        cfg.inference.prompts_file = to_absolute_path(cfg.inference.prompts_file)
    if cfg.anchor.enabled:
        anchor.resolve_paths(cfg)
    if cfg.semantic_guidance.enabled:
        semantic_guidance.resolve_paths(cfg)


def _merge_callback_kwargs(callback_kwargs_list: list[dict]) -> dict:
    """Compose multiple Diffusers step-end callbacks into one callback config."""
    callback_kwargs_list = [kwargs for kwargs in callback_kwargs_list if kwargs]
    if not callback_kwargs_list:
        return {}
    if len(callback_kwargs_list) == 1:
        return callback_kwargs_list[0]

    callbacks = [kwargs["callback_on_step_end"] for kwargs in callback_kwargs_list]
    tensor_inputs = []
    seen_inputs = set()
    for kwargs in callback_kwargs_list:
        for name in kwargs.get("callback_on_step_end_tensor_inputs", []):
            if name not in seen_inputs:
                tensor_inputs.append(name)
                seen_inputs.add(name)

    def _callback(pipe, step_idx, timestep, callback_kwargs):
        working_kwargs = dict(callback_kwargs)
        returned_kwargs = {}
        for callback in callbacks:
            updates = callback(pipe, step_idx, timestep, working_kwargs) or {}
            working_kwargs.update(updates)
            returned_kwargs.update(updates)
        return returned_kwargs

    return {
        "callback_on_step_end": _callback,
        "callback_on_step_end_tensor_inputs": tensor_inputs,
    }


def _semantic_diagnostics_path(cfg, output_dir: Path, prompt_idx: int, sample_idx: int, seed: int) -> Path | None:
    sg = cfg.semantic_guidance
    if not getattr(sg, "diagnostics_enabled", False):
        return None

    diagnostics_dir = getattr(sg, "diagnostics_dir", None)
    if diagnostics_dir:
        base_dir = Path(diagnostics_dir)
        if not base_dir.is_absolute():
            base_dir = output_dir / base_dir
    else:
        base_dir = output_dir / "semantic_guidance_diagnostics"
    return base_dir / f"prompt_{prompt_idx:02d}_sample_{sample_idx:02d}_seed{seed}.csv"


def resolve_model_path(model_path: str) -> str:
    path = Path(model_path)
    if path.is_absolute() or model_path.startswith((".", "..")) or path.exists():
        return to_absolute_path(model_path)
    return model_path


def _load_base_pipeline(base_model: str, weight_dtype: torch.dtype):
    return StableDiffusionPipeline.from_pretrained(
        base_model,
        torch_dtype=weight_dtype,
        safety_checker=None,
    )


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
      3. Default to "image" for saved pipelines and plain HF model ids.
    """
    resolved_model_path = resolve_model_path(model_path)
    model_dir = Path(resolved_model_path)
    lora_config_path = model_dir / "lora_config.json"
    lora_weights_path = model_dir / "lora_weights.pt"

    if model_dir.is_dir() and lora_config_path.exists() and lora_weights_path.exists():
        with open(lora_config_path, "r") as f:
            lora_config = json.load(f)
        task = task_override or lora_config.get("task", "image")
        if task != "image":
            raise ValueError(f"Only image LoRA generation is supported, got task={task!r}")

        pipeline = _load_base_pipeline(lora_config["pretrained_model"], weight_dtype).to(device)

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

    task = task_override or "image"
    if task != "image":
        raise ValueError(f"Only image generation is supported, got task={task!r}")

    pipeline = _load_base_pipeline(resolved_model_path, weight_dtype).to(device)
    if model_dir.is_dir():
        print(f"Loaded full pipeline from {model_dir} (task={task})")
    else:
        print(f"Loaded base model {resolved_model_path} (task={task})")
    return pipeline, task


@hydra.main(version_base=None, config_path="conf", config_name="generate")
def main(cfg: DictConfig):
    validate_config(cfg)
    resolve_config_paths(cfg)
    resolved_cfg_yaml = OmegaConf.to_yaml(cfg, resolve=True)
    print(resolved_cfg_yaml)

    output_dir = Path(cfg.inference.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "generate_config.yaml", "w") as f:
        f.write(resolved_cfg_yaml)
    started_at = datetime.now(timezone.utc)
    start_time = time.time()

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

    subject_latents = None
    if cfg.anchor.enabled:
        subject_latents = anchor.prepare(cfg, pipeline, device, weight_dtype)

    semantic_state = None
    if cfg.semantic_guidance.enabled:
        semantic_state = semantic_guidance.prepare(cfg, pipeline, device, weight_dtype)

    metadata = []

    for prompt_idx, prompt in enumerate(prompts):
        prompt_dir = output_dir / f"prompt_{prompt_idx:02d}"
        prompt_dir.mkdir(exist_ok=True)

        for i in range(cfg.inference.num_images_per_prompt):
            seed = cfg.runtime.seed + i
            generator = torch.Generator(device).manual_seed(seed)

            step_callbacks = []
            if subject_latents is not None:
                step_callbacks.append(
                    anchor.build_callback_kwargs(
                        cfg,
                        subject_latents,
                        pipeline.scheduler,
                        cfg.inference.num_inference_steps,
                        device,
                        seed,
                    )
                )
            if semantic_state is not None:
                step_callbacks.append(
                    semantic_guidance.build_callback_kwargs(
                        cfg,
                        semantic_state,
                        cfg.inference.num_inference_steps,
                        diagnostics_path=_semantic_diagnostics_path(cfg, output_dir, prompt_idx, i, seed),
                    )
                )
            extra_kwargs = _merge_callback_kwargs(step_callbacks)
            image = pipeline(
                prompt,
                height=cfg.inference.height,
                width=cfg.inference.width,
                num_inference_steps=cfg.inference.num_inference_steps,
                guidance_scale=cfg.inference.guidance_scale,
                generator=generator,
                **extra_kwargs,
            ).images[0]
            filename = f"img_{i:02d}_seed{seed}.png"
            image_path = prompt_dir / filename
            image.save(image_path)

            entry = {
                "prompt": prompt,
                "prompt_idx": prompt_idx,
                "sample_idx": i,
                "seed": seed,
                "filename": str(image_path.relative_to(output_dir)),
            }
            metadata.append(entry)

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    summary = {
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": time.time() - start_time,
        "task": task,
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "model_path": cfg.model.model_path,
        "num_prompts": len(prompts),
        "num_outputs": len(metadata),
        "num_images_per_prompt": cfg.inference.num_images_per_prompt,
        "anchor": OmegaConf.to_container(cfg.anchor, resolve=True),
        "semantic_guidance": OmegaConf.to_container(cfg.semantic_guidance, resolve=True),
    }
    with open(output_dir / "generate_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Generated {len(metadata)} images in {output_dir}")


if __name__ == "__main__":
    main()
