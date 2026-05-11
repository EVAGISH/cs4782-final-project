import json
from pathlib import Path

import hydra
import torch
from diffusers import Flux2KleinPipeline
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from lora_flux2 import (
    get_lora_parameter_names,
    patch_flux2_transformer_with_lora,
    summarize_lora_state_dict,
)


WEIGHT_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def validate_config(cfg: DictConfig):
    if cfg.inference.prompts_file is None and not cfg.inference.prompts:
        raise ValueError("Provide inference.prompts or inference.prompts_file")
    if cfg.inference.num_images_per_prompt < 1:
        raise ValueError("inference.num_images_per_prompt must be at least 1")
    if cfg.inference.resolution % 16 != 0:
        raise ValueError("inference.resolution must be divisible by 16")
    if cfg.inference.weight_dtype not in WEIGHT_DTYPES:
        raise ValueError(f"weight_dtype must be one of {sorted(WEIGHT_DTYPES)}")


def resolve_config_paths(cfg: DictConfig):
    cfg.inference.output_dir = to_absolute_path(cfg.inference.output_dir)
    if cfg.inference.prompts_file is not None:
        cfg.inference.prompts_file = to_absolute_path(cfg.inference.prompts_file)


def resolve_model_path(model_path: str) -> str:
    path = Path(model_path)
    if path.is_absolute() or model_path.startswith((".", "..")) or path.exists():
        return to_absolute_path(model_path)
    return model_path


def load_pipeline(model_path: str, device: torch.device, weight_dtype: torch.dtype):
    """Load a Flux2KleinPipeline. Supports three sources:
      1. LoRA dir (lora_config.json + lora_weights.pt) — load base, patch transformer, load LoRA.
      2. Full pipeline dir saved via save_pretrained.
      3. Bare HF model id (e.g., "black-forest-labs/FLUX.2-klein-4B").
    """
    resolved = resolve_model_path(model_path)
    model_dir = Path(resolved)
    lora_config_path = model_dir / "lora_config.json"
    lora_weights_path = model_dir / "lora_weights.pt"

    if model_dir.is_dir() and lora_config_path.exists() and lora_weights_path.exists():
        with open(lora_config_path, "r") as f:
            lora_config = json.load(f)
        if lora_config.get("model_kind") != "flux2_klein":
            raise ValueError(
                f"{lora_config_path} has model_kind={lora_config.get('model_kind')!r}; "
                f"expected 'flux2_klein'. Use the SD generate.py for SD/UNet LoRAs."
            )

        pipeline = Flux2KleinPipeline.from_pretrained(
            lora_config["pretrained_model"], torch_dtype=weight_dtype,
        ).to(device)

        patch_flux2_transformer_with_lora(
            pipeline.transformer, rank=lora_config["rank"], alpha=lora_config["alpha"],
        )
        state = torch.load(lora_weights_path, map_location="cpu")
        expected = get_lora_parameter_names(pipeline.transformer)
        loaded = set(state)
        missing = sorted(expected - loaded)
        unexpected = sorted(loaded - expected)
        if missing:
            raise RuntimeError(f"Missing LoRA keys when loading weights: {missing}")
        if unexpected:
            raise RuntimeError(f"Unexpected LoRA keys when loading weights: {unexpected}")

        lora_summary = summarize_lora_state_dict(state)
        state = {k: v.to(device=device, dtype=weight_dtype) for k, v in state.items()}
        _, unexpected_load = pipeline.transformer.load_state_dict(state, strict=False)
        if unexpected_load:
            raise RuntimeError(f"Unexpected keys when loading LoRA into transformer: {unexpected_load}")
        pipeline.transformer.to(device)
        print(
            f"Loaded FLUX2 LoRA from {model_dir} "
            f"(base={lora_config['pretrained_model']}, rank={lora_config['rank']}, "
            f"tensors={lora_summary['num_tensors']}, "
            f"mean_abs={lora_summary['mean_abs']:.6f}, "
            f"max_abs={lora_summary['max_abs']:.6f}, "
            f"sha256={lora_summary['sha256']})"
        )
        return pipeline

    pipeline = Flux2KleinPipeline.from_pretrained(resolved, torch_dtype=weight_dtype).to(device)
    if model_dir.is_dir():
        print(f"Loaded full pipeline from {model_dir}")
    else:
        print(f"Loaded base FLUX2 pipeline {resolved}")
    return pipeline


@hydra.main(version_base=None, config_path="conf/flux2", config_name="generate")
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
    weight_dtype = WEIGHT_DTYPES[cfg.inference.weight_dtype]
    print(f"Using device: {device}")
    print(f"Using dtype: {weight_dtype}")

    pipeline = load_pipeline(cfg.model.model_path, device, weight_dtype)

    output_dir = Path(cfg.inference.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = []
    res = cfg.inference.resolution

    # Anchor-latent biasing (present in generate.py for SD) is intentionally not ported:
    # FLUX2 latents are 2x2-packed and BN-normalized, so the existing math doesn't transfer.
    for prompt_idx, prompt in enumerate(prompts):
        prompt_dir = output_dir / f"prompt_{prompt_idx:02d}"
        prompt_dir.mkdir(exist_ok=True)

        for i in range(cfg.inference.num_images_per_prompt):
            seed = cfg.runtime.seed + i
            generator = torch.Generator(device).manual_seed(seed)

            image = pipeline(
                prompt=prompt,
                height=res,
                width=res,
                num_inference_steps=cfg.inference.num_inference_steps,
                guidance_scale=cfg.inference.guidance_scale,
                generator=generator,
            ).images[0]

            filename = f"img_{i:02d}_seed{seed}.png"
            image.save(prompt_dir / filename)
            metadata.append({
                "prompt": prompt,
                "prompt_idx": prompt_idx,
                "sample_idx": i,
                "seed": seed,
                "filename": str(prompt_dir / filename),
            })

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Generated {len(metadata)} images in {output_dir}")


if __name__ == "__main__":
    main()
