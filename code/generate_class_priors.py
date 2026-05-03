import random
from pathlib import Path

import hydra
import numpy as np
import torch
from diffusers import StableDiffusionPipeline
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from class_priors import append_class_prior_images, count_class_prior_images, pil_to_uint8_rgb


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def validate_config(cfg: DictConfig):
    if cfg.prior_generation.num_images < 1:
        raise ValueError("prior_generation.num_images must be at least 1")
    if cfg.prior_generation.batch_size < 1:
        raise ValueError("prior_generation.batch_size must be at least 1")


def resolve_config_paths(cfg: DictConfig):
    cfg.prior_generation.output_npz = to_absolute_path(cfg.prior_generation.output_npz)
    if cfg.prior_generation.preview_dir is not None:
        cfg.prior_generation.preview_dir = to_absolute_path(cfg.prior_generation.preview_dir)


@hydra.main(version_base=None, config_path="conf", config_name="generate_class_priors")
def main(cfg: DictConfig):
    validate_config(cfg)
    resolve_config_paths(cfg)
    print(OmegaConf.to_yaml(cfg))

    torch.manual_seed(cfg.runtime.seed)
    random.seed(cfg.runtime.seed)

    device = get_device()
    weight_dtype = torch.float32
    print(f"Using device: {device}")
    print(f"Using dtype: {weight_dtype}")
    output_npz = Path(cfg.prior_generation.output_npz)

    n_existing = count_class_prior_images(output_npz)
    if n_existing >= cfg.prior_generation.num_images:
        print(f"{output_npz} already contains {n_existing} class priors; nothing to generate.")
        return

    num_to_generate = cfg.prior_generation.num_images - n_existing
    print(f"Generating {num_to_generate} class priors into {output_npz}...")

    preview_dir = None
    if cfg.prior_generation.save_previews:
        if cfg.prior_generation.preview_dir is None:
            preview_dir = output_npz.parent / "preview"
        else:
            preview_dir = Path(cfg.prior_generation.preview_dir)
        preview_dir.mkdir(parents=True, exist_ok=True)

    pipeline = StableDiffusionPipeline.from_pretrained(
        cfg.model.pretrained_model,
        torch_dtype=weight_dtype,
        safety_checker=None,
    )
    pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)

    rows: list[np.ndarray] = []
    preview_idx = n_existing
    batch_size = cfg.prior_generation.batch_size
    for i in tqdm(range(0, num_to_generate, batch_size), desc="Generating class priors"):
        current_batch = min(batch_size, num_to_generate - i)
        images = pipeline(
            [cfg.prior_generation.class_prompt] * current_batch,
            num_inference_steps=cfg.prior_generation.num_inference_steps,
            guidance_scale=cfg.prior_generation.guidance_scale,
        ).images
        for img in images:
            rows.append(pil_to_uint8_rgb(img))
            if preview_dir is not None:
                img.save(preview_dir / f"class_{preview_idx:04d}.png")
                preview_idx += 1

    append_class_prior_images(output_npz, np.stack(rows, axis=0))
    print(f"Saved {cfg.prior_generation.num_images} total class priors to {output_npz}")

    pipeline.to("cpu")
    del pipeline
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
