"""Generate class prior images for one or more classes using base SD.

Usage:
    python code/generate_class_images.py \
        --pretrained_model runwayml/stable-diffusion-v1-5 \
        --classes dog cat "stuffed bear" \
        --output_root data/class_images \
        --num_images 200
"""

import argparse
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline
from tqdm import tqdm


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def slugify(s):
    return s.lower().replace(" ", "_")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--classes", type=str, nargs="+", required=True,
                        help="Class nouns to generate priors for, e.g. dog cat 'stuffed bear'")
    parser.add_argument("--output_root", type=str, default="data/class_images")
    parser.add_argument("--num_images", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")

    weight_dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipeline = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model,
        torch_dtype=weight_dtype,
        safety_checker=None,
    ).to(device)
    pipeline.set_progress_bar_config(disable=True)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for class_noun in args.classes:
        class_dir = output_root / slugify(class_noun)
        class_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(class_dir.glob("class_*.png"))
        if len(existing) >= args.num_images:
            print(f"[{class_noun}] already has {len(existing)} images, skipping.")
            continue

        prompt = f"a {class_noun}"
        to_make = args.num_images - len(existing)
        print(f"[{class_noun}] generating {to_make} images with prompt: {prompt!r}")

        for i in tqdm(range(0, to_make, args.batch_size)):
            cur = min(args.batch_size, to_make - i)
            images = pipeline(
                [prompt] * cur,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
            ).images
            for j, img in enumerate(images):
                idx = len(existing) + i + j
                img.save(class_dir / f"class_{idx:04d}.png")

        print(f"[{class_noun}] -> {class_dir}")


if __name__ == "__main__":
    main()
