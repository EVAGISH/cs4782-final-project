import argparse
import json
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline

from train_dreambooth import patch_unet_with_lora


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to a saved DreamBooth pipeline, or to base SD when using --lora_path")
    parser.add_argument("--lora_path", type=str, default=None,
                        help="Path to a directory containing lora_weights.pt and lora_config.json")
    parser.add_argument("--prompts", type=str, nargs="+", default=None)
    parser.add_argument("--prompts_file", type=str, default=None,
                        help="Path to a plain text file with one prompt per line")
    parser.add_argument("--prompts_template", type=str, default=None,
                        help="Path to a JSON list of prompt templates with {V} and {class} placeholders")
    parser.add_argument("--class_noun", type=str, default=None,
                        help="Class noun used to fill {class} in --prompts_template")
    parser.add_argument("--v_token", type=str, default="[V]",
                        help="Identifier substituted for {V}; pass empty string for the base SD baseline")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_images_per_prompt", type=int, default=4)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.prompts_template:
        if args.class_noun is None:
            raise ValueError("--class_noun is required when using --prompts_template")
        with open(args.prompts_template, "r") as f:
            templates = json.load(f)
        v = args.v_token if args.v_token else ""
        prompts = []
        for t in templates:
            resolved = t.format(V=v, **{"class": args.class_noun})
            resolved = " ".join(resolved.split())
            prompts.append(resolved)
    elif args.prompts_file:
        with open(args.prompts_file, "r") as f:
            prompts = [line.strip() for line in f if line.strip()]
    elif args.prompts:
        prompts = args.prompts
    else:
        raise ValueError("Provide one of --prompts, --prompts_file, or --prompts_template")

    device = get_device()
    print(f"Using device: {device}")

    weight_dtype = torch.float16 if device.type == "cuda" else torch.float32

    pipeline = StableDiffusionPipeline.from_pretrained(
        args.model_path,
        torch_dtype=weight_dtype,
        safety_checker=None,
    ).to(device)

    if args.lora_path:
        lora_dir = Path(args.lora_path)
        with open(lora_dir / "lora_config.json", "r") as f:
            lora_config = json.load(f)
        patch_unet_with_lora(pipeline.unet, rank=lora_config["rank"], alpha=lora_config["alpha"])
        state = torch.load(lora_dir / "lora_weights.pt", map_location="cpu")
        state = {k: v.to(device=device, dtype=weight_dtype) for k, v in state.items()}
        missing, unexpected = pipeline.unet.load_state_dict(state, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected keys when loading LoRA: {unexpected}")
        pipeline.unet.to(device)
        print(f"Loaded LoRA weights from {lora_dir} (rank={lora_config['rank']})")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = []

    for prompt_idx, prompt in enumerate(prompts):
        prompt_dir = output_dir / f"prompt_{prompt_idx:02d}"
        prompt_dir.mkdir(exist_ok=True)

        for i in range(args.num_images_per_prompt):
            seed = args.seed + i
            generator = torch.Generator(device).manual_seed(seed)

            image = pipeline(
                prompt,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
            ).images[0]

            filename = f"img_{i:02d}_seed{seed}.png"
            image.save(prompt_dir / filename)

            metadata.append({
                "prompt": prompt,
                "prompt_idx": prompt_idx,
                "image_idx": i,
                "seed": seed,
                "filename": str(prompt_dir / filename),
            })

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Generated {len(metadata)} images in {output_dir}")


if __name__ == "__main__":
    main()
