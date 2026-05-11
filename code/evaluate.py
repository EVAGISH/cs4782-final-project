import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_images_dir", type=str, required=True)
    parser.add_argument("--generated_images_dir", type=str, required=True)
    parser.add_argument("--prompts_file", type=str, default=None)
    parser.add_argument("--output_file", type=str, default="results/metrics.json")
    return parser.parse_args()


def load_images(image_dir):
    image_dir = Path(image_dir)
    images = []
    paths = []
    for p in sorted(image_dir.rglob("*.png")):
        with Image.open(p) as source:
            img = source.convert("RGB")
        img.load()
        images.append(img)
        paths.append(str(p))
    for p in sorted(image_dir.rglob("*.jpg")):
        with Image.open(p) as source:
            img = source.convert("RGB")
        img.load()
        images.append(img)
        paths.append(str(p))
    return images, paths


def load_dino_model(device="cuda"):
    model = torch.hub.load("facebookresearch/dino:main", "dino_vits16")
    model.eval().to(device)

    transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, transform


def compute_dino_embeddings(images, model, transform, device="cuda"):
    embeddings = []
    with torch.no_grad():
        for img in images:
            tensor = transform(img).unsqueeze(0).to(device)
            emb = model(tensor)
            embeddings.append(F.normalize(emb, dim=-1))

    return torch.cat(embeddings, dim=0)


def _as_image_tensor(emb, model):
    if isinstance(emb, torch.Tensor):
        return emb
    if hasattr(emb, "image_embeds") and emb.image_embeds is not None:
        return emb.image_embeds
    if hasattr(emb, "pooler_output") and emb.pooler_output is not None:
        pooled = emb.pooler_output
        proj_dim = getattr(model.config, "projection_dim", None)
        if proj_dim is not None and pooled.shape[-1] == proj_dim:
            return pooled
        return model.visual_projection(pooled)
    raise TypeError(f"Unexpected CLIP image output: {type(emb)}")


def _as_text_tensor(emb, model):
    if isinstance(emb, torch.Tensor):
        return emb
    if hasattr(emb, "text_embeds") and emb.text_embeds is not None:
        return emb.text_embeds
    if hasattr(emb, "pooler_output") and emb.pooler_output is not None:
        pooled = emb.pooler_output
        proj_dim = getattr(model.config, "projection_dim", None)
        if proj_dim is not None and pooled.shape[-1] == proj_dim:
            return pooled
        return model.text_projection(pooled)
    raise TypeError(f"Unexpected CLIP text output: {type(emb)}")


def compute_clip_image_embeddings(images, processor, model, device="cuda"):
    embeddings = []
    with torch.no_grad():
        for img in images:
            inputs = processor(images=img, return_tensors="pt").to(device)
            emb = _feature_tensor(model.get_image_features(**inputs))
            embeddings.append(F.normalize(emb, dim=-1))
    return torch.cat(embeddings, dim=0)


def compute_clip_text_embeddings(prompts, tokenizer, model, device="cuda"):
    embeddings = []
    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
            emb = _feature_tensor(model.get_text_features(**inputs))
            embeddings.append(F.normalize(emb, dim=-1))
    return torch.cat(embeddings, dim=0)


def _feature_tensor(output):
    if torch.is_tensor(output):
        return output
    for attr in ("image_embeds", "text_embeds", "pooler_output"):
        value = getattr(output, attr, None)
        if torch.is_tensor(value):
            return value
    if hasattr(output, "to_tuple"):
        for value in output.to_tuple():
            if torch.is_tensor(value):
                return value
    raise TypeError(f"Could not extract a tensor embedding from {type(output).__name__}")


def pairwise_cosine_similarity(a, b):
    return (a @ b.T).mean().item()


def pairwise_cosine_stats(a, b):
    values = (a @ b.T).detach().cpu().float().flatten()
    return {
        "mean": values.mean().item(),
        "std": values.std(unbiased=False).item() if values.numel() > 1 else 0.0,
        "min": values.min().item(),
        "max": values.max().item(),
        "count": values.numel(),
    }


def scalar_stats(values):
    if not values:
        return None
    tensor = torch.tensor(values, dtype=torch.float32)
    return {
        "mean": tensor.mean().item(),
        "std": tensor.std(unbiased=False).item() if tensor.numel() > 1 else 0.0,
        "min": tensor.min().item(),
        "max": tensor.max().item(),
        "count": tensor.numel(),
    }


def main():
    args = parse_args()
    started_at = datetime.now(timezone.utc)
    start_time = time.time()

    real_images, real_paths = load_images(args.real_images_dir)
    generated_images, generated_paths = load_images(args.generated_images_dir)

    print(f"Loaded {len(real_images)} real images, {len(generated_images)} generated images")
    if not real_images:
        raise ValueError(f"No real images found in {args.real_images_dir}")
    if not generated_images:
        raise ValueError(f"No generated images found in {args.generated_images_dir}")

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    print("Computing DINO embeddings...")
    dino_model, dino_transform = load_dino_model(device)
    real_dino = compute_dino_embeddings(real_images, dino_model, dino_transform, device)
    gen_dino = compute_dino_embeddings(generated_images, dino_model, dino_transform, device)
    dino_stats = pairwise_cosine_stats(gen_dino, real_dino)
    dino_score = dino_stats["mean"]

    print("Computing CLIP embeddings...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    real_clip = compute_clip_image_embeddings(real_images, clip_processor, clip_model, device)
    gen_clip = compute_clip_image_embeddings(generated_images, clip_processor, clip_model, device)
    clip_i_stats = pairwise_cosine_stats(gen_clip, real_clip)
    clip_i_score = clip_i_stats["mean"]

    clip_t_score = None
    clip_t_stats = None
    clip_t_rows = []
    if args.prompts_file:
        prompts_path = Path(args.prompts_file)
        metadata_dir = prompts_path.parent
        generated_root = Path(args.generated_images_dir)
        with open(prompts_path, "r") as f:
            metadata = json.load(f)

        prompt_scores = []
        for entry in metadata:
            if "prompt" not in entry or "filename" not in entry:
                continue
            prompt = entry["prompt"]
            img_path = Path(entry["filename"])
            if not img_path.is_absolute():
                metadata_relative = metadata_dir / img_path
                generated_relative = generated_root / img_path
                img_path = metadata_relative if metadata_relative.exists() else generated_relative
            with Image.open(img_path) as source:
                img = source.convert("RGB")
            img.load()

            img_emb = compute_clip_image_embeddings([img], clip_processor, clip_model, device)
            txt_emb = compute_clip_text_embeddings([prompt], clip_tokenizer, clip_model, device)
            score = (img_emb @ txt_emb.T).item()
            prompt_scores.append(score)
            clip_t_rows.append({
                "prompt": prompt,
                "filename": str(img_path),
                "clip_t": score,
            })

        clip_t_stats = scalar_stats(prompt_scores)
        clip_t_score = clip_t_stats["mean"] if clip_t_stats is not None else None
        if clip_t_stats is None:
            print(f"No prompt/filename rows found in {args.prompts_file}; skipping CLIP-T.")

    results = {
        "dino": dino_score,
        "dino_stats": dino_stats,
        "clip_i": clip_i_score,
        "clip_i_stats": clip_i_stats,
        "clip_t": clip_t_score,
        "clip_t_stats": clip_t_stats,
        "clip_t_rows": clip_t_rows,
        "num_real_images": len(real_images),
        "num_generated_images": len(generated_images),
        "real_images_dir": str(args.real_images_dir),
        "generated_images_dir": str(args.generated_images_dir),
        "real_image_paths": real_paths,
        "generated_image_paths": generated_paths,
        "prompts_file": args.prompts_file,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": time.time() - start_time,
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "torch_version": torch.__version__,
    }

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults:")
    print(f"  DINO:   {dino_score:.4f}")
    print(f"  CLIP-I: {clip_i_score:.4f}")
    if clip_t_score is not None:
        print(f"  CLIP-T: {clip_t_score:.4f}")
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
