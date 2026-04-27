import argparse
import json
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
        images.append(Image.open(p).convert("RGB"))
        paths.append(str(p))
    for p in sorted(image_dir.rglob("*.jpg")):
        images.append(Image.open(p).convert("RGB"))
        paths.append(str(p))
    return images, paths


def compute_dino_embeddings(images, device="cuda"):
    model = torch.hub.load("facebookresearch/dino:main", "dino_vits16")
    model.eval().to(device)

    transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    embeddings = []
    with torch.no_grad():
        for img in images:
            tensor = transform(img).unsqueeze(0).to(device)
            emb = model(tensor)
            embeddings.append(F.normalize(emb, dim=-1))

    return torch.cat(embeddings, dim=0)


def compute_clip_image_embeddings(images, processor, model, device="cuda"):
    embeddings = []
    with torch.no_grad():
        for img in images:
            inputs = processor(images=img, return_tensors="pt").to(device)
            emb = model.get_image_features(**inputs)
            embeddings.append(F.normalize(emb, dim=-1))
    return torch.cat(embeddings, dim=0)


def compute_clip_text_embeddings(prompts, tokenizer, model, device="cuda"):
    embeddings = []
    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
            emb = model.get_text_features(**inputs)
            embeddings.append(F.normalize(emb, dim=-1))
    return torch.cat(embeddings, dim=0)


def pairwise_cosine_similarity(a, b):
    return (a @ b.T).mean().item()


def main():
    args = parse_args()

    real_images, _ = load_images(args.real_images_dir)
    generated_images, _ = load_images(args.generated_images_dir)

    print(f"Loaded {len(real_images)} real images, {len(generated_images)} generated images")

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    print("Computing DINO embeddings...")
    real_dino = compute_dino_embeddings(real_images, device)
    gen_dino = compute_dino_embeddings(generated_images, device)
    dino_score = pairwise_cosine_similarity(gen_dino, real_dino)

    print("Computing CLIP embeddings...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    real_clip = compute_clip_image_embeddings(real_images, clip_processor, clip_model, device)
    gen_clip = compute_clip_image_embeddings(generated_images, clip_processor, clip_model, device)
    clip_i_score = pairwise_cosine_similarity(gen_clip, real_clip)

    clip_t_score = None
    if args.prompts_file:
        with open(args.prompts_file, "r") as f:
            metadata = json.load(f)

        prompt_scores = []
        for entry in metadata:
            prompt = entry["prompt"]
            img_path = entry["filename"]
            img = Image.open(img_path).convert("RGB")

            img_emb = compute_clip_image_embeddings([img], clip_processor, clip_model, device)
            txt_emb = compute_clip_text_embeddings([prompt], clip_tokenizer, clip_model, device)
            prompt_scores.append((img_emb @ txt_emb.T).item())

        clip_t_score = sum(prompt_scores) / len(prompt_scores)

    results = {
        "dino": dino_score,
        "clip_i": clip_i_score,
        "clip_t": clip_t_score,
        "num_real_images": len(real_images),
        "num_generated_images": len(generated_images),
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
