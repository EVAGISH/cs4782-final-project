"""End-to-end orchestrator: train -> generate -> evaluate for the full matrix.

Edit the SUBJECTS / METHODS section below if you need to change scope. Then:

    python code/run_pipeline.py --stages train,generate,evaluate

Stages can be run independently:
    --stages train       # only training (~2 hrs on RTX 4090)
    --stages generate    # only inference  (~60 min)
    --stages evaluate    # only evaluate.py over already-generated images

Each stage skips cells whose outputs already exist, so it's safe to re-run.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

sys.path.insert(0, str(Path(__file__).parent))
from class_priors import append_class_prior_images, pil_to_uint8_rgb  # noqa: E402


# ---------------------------------------------------------------------------
# Configuration -- edit subjects/methods/paths here
# ---------------------------------------------------------------------------

PRETRAINED_MODEL = "runwayml/stable-diffusion-v1-5"
INSTANCE_DATA_ROOT = Path("data/instance_images")
CLASS_DATA_ROOT = Path("data/class_images")
CLASS_PRIORS_ROOT = Path("class_priors")
RESULTS_ROOT = Path("results")
PROMPTS_TEMPLATE = Path("code/prompts.json")

# subject_dir_name -> class noun (used in prompts and class-image folder name)
SUBJECTS = {
    "dog":          "dog",
    "cat":          "cat",
    "backpack":     "backpack",
    "bear_plushie": "stuffed bear",
}

# Methods to run for each subject. "base" = no training, just inference.
METHODS = ["base", "full", "lora"]

V_TOKEN = "sks"
MAX_TRAIN_STEPS = 1000
LEARNING_RATE = 5e-6
LEARNING_RATE_LORA = 1e-4
PRIOR_LOSS_WEIGHT_FULL = 1.0
PRIOR_LOSS_WEIGHT_LORA = 0.5
NUM_CLASS_IMAGES = 200
LORA_RANK = 16
LORA_ALPHA = 16.0
NUM_IMAGES_PER_PROMPT = 4
GUIDANCE_SCALE = 7.5
NUM_INFERENCE_STEPS = 50


_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def pack_class_images_to_npz(class_images_dir: Path, npz_path: Path) -> bool:
    """Pack a directory of class images into an .npz file. Returns True if packed."""
    if npz_path.exists():
        return True
    images = [
        pil_to_uint8_rgb(PILImage.open(p))
        for p in sorted(class_images_dir.iterdir())
        if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
    ]
    if not images:
        return False
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(npz_path), images=np.stack(images, axis=0))
    print(f"Packed {len(images)} class images -> {npz_path}")
    return True


def slugify(s):
    return s.lower().replace(" ", "_")


def model_dir_for(subject, method):
    return RESULTS_ROOT / f"{subject}_{method}"


def results_dir_for(subject, method):
    return RESULTS_ROOT / f"{subject}_{method}_results"


def class_dir_for(class_noun):
    return CLASS_DATA_ROOT / slugify(class_noun)


def run(cmd):
    print(f"\n>>> {' '.join(str(c) for c in cmd)}")
    subprocess.run([str(c) for c in cmd], check=True)


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def stage_train():
    for subject, class_noun in SUBJECTS.items():
        instance_dir = INSTANCE_DATA_ROOT / subject
        if not instance_dir.exists():
            print(f"SKIP train {subject}: {instance_dir} not found")
            continue

        class_dir = class_dir_for(class_noun)
        npz_path = CLASS_PRIORS_ROOT / f"{subject}_class_priors.npz"
        if not pack_class_images_to_npz(class_dir, npz_path):
            print(f"SKIP train {subject}: class images missing at {class_dir}")
            continue

        for method in METHODS:
            if method == "base":
                continue  # no training for the base SD baseline
            output_dir = model_dir_for(subject, method)
            done_marker = output_dir / "run_stats.json"
            if done_marker.exists():
                print(f"SKIP train {subject}/{method}: already complete")
                continue

            prior_w = PRIOR_LOSS_WEIGHT_LORA if method == "lora" else PRIOR_LOSS_WEIGHT_FULL
            lr = LEARNING_RATE_LORA if method == "lora" else LEARNING_RATE
            mode = "lora" if method == "lora" else "full"
            cmd = [
                sys.executable, "code/train_dreambooth.py",
                f"subject={subject}",
                f"mode={mode}",
                f"training.output_dir={str(output_dir)}",
                f"training.prior_loss_weight={prior_w}",
                f"training.max_train_steps={MAX_TRAIN_STEPS}",
                f"training.learning_rate={lr}",
                f"data.class_images_npz={str(npz_path)}",
            ]
            if method == "lora":
                cmd += [
                    f"lora.rank={LORA_RANK}",
                    f"lora.alpha={LORA_ALPHA}",
                ]
            run(cmd)


def stage_generate():
    for subject, class_noun in SUBJECTS.items():
        for method in METHODS:
            results_dir = results_dir_for(subject, method)
            done_marker = results_dir / "metadata.json"
            if done_marker.exists():
                print(f"SKIP generate {subject}/{method}: already complete")
                continue

            cmd = [
                sys.executable, "code/generate.py",
                f"subject={subject}",
                f"inference.output_dir={str(results_dir)}",
                f"inference.num_images_per_prompt={NUM_IMAGES_PER_PROMPT}",
                f"inference.guidance_scale={GUIDANCE_SCALE}",
                f"inference.num_inference_steps={NUM_INFERENCE_STEPS}",
            ]

            if method == "base":
                cmd += [
                    "mode=full",
                    f"model.model_path={PRETRAINED_MODEL}",
                ]
            elif method == "full":
                model_dir = model_dir_for(subject, "full")
                if not (model_dir / "model_index.json").exists():
                    print(f"SKIP generate {subject}/full: trained model not found at {model_dir}")
                    continue
                cmd += [
                    "mode=full",
                    f"model.model_path={str(model_dir)}",
                ]
            elif method == "lora":
                model_dir = model_dir_for(subject, "lora")
                if not (model_dir / "lora_weights.pt").exists():
                    print(f"SKIP generate {subject}/lora: LoRA weights not found at {model_dir}")
                    continue
                cmd += [
                    "mode=lora",
                    f"model.model_path={str(model_dir)}",
                ]
            run(cmd)


def stage_evaluate():
    for subject in SUBJECTS:
        real_dir = INSTANCE_DATA_ROOT / subject
        if not real_dir.exists():
            print(f"SKIP evaluate {subject}: {real_dir} not found")
            continue
        for method in METHODS:
            results_dir = results_dir_for(subject, method)
            metadata = results_dir / "metadata.json"
            metrics = results_dir / "metrics.json"
            if not metadata.exists():
                print(f"SKIP evaluate {subject}/{method}: no generations at {results_dir}")
                continue
            if metrics.exists():
                print(f"SKIP evaluate {subject}/{method}: metrics already computed")
                continue
            cmd = [
                sys.executable, "code/evaluate.py",
                "--real_images_dir", str(real_dir),
                "--generated_images_dir", str(results_dir),
                "--prompts_file", str(metadata),
                "--output_file", str(metrics),
            ]
            run(cmd)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", type=str, default="train,generate,evaluate",
                        help="Comma-separated subset of: train, generate, evaluate")
    return parser.parse_args()


def main():
    args = parse_args()
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    if "train" in stages:
        print("\n========== STAGE: train ==========")
        stage_train()
    if "generate" in stages:
        print("\n========== STAGE: generate ==========")
        stage_generate()
    if "evaluate" in stages:
        print("\n========== STAGE: evaluate ==========")
        stage_evaluate()


if __name__ == "__main__":
    main()
