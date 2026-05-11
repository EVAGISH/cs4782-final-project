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


# ---------------------------------------------------------------------------
# Configuration -- edit subjects/methods/paths here
# ---------------------------------------------------------------------------

PRETRAINED_MODEL = "runwayml/stable-diffusion-v1-5"
INSTANCE_DATA_ROOT = Path("dreambooth/dataset")
CLASS_DATA_ROOT = Path("data/class_images")
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
        class_dir = class_dir_for(class_noun)
        if not instance_dir.exists():
            print(f"SKIP train {subject}: {instance_dir} not found")
            continue
        if not class_dir.exists() or not any(class_dir.iterdir()):
            print(f"SKIP train {subject}: class images missing at {class_dir} "
                  f"(run generate_class_images.py first)")
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
            cmd = [
                sys.executable, "code/train_dreambooth.py",
                "--pretrained_model", PRETRAINED_MODEL,
                "--instance_data_dir", str(instance_dir),
                "--class_data_dir", str(class_dir),
                "--output_dir", str(output_dir),
                "--instance_prompt", f"a photo of {V_TOKEN} {class_noun}",
                "--class_prompt", f"a photo of a {class_noun}",
                "--num_class_images", str(NUM_CLASS_IMAGES),
                "--max_train_steps", str(MAX_TRAIN_STEPS),
                "--learning_rate", str(LEARNING_RATE_LORA if method == "lora" else LEARNING_RATE),
                "--prior_loss_weight", str(prior_w),
                "--mixed_precision", "no",
            ]
            if method == "lora":
                cmd += [
                    "--use_lora",
                    "--lora_rank", str(LORA_RANK),
                    "--lora_alpha", str(LORA_ALPHA),
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
                "--prompts_template", str(PROMPTS_TEMPLATE),
                "--class_noun", class_noun,
                "--output_dir", str(results_dir),
                "--num_images_per_prompt", str(NUM_IMAGES_PER_PROMPT),
                "--guidance_scale", str(GUIDANCE_SCALE),
                "--num_inference_steps", str(NUM_INFERENCE_STEPS),
            ]

            if method == "base":
                cmd += [
                    "--model_path", PRETRAINED_MODEL,
                    "--v_token", "",
                ]
            elif method == "full":
                model_dir = model_dir_for(subject, "full")
                if not (model_dir / "model_index.json").exists():
                    print(f"SKIP generate {subject}/full: trained model not found at {model_dir}")
                    continue
                cmd += [
                    "--model_path", str(model_dir),
                    "--v_token", V_TOKEN,
                ]
            elif method == "lora":
                model_dir = model_dir_for(subject, "lora")
                if not (model_dir / "lora_weights.pt").exists():
                    print(f"SKIP generate {subject}/lora: LoRA weights not found at {model_dir}")
                    continue
                cmd += [
                    "--model_path", PRETRAINED_MODEL,
                    "--lora_path", str(model_dir),
                    "--v_token", V_TOKEN,
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
