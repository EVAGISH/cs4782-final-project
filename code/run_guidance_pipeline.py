#!/usr/bin/env python3
"""Run the local guidance experiment matrix.

The script is intentionally a thin orchestrator around the standalone project
entry points:

    generate_class_priors.py -> train_dreambooth.py -> generate.py -> evaluate.py

It skips completed cells, so interrupted runs can be resumed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
DEFAULT_RESULTS_ROOT = ROOT / "results" / "local_runs"
DEFAULT_SUBJECTS = ("backpack", "dog")
DEFAULT_SEEDS = (0, 1, 2)


def anchor_overrides(weight: float) -> dict:
    return {
        "anchor.enabled": True,
        "anchor.mode": "soft_nearest",
        "anchor.schedule": "linear_decay",
        "anchor.weight": weight,
        "anchor.temperature": 0.25,
        "anchor.consistent_noise": True,
        "semantic_guidance.enabled": False,
    }


def combined_overrides() -> dict:
    overrides = anchor_overrides(0.1)
    overrides.update(
        {
            "semantic_guidance.enabled": True,
            "semantic_guidance.weight": 0.02,
            "semantic_guidance.schedule": "warmup_hold_decay",
            "semantic_guidance.every_n_steps": 5,
            "semantic_guidance.start_step_frac": 0.2,
            "semantic_guidance.hold_until_frac": 0.6,
            "semantic_guidance.end_step_frac": 0.7,
        }
    )
    return overrides


def dino_rms_overrides(weight: float, max_update_norm_ratio: float) -> dict:
    return {
        "anchor.enabled": False,
        "semantic_guidance.enabled": True,
        "semantic_guidance.weight": weight,
        "semantic_guidance.gradient_normalization": "rms",
        "semantic_guidance.max_update_norm_ratio": max_update_norm_ratio,
        "semantic_guidance.diagnostics_enabled": False,
        "semantic_guidance.schedule": "constant_window",
        "semantic_guidance.every_n_steps": 1,
        "semantic_guidance.start_step_frac": 0.1,
        "semantic_guidance.hold_until_frac": 0.7,
        "semantic_guidance.end_step_frac": 0.8,
    }


DEFAULT_VARIANTS = [
    {
        "name": "baseline",
        "label": "Trained LoRA, no guidance",
        "model_source": "trained",
        "overrides": {"anchor.enabled": False, "semantic_guidance.enabled": False},
    },
    {
        "name": "anchor_soft_linear_w010",
        "label": "Trained LoRA + Anchor .10",
        "model_source": "trained",
        "overrides": anchor_overrides(0.1),
    },
    {
        "name": "anchor_soft_linear_w020",
        "label": "Trained LoRA + Anchor .20",
        "model_source": "trained",
        "overrides": anchor_overrides(0.2),
    },
    {
        "name": "combined_anchor010_dino002",
        "label": "Trained LoRA + Anchor .10 + DINO .02",
        "model_source": "trained",
        "overrides": combined_overrides(),
    },
    {
        "name": "base_baseline",
        "label": "Base SD, no guidance",
        "model_source": "base",
        "model_path": "runwayml/stable-diffusion-v1-5",
        "overrides": {"anchor.enabled": False, "semantic_guidance.enabled": False},
    },
    {
        "name": "base_anchor_soft_linear_w010",
        "label": "Anchor .10",
        "model_source": "base",
        "model_path": "runwayml/stable-diffusion-v1-5",
        "overrides": anchor_overrides(0.1),
    },
    {
        "name": "base_anchor_soft_linear_w020",
        "label": "Anchor .20",
        "model_source": "base",
        "model_path": "runwayml/stable-diffusion-v1-5",
        "overrides": anchor_overrides(0.2),
    },
    {
        "name": "base_combined_anchor010_dino002",
        "label": "Anchor .10 + DINO .02",
        "model_source": "base",
        "model_path": "runwayml/stable-diffusion-v1-5",
        "overrides": combined_overrides(),
    },
    {
        "name": "base_dino_rms_very_strong_w200",
        "label": "DINO RMS very strong .20",
        "model_source": "base",
        "model_path": "runwayml/stable-diffusion-v1-5",
        "overrides": dino_rms_overrides(0.2, 0.25),
    },
    {
        "name": "base_dino_rms_very_strong_w400",
        "label": "DINO RMS very strong .40",
        "model_source": "base",
        "model_path": "runwayml/stable-diffusion-v1-5",
        "subjects": ["backpack"],
        "overrides": dino_rms_overrides(0.4, 0.5),
    },
    {
        "name": "base_dino_rms_very_strong_w800",
        "label": "DINO RMS very strong .80",
        "model_source": "base",
        "model_path": "runwayml/stable-diffusion-v1-5",
        "subjects": ["backpack"],
        "overrides": dino_rms_overrides(0.8, 1.0),
    },
    {
        "name": "base_dino_rms_very_strong_w1000",
        "label": "DINO RMS very strong 1.00",
        "model_source": "base",
        "model_path": "runwayml/stable-diffusion-v1-5",
        "subjects": ["backpack"],
        "overrides": dino_rms_overrides(1.0, 1.25),
    },
]


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stages",
        default="priors,train,generate,evaluate,aggregate",
        help="Comma-separated subset of: priors, train, generate, evaluate, aggregate.",
    )
    parser.add_argument("--subjects", default=",".join(DEFAULT_SUBJECTS))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--variants-file", type=Path, default=None)
    parser.add_argument("--variants", default=None, help="Comma-separated variant names. Defaults to all curated variants.")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--class-prior-images", type=int, default=200)
    parser.add_argument("--class-prior-batch-size", type=int, default=8)
    parser.add_argument("--max-train-steps", type=int, default=1500)
    parser.add_argument("--num-images-per-prompt", type=int, default=4)
    parser.add_argument("--quick", action="store_true", help="Small smoke-sized run for checking the local setup.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run(cmd: list[str], dry_run: bool = False) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(CODE) + os.pathsep + env.get("PYTHONPATH", "")
    print("\n>>>", " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def load_variants(path: Path | None, names: list[str] | None) -> list[dict]:
    if path is None:
        variants = DEFAULT_VARIANTS
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        variants = data["variants"]
    if names is not None:
        wanted = set(names)
        variants = [variant for variant in variants if variant["name"] in wanted]
        missing = wanted - {variant["name"] for variant in variants}
        if missing:
            raise ValueError(f"Unknown variant names: {sorted(missing)}")
    return variants


def subject_train_dir(results_root: Path, subject: str, seed: int) -> Path:
    return results_root / "trained_lora" / subject / f"seed_{seed}"


def output_dir(results_root: Path, subject: str, seed: int, variant: str) -> Path:
    return results_root / "results" / subject / f"seed_{seed}" / variant


def variant_applies(variant: dict, subject: str) -> bool:
    subjects = variant.get("subjects")
    return subjects is None or subject in subjects


def hydra_overrides(mapping: dict) -> list[str]:
    def format_value(value) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)

    return [f"{key}={format_value(value)}" for key, value in mapping.items()]


def has_trained_variants(variants: list[dict]) -> bool:
    return any(variant.get("model_source", "base") == "trained" for variant in variants)


def stage_priors(args: argparse.Namespace, subjects: list[str], dry_run: bool) -> None:
    for subject in subjects:
        output_npz = ROOT / "class_priors" / f"{subject}_class_priors.npz"
        if output_npz.exists():
            print(f"SKIP priors {subject}: {output_npz} exists")
            continue
        cmd = [
            sys.executable,
            str(CODE / "generate_class_priors.py"),
            f"subject={subject}",
            f"prior_generation.num_images={args.class_prior_images}",
            f"prior_generation.batch_size={args.class_prior_batch_size}",
        ]
        run(cmd, dry_run)


def stage_train(args: argparse.Namespace, subjects: list[str], seeds: list[int], variants: list[dict], dry_run: bool) -> None:
    if not has_trained_variants(variants):
        return
    for subject in subjects:
        for seed in seeds:
            train_dir = subject_train_dir(args.results_root, subject, seed)
            if (train_dir / "lora_weights.pt").exists():
                print(f"SKIP train {subject}/seed_{seed}: {train_dir} exists")
                continue
            cmd = [
                sys.executable,
                str(CODE / "train_dreambooth.py"),
                f"subject={subject}",
                f"training.seed={seed}",
                f"training.output_dir={train_dir}",
                f"training.max_train_steps={args.max_train_steps}",
            ]
            run(cmd, dry_run)


def stage_generate(
    args: argparse.Namespace,
    subjects: list[str],
    seeds: list[int],
    variants: list[dict],
    dry_run: bool,
) -> None:
    for subject in subjects:
        for seed in seeds:
            for variant in variants:
                if not variant_applies(variant, subject):
                    continue
                variant_name = variant["name"]
                out_dir = output_dir(args.results_root, subject, seed, variant_name)
                if (out_dir / "metadata.json").exists():
                    print(f"SKIP generate {subject}/seed_{seed}/{variant_name}: metadata exists")
                    continue

                model_source = variant.get("model_source", "base")
                model_path = (
                    subject_train_dir(args.results_root, subject, seed)
                    if model_source == "trained"
                    else variant.get("model_path", "runwayml/stable-diffusion-v1-5")
                )
                overrides = {
                    "subject": subject,
                    "task": "image",
                    "model.model_path": model_path,
                    "inference.output_dir": out_dir,
                    "inference.num_images_per_prompt": args.num_images_per_prompt,
                    "runtime.seed": seed * 1000,
                }
                overrides.update(variant["overrides"])
                cmd = [sys.executable, str(CODE / "generate.py"), *hydra_overrides(overrides)]
                run(cmd, dry_run)


def stage_evaluate(
    args: argparse.Namespace,
    subjects: list[str],
    seeds: list[int],
    variants: list[dict],
    dry_run: bool,
) -> None:
    for subject in subjects:
        real_dir = ROOT / "dataset" / subject
        for seed in seeds:
            for variant in variants:
                if not variant_applies(variant, subject):
                    continue
                out_dir = output_dir(args.results_root, subject, seed, variant["name"])
                metadata = out_dir / "metadata.json"
                metrics = out_dir / "metrics.json"
                if metrics.exists():
                    print(f"SKIP evaluate {subject}/seed_{seed}/{variant['name']}: metrics exists")
                    continue
                if not metadata.exists():
                    print(f"SKIP evaluate {subject}/seed_{seed}/{variant['name']}: no metadata at {metadata}")
                    continue
                cmd = [
                    sys.executable,
                    str(CODE / "evaluate.py"),
                    "--real_images_dir",
                    str(real_dir),
                    "--generated_images_dir",
                    str(out_dir),
                    "--prompts_file",
                    str(metadata),
                    "--output_file",
                    str(metrics),
                ]
                run(cmd, dry_run)


def stage_aggregate(args: argparse.Namespace, variants: list[dict]) -> None:
    variant_labels = {variant["name"]: variant.get("label", variant["name"]) for variant in variants}
    rows = []
    for metrics_path in sorted((args.results_root / "results").glob("*/*/*/metrics.json")):
        variant_dir = metrics_path.parent
        seed_dir = variant_dir.parent
        subject_dir = seed_dir.parent
        variant = variant_dir.name
        seed = int(seed_dir.name.removeprefix("seed_"))
        subject = subject_dir.name
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "subject": subject,
                "train_seed": seed,
                "variant": variant,
                "label": variant_labels.get(variant, variant),
                "dino": metrics.get("dino"),
                "clip_i": metrics.get("clip_i"),
                "clip_t": metrics.get("clip_t"),
                "num_generated_images": metrics.get("num_generated_images"),
                "metrics_path": str(metrics_path),
            }
        )

    output = args.results_root / "metrics_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "subject",
        "train_seed",
        "variant",
        "label",
        "dino",
        "clip_i",
        "clip_t",
        "num_generated_images",
        "metrics_path",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output}")


def main() -> None:
    args = parse_args()
    if args.quick:
        args.class_prior_images = min(args.class_prior_images, 4)
        args.class_prior_batch_size = min(args.class_prior_batch_size, 2)
        args.max_train_steps = min(args.max_train_steps, 2)
        args.num_images_per_prompt = min(args.num_images_per_prompt, 1)
        args.seeds = "0"
        args.variants = args.variants or "base_baseline,base_anchor_soft_linear_w010,base_dino_rms_very_strong_w200"

    subjects = parse_csv_strings(args.subjects)
    seeds = parse_csv_ints(args.seeds)
    variant_names = parse_csv_strings(args.variants) if args.variants else None
    variants = load_variants(args.variants_file, variant_names)
    stages = set(parse_csv_strings(args.stages))

    if "priors" in stages and has_trained_variants(variants):
        stage_priors(args, subjects, args.dry_run)
    elif "priors" in stages:
        print("SKIP priors: no trained LoRA variants selected")
    if "train" in stages:
        stage_train(args, subjects, seeds, variants, args.dry_run)
    if "generate" in stages:
        stage_generate(args, subjects, seeds, variants, args.dry_run)
    if "evaluate" in stages:
        stage_evaluate(args, subjects, seeds, variants, args.dry_run)
    if "aggregate" in stages and not args.dry_run:
        stage_aggregate(args, variants)
    elif "aggregate" in stages:
        print("SKIP aggregate: dry run")


if __name__ == "__main__":
    main()
