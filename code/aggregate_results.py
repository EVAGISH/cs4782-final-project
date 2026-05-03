"""Aggregate metrics.json + run_stats.json across all results into one CSV.

Walks the results/ tree, pairs each generated-images directory (which contains
metrics.json from evaluate.py) with the matching trained-model directory (which
contains run_stats.json from train_dreambooth.py) by stripping a trailing
"_results" suffix.

Output: results/all_metrics.csv  -- single source of truth for the poster.
"""

import argparse
import csv
import json
from pathlib import Path


COLUMNS = [
    "subject",
    "method",
    "dino",
    "clip_i",
    "clip_t",
    "trainable_params",
    "trainable_pct",
    "train_time_sec",
    "peak_vram_gb",
    "ckpt_size_mb",
    "max_train_steps",
    "lora_rank",
    "prior_loss_weight",
    "results_dir",
    "model_dir",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", type=str, default="results")
    parser.add_argument("--output", type=str, default="results/all_metrics.csv")
    return parser.parse_args()


KNOWN_SUBJECTS = ["bear_plushie", "backpack", "dog", "cat"]
KNOWN_METHODS = {"base", "full", "lora"}


def infer_subject_method(results_dir: Path):
    """Split 'backpack_lora_results' into ('backpack', 'lora'). Handles
    multi-word subjects like 'bear_plushie' by checking the longest known
    subject prefix first."""
    stem = results_dir.name
    if stem.endswith("_results"):
        stem = stem[: -len("_results")]
    for subj in sorted(KNOWN_SUBJECTS, key=len, reverse=True):
        if stem == subj:
            return subj, "unknown"
        if stem.startswith(subj + "_"):
            return subj, stem[len(subj) + 1:]
    parts = stem.split("_")
    if len(parts) >= 2:
        return parts[0], "_".join(parts[1:])
    return stem, "unknown"


def find_matching_model_dir(results_dir: Path):
    """Given results/<name>_results/, look for results/<name>/ holding run_stats.json."""
    name = results_dir.name
    if name.endswith("_results"):
        candidate = results_dir.parent / name[: -len("_results")]
        if candidate.is_dir():
            return candidate
    return None


def main():
    args = parse_args()
    root = Path(args.results_root)
    rows = []

    for metrics_path in sorted(root.rglob("metrics.json")):
        results_dir = metrics_path.parent
        with open(metrics_path) as f:
            metrics = json.load(f)

        subject, method = infer_subject_method(results_dir)
        model_dir = find_matching_model_dir(results_dir)

        row = {
            "subject": subject,
            "method": method,
            "dino": metrics.get("dino"),
            "clip_i": metrics.get("clip_i"),
            "clip_t": metrics.get("clip_t"),
            "results_dir": str(results_dir),
            "model_dir": str(model_dir) if model_dir else "",
        }

        if model_dir is not None:
            stats_path = model_dir / "run_stats.json"
            if stats_path.exists():
                with open(stats_path) as f:
                    stats = json.load(f)
                if stats.get("subject"):
                    row["subject"] = stats["subject"]
                if stats.get("method"):
                    row["method"] = stats["method"]
                for key in (
                    "trainable_params",
                    "trainable_pct",
                    "train_time_sec",
                    "peak_vram_gb",
                    "ckpt_size_mb",
                    "max_train_steps",
                    "lora_rank",
                    "prior_loss_weight",
                ):
                    row[key] = stats.get(key)

        if row["method"] not in KNOWN_METHODS:
            continue
        rows.append(row)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in COLUMNS})

    print(f"Wrote {len(rows)} rows to {output_path}")
    for row in rows:
        dino = row.get("dino")
        clip_i = row.get("clip_i")
        clip_t = row.get("clip_t")
        dino_s = f"{dino:.3f}" if isinstance(dino, (int, float)) else "  -  "
        ci_s = f"{clip_i:.3f}" if isinstance(clip_i, (int, float)) else "  -  "
        ct_s = f"{clip_t:.3f}" if isinstance(clip_t, (int, float)) else "  -  "
        print(f"  {row['subject']:>14} | {row['method']:>10} | "
              f"DINO {dino_s} | CLIP-I {ci_s} | CLIP-T {ct_s}")


if __name__ == "__main__":
    main()
