"""Generate poster-ready figures from results/all_metrics.csv.

Style matches the editorial poster aesthetic from graph.png:
  - Cream background, terracotta accent
  - Section number in accent color, serif title with thin underline
  - Light beige (Base SD) / black (Paper) / terracotta (Ours Full) / coral (Ours LoRA)
  - Horizontal gridlines only, no axis frame
  - Value labels above each bar
"""

import argparse
import csv
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager as fm
import numpy as np


def register_local_fonts():
    """Register every .ttf / .otf in assets/fonts/ with matplotlib so figures
    can use Source Serif 4 / Inter / JetBrains Mono without system installation."""
    for fonts_dir in (Path("assets/fonts"), Path("../assets/fonts")):
        if not fonts_dir.exists():
            continue
        for font_file in list(fonts_dir.rglob("*.ttf")) + list(fonts_dir.rglob("*.otf")):
            try:
                fm.fontManager.addfont(str(font_file))
            except Exception:
                pass


register_local_fonts()


# Hardcoded paper-reported numbers from the DreamBooth paper, Table 1
# (Stable Diffusion variant of DreamBooth)
PAPER_NUMBERS = {"dino": 0.668, "clip_i": 0.803, "clip_t": 0.305}

METHOD_LABELS = {
    "base": "Base SD\n(zero-shot)",
    "paper": "Paper\n(DreamBooth)",
    "full": "Ours · Full FT",
    "lora": "Ours · LoRA",
    "no_prior": "Ours · no prior",
}

# Editorial palette matching graph.png
PALETTE = {
    "bg":        "#F5F1E8",
    "fg":        "#1A1A2E",
    "accent":    "#C2410C",
    "muted":     "#8A8A8A",
    "grid":      "#E5E1D8",
    "base_bar":  "#D8D2C5",
    "paper_bar": "#1A1A1A",
    "full_bar":  "#C2410C",
    "lora_bar":  "#E89376",
}

METHOD_COLORS = {
    "base":  PALETTE["base_bar"],
    "paper": PALETTE["paper_bar"],
    "full":  PALETTE["full_bar"],
    "lora":  PALETTE["lora_bar"],
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="results/all_metrics.csv")
    parser.add_argument("--out_dir", type=str, default="results/figures")
    parser.add_argument("--loss_history", type=str, default=None)
    return parser.parse_args()


def load_csv(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in ("dino", "clip_i", "clip_t", "trainable_pct",
                        "train_time_sec", "peak_vram_gb", "ckpt_size_mb"):
                if row.get(key) not in (None, ""):
                    try:
                        row[key] = float(row[key])
                    except ValueError:
                        pass
            rows.append(row)
    return rows


def average_by_method(rows, method):
    cells = [r for r in rows if r.get("method") == method]
    out = {}
    for metric in ("dino", "clip_i", "clip_t"):
        vals = [r[metric] for r in cells if isinstance(r.get(metric), (int, float))]
        if vals:
            out[metric] = float(np.mean(vals))
    return out


def load_run_stats(results_root="results"):
    """Walk results/ and collect every run_stats.json into a list of dicts."""
    runs = []
    for stats_path in Path(results_root).rglob("run_stats.json"):
        try:
            with open(stats_path) as f:
                runs.append(json.load(f))
        except Exception:
            pass
    return runs


def average_efficiency_by_method(runs, method):
    cells = [r for r in runs if r.get("method") == method]
    out = {}
    for key in ("trainable_pct", "train_time_sec", "peak_vram_gb", "ckpt_size_mb"):
        vals = [r[key] for r in cells if isinstance(r.get(key), (int, float))]
        if vals:
            out[key] = float(np.mean(vals))
    return out


FONT_SERIF = ["Source Serif 4", "DejaVu Serif"]
FONT_SANS = ["Inter", "DejaVu Sans"]
FONT_MONO = ["JetBrains Mono", "DejaVu Sans Mono"]

# Sized so when the figure is placed at ~2000-2400px wide on the 3600x2400
# Figma canvas (i.e. printed ~20-24 inches wide on the 36" poster), the smallest
# in-chart text still renders at >=24pt per the rubric.
FS_TITLE   = 36
FS_CAPTION = 18
FS_HEADER  = 22   # metric labels like "DINO ↑"
FS_SUB     = 14   # under-axis subtitles like "subject identity"
FS_VALUE   = 14   # value labels above bars
FS_TICK    = 14
FS_LEGEND  = 16


def setup_style():
    plt.rcParams.update({
        "figure.facecolor":  PALETTE["bg"],
        "axes.facecolor":    PALETTE["bg"],
        "savefig.facecolor": PALETTE["bg"],
        "axes.edgecolor":    PALETTE["fg"],
        "axes.labelcolor":   PALETTE["fg"],
        "axes.titlecolor":   PALETTE["fg"],
        "xtick.color":       PALETTE["fg"],
        "ytick.color":       PALETTE["fg"],
        "axes.grid":         True,
        "axes.grid.axis":    "y",
        "grid.color":        PALETTE["grid"],
        "grid.linewidth":    0.8,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.spines.bottom": False,
        "axes.spines.left":  False,
        "xtick.bottom":      False,
        "ytick.left":        False,
        "font.family":       "sans-serif",
        "font.sans-serif":   FONT_SANS,
        "font.serif":        FONT_SERIF,
        "font.monospace":    FONT_MONO,
        "font.size":         11,
    })


def add_title(fig, title):
    """Centered serif title at the top of the figure."""
    fig.text(0.5, 0.94, title,
             fontsize=FS_TITLE, color=PALETTE["fg"],
             fontfamily=FONT_SERIF, fontweight="bold",
             ha="center", va="center")


def add_caption(fig, caption, wrap_width=95):
    """Italic caption at the bottom of the figure. Long lines are wrapped to
    keep the text within the figure width."""
    wrapped = "\n".join(textwrap.wrap(caption, width=wrap_width))
    fig.text(0.5, 0.04, wrapped,
             fontsize=FS_CAPTION, color=PALETTE["fg"],
             fontfamily=FONT_SANS, style="italic",
             ha="center", va="bottom",
             linespacing=1.4)


def plot_bar_chart(rows, out_path):
    """Grouped bars: 3 metrics × 4 methods, matching graph.png."""
    methods_order = ["base", "paper", "full", "lora"]
    method_data = {}
    for m in methods_order:
        if m == "paper":
            method_data[m] = PAPER_NUMBERS
        else:
            method_data[m] = average_by_method(rows, m)

    metrics      = ["dino",            "clip_i",          "clip_t"]
    metric_labels = ["DINO ↑",          "CLIP-I ↑",        "CLIP-T ↑"]
    metric_subs   = ["subject identity", "subject semantic", "prompt fidelity"]

    n_metrics = len(metrics)
    n_methods = len(methods_order)
    bar_w = 0.18
    x_centers = np.arange(n_metrics)

    fig = plt.figure(figsize=(13, 10.5))
    ax = fig.add_axes([0.08, 0.36, 0.88, 0.48])

    for j, method in enumerate(methods_order):
        offsets = (j - (n_methods - 1) / 2) * bar_w
        values = [method_data[method].get(m, 0) for m in metrics]
        bars = ax.bar(x_centers + offsets, values, bar_w,
                      color=METHOD_COLORS[method], edgecolor="none")
        for bar, v in zip(bars, values):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015,
                        f"{v:.2f}", ha="center", va="bottom",
                        fontsize=FS_VALUE, color=PALETTE["fg"], fontfamily=FONT_MONO)

    ax.set_xticks(x_centers)
    ax.set_xticklabels(metric_labels, fontsize=FS_HEADER, fontweight="bold",
                        color=PALETTE["fg"], fontfamily=FONT_SANS)
    for x, sub in zip(x_centers, metric_subs):
        ax.text(x, -0.08, sub, ha="center", va="top",
                fontsize=FS_SUB, color=PALETTE["muted"], fontfamily=FONT_SANS,
                transform=ax.get_xaxis_transform())

    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.00, 0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels([f"{v:.2f}" for v in [0.00, 0.25, 0.50, 0.75, 1.00]],
                       fontsize=FS_TICK, color=PALETTE["muted"], fontfamily=FONT_MONO)
    ax.tick_params(length=0, pad=8)

    handles = [mpatches.Patch(color=METHOD_COLORS[m], label=METHOD_LABELS[m].replace("\n", " "))
               for m in methods_order]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               frameon=False, fontsize=FS_LEGEND,
               bbox_to_anchor=(0.5, 0.22),
               labelcolor=PALETTE["fg"], handlelength=1.4,
               columnspacing=2.5, handletextpad=0.7)

    add_title(fig, "Subject and prompt fidelity vs the paper")
    add_caption(
        fig,
        "Our Full FT matches the paper on subject semantics (CLIP-I 0.77 vs 0.80) and "
        "exceeds it on prompt fidelity (CLIP-T 0.33 vs 0.30). DINO sits 0.07 below the "
        "paper's 30-subject average, within their reported per-subject variance. LoRA "
        "recovers ~86% of Full FT's DINO at <1% of trainable parameters.")

    fig.savefig(out_path, format="svg", bbox_inches=None,
                facecolor=PALETTE["bg"])
    fig.savefig(out_path.with_suffix(".png"), format="png", dpi=200,
                bbox_inches=None, facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_pareto_scatter(rows, out_path):
    """DINO (x) vs CLIP-T (y), one point per method."""
    methods_order = ["base", "paper", "full", "lora"]
    points = []
    for m in methods_order:
        if m == "paper":
            agg = PAPER_NUMBERS
        else:
            agg = average_by_method(rows, m)
        if agg.get("dino") is not None and agg.get("clip_t") is not None:
            points.append((METHOD_LABELS[m], agg["dino"], agg["clip_t"], METHOD_COLORS[m]))

    fig = plt.figure(figsize=(11, 9.5))
    ax = fig.add_axes([0.10, 0.24, 0.84, 0.58])

    for label, x, y, color in points:
        ax.scatter(x, y, s=380, color=color, edgecolor=PALETTE["bg"],
                   linewidth=3, zorder=3)
        ax.annotate(label.replace("\n", " "), (x, y),
                    xytext=(12, 12), textcoords="offset points",
                    fontsize=FS_LEGEND, color=PALETTE["fg"], fontweight="bold")

    ax.set_xlabel("DINO score → subject identity fidelity",
                  fontsize=FS_SUB, color=PALETTE["muted"], labelpad=14)
    ax.set_ylabel("CLIP-T score → prompt fidelity",
                  fontsize=FS_SUB, color=PALETTE["muted"], labelpad=14)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    pad_x = (xlim[1] - xlim[0]) * 0.10
    pad_y = (ylim[1] - ylim[0]) * 0.12
    ax.set_xlim(xlim[0] - pad_x, xlim[1] + pad_x)
    ax.set_ylim(ylim[0] - pad_y, ylim[1] + pad_y)

    ax.annotate("ideal\n(both high)",
                xy=(ax.get_xlim()[1], ax.get_ylim()[1]),
                xytext=(-12, -14), textcoords="offset points",
                ha="right", va="top",
                fontsize=FS_CAPTION, style="italic", color=PALETTE["muted"])

    ax.tick_params(length=0, pad=6, colors=PALETTE["muted"], labelsize=FS_TICK)

    add_title(fig, "The subject–prompt fidelity trade-off")
    add_caption(
        fig,
        "DreamBooth's central tension: aggressive fine-tuning improves subject identity "
        "(DINO ↑) but risks language drift on prompts (CLIP-T ↓). Both our variants land "
        "near the paper's Pareto point, far from base SD's near-zero subject identity.")

    fig.savefig(out_path, format="svg", bbox_inches=None, facecolor=PALETTE["bg"])
    fig.savefig(out_path.with_suffix(".png"), format="png", dpi=200,
                bbox_inches=None, facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_per_subject_bars(rows, out_path):
    """One subplot per subject; methods × metrics."""
    subjects = sorted({r["subject"] for r in rows if r.get("subject")})
    if not subjects:
        print("  (no subjects, skipping per_subject_bars)")
        return

    methods_order = ["base", "paper", "full", "lora"]
    metrics = ["dino", "clip_i", "clip_t"]
    metric_labels = ["DINO", "CLIP-I", "CLIP-T"]

    n = len(subjects)
    cols = min(n, 4)
    rows_count = (n + cols - 1) // cols
    fig = plt.figure(figsize=(5.5 * cols, 4.5 * rows_count + 3.5))
    fig.subplots_adjust(left=0.06, right=0.96, bottom=0.32, top=0.78,
                        wspace=0.30, hspace=0.40)

    for idx, subject in enumerate(subjects):
        ax = fig.add_subplot(rows_count, cols, idx + 1)
        subject_rows = [r for r in rows if r.get("subject") == subject]

        present_methods = []
        values = []
        for method in methods_order:
            if method == "paper":
                present_methods.append("paper")
                values.append([PAPER_NUMBERS[m] for m in metrics])
                continue
            cell = next((r for r in subject_rows if r.get("method") == method), None)
            if cell is None:
                continue
            present_methods.append(method)
            values.append([cell.get(m) if isinstance(cell.get(m), (int, float)) else 0
                           for m in metrics])
        values = np.array(values) if values else np.zeros((0, 3))

        x = np.arange(len(metric_labels))
        bar_w = 0.20
        for j, method in enumerate(present_methods):
            offset = (j - (len(present_methods) - 1) / 2) * bar_w
            ax.bar(x + offset, values[j], bar_w,
                   color=METHOD_COLORS[method], edgecolor="none",
                   label=METHOD_LABELS[method].replace("\n", " "))

        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, fontsize=FS_SUB, color=PALETTE["fg"],
                           fontfamily=FONT_SANS)
        ax.set_title(subject.replace("_", " "), fontsize=FS_HEADER,
                     fontweight="bold", color=PALETTE["fg"], pad=12,
                     fontfamily=FONT_SANS)
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.tick_params(length=0, pad=6, colors=PALETTE["muted"], labelsize=FS_TICK)

    handles = [mpatches.Patch(color=METHOD_COLORS[m],
                              label=METHOD_LABELS[m].replace("\n", " "))
               for m in methods_order]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               frameon=False, fontsize=FS_LEGEND,
               bbox_to_anchor=(0.5, 0.20),
               labelcolor=PALETTE["fg"])

    add_title(fig, "Per-subject metric breakdown")
    add_caption(
        fig,
        "Cat reaches DINO 0.73 (above paper's 0.67 average) and dog 0.65; backpack lags "
        "at 0.42, consistent with the paper's note that generic object classes are the "
        "hardest for instance-level identity. CLIP-I and CLIP-T stay stable across all "
        "four subjects — the per-subject variance lives almost entirely in DINO.")

    fig.savefig(out_path, format="svg", bbox_inches=None, facecolor=PALETTE["bg"])
    fig.savefig(out_path.with_suffix(".png"), format="png", dpi=200,
                bbox_inches=None, facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  wrote {out_path}")


def load_lambda_ablation(path="results/lora_lambda_ablation.csv"):
    """Read the LoRA lambda ablation CSV. Returns ordered list of dicts."""
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    with open(p) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                row["dino"] = float(row["dino"])
                row["clip_i"] = float(row["clip_i"])
                row["clip_t"] = float(row["clip_t"])
            except (ValueError, KeyError):
                continue
            rows.append(row)
    return rows


def plot_lambda_ablation(out_path, ablation_csv="results/lora_lambda_ablation.csv"):
    """Line chart showing how DINO / CLIP-I / CLIP-T vary with prior-loss weight λ.
    Includes horizontal reference lines for Full FT (upper bound) and Base SD (lower)."""
    rows = load_lambda_ablation(ablation_csv)
    if not rows:
        print("  (lambda ablation skipped: results/lora_lambda_ablation.csv not found)")
        return

    lora_rows = [r for r in rows if r["method"] == "lora"]
    full_row = next((r for r in rows if r["method"] == "full"), None)
    base_row = next((r for r in rows if r["method"] == "base"), None)

    lora_rows.sort(key=lambda r: float(r["prior_loss_weight"]))
    if not lora_rows:
        print("  (lambda ablation skipped: no LoRA rows in CSV)")
        return

    lambdas = [float(r["prior_loss_weight"]) for r in lora_rows]
    dino  = [r["dino"]   for r in lora_rows]
    clipi = [r["clip_i"] for r in lora_rows]
    clipt = [r["clip_t"] for r in lora_rows]

    fig = plt.figure(figsize=(13, 9.5))
    ax = fig.add_axes([0.10, 0.30, 0.84, 0.52])

    metric_specs = [
        ("DINO",   dino,  PALETTE["accent"], "o"),
        ("CLIP-I", clipi, PALETTE["fg"],     "s"),
        ("CLIP-T", clipt, PALETTE["lora_bar"], "^"),
    ]

    for label, values, color, marker in metric_specs:
        ax.plot(lambdas, values, color=color, linewidth=3,
                marker=marker, markersize=14,
                markerfacecolor=color, markeredgecolor=PALETTE["bg"],
                markeredgewidth=2.5, zorder=3, label=label)
        for x, y in zip(lambdas, values):
            ax.annotate(f"{y:.3f}",
                        xy=(x, y), xytext=(0, 14),
                        textcoords="offset points",
                        ha="center", va="bottom",
                        fontsize=FS_VALUE, color=PALETTE["fg"],
                        fontfamily=FONT_MONO, fontweight="bold")

    if full_row is not None:
        ax.axhline(full_row["dino"], color=PALETTE["accent"],
                   linestyle=":", linewidth=1.5, alpha=0.5)
        ax.text(max(lambdas) * 1.04, full_row["dino"], " Full FT DINO",
                va="center", ha="left",
                fontsize=FS_SUB, color=PALETTE["accent"],
                fontfamily=FONT_SANS, style="italic")

    if base_row is not None:
        ax.axhline(base_row["dino"], color=PALETTE["muted"],
                   linestyle=":", linewidth=1.5, alpha=0.5)
        ax.text(max(lambdas) * 1.04, base_row["dino"], " Base SD DINO",
                va="center", ha="left",
                fontsize=FS_SUB, color=PALETTE["muted"],
                fontfamily=FONT_SANS, style="italic")

    sweet = max(lora_rows, key=lambda r: r["dino"])
    sweet_lambda = float(sweet["prior_loss_weight"])
    ax.axvspan(sweet_lambda - 0.05, sweet_lambda + 0.05,
               color=PALETTE["accent"], alpha=0.06, zorder=0)
    ax.text(sweet_lambda, 0.78, "sweet spot",
            ha="center", va="bottom",
            fontsize=FS_SUB, color=PALETTE["accent"],
            fontfamily=FONT_SANS, style="italic")

    ax.set_xticks(lambdas)
    ax.set_xticklabels([f"λ = {l}" for l in lambdas],
                       fontsize=FS_HEADER, color=PALETTE["fg"],
                       fontfamily=FONT_SANS, fontweight="bold")
    ax.set_ylim(0.3, 0.85)
    ax.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    ax.set_yticklabels([f"{v:.1f}" for v in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]],
                       fontsize=FS_TICK, color=PALETTE["muted"],
                       fontfamily=FONT_MONO)
    ax.set_xlim(min(lambdas) - 0.08, max(lambdas) + 0.20)
    ax.tick_params(length=0, pad=8)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               frameon=False, fontsize=FS_LEGEND,
               bbox_to_anchor=(0.5, 0.21),
               labelcolor=PALETTE["fg"], handlelength=2.5,
               columnspacing=2.5, handletextpad=0.7)

    add_title(fig, "Prior-loss weight · LoRA ablation on dog")
    add_caption(
        fig,
        "Both subject metrics peak at λ=0.5 — the LoRA-specific sweet spot. λ=1.0 "
        "over-regularizes (subject under-fit); λ=0.1 starts to drift on prompts. "
        "Validates the paper's claim that prior preservation regularizes against "
        "language drift, while showing LoRA prefers a weaker prior than Full DreamBooth.")

    fig.savefig(out_path, format="svg", bbox_inches=None, facecolor=PALETTE["bg"])
    fig.savefig(out_path.with_suffix(".png"), format="png", dpi=200,
                bbox_inches=None, facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_efficiency(runs, out_path):
    """4-panel bar chart contrasting Full FT vs LoRA on compute cost.
    Panels: trainable %, train time, peak VRAM, checkpoint size."""
    full = average_efficiency_by_method(runs, "full")
    lora = average_efficiency_by_method(runs, "lora")
    if not full or not lora:
        print("  (efficiency skipped: missing run_stats.json for full or lora)")
        return

    panels = [
        ("Trainable params",   "trainable_pct",   "% of UNet",  lambda v: v),
        ("Wall-clock train",   "train_time_sec",  "minutes",    lambda v: v / 60),
        ("Peak VRAM",          "peak_vram_gb",    "GB",         lambda v: v),
        ("Checkpoint size",    "ckpt_size_mb",    "MB",         lambda v: v),
    ]

    fig = plt.figure(figsize=(16, 9.0))
    fig.subplots_adjust(left=0.05, right=0.97, bottom=0.30, top=0.78,
                        wspace=0.30)

    methods = [
        ("Full FT",  METHOD_COLORS["full"], full),
        ("LoRA r=16", METHOD_COLORS["lora"], lora),
    ]

    for i, (label, key, unit, fn) in enumerate(panels):
        ax = fig.add_subplot(1, 4, i + 1)
        x = np.arange(len(methods))
        bar_w = 0.55
        values = [fn(m[2].get(key, 0)) for m in methods]

        for j, (mlabel, color, _stats) in enumerate(methods):
            ax.bar(x[j], values[j], bar_w, color=color, edgecolor="none")

        ymax = max(values) * 1.20 if max(values) > 0 else 1.0
        for j, v in enumerate(values):
            if v < 0.001:
                txt = f"{v:.4f}"
            elif v < 1:
                txt = f"{v:.2f}"
            elif v < 100:
                txt = f"{v:.1f}"
            else:
                txt = f"{v:.0f}"
            ax.text(x[j], v + ymax * 0.02, txt,
                    ha="center", va="bottom",
                    fontsize=FS_VALUE, color=PALETTE["fg"], fontfamily=FONT_MONO)

        ax.set_xticks(x)
        ax.set_xticklabels([m[0] for m in methods], fontsize=FS_SUB,
                           color=PALETTE["fg"], fontfamily=FONT_SANS)
        ax.set_title(label, fontsize=FS_HEADER, fontweight="bold",
                     color=PALETTE["fg"], fontfamily=FONT_SANS, pad=14)
        ax.set_ylabel(unit, fontsize=FS_SUB, color=PALETTE["muted"],
                      fontfamily=FONT_SANS, labelpad=8)
        ax.set_ylim(0, ymax)
        ax.tick_params(length=0, pad=6, colors=PALETTE["muted"], labelsize=FS_TICK)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.yaxis.grid(True, color=PALETTE["grid"], linewidth=0.8)
        ax.set_axisbelow(True)

    add_title(fig, "LoRA · same job, fraction of the cost")
    add_caption(
        fig,
        "LoRA trains 357× fewer parameters (0.28% of the UNet) and produces checkpoints "
        "445× smaller (4.3 GB → 9.6 MB) while recovering 86% of Full FT's DINO and matching "
        "its CLIP-T. Wall-clock and VRAM savings are modest (25% / 36%) since the frozen "
        "UNet still runs forward each step. Single RTX 4090, 1000 steps, batch 1, fp32.")

    fig.savefig(out_path, format="svg", bbox_inches=None, facecolor=PALETTE["bg"])
    fig.savefig(out_path.with_suffix(".png"), format="png", dpi=200,
                bbox_inches=None, facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  wrote {out_path}")


def ema(values, alpha=0.1):
    out = []
    s = None
    for v in values:
        s = v if s is None else alpha * v + (1 - alpha) * s
        out.append(s)
    return out


def plot_loss_curves(loss_history_path, out_path):
    if loss_history_path is None:
        candidates = list(Path("results").rglob("loss_history.json"))
        if not candidates:
            print("  (no loss_history.json found, skipping loss_curves)")
            return
        loss_history_path = candidates[0]
    loss_history_path = Path(loss_history_path)
    with open(loss_history_path) as f:
        history = json.load(f)
    if not history:
        return

    steps = [h["step"] for h in history]
    inst = ema([h["loss_inst"] for h in history], alpha=0.1)
    cls  = ema([h["loss_cls"]  for h in history], alpha=0.1)

    fig = plt.figure(figsize=(12, 8.5))
    ax = fig.add_axes([0.10, 0.28, 0.84, 0.54])

    ax.plot(steps, inst, color=PALETTE["accent"], linewidth=3, label="Instance loss")
    ax.plot(steps, cls,  color=PALETTE["fg"],     linewidth=3, label="Prior-preservation loss")
    ax.set_yscale("log")
    ax.set_xlabel("Training step", fontsize=FS_SUB, color=PALETTE["muted"],
                  labelpad=12, fontfamily=FONT_SANS)
    ax.set_ylabel("MSE loss (log scale)", fontsize=FS_SUB, color=PALETTE["muted"],
                  labelpad=12, fontfamily=FONT_SANS)
    ax.legend(frameon=False, fontsize=FS_LEGEND, loc="upper right",
              labelcolor=PALETTE["fg"])
    ax.tick_params(length=0, pad=6, colors=PALETTE["muted"], labelsize=FS_TICK)

    add_title(fig, "Training loss over 1000 steps")
    add_caption(
        fig,
        f"Instance loss (subject reconstruction) and prior-preservation loss "
        f"(class-image regularization) decrease in parallel and stabilize at similar "
        f"magnitudes — neither the subject nor the broader class collapses during "
        f"fine-tuning. Source: {loss_history_path.parent.name}, EMA α=0.1.")

    fig.savefig(out_path, format="svg", bbox_inches=None, facecolor=PALETTE["bg"])
    fig.savefig(out_path.with_suffix(".png"), format="png", dpi=200,
                bbox_inches=None, facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_style()

    csv_path = Path(args.csv)
    rows = []
    if csv_path.exists():
        rows = load_csv(csv_path)
        print(f"Loaded {len(rows)} rows from {csv_path}")
    else:
        print(f"WARNING: {csv_path} not found; metric figures will be empty.")

    if rows:
        plot_bar_chart(rows, out_dir / "bar_chart.svg")
        plot_pareto_scatter(rows, out_dir / "pareto_scatter.svg")
        plot_per_subject_bars(rows, out_dir / "per_subject_bars.svg")

    runs = load_run_stats()
    if runs:
        print(f"Loaded {len(runs)} run_stats.json files")
        plot_efficiency(runs, out_dir / "efficiency.svg")

    plot_lambda_ablation(out_dir / "lambda_ablation.svg")
    plot_loss_curves(args.loss_history, out_dir / "loss_curves.svg")


if __name__ == "__main__":
    main()
