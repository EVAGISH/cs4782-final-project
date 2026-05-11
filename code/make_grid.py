"""Compose poster image grids from generated outputs and reference photos.

Five named figures are produced into results/figures/:
  - V1_hero.png               4 reference imgs of the hero subject + 8 generated
                              recontextualizations (single horizontal strip)
  - V3_recontext.png          1 subject x 8 prompts (best seed each)
  - V4_method_comparison.png  4 subjects (rows) x [Reference, Base SD, Full, LoRA]
  - V7_ablation.png           2x2 with/without prior loss x early/late training
  - V8_failures.png           3 hand-picked failures, side-by-side

All paths and per-cell picks are configured at the top of this file. Edit them
after eyeballing the generated images, then re-run.
"""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
TILE = 320                # pixel size each image is rendered at in the grid
GUTTER = 12               # spacing between tiles
LABEL_HEIGHT = 56         # space reserved for column / row labels
TITLE_HEIGHT = 64         # space reserved for grid title
BG = (250, 247, 242)      # cream background (#FAF7F2)
FG = (26, 26, 46)         # near-black text  (#1A1A2E)
ACCENT = (194, 65, 12)    # terracotta       (#C2410C)
BORDER = (220, 215, 205)  # 1px border around each tile

# Paths
DREAMBOOTH_DATASET = Path("dreambooth/dataset")
RESULTS_ROOT = Path("results")
OUT_DIR = Path("results/figures")


# ---------------------------------------------------------------------------
# Configuration -- EDIT AFTER LOOKING AT YOUR GENERATIONS
# ---------------------------------------------------------------------------

# The 8 standardized prompts in code/prompts.json, in the same order
PROMPT_LABELS = [
    "Times Square",
    "On the moon",
    "Snowy forest",
    "LEGO bricks",
    "Chef's hat",
    "Watercolor",
    "Van Gogh style",
    "Underwater",
]

# Subject -> class noun (used to build base-SD prompts and grid labels)
SUBJECTS = {
    "dog": "dog",
    "cat": "cat",
    "backpack": "backpack",
    "bear_plushie": "stuffed bear",
}

# The hero subject for V1 / V3 -- pick the most visually charismatic one
HERO_SUBJECT = "dog"

# Which method's generations to use for V1 hero strip and V3 recontext grid.
# "full" usually looks best; switch to "lora" if you prefer those crops.
HERO_METHOD = "full"

# For each (subject, prompt_idx) cell in V4, which seed file to use.
# Seeds are 0-3 (4 seeds per prompt). Default to seed 0; override after curating.
V4_SEED_PICKS = {
    # ("dog", 0): 2,       # for prompt index 0 ("Times Square"), use seed 2
    # ("backpack", 4): 1,  # etc.
}

# Which prompt index to use as the row's "single representative" prompt in V4.
# Pick a prompt that works well across all 4 subjects (LEGO and watercolor are safe bets).
V4_PROMPT_IDX = 3   # LEGO bricks

# V7 ablation: directories for the 4 cells. Fill in once you have:
#   results/backpack_full/  vs  results/backpack_no_prior/
# at two checkpoints (you'll need to save intermediate checkpoints, or just use
# a single late checkpoint and one early-saved snapshot).
V7_CELLS = {
    # ("with_prior", "early"): "results/backpack_full_step200_results/prompt_03/img_00_seed0.png",
    # ("with_prior", "late"):  "results/backpack_full_results/prompt_03/img_00_seed0.png",
    # ("no_prior",   "early"): "results/backpack_no_prior_step200_results/prompt_03/img_00_seed0.png",
    # ("no_prior",   "late"):  "results/backpack_no_prior_results/prompt_03/img_00_seed0.png",
}

# V8 failures: list of (caption, image_path) tuples, hand-picked
V8_FAILURES = [
    # ("Subject blends into background", "results/cat_full_results/prompt_07/img_02_seed2.png"),
    # ("Anatomy drift",                  "results/dog_full_results/prompt_06/img_01_seed1.png"),
    # ("Prompt ignored (language drift)","results/backpack_no_prior_results/prompt_01/img_03_seed3.png"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_font(size, bold=False):
    # Try a sensible list of system fonts across Windows / macOS / Linux.
    if bold:
        candidates = [
            "C:/Windows/Fonts/Inter-Bold.ttf",
            "C:/Windows/Fonts/segoeuib.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/System/Library/Fonts/HelveticaNeue.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ]
    else:
        candidates = [
            "C:/Windows/Fonts/Inter-Regular.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]
    for c in candidates:
        if Path(c).exists():
            try:
                return ImageFont.truetype(c, size)
            except Exception:
                pass
    return ImageFont.load_default()


def fit_image(path, size=TILE):
    """Open an image, center-crop to square, resize to (size, size), add 1px border."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side)).resize((size, size), Image.LANCZOS)

    bordered = Image.new("RGB", (size, size), BORDER)
    bordered.paste(img, (0, 0))
    draw = ImageDraw.Draw(bordered)
    draw.rectangle([0, 0, size - 1, size - 1], outline=BORDER, width=1)
    return bordered


def measure_text(draw, text, font):
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def text_centered(draw, text, x, y, w, font, color=FG):
    tw, th = measure_text(draw, text, font)
    draw.text((x + (w - tw) / 2, y + (LABEL_HEIGHT - th) / 2), text, fill=color, font=font)


def text_left(draw, text, x, y, h, font, color=FG):
    _, th = measure_text(draw, text, font)
    draw.text((x, y + (h - th) / 2), text, fill=color, font=font)


def list_subject_refs(subject, max_n=4):
    folder = DREAMBOOTH_DATASET / subject
    if not folder.exists():
        return []
    files = sorted([p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    return files[:max_n]


def find_generated(subject, method, prompt_idx, seed=0):
    """Look up results/<subject>_<method>_results/prompt_XX/img_YY_seedZ.png."""
    base = RESULTS_ROOT / f"{subject}_{method}_results"
    if not base.exists():
        # try the legacy naming we saw in the existing repo (e.g., backpack_v9_results)
        candidates = list(RESULTS_ROOT.glob(f"{subject}*_results"))
        if not candidates:
            return None
        base = candidates[0]
    prompt_dir = base / f"prompt_{prompt_idx:02d}"
    if not prompt_dir.exists():
        return None
    matches = sorted(prompt_dir.glob(f"img_*_seed{seed}.png"))
    if matches:
        return matches[0]
    matches = sorted(prompt_dir.glob("img_*.png"))
    return matches[0] if matches else None


# ---------------------------------------------------------------------------
# Grid builders
# ---------------------------------------------------------------------------

def build_v1_hero(out_path):
    """4 reference imgs | divider | 8 generated imgs in different contexts."""
    refs = list_subject_refs(HERO_SUBJECT, max_n=4)
    gens = [find_generated(HERO_SUBJECT, HERO_METHOD, i, seed=0)
            for i in range(len(PROMPT_LABELS))]
    gens = [g for g in gens if g is not None]
    if not refs or not gens:
        print(f"  V1 skipped: refs={len(refs)} gens={len(gens)} for {HERO_SUBJECT}")
        return

    n_left = len(refs)
    n_right = len(gens)
    divider_w = 80
    width = n_left * TILE + (n_left - 1) * GUTTER + divider_w + n_right * TILE + (n_right - 1) * GUTTER
    height = TILE + LABEL_HEIGHT
    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)
    label_font = load_font(20, bold=True)

    x = 0
    for ref in refs:
        canvas.paste(fit_image(ref, TILE), (x, LABEL_HEIGHT))
        x += TILE + GUTTER
    text_centered(draw, "Reference photos",
                  0, 0, n_left * TILE + (n_left - 1) * GUTTER, label_font, color=FG)

    x -= GUTTER
    arrow_x = x + divider_w // 2
    draw.line([(x + 20, LABEL_HEIGHT + TILE / 2), (x + divider_w - 20, LABEL_HEIGHT + TILE / 2)],
              fill=ACCENT, width=4)
    draw.polygon([
        (x + divider_w - 20, LABEL_HEIGHT + TILE / 2 - 12),
        (x + divider_w - 20, LABEL_HEIGHT + TILE / 2 + 12),
        (x + divider_w - 5,  LABEL_HEIGHT + TILE / 2),
    ], fill=ACCENT)
    x += divider_w

    right_start = x
    for gen in gens:
        canvas.paste(fit_image(gen, TILE), (x, LABEL_HEIGHT))
        x += TILE + GUTTER
    right_width = n_right * TILE + (n_right - 1) * GUTTER
    text_centered(draw, "Generated in novel contexts",
                  right_start, 0, right_width, label_font, color=ACCENT)

    canvas.save(out_path, dpi=(300, 300))
    print(f"  wrote {out_path}")


def build_v3_recontext(out_path):
    """1 subject x 8 prompts in a 2x4 grid with prompt labels under each tile."""
    cells = [(label, find_generated(HERO_SUBJECT, HERO_METHOD, i, seed=0))
             for i, label in enumerate(PROMPT_LABELS)]
    cells = [(label, p) for label, p in cells if p is not None]
    if not cells:
        print(f"  V3 skipped: no generations found for {HERO_SUBJECT}/{HERO_METHOD}")
        return

    cols = 4
    rows = (len(cells) + cols - 1) // cols
    cell_h = TILE + LABEL_HEIGHT
    width = cols * TILE + (cols - 1) * GUTTER
    height = TITLE_HEIGHT + rows * cell_h + (rows - 1) * GUTTER
    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(28, bold=True)
    label_font = load_font(18)

    title = f"One subject ({HERO_SUBJECT.replace('_', ' ')}), eight contexts"
    text_centered(draw, title, 0, 0, width, title_font, color=FG)

    for idx, (label, path) in enumerate(cells):
        r, c = divmod(idx, cols)
        x = c * (TILE + GUTTER)
        y = TITLE_HEIGHT + r * (cell_h + GUTTER)
        canvas.paste(fit_image(path, TILE), (x, y))
        text_centered(draw, label, x, y + TILE, TILE, label_font, color=FG)

    canvas.save(out_path, dpi=(300, 300))
    print(f"  wrote {out_path}")


def build_v4_method_comparison(out_path):
    """4 subject rows x 4 method columns. Same prompt across the row."""
    method_cols = [
        ("Reference", "ref"),
        ("Base SD", "base"),
        ("DreamBooth Full", "full"),
        ("DreamBooth + LoRA", "lora"),
    ]
    subject_keys = list(SUBJECTS.keys())

    label_w = 140
    n_cols = len(method_cols)
    width = label_w + n_cols * TILE + (n_cols - 1) * GUTTER
    cell_h = TILE
    height = TITLE_HEIGHT + LABEL_HEIGHT + len(subject_keys) * cell_h + (len(subject_keys) - 1) * GUTTER
    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(28, bold=True)
    col_font = load_font(20, bold=True)
    row_font = load_font(18, bold=True)

    prompt_label = PROMPT_LABELS[V4_PROMPT_IDX]
    text_centered(draw, f"Method comparison  -  prompt: \"{prompt_label}\"",
                  0, 0, width, title_font, color=FG)

    for j, (label, _key) in enumerate(method_cols):
        x = label_w + j * (TILE + GUTTER)
        text_centered(draw, label, x, TITLE_HEIGHT, TILE, col_font, color=ACCENT)

    for i, subject in enumerate(subject_keys):
        y = TITLE_HEIGHT + LABEL_HEIGHT + i * (cell_h + GUTTER)
        text_left(draw, subject.replace("_", " "), 8, y, cell_h, row_font, color=FG)

        for j, (_label, key) in enumerate(method_cols):
            x = label_w + j * (TILE + GUTTER)
            if key == "ref":
                refs = list_subject_refs(subject, max_n=1)
                path = refs[0] if refs else None
            else:
                seed = V4_SEED_PICKS.get((subject, V4_PROMPT_IDX), 0)
                path = find_generated(subject, key, V4_PROMPT_IDX, seed=seed)

            if path is None:
                draw.rectangle([x, y, x + TILE, y + TILE], outline=BORDER, width=1)
                text_centered(draw, "missing", x, y + TILE / 2 - LABEL_HEIGHT / 2,
                              TILE, col_font, color=BORDER)
            else:
                canvas.paste(fit_image(path, TILE), (x, y))

    canvas.save(out_path, dpi=(300, 300))
    print(f"  wrote {out_path}")


def build_v7_ablation(out_path):
    """2x2: rows = with/without prior loss, cols = early / late training step."""
    if not V7_CELLS:
        print("  V7 skipped: V7_CELLS not configured (run prior-loss ablation first)")
        return

    rows = ["with_prior", "no_prior"]
    cols = ["early", "late"]
    row_labels = ["With prior loss", "Without prior loss"]
    col_labels = ["Early in training", "Late in training"]

    label_w = 200
    width = label_w + len(cols) * TILE + (len(cols) - 1) * GUTTER
    height = TITLE_HEIGHT + LABEL_HEIGHT + len(rows) * TILE + (len(rows) - 1) * GUTTER
    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(26, bold=True)
    label_font = load_font(18, bold=True)

    text_centered(draw, "Prior-preservation loss ablation",
                  0, 0, width, title_font, color=FG)

    for j, label in enumerate(col_labels):
        x = label_w + j * (TILE + GUTTER)
        text_centered(draw, label, x, TITLE_HEIGHT, TILE, label_font, color=ACCENT)

    for i, (row_key, row_label) in enumerate(zip(rows, row_labels)):
        y = TITLE_HEIGHT + LABEL_HEIGHT + i * (TILE + GUTTER)
        text_left(draw, row_label, 8, y, TILE, label_font, color=FG)
        for j, col_key in enumerate(cols):
            x = label_w + j * (TILE + GUTTER)
            path = V7_CELLS.get((row_key, col_key))
            if path and Path(path).exists():
                canvas.paste(fit_image(path, TILE), (x, y))
            else:
                draw.rectangle([x, y, x + TILE, y + TILE], outline=BORDER, width=1)

    canvas.save(out_path, dpi=(300, 300))
    print(f"  wrote {out_path}")


def build_v8_failures(out_path):
    """Horizontal strip of cherry-picked failure images with captions."""
    if not V8_FAILURES:
        print("  V8 skipped: V8_FAILURES not configured")
        return

    n = len(V8_FAILURES)
    width = n * TILE + (n - 1) * GUTTER
    height = TITLE_HEIGHT + TILE + LABEL_HEIGHT
    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(26, bold=True)
    cap_font = load_font(16)

    text_centered(draw, "Failure modes", 0, 0, width, title_font, color=FG)

    for i, (caption, path) in enumerate(V8_FAILURES):
        x = i * (TILE + GUTTER)
        y = TITLE_HEIGHT
        if Path(path).exists():
            canvas.paste(fit_image(path, TILE), (x, y))
        else:
            draw.rectangle([x, y, x + TILE, y + TILE], outline=BORDER, width=1)
        text_centered(draw, caption, x, y + TILE, TILE, cap_font, color=FG)

    canvas.save(out_path, dpi=(300, 300))
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default=str(OUT_DIR))
    parser.add_argument("--only", type=str, nargs="+", default=None,
                        choices=["v1", "v3", "v4", "v7", "v8"],
                        help="If set, only build these grids")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    only = set(args.only) if args.only else None

    if only is None or "v1" in only:
        build_v1_hero(out_dir / "V1_hero.png")
    if only is None or "v3" in only:
        build_v3_recontext(out_dir / "V3_recontext.png")
    if only is None or "v4" in only:
        build_v4_method_comparison(out_dir / "V4_method_comparison.png")
    if only is None or "v7" in only:
        build_v7_ablation(out_dir / "V7_ablation.png")
    if only is None or "v8" in only:
        build_v8_failures(out_dir / "V8_failures.png")


if __name__ == "__main__":
    main()
