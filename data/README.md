# Data

This re-implementation uses two image sources. Neither is checked in to this
repository — both are obtained at setup time.

## 1. Subject reference images (DreamBooth dataset)

Source: official DreamBooth dataset released by the paper authors at
<https://github.com/google/dreambooth>.

Setup (run once, from the repo root):

```bash
git clone https://github.com/google/dreambooth.git
```

This places 30 subjects under `dreambooth/dataset/<subject>/`. The subjects
used by this project are:

| Folder                       | Class noun       | Type   |
|------------------------------|------------------|--------|
| `dreambooth/dataset/dog`          | dog              | live   |
| `dreambooth/dataset/cat`          | cat              | live   |
| `dreambooth/dataset/backpack`     | backpack         | object |
| `dreambooth/dataset/bear_plushie` | stuffed bear     | object |

Each subject has 4–6 reference photos. The dataset's
`prompts_and_classes.txt` lists the canonical prompts the paper used; the
8-prompt subset used here lives in `code/prompts.json`.

## 2. Class prior images (auto-generated)

DreamBooth's prior-preservation loss requires ~200 generic images per class,
sampled from the frozen base Stable Diffusion. These are generated locally —
they are not downloaded.

Setup (one-time, ~5 minutes per class on an RTX 4090):

```bash
python code/generate_class_images.py \
  --classes dog cat backpack "stuffed bear" \
  --num_images 200
```

Output: `data/class_images/<slugified_class>/class_0000.png … class_0199.png`.

Re-running the script is idempotent — it only generates the gap up to
`--num_images`.

## Directory layout (after setup)

```
data/
  class_images/
    dog/             class_0000.png … class_0199.png
    cat/             class_0000.png … class_0199.png
    backpack/        class_0000.png … class_0199.png
    stuffed_bear/    class_0000.png … class_0199.png
dreambooth/          (cloned externally, gitignored)
  dataset/
    dog/, cat/, backpack/, bear_plushie/, …
```

## Storage and licensing notes

- The DreamBooth dataset is released under the license described in its
  upstream repository; refer to that repo for terms. We do not redistribute
  the images.
- Class prior images are derived from Stable Diffusion v1.5 outputs and live
  only in your local working tree (`data/class_images/` is gitignored as
  part of the standard `*.png` ML-artifact pattern when applicable).
