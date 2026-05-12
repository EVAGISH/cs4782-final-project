# Data

This re-implementation uses two image sources.

## 1. Subject reference images (DreamBooth dataset)

Source: official DreamBooth dataset released by the paper authors at
<https://github.com/google/dreambooth>.

The 4 subjects used by this project are checked into this repo under
`data/instance_images/<subject>/`:

| Folder                            | Class noun       | Type   |
|-----------------------------------|------------------|--------|
| `data/instance_images/dog`          | dog              | live   |
| `data/instance_images/cat`          | cat              | live   |
| `data/instance_images/backpack`     | backpack         | object |
| `data/instance_images/bear_plushie` | stuffed bear     | object |

Each subject has 4–6 reference photos. The 8-prompt recontextualization
subset used here lives in `code/prompts.json`. To work with additional
subjects from the full DreamBooth dataset, clone the upstream repo:

```bash
git clone https://github.com/google/dreambooth.git
```

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
  instance_images/   (checked into the repo)
    dog/, cat/, backpack/, bear_plushie/   (4–6 jpgs per subject)
  class_images/      (auto-generated, gitignored)
    dog/             class_0000.png … class_0199.png
    cat/             class_0000.png … class_0199.png
    backpack/        class_0000.png … class_0199.png
    stuffed_bear/    class_0000.png … class_0199.png
```

## Storage and licensing notes

- The DreamBooth dataset is released under the license described in its
  upstream repository; refer to that repo for terms. We do not redistribute
  the images.
- Class prior images are derived from Stable Diffusion v1.5 outputs and live
  only in your local working tree (`data/class_images/` is gitignored as
  part of the standard `*.png` ML-artifact pattern when applicable).
