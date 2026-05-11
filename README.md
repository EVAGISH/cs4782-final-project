# DreamBooth + LoRA + VideoGen

## Introduction

This project implements the Dreambooth technique for fine-tuning image diffusion models, and extends it with LoRA and video generation.


## Project Setup

### Prerequisites

- **Python** 3.10 or newer (3.11+ recommended).
- **GPU** strongly recommended for training and for fast inference. The code uses CUDA when available, otherwise Apple **MPS**, then **CPU**.
- Enough disk space for Stable Diffusion weights and generated class images (several GB).

### Environment

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r code/requirements.txt
```

Install **PyTorch** matching your platform if `pip install torch` from the generic index is not what you want (see [pytorch.org](https://pytorch.org/get-started/locally/) for the CUDA wheel you need).

**Optional — 8-bit optimizer (CUDA only):** if `bitsandbytes` is installed, training uses 8-bit AdamW to save VRAM. It is not listed in `requirements.txt`; install it only on a CUDA setup where the package is supported.

### Hugging Face models

Training defaults to `runwayml/stable-diffusion-v1-5`. The first run downloads weights from the Hugging Face Hub. If access is denied, log in:

```bash
pip install huggingface_hub
huggingface-cli login
```

Accept any model license on the model card in the browser if prompted.

### Data layout

Place instance images under a directory you pass as `--instance_data_dir` (see `data/instance_images/` for an example). Class prior images are generated under `--class_data_dir` when you run training if that folder does not already contain enough images.

### Running scripts

Run Python from the `code/` directory (or use `python code/train_dreambooth.py` with paths adjusted accordingly):

```bash
cd code
python train_dreambooth.py --help
python generate.py --help
python evaluate.py --help
```

`evaluate.py` loads DINO via `torch.hub` the first time you run it; that step needs network access.
