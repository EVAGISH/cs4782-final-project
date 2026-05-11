# Reproducing DreamBooth: Subject-Driven Generation with Diffusion Models

CS 4782 final project · Cornell University · Spring 2026

This repository re-implements **DreamBooth** (Ruiz et al., CVPR 2023) on
Stable Diffusion v1.5, evaluates it with the paper's three fidelity metrics
(DINO, CLIP-I, CLIP-T), and adds a LoRA-based variant for an efficiency
comparison the paper does not study.

## 1. Introduction

DreamBooth fine-tunes a text-to-image diffusion model on 4–6 photos of a
*specific* subject so that the model can subsequently generate that subject
in novel contexts described by text prompts. The key technical contributions
of the paper are (i) tying a unique-identifier token `[V]` to the subject and
(ii) a **class-specific prior-preservation loss** that prevents the model
from collapsing onto the few training images and forgetting the broader
class.

This repository re-implements the method from scratch on top of
`diffusers`, evaluates it on 4 subjects (dog, cat, backpack, stuffed bear),
and compares full-UNet fine-tuning against a LoRA-only variant.

**Paper:** Ruiz, N., Li, Y., Jampani, V., Pritch, Y., Rubinstein, M.,
Aberman, K. *DreamBooth: Fine Tuning Text-to-Image Diffusion Models for
Subject-Driven Generation.* CVPR 2023. <https://arxiv.org/abs/2208.12242>

## 2. Chosen Result

We aim to reproduce the **subject-driven recontextualization** result and the
quantitative DINO / CLIP-I / CLIP-T scores from **Table 1 of the paper**
(Stable Diffusion DreamBooth row): DINO ≈ 0.668, CLIP-I ≈ 0.803, CLIP-T ≈
0.305.

The qualitative target is the recontextualization grid concept used
throughout the paper (e.g., Figures 1 and 5): one specific subject placed in
many novel contexts that the base model would otherwise be unable to produce.

## 3. GitHub Contents

```
code/
  train_dreambooth.py        DreamBooth training (full + LoRA)
  generate.py                Inference / sampling
  evaluate.py                DINO + CLIP-I + CLIP-T computation
  generate_class_images.py   Auto-generate class prior images for new classes
  run_pipeline.py            End-to-end orchestrator (train -> gen -> eval)
  aggregate_results.py       Walk results/, emit single all_metrics.csv
  plot_metrics.py            Bar chart, Pareto scatter, per-subject, loss curves
  make_grid.py               Image-grid composer for the poster figures
  prompts.json               The 8 standardized recontextualization prompts
  requirements.txt           Python dependencies
data/
  README.md                  How to obtain reference and class images
  class_images/              Auto-generated; gitignored
results/                     Trained models, generated images, metrics, figures
poster/                      Final poster PDF
report/                      2-page report PDF
```

## 4. Re-implementation Details

- **Base model:** `runwayml/stable-diffusion-v1-5`.
- **Loss:** standard DreamBooth — instance MSE plus prior-preservation MSE
  with weight λ = 1.0. 200 class images sampled from the frozen base model
  per class.
- **Training:** 1000 steps, batch size 1, learning rate 5e-6, fp16 mixed
  precision, AdamW (8-bit via `bitsandbytes` when available), gradient
  clipping `max_norm=1.0`.
- **LoRA variant:** rank `r = 16`, alpha `α = 16` (scaling = 1.0). Adapters
  applied only to the UNet attention linears (`to_q`, `to_k`, `to_v`,
  `to_out[0]`) across all self- and cross-attention blocks. VAE, text
  encoder, and all non-attention UNet weights remain frozen.
- **Subjects:** dog, cat, backpack, bear_plushie from the official DreamBooth
  dataset.
- **Prompts:** an 8-prompt recontextualization set in `code/prompts.json`,
  applied identically across every (subject, method) cell so the
  qualitative comparison grid is directly comparable.
- **Evaluation:** DINO ViT-S/16 features for subject-identity fidelity, CLIP
  ViT-B/32 image embeddings for semantic subject fidelity, and CLIP text-image
  alignment for prompt fidelity. Implemented in `code/evaluate.py`.

## 5. Reproduction Steps

Hardware: a single CUDA GPU with ≥16 GB VRAM. We trained on an **NVIDIA
RTX 4090** (24 GB). CPU and Apple-Silicon MPS code paths exist as fallbacks
but are slow.

### Setup

```bash
git clone <this-repo-url>
cd cs4782-final-project
git clone https://github.com/google/dreambooth.git           # subject images
pip install -r code/requirements.txt
```

### One-shot reproduction

```bash
python code/generate_class_images.py \
  --classes dog cat backpack "stuffed bear" --num_images 200    # ~20 min

python code/run_pipeline.py --stages train,generate,evaluate    # ~3.5 hrs

python code/aggregate_results.py
python code/plot_metrics.py
python code/make_grid.py
```

After the final three commands, every figure used by the poster lives in
`results/figures/` and every numeric value lives in `results/all_metrics.csv`.

### Optional ablations

Prior-preservation loss off (illustrates language drift):

```bash
python code/train_dreambooth.py \
  --pretrained_model runwayml/stable-diffusion-v1-5 \
  --instance_data_dir dreambooth/dataset/backpack \
  --class_data_dir data/class_images/backpack \
  --output_dir results/backpack_no_prior \
  --instance_prompt "a [V] backpack" --class_prompt "a backpack" \
  --max_train_steps 1000 --mixed_precision fp16 --prior_loss_weight 0
```

## 6. Results / Insights

[FILL after final runs — copy from results/all_metrics.csv]

| Method                 | DINO  | CLIP-I | CLIP-T |
|------------------------|-------|--------|--------|
| Paper (Table 1)        | 0.668 | 0.803  | 0.305  |
| Base SD (no fine-tune) | [FILL]| [FILL] | [FILL] |
| DreamBooth Full (ours) | [FILL]| [FILL] | [FILL] |
| DreamBooth LoRA (ours) | [FILL]| [FILL] | [FILL] |

LoRA efficiency vs. full fine-tuning (single RTX 4090, 1000 steps):

| Variant | Trainable params | % of UNet | Train time | Peak VRAM | Checkpoint |
|---------|------------------|-----------|------------|-----------|------------|
| Full    | [FILL]           | 100%      | [FILL]     | [FILL]    | ~3.4 GB    |
| LoRA r=4| [FILL]           | <1%       | [FILL]     | [FILL]    | [FILL] MB  |

Qualitative outputs are in `results/<subject>_<method>_results/` and the
composed poster figures in `results/figures/`.

## 7. Conclusion

[FILL after analysis — 2-3 sentences on whether the paper reproduced cleanly,
how LoRA compared, and what the prior-loss ablation showed.]

## 8. References

1. Ruiz, N., Li, Y., Jampani, V., Pritch, Y., Rubinstein, M., Aberman, K.
   *DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven
   Generation.* CVPR 2023. arXiv:2208.12242.
2. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.
   *High-Resolution Image Synthesis with Latent Diffusion Models.*
   CVPR 2022. (Stable Diffusion.)
3. Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L.,
   Chen, W. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022.
   arXiv:2106.09685.
4. Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J., Bojanowski, P.,
   Joulin, A. *Emerging Properties in Self-Supervised Vision Transformers.*
   ICCV 2021. (DINO.)
5. Radford, A. et al. *Learning Transferable Visual Models From Natural
   Language Supervision.* ICML 2021. (CLIP.)
6. von Platen, P. et al. *Diffusers: State-of-the-art Diffusion Models.*
   <https://github.com/huggingface/diffusers>.

## 9. Acknowledgements

This work was completed as the final project for **CS 4782: Introduction to
Deep Learning** at Cornell University, Spring 2026, taught by [FILL: instructor].
We thank the course staff for guidance and feedback.

The DreamBooth dataset is provided by the original paper's authors at
<https://github.com/google/dreambooth>. Stable Diffusion v1.5 weights are
distributed by Runway via Hugging Face. Implementation builds on the
`diffusers` library by Hugging Face.
