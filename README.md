# Reproducing DreamBooth: Subject-Driven Generation with Diffusion Models

CS 4782 final project · Cornell University · Spring 2026

This repository re-implements **DreamBooth** (Ruiz et al., CVPR 2023) on
Stable Diffusion v1.5, evaluates it with the paper's three fidelity metrics
(DINO, CLIP-I, CLIP-T), and attempts three extensions: a **LoRA** variant
of the fine-tuning, an adaptation of the DreamBooth formulation to a
**text-to-video** model (ModelScopeT2V), and a **classifier-guidance**
attempt at weight-free personalization (which we show fails).

## 1. Introduction

DreamBooth "personalizes" a text-to-image diffusion model using only 3–5
images of a specific subject: it fine-tunes the model to bind a rare-token
identifier to that subject so it can subsequently be placed in novel
contexts via text prompts. The paper's key technical contribution is a
**class-specific prior-preservation loss** that prevents language drift and
maintains output diversity during fine-tuning.

We re-implement DreamBooth from scratch on top of `diffusers` and evaluate
it on 4 subjects (dog, cat, backpack, bear plushie). Beyond the paper we
also try (1) **LoRA** fine-tuning, which we reasoned would have a
regularizing effect similar to the prior-preservation loss; (2) adapting
the formulation to **ModelScopeT2V**, a text-to-video latent diffusion
model, to see whether DreamBooth scales to much higher-dimensional latent
spaces; and (3) **classifier guidance** as a weight-free alternative to
fine-tuning, which we show naively fails.

**Paper:** Ruiz, N., Li, Y., Jampani, V., Pritch, Y., Rubinstein, M.,
Aberman, K. *DreamBooth: Fine Tuning Text-to-Image Diffusion Models for
Subject-Driven Generation.* CVPR 2023. <https://arxiv.org/abs/2208.12242>

## 2. Chosen Result

We reproduce the **subject-driven recontextualization** result and the
quantitative DINO / CLIP-I / CLIP-T scores from **Table 1 of the paper**
(Stable Diffusion DreamBooth row): DINO ≈ 0.668, CLIP-I ≈ 0.803, CLIP-T ≈
0.305. DINO is the primary fidelity signal because, as a self-supervised vision
transformer trained without class labels, its features discriminate between
*different members of the same class*.

The qualitative target is the recontextualization grid concept used
throughout the paper (e.g., Figures 1 and 5): one specific subject placed in
many novel contexts that the base model would otherwise be unable to produce.

## 3. GitHub Contents

```
code/
  train_dreambooth.py        DreamBooth training (full + LoRA)
  generate.py                Inference / sampling
  evaluate.py                DINO + CLIP-I + CLIP-T computation
  lora.py                    LoRALinear module + UNet attention patching
  anchor.py                  Classifier-guidance experiments (4 target strategies)
  class_priors.py            Class-prior .npz loader used by the training dataset
  generate_class_images.py   Auto-generate class prior images for new classes
  generate_class_priors.py   Pack class images into the .npz format
  run_pipeline.py            End-to-end orchestrator (train -> gen -> eval)
  run_guidance_pipeline.py   Sweep wrapper for the classifier-guidance experiments
  aggregate_results.py       Walk results/, emit a single all_metrics.csv
  plot_metrics.py            Bar charts, Pareto scatter, per-subject, loss curves
  make_grid.py               Image-grid composer for poster figures
  find_rare_tokens.py        Helper for picking rare-token identifiers
  setup_fonts.py             Font installer used by make_grid.py
  hydra_compat.py            Hydra/OmegaConf compatibility shim
  prompts.json               The 8 standardized recontextualization prompts
  requirements.txt           Python dependencies
  conf/                      Hydra configs (train_dreambooth, generate, subject/);
                             the text-to-video (ModelScopeT2V) variant is configured
                             via the same Hydra config tree with task=video
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
- **Subjects:** dog, cat, backpack, bear_plushie from the official
  DreamBooth dataset.
- **Prompts:** an 8-prompt recontextualization set in `code/prompts.json`,
  applied identically across every (subject, method) cell so the
  qualitative comparison grid is directly comparable.
- **Evaluation:** DINO ViT-S/16 features for subject-identity fidelity, CLIP
  ViT-B/32 image embeddings for semantic subject fidelity, and CLIP text-image
  alignment for prompt fidelity. Implemented in `code/evaluate.py`.

### Extensions

- **LoRA fine-tuning.** Low-rank adapters on the `to_q`/`to_k`/`to_v`
  projections across all 32 UNet attention blocks (96 A/B pairs), rank
  `r = 16`, learning rate `1e-4`, prior-loss weight λ = 0.5.
- **Text-to-video (ModelScopeT2V).** The same DreamBooth objective applied
  to `ali-vilab/text-to-video-ms-1.7b`, whose spatio-temporal UNet operates
  on blocks of frame latents; the denoising loss transfers without
  modification. Selected via `task=video` in the Hydra config.
- **Classifier guidance.** A weight-free alternative in which each
  denoising step adds an L2 pull on the current latent toward a
  forward-diffused subject latent. We tried four target-selection
  strategies - the pooled mean of subject latents, one fixed sample per
  generation, a fresh resample per step, and the nearest noised subject
  latent - and swept guidance weights and schedules over each.
  Implementation in `code/anchor.py`.

### Configs

All entry points use [Hydra](https://hydra.cc/) with configs under
`code/conf/`. Per-subject overrides live in `code/conf/subject/<name>.yaml`;
CLI overrides use dotted keys (e.g.,
`subject=dog lora.enabled=true lora.rank=16`).

## 5. Reproduction Steps

Hardware: a single CUDA GPU with ≥16 GB VRAM. We trained on an **NVIDIA
RTX 4090** (24 GB). CPU and Apple-Silicon MPS code paths exist as fallbacks
but are slow.

### Setup

```bash
git clone <this-repo-url>
cd cs4782-final-project
pip install -r code/requirements.txt
```

### One-shot reproduction

**Note:** Every stage in `run_pipeline.py` (train / generate / evaluate) skips cells whose
outputs already exist, so re-running is idempotent and safe. If you need to
regenerate something, delete the corresponding marker file (`run_stats.json`
for train, `metadata.json` for generate, `metrics.json` for evaluate) and re-run.

```bash
# Generate 200 class-prior images per class (~20 min)
python code/generate_class_images.py \
  --classes dog cat backpack "stuffed bear" --num_images 200

# Train, generate, and evaluate every (subject, method) cell (~3.5 hrs)
python code/run_pipeline.py --stages train,generate,evaluate

# Aggregate metrics and build poster figures
python code/aggregate_results.py
python code/plot_metrics.py
python code/make_grid.py
```

After the final three commands, every figure used by the poster lives in
`results/figures/` and every numeric value lives in `results/all_metrics.csv`.

### Single-run invocations (Hydra overrides)

```bash
# Train one subject with full fine-tuning
python code/train_dreambooth.py subject=dog lora.enabled=false \
  training.learning_rate=5e-6

# Train one subject with LoRA
python code/train_dreambooth.py subject=dog lora.enabled=true lora.rank=16 \
  training.learning_rate=1e-4

# Generate from a trained checkpoint
python code/generate.py subject=dog model.model_path=results/dog_lora

# Evaluate generated images
python code/evaluate.py \
  --real_images_dir data/instance_images/dog \
  --generated_images_dir results/dog_lora_results \
  --prompts_file results/dog_lora_results/metadata.json \
  --output_file results/dog_lora_results/metrics.json
```

### Prior-loss ablation

```bash
# Turn off prior preservation to illustrate language drift
python code/train_dreambooth.py subject=backpack \
  training.prior_loss_weight=0 \
  training.output_dir=results/backpack_no_prior
```

### Classifier-guidance experiments

```bash
# Single classifier-guidance run (strategies: pooled, random_per_sample,
# random_per_step, nearest)
python code/generate.py subject=dog \
  model.model_path=runwayml/stable-diffusion-v1-5 \
  anchor.enabled=true anchor.mode=nearest anchor.weight=0.15

# Sweep multiple guidance configs
python code/run_guidance_pipeline.py
```

## 6. Results / Insights


| Method                 | DINO  | CLIP-I | CLIP-T |
|------------------------|-------|--------|--------|
| Paper (Table 1)        | 0.668 | 0.803  | 0.305  |
| Base SD (no fine-tune) | 0.29  | 0.66 | 0.34 |
| DreamBooth Full (ours) | 0.60| 0.77 | 0.33 |
| DreamBooth LoRA (ours) | 0.52| 0.74 | 0.33 |

Both methods dramatically improve subject fidelity over the base model (DINO +0.31/+0.23), with LoRA recovering ~87% of Full FT's DINO score while prompt alignment (CLIP-T) remains unchanged across all variants. The full fine-tuning approach directly reproduced from the paper is largely comparable in terms of results.

LoRA efficiency vs. full fine-tuning (single RTX 4090, 1000 steps):

| Variant  | Trainable params | % of UNet | Train time | Peak VRAM | Checkpoint |
|----------|------------------|-----------|------------|-----------|------------|
| Full     | 859,520,964      | 100%      | ~12 min    | 11.95 GB  | ~4.27 GB   |
| LoRA r=16| 2,390,016        | 0.28%     | ~9 min     | 7.61 GB   | ~9.6 MB    |

Qualitative outputs are in `results/<subject>_<method>_results/` and the
composed poster figures in `results/figures/`.

## 7. Conclusion

DreamBooth reproduces cleanly on Stable Diffusion v1.5: our Full FT scores
land within the paper's reported per-subject variance, and LoRA proves a
strong drop-in alternative that recovers roughly 86% of Full FT's subject
fidelity at 0.28% of the trainable parameters while producing checkpoints
445 times smaller. LoRA also prefers a weaker prior than full fine-tuning,
with the optimum shifting from λ = 1.0 to λ = 0.5, suggesting that the
low-rank update itself does some of the regularizing work prior
preservation was designed to handle. By contrast, our classifier-guidance
experiments all failed, supporting the view that this kind of
personalization genuinely requires weight-level updates.

## 8. References

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L.,
& Chen, W. (2021). LoRA: Low-rank adaptation of large language models.
*arXiv preprint* arXiv:2106.09685.

Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022).
High-resolution image synthesis with latent diffusion models. In
*Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR)* (pp. 10684–10695).

Ruiz, N., Li, Y., Jampani, V., Pritch, Y., Rubinstein, M., & Aberman, K.
(2023). DreamBooth: Fine tuning text-to-image diffusion models for
subject-driven generation. In *Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition (CVPR)* (pp. 22500–22510).

Wang, J., Yuan, H., Chen, D., Zhang, Y., Wang, X., & Zhang, S. (2023).
ModelScope text-to-video technical report. *arXiv preprint* arXiv:2308.06571.

## 9. Acknowledgements

This work was completed as the final project for **CS 4782: Introduction to
Deep Learning** at Cornell University, Spring 2026, taught by [FILL: instructor].
We thank the course staff for guidance and feedback.

The DreamBooth dataset is provided by the original paper's authors at
<https://github.com/google/dreambooth>. Stable Diffusion v1.5 weights are
distributed by Runway via Hugging Face. ModelScopeT2V weights are
distributed by Alibaba DAMO Academy via Hugging Face. Implementation
builds on the `diffusers` library by Hugging Face.
