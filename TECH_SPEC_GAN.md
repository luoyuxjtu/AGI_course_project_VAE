# GAN Image Inpainting Practice Project — Technical Specification

> This document is for Claude Code to read. It describes the complete project goals, scope,
> technical details, and workflow constraints. Please read it fully before writing code and
> strictly follow the workflow constraints in Section 3.
>
> NOTE: This project was migrated from a previous VAE project. Section 11 lists what to reuse
> from the existing code vs. what to replace. Read Section 11 to understand the migration.

---

## 1. Project Background & Goals

This is an entry-level hands-on practice project for AI enthusiasts. The goal is to implement
and train a Generative Adversarial Network (GAN) to perform an **image inpainting** (corrupted
image completion) task, building intuitive understanding of generative AI and adversarial training.

The project consists of three modules:

1. **Application Practice** — Train a GAN-based inpainting model on COCO 2017. Given an image
   with a missing (masked) region, the generator fills in plausible content. Evaluate on unseen
   validation images: show masked input → completed output → ground truth.
2. **Model Improvements** — Run controlled comparison experiments across three improvement axes:
   generation quality (adversarial loss ablation), computational efficiency (mixed precision),
   and model lightweighting (channel reduction).
3. **Theoretical Analysis** — Based on experimental results, discuss three topics: adversarial
   training dynamics, the trade-off between reconstruction loss and adversarial loss, and why
   adversarial training avoids the blurriness of pixel-wise losses.

This is an educational exercise, **not pursuing SOTA**. Code should prioritize clarity,
readability, reproducibility, and modularity.

---

## 2. Scope & Constraints

To ensure completion within one week, scope is intentionally simplified:

- **Task**: Image inpainting. Generator is **deterministic and conditional** (conditioned on the
  masked image); the "generative" aspect comes from the adversarial discriminator. Do NOT add a
  noise/latent prior input, do NOT implement prior sampling or latent interpolation (those were
  VAE concepts and no longer apply).
- **"Unseen" definition uses sample-level hold-out**: train on COCO `train2017`, evaluate
  inpainting on COCO `val2017` (images never seen during training).
- **Dataset**: COCO 2017, loaded via a custom flat-directory Dataset class (no annotations needed).
- **Image resolution**: 256×256.
- **Architecture**: Context-Encoder-style generator (encoder–bottleneck–decoder, NO skip
  connections) + PatchGAN discriminator. Do NOT implement U-Net skip connections, multi-scale
  discriminators, partial/gated convolutions, or attention.
- **Mask**: Random rectangular hole(s), configurable size. Do NOT implement free-form/irregular
  masks.
- **Improvement items**: Only three lightweight improvements (Section 6). No additional ones.
- **Theoretical analysis**: Observational discussion based on results. No formula derivation or
  new algorithm implementation.

---

## 3. Workflow Constraints (Critical — Must Follow)

This project separates development from execution. Claude Code must strictly follow:

1. **Claude Code is responsible only for writing/modifying all project code in the GitHub cloud
   repository.** The user will later pull it to a GPU server to run. Claude Code's environment
   has **no GPU and no datasets**.
2. **Do not run training code, do not run any code requiring GPU or datasets, do not download
   datasets for testing.**
3. **Do not do functional testing in intermediate steps.** The only permitted and recommended
   check is **syntax checking**: run `python -m py_compile <file>` on new or modified Python
   files. This requires no torch installation or data and only verifies syntax correctness.
4. Code must be in a state that can run directly on the GPU server: no hardcoded paths,
   all dependencies declared in `requirements.txt`, entry points clearly documented in README.
5. **Commit after each completed step** with clear commit messages.
6. Real execution verification is the **final step**, completed by the user on the GPU server.

---

## 4. Repository Structure

Filenames are kept identical to the previous project to make in-place modification clean:

```
gan-inpainting/
├── README.md                 # Project description + GPU server run instructions
├── requirements.txt          # Python dependencies
├── .gitignore                # Ignore data/, outputs/, __pycache__, *.pt, etc.
├── configs/
│   ├── baseline.yaml         # Full GAN (reconstruction + adversarial)
│   ├── recon_only.yaml       # lambda_adv=0 (pure reconstruction, no discriminator)
│   └── lite.yaml             # Full GAN, lightweight (half channels)
├── scripts/
│   ├── download_data.sh      # Download and extract COCO 2017 to data/
│   └── run_all.sh            # Run all experiments + comparison analysis
└── src/
    ├── __init__.py
    ├── config.py             # Load YAML config        (REUSE, extend fields)
    ├── dataset.py            # COCO loading + masking   (EXTEND)
    ├── model.py              # Generator + Discriminator(REPLACE)
    ├── losses.py             # Adversarial + recon loss (REPLACE)
    ├── utils.py              # Seed, checkpoint, images  (REUSE, minor change)
    ├── train.py              # Adversarial training loop (REWRITE)
    ├── evaluate.py           # Inpainting visualization  (REWRITE)
    └── compare.py            # Cross-experiment compare  (MINOR CHANGE)
```

All outputs go to `outputs/` (git-ignored).

---

## 5. Technical Specification for Each Module

### 5.1 Configuration System (`src/config.py`)

The YAML-reading mechanism is unchanged. Update the config fields. Config must include:

```yaml
exp_name: baseline          # determines outputs/ subdirectory
seed: 42
data_dir: data/coco2017     # contains train2017/ and val2017/ flat image dirs
image_size: 256
batch_size: 32
num_workers: 4

base_channels: 32           # base channels for BOTH generator and discriminator; 16 for lite
bottleneck_dim: 1024        # generator bottleneck dimension (deterministic, NOT a sampled latent)

lambda_rec: 100.0           # weight on L1 reconstruction loss
lambda_adv: 1.0             # weight on adversarial loss; set 0 for recon_only experiment
adv_warmup_epochs: 5        # train recon-only for first N epochs, then enable adversarial (stabilizes GAN)

mask_min_ratio: 0.25        # min hole side length as fraction of image_size
mask_max_ratio: 0.5         # max hole side length as fraction of image_size

epochs: 100
lr_g: 0.0002                # generator learning rate
lr_d: 0.0002                # discriminator learning rate
beta1: 0.5                  # Adam beta1 (standard for GAN training)
use_amp: true               # mixed precision

num_vis: 16                 # number of inpainting examples to visualize
compute_fid: false          # optional FID (requires torchmetrics)
```

### 5.2 Data Loading & Masking (`src/dataset.py`)

COCO 2017 stores images in flat directories (`train2017/*.jpg`, `val2017/*.jpg`), no class
subdirectories, and VAE/GAN training here is unsupervised so no annotations are needed.

Implement:

- `CocoImageDataset(Dataset)`: constructor takes `root_dir` and `transform`; collects all
  `.jpg`/`.jpeg` paths via `glob`; `__getitem__` opens with PIL, `.convert("RGB")`, applies
  transform, returns the image tensor.
- Transforms: train = `Resize(image_size)` → `CenterCrop(image_size)` → `RandomHorizontalFlip()`
  → `ToTensor()` (range [0,1]); val = same without the flip. **Do not normalize beyond [0,1]**
  (generator output is Sigmoid).
- `generate_mask(image_size, min_ratio, max_ratio)`: returns a binary mask tensor of shape
  `(1, H, W)`, where **1 marks the missing (hole) region** and 0 marks known pixels. Pick a random
  rectangle whose side lengths are random fractions in `[min_ratio, max_ratio]` of `image_size`,
  placed at a random valid position; set that region to 1.
- A collate or per-item helper that, given an original image tensor `x` and a mask `m`, produces
  the **masked image** `x_masked = x * (1 - m)` (hole pixels zeroed).
- `get_dataloaders(config)`: builds train/val datasets from `{data_dir}/train2017` and
  `{data_dir}/val2017` and returns `(train_loader, val_loader)`. Masks should be generated fresh
  per batch (random each time) — generate them inside the training/eval loop or via a transform,
  so each image gets varied holes across epochs.

> Convention to fix everywhere: **mask==1 → missing region to be filled**. Generator input is the
> 4-channel concatenation `[x_masked (3ch), mask (1ch)]`. The composited completion is
> `x_completed = x_masked + m * G(input)` (known pixels copied exactly, generated content only in
> the hole).

### 5.3 Model (`src/model.py`)

Replace the old `ConvVAE`. Implement two classes:

**`Generator`** — Context-Encoder style, derived from the old VAE encoder/decoder but with the
probabilistic parts removed (no `mu`, no `logvar`, no reparameterization) and a 4-channel input:

- Input: 4 channels `[masked_rgb (3), mask (1)]`.
- Encoder: downsampling blocks `Conv(k=4, s=2, p=1) + BatchNorm + ReLU`; number of layers =
  `log2(image_size) - 2` (256→4 = 6 layers); channels double from `base_channels`
  (32→64→128→256→512→1024).
- Bottleneck: flatten and a linear layer to `bottleneck_dim`, then a linear layer back to the
  flattened feature size and reshape (a deterministic bottleneck, NO sampling). This is the only
  place that differs structurally from the VAE bottleneck.
- Decoder: mirror with `ConvTranspose(k=4, s=2, p=1) + BatchNorm + ReLU`; final layer outputs
  3 channels + **Sigmoid** (range [0,1]). NO skip connections.
- `forward(x_masked, mask)` returns the raw generated RGB `G_out`. Provide a helper
  `complete(x_masked, mask, G_out)` (or compute inside forward) returning
  `x_completed = x_masked + mask * G_out`.

**`Discriminator`** — PatchGAN:

- Input: 3-channel RGB image (real original OR composited completion).
- A stack of `Conv(k=4, s=2, p=1) + (InstanceNorm or BatchNorm, except first layer) + LeakyReLU(0.2)`
  blocks (about 4–5 of them), channels starting from `base_channels` and doubling, ending in a
  `Conv` to **1 output channel** producing a patch score map `(N, 1, h, w)`. **No final Sigmoid**
  (use logits; loss applies `BCEWithLogitsLoss`).

Both models scale with `base_channels` (full=32, lite=16).

### 5.4 Losses (`src/losses.py`)

Replace the ELBO. Implement:

- `reconstruction_loss(x_completed, x_real)`: **L1 loss** over the full image (since known pixels
  are copied exactly, the error concentrates on the hole). Return a scalar.
- Adversarial loss via `torch.nn.BCEWithLogitsLoss` (classic non-saturating GAN):
  - `discriminator_loss(d_real_logits, d_fake_logits)` = BCE(d_real, 1) + BCE(d_fake, 0).
    (Caller passes `d_fake_logits` computed on `x_completed.detach()`.)
  - `generator_adv_loss(d_fake_logits)` = BCE(d_fake, 1) (non-saturating: generator wants the
    discriminator to label completions as real).
- `generator_total_loss = lambda_rec * recon + lambda_adv * adv`.
- Each function returns the scalar component(s) so the training loop can log `d_loss`,
  `g_adv`, `g_recon` separately.
- Comment noting: L1 alone tends to produce blurry/averaged fills; the adversarial term pushes
  completions toward the manifold of realistic images — this is the central point of the
  theoretical analysis.

### 5.5 Utils (`src/utils.py`)

Mostly reused. Required:

- `set_seed(seed)` — random/numpy/torch(+CUDA). (unchanged)
- `save_checkpoint` / `load_checkpoint` — now must store/restore **both** generator and
  discriminator state dicts plus both optimizer states.
- `save_image_grid(tensor, path)` via `torchvision.utils.make_grid`. (unchanged)
- Metrics logger appending per-epoch `d_loss`, `g_adv`, `g_recon`, val L1, and epoch time to
  `metrics.json`.

### 5.6 Training (`src/train.py`)

Rewrite for adversarial training. Entry: `python -m src.train --config configs/baseline.yaml`.

- Build dataloaders, Generator, Discriminator, **two Adam optimizers** (`opt_g` with `lr_g`,
  `opt_d` with `lr_d`, both `betas=(beta1, 0.999)`).
- Per training batch:
  1. Load real images `x_real`; generate masks; compute `x_masked`.
  2. `G_out = G(x_masked, mask)`; `x_completed = x_masked + mask * G_out`.
  3. **Update D** (only if `lambda_adv > 0` and current epoch ≥ `adv_warmup_epochs`):
     real logits on `x_real`, fake logits on `x_completed.detach()`; `d_loss`; `opt_d` step.
  4. **Update G**: `g_recon = L1(x_completed, x_real)`; if adversarial is active, also
     `g_adv = generator_adv_loss(D(x_completed))`, else `g_adv = 0`;
     `g_total = lambda_rec * g_recon + lambda_adv * g_adv`; `opt_g` step.
- `adv_warmup_epochs`: for the first N epochs train generator with reconstruction only (D not
  updated, `g_adv` disabled) to stabilize early training, then enable the adversarial game.
- `use_amp`: wrap forward/backward in `torch.cuda.amp.autocast` with a `GradScaler` per optimizer.
- Each epoch: log losses + time; run a quick val pass computing val L1; save `last.pt`, and
  `best.pt` by lowest val L1. Checkpoints contain both G and D.
- **After training, automatically call evaluation** (`src/evaluate.py` main function) so one
  command does train + eval.
- Outputs to `outputs/{exp_name}/`.

### 5.7 Evaluation (`src/evaluate.py`)

Rewrite for inpainting. Load `best.pt` generator and produce to `outputs/{exp_name}/`:

- `inpainting.png`: `num_vis` rows, each row = `[masked_input | completed | ground_truth]`
  side by side, on **validation** images (the core unseen-sample inpainting task). Use a fixed
  seed for mask generation here so all experiments are visualized on comparable masks.
- `loss_curve.png`: `d_loss`, `g_adv`, `g_recon`, and val L1 vs epoch (read from `metrics.json`).
- `metrics.json`: final val L1, PSNR and SSIM on val (compute via `torchmetrics` if available,
  else a simple PSNR; wrap in try/except so failure doesn't break the pipeline), generator and
  discriminator parameter counts, and optional FID between val originals and completions.
- Do NOT produce prior-sample grids or latent interpolations (not applicable to a deterministic
  conditional generator).

### 5.8 Comparison Analysis (`src/compare.py`)

Minor change of intent. Entry: `python -m src.compare`:

- `outputs/comparison/gan_vs_recon.png`: place the `baseline` (full GAN) and `recon_only`
  completions on the **same** validation images side by side, to show how the adversarial loss
  changes sharpness/realism versus pure L1 (which looks blurry).
- `outputs/comparison/summary.md`: table of final val L1 / PSNR / SSIM, generator + discriminator
  parameter counts, and optional FID per experiment; highlight full vs lite parameter difference.
- Tolerate missing experiment outputs (skip with a notice, don't crash).

---

## 6. Implementation of Three Improvement Items

Implemented via config switching:

1. **Generation quality — adversarial loss ablation**: `baseline.yaml` (`lambda_adv=1`, full GAN)
   vs `recon_only.yaml` (`lambda_adv=0`, generator trained with L1 only, discriminator unused).
   Shows what the adversarial term contributes (sharper, more realistic fills vs blurry averages).
2. **Computational efficiency — mixed precision**: `use_amp` toggles AMP; training logs record
   per-epoch time to compare speed.
3. **Lightweighting — channel reduction**: `lite.yaml` sets `base_channels=16` (both G and D);
   evaluation records parameter counts; `compare.py` contrasts full vs lite.

---

## 7. Run Scripts

- `scripts/download_data.sh`: download COCO 2017 train/val images via `wget`, extract to
  `data/coco2017/` so the structure is `data/coco2017/train2017/*.jpg` (~118k, ~18GB) and
  `data/coco2017/val2017/*.jpg` (~5k, ~1GB). URLs:
  `http://images.cocodataset.org/zips/train2017.zip` and
  `http://images.cocodataset.org/zips/val2017.zip`. Delete the zips after extraction.
- `scripts/run_all.sh`: run `train` for `baseline.yaml`, `recon_only.yaml`, `lite.yaml`, then
  `python -m src.compare`.

---

## 8. Dependencies & Environment

`requirements.txt`: `torch`, `torchvision`, `numpy`, `pyyaml`, `matplotlib`, `tqdm`, `pillow`,
`torchmetrics` (optional FID/PSNR/SSIM). Avoid over-pinning versions.

---

## 9. Coding Standards

- Python 3.10+, PyTorch. Modular, single responsibility, docstrings, type hints for key functions.
- No hardcoded absolute paths; all paths from config/script args.
- Beginner-oriented comments explaining GAN key steps (adversarial game, G/D alternating updates,
  why L1 alone blurs, the mask convention).

---

## 10. Deliverables Checklist

- [ ] Updated `src/` code (config, dataset+masking, model G+D, losses, utils, train, evaluate, compare)
- [ ] Three `configs/*.yaml` (baseline, recon_only, lite)
- [ ] Two `scripts/*.sh`
- [ ] `requirements.txt`, `.gitignore`
- [ ] `README.md` with project description + GPU server run instructions
- [ ] All Python files pass `python -m py_compile`
- [ ] No training/GPU code executed, no datasets downloaded during development

---

## 11. Migration Notes (from the previous VAE project)

The repository already contains a VAE project. Migrate in place, reusing structure where possible:

- **Reuse as-is**: `config.py` loading mechanism, `utils.py` seed/image-grid/logging,
  the COCO file-collection logic in `dataset.py`, the overall repo layout and workflow.
- **Extend**: `dataset.py` — keep COCO loading, ADD `generate_mask` and the masked-image helper,
  and have `__getitem__`/loaders deliver what the inpainting loop needs.
- **Replace**: `model.py` — turn the VAE encoder/decoder into the `Generator` (remove
  `mu`/`logvar`/reparameterization, add 4-channel input and a deterministic bottleneck), and ADD
  the `Discriminator`. `losses.py` — ELBO is gone; implement L1 + BCEWithLogits adversarial losses.
- **Rewrite**: `train.py` (two optimizers, G/D alternating updates, adv warmup) and `evaluate.py`
  (inpainting triplet visualization; drop prior sampling and interpolation).
- **Minor**: `compare.py` (β comparison → GAN-vs-recon comparison), `configs/` (new set),
  `run_all.sh` (new experiment list), README.
- **Unchanged**: `download_data.sh` (still COCO 2017).
