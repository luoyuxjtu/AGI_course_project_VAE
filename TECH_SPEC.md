# VAE Generative Model Practice Project — Technical Specification

> This document is for Claude Code to read. It describes the complete project goals, scope,
> technical details, and workflow constraints. Please read it fully before writing code and
> strictly follow the workflow constraints in Section 3.

---

## 1. Project Background & Goals

This is an entry-level hands-on practice project for AI enthusiasts. The goal is to implement
and train a Variational Autoencoder (VAE) from scratch to build intuitive understanding of
generative AI.

The project consists of three modules:

1. **Application Practice** — Train a VAE on ImageNet-style data, completing generative tasks
   on unseen samples: sampling from the prior to generate new images, reconstructing unseen
   samples, and latent space interpolation.
2. **Model Improvements** — Run controlled comparison experiments across three improvement axes:
   generation quality, computational efficiency, and model lightweighting.
3. **Theoretical Analysis** — Based on experimental results, discuss three theoretical assumptions:
   prior distribution, approximate posterior distribution, and Gaussian assumptions.

This is an educational exercise, **not pursuing SOTA**. Code should prioritize clarity,
readability, reproducibility, and modularity.

---

## 2. Scope & Constraints

To ensure completion within one week, scope is intentionally simplified:

- **"Unseen" definition uses sample-level hold-out**: Use the dataset's built-in train/val split.
  The validation set represents samples the model has never seen. **Do not** use class-level
  hold-out.
- **Dataset**: Imagenette (a 10-class subset of ImageNet), loaded via
  `torchvision.datasets.ImageFolder`. Imagenette naturally provides `train/` and `val/`
  directories, meeting sample-level unseen requirements.
- **Image resolution**: Default 64×64 (configurable to 32×32 for speed via config).
- **Model**: Standard convolutional VAE (ConvVAE). Do not implement hierarchical VAE,
  normalizing flows, or other complex structures.
- **Improvement items**: Only three lightweight improvements (Section 6). No additional
  enhancements.
- **Theoretical analysis**: Based on experimental results, do observational discussions.
  Do not require formula derivation or new algorithm implementation.

---

## 3. Workflow Constraints (Critical — Must Follow)

This project separates development from execution. Claude Code must strictly follow:

1. **Claude Code is responsible only for writing all project code in the GitHub cloud
   repository.** The user will later pull it to a GPU server to run. Claude Code's
   environment has **no GPU and no datasets**.
2. **Do not run training code, do not run any code requiring GPU or datasets, do not download
   datasets for testing.**
3. **Do not do functional testing in intermediate steps.** The only permitted and recommended
   check is **syntax checking**: run `python -m py_compile <file>` on new or modified Python
   files. This requires no torch installation or data, only verifies syntax correctness, and
   prevents users from pulling broken code.
4. Code must be in a state that can run directly on the GPU server: no hardcoded paths,
   all dependencies declared in `requirements.txt`, entry points clearly documented in README.
5. **Commit after each completed step** with clear commit messages describing the step's
   content, enabling users to track progress.
6. Real execution verification is the **final step** of the entire process, completed by the
   user on the GPU server, not during development.

---

## 4. Repository Structure

Organize the project as follows:

```
vae-practice/
├── README.md                 # Project description + GPU server run instructions
├── requirements.txt          # Python dependencies
├── .gitignore                # Ignore data/, outputs/, __pycache__, etc.
├── configs/
│   ├── baseline.yaml         # beta=1, full model (baseline)
│   ├── beta_0.5.yaml         # beta=0.5
│   ├── beta_4.yaml           # beta=4
│   └── lite.yaml             # beta=1, lightweight (half channels)
├── scripts/
│   ├── download_data.sh      # Download and extract Imagenette to data/
│   └── run_all.sh            # Run 4 experiments + comparison analysis
└── src/
    ├── __init__.py
    ├── config.py             # Load/merge YAML config
    ├── dataset.py            # Data loading
    ├── model.py              # ConvVAE model
    ├── losses.py             # ELBO loss
    ├── utils.py              # Random seed, checkpoint, image saving, logging
    ├── train.py              # Training entry point (auto-calls evaluation after training)
    ├── evaluate.py           # Generation / reconstruction / interpolation / metrics
    └── compare.py            # Cross-experiment comparison + analysis plots
```

All outputs go to `outputs/` (git-ignored), not in version control.

---

## 5. Technical Specification for Each Module

### 5.1 Configuration System (`src/config.py`)

- Use YAML files to manage all hyperparameters; provide a function to read yaml and return
  a config object (dict or simple dataclass).
- Training/evaluation scripts specify config via `--config <path>`.
- Config must include at least:

  ```yaml
  exp_name: baseline          # Experiment name, determines outputs/ subdirectory
  seed: 42
  data_dir: data/imagenette2-160
  image_size: 64
  batch_size: 128
  num_workers: 4

  latent_dim: 128
  base_channels: 32           # Change to 16 for lite config
  beta: 1.0
  kl_anneal_epochs: 0         # >0: linearly anneal beta over first N epochs; 0: disabled

  epochs: 50
  lr: 0.001
  use_amp: true               # Whether to use mixed precision

  num_samples: 64             # Number of samples to generate during evaluation
  num_interpolations: 8       # Number of interpolation sequences
  compute_fid: false          # Whether to compute FID (optional, requires torchmetrics)
  ```

### 5.2 Data Loading (`src/dataset.py`)

- Use `torchvision.datasets.ImageFolder` to load `{data_dir}/train` and `{data_dir}/val`
  separately.
- Train set transforms: `Resize(image_size)` → `CenterCrop(image_size)` →
  `RandomHorizontalFlip()` → `ToTensor()` (outputs range [0,1]).
- Val set transforms: `Resize(image_size)` → `CenterCrop(image_size)` → `ToTensor()`.
- **Do not normalize beyond [0,1]** (decoder output is Sigmoid, reconstruction target is [0,1]).
- Provide a function returning `(train_loader, val_loader)`.

### 5.3 Model (`src/model.py`)

Implement standard convolutional VAE, class name: `ConvVAE`:

**Encoder** — Composed of downsampling blocks: `Conv(kernel=4, stride=2, padding=1) + BatchNorm + ReLU`.
Number of downsampling layers computed automatically from `image_size` (each layer halves spatial
dims, down to 4×4; e.g., 64→4 needs 4 layers, 32→4 needs 3 layers; assume image_size is power of 2).
Channel count doubles each layer starting from `base_channels`. After flattening features, apply
two linear layers to output `mu` and `logvar` (both of dimension `latent_dim`).

**Reparameterization** — `z = mu + exp(0.5 * logvar) * eps`, where `eps ~ N(0, I)`.

**Decoder** — Linear layer maps `z` to flattened feature size and reshape. Then mirror encoder with
upsampling blocks: `ConvTranspose(kernel=4, stride=2, padding=1) + BatchNorm + ReLU`. Final layer
outputs 3 channels, followed by **Sigmoid** (range [0,1]).

`forward` returns `(x_recon, mu, logvar)`. Also provide `sample(n)` method: sample `n` vectors
from `N(0, I)` and decode them (for generation tasks).

`base_channels` is the key lever for model size: full model uses 32, lite model uses 16.

### 5.4 Loss (`src/losses.py`)

Implement ELBO loss:

- **Reconstruction term**: MSE between `x_recon` and `x`, summed over pixels, averaged over batch.
  (MSE reconstruction corresponds to assuming Gaussian likelihood with fixed variance — this is
  discussed in theoretical analysis. Keep MSE for now.)
- **KL term**: Analytical form `KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))`, averaged over batch.
- **Total loss**: `loss = recon + beta * KL`.
- Function should return three scalars separately: `recon`, `KL`, `total`, for separate logging.

### 5.5 Utils (`src/utils.py`)

Provide these utility functions:

- `set_seed(seed)`: Fix random seeds for `random`, `numpy`, `torch` (including CUDA).
- `save_checkpoint` / `load_checkpoint`: Save/load model weights and training state.
- `save_image_grid(tensor, path)`: Save image grid using `torchvision.utils.make_grid`.
- Simple metrics logger: Append each epoch's train/val recon/KL/total losses to `metrics.json`
  or csv for later visualization.

### 5.6 Training (`src/train.py`)

- Entry point: `python -m src.train --config configs/baseline.yaml`.
- Flow: Read config → set seed → build dataloader/model/Adam optimizer → training loop.
- Training loop requirements:
  - If `use_amp` is true, use `torch.cuda.amp` (autocast + GradScaler).
  - If `kl_anneal_epochs > 0`, linearly anneal beta from 0 to target over first N epochs.
  - After each epoch, evaluate on validation set, log metrics.
  - Save `last.pt` and `best.pt` (based on validation loss).
- **After training, automatically call evaluation** (main function from `src/evaluate.py`),
  producing all evaluation outputs. This way, one GPU command completes train + eval for one
  experiment.
- All outputs go to `outputs/{exp_name}/`.

### 5.7 Evaluation (`src/evaluate.py`)

Load `best.pt` and produce the following to `outputs/{exp_name}/`:

- `samples.png`: Grid of images sampled from prior N(0,I) and decoded (**generation task**).
- `reconstructions.png`: One validation batch with original images (top) and reconstructions
  (bottom) side-by-side (**unseen sample reconstruction task**).
- `interpolations.png`: Linear interpolations between latent means of several image pairs,
  decoded. Each row is one interpolation sequence (**latent space interpolation task**).
- `loss_curve.png`: Curves of total/recon/KL losses (train and val) vs epoch, read from metrics.json.
- `metrics.json`: Final numerical metrics (final losses, model parameter count, optional FID).
- If `compute_fid` is true, use `torchmetrics` FID to compute FID between validation set and
  generated samples, write to `metrics.json`. If FID computation fails, catch exception and skip
  without breaking the pipeline.
- Simultaneously count and log total model parameters for lightweighting comparison.

### 5.8 Comparison Analysis (`src/compare.py`)

Run after all experiments, entry point: `python -m src.compare`:

- Read `metrics.json` and outputs from all experiments in `outputs/`.
- Generate `outputs/comparison/beta_comparison.png`: Side-by-side display of generated/reconstructed
  images from baseline, beta_0.5, and beta_4, showing beta's effect on reconstruction clarity and
  latent space structure.
- Generate summary table `outputs/comparison/summary.md` or `.csv`: List final recon/KL losses,
  parameter counts, and optional FID for each experiment. Highlight parameter count difference
  between full and lite models.
- These outputs directly support reports for "Model Improvements" and "Theoretical Analysis" modules.

---

## 6. Implementation of Three Improvement Items

Improvements are implemented via config switching, no code branching needed:

1. **Generation Quality — β-VAE Comparison**: Use three configs (`baseline.yaml` with β=1,
   `beta_0.5.yaml`, `beta_4.yaml`) to compare trade-offs between reconstruction clarity and
   latent space regularization across different β values.
2. **Computational Efficiency — Mixed Precision**: `use_amp` config controls AMP. Training logs
   should record time per epoch to compare speed differences with AMP on/off.
3. **Lightweighting — Channel Reduction**: `configs/lite.yaml` reduces `base_channels` from 32
   to 16. Evaluation records parameter count. `compare.py` contrasts full vs lite.

---

## 7. Run Scripts

- `scripts/download_data.sh`: Download Imagenette 160px version using `wget`
  (`https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz`), extract to `data/`.
  After extraction: `data/imagenette2-160/{train,val}/<class>/<image>.JPEG`.
- `scripts/run_all.sh`: Sequentially run `python -m src.train --config configs/baseline.yaml`,
  `beta_0.5.yaml`, `beta_4.yaml`, `lite.yaml`, then `python -m src.compare`.

---

## 8. Dependencies & Environment

`requirements.txt` must include: `torch`, `torchvision`, `numpy`, `pyyaml`, `matplotlib`,
`tqdm`, `pillow`, `torchmetrics` (for optional FID). Avoid over-pinning versions to maintain
compatibility with GPU server CUDA environment.

---

## 9. Coding Standards

- Python 3.10+, PyTorch.
- Modular design with single responsibility; clear docstrings for functions and classes.
- Type hints for key functions.
- No hardcoded absolute paths; all paths from config or script arguments.
- Consistent code style, clear variable names.
- Comments geared toward beginners, explaining VAE key steps (reparameterization, ELBO, KL).

---

## 10. Deliverables Checklist

After Claude Code completes, the repository should have:

- [ ] Complete `src/` code (config, dataset, model, losses, utils, train, evaluate, compare)
- [ ] Four `configs/*.yaml` files
- [ ] Two `scripts/*.sh` files
- [ ] `requirements.txt`, `.gitignore`
- [ ] `README.md` with project description and complete GPU server run instructions
- [ ] All Python files pass `python -m py_compile` syntax check
- [ ] No training/GPU code executed, no datasets downloaded during development

Finally, the user pulls and verifies by running on GPU server.
