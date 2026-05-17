# VAE Generative Model Practice Project

An entry-level hands-on implementation of a **Variational Autoencoder (ConvVAE)**
trained on [COCO 2017](https://cocodataset.org/) at 256×256 resolution. The goal
is to build intuitive understanding of generative AI through implementation and
experiment, not to pursue state-of-the-art results.

---

## Project Modules

### Module 1 — Application Practice

Train a ConvVAE on COCO 2017 (256×256) and apply it to three generative tasks
on **unseen** validation samples:

| Task | Output file |
|---|---|
| **Generation** — sample z ~ N(0,I), decode to new images | `samples.png` |
| **Reconstruction** — encode val images, decode back | `reconstructions.png` |
| **Interpolation** — linearly interpolate between latent means | `interpolations.png` |

COCO's train2017 / val2017 split provides the sample-level held-out set
(the model never sees val2017 images during training).

### Module 2 — Model Improvements

Three improvements, each controlled purely by config (no code changes):

| Improvement | Mechanism | Configs |
|---|---|---|
| **Generation quality** | β-VAE: vary KL weight β | `baseline` / `beta_0.5` / `beta_4` |
| **Computational efficiency** | Mixed-precision AMP (`use_amp: true`) | all configs |
| **Lightweighting** | Halve encoder/decoder channels | `baseline` vs `lite` |

### Module 3 — Theoretical Analysis

Based on experimental results, discuss three theoretical assumptions of the
standard VAE:

1. **Prior distribution** — p(z) = N(0,I): does the β-VAE comparison reveal
   how well the learned posterior matches this prior?
2. **Approximate posterior** — q(z|x) = N(μ,diag(σ²)): limitations of the
   diagonal Gaussian family.
3. **Gaussian likelihood** — p(x|z) ∝ exp(−‖x−x̂‖²): using MSE reconstruction
   assumes fixed variance; what does this mean for sample sharpness?

---

## Repository Structure

```
├── configs/
│   ├── baseline.yaml       # β=1.0, base_channels=32  (reference)
│   ├── beta_0.5.yaml       # β=0.5, base_channels=32
│   ├── beta_4.yaml         # β=4.0, base_channels=32
│   └── lite.yaml           # β=1.0, base_channels=16  (lightweight)
├── scripts/
│   ├── download_data.sh    # Download COCO 2017 train+val to data/coco2017/
│   └── run_all.sh          # Run all 4 experiments + comparison
├── src/
│   ├── __init__.py
│   ├── config.py           # YAML loader, --config argument parsing
│   ├── dataset.py          # CocoImageDataset + DataLoaders (train + val)
│   ├── model.py            # ConvVAE (encoder / reparameterisation / decoder)
│   ├── losses.py           # ELBO loss: MSE reconstruction + β·KL
│   ├── utils.py            # set_seed, checkpoints, image grid, MetricsLogger
│   ├── train.py            # Training loop (AMP, KL annealing, auto-eval)
│   ├── evaluate.py         # Generation, reconstruction, interpolation, metrics
│   └── compare.py          # Cross-experiment plots and summary table
├── README.md
├── requirements.txt
└── .gitignore
```

All experiment outputs go to `outputs/` (git-ignored):

```
outputs/
├── baseline/               # checkpoints, metrics.json, eval images
├── beta_0.5/
├── beta_4/
├── lite/
└── comparison/             # beta_comparison.png, summary.md
```

---

## Running on a GPU Server

### Step 0 — Clone the repository

```bash
git clone <repo-url>
cd AGI_course_project_VAE
```

If you already cloned it, pull the latest code:

```bash
git pull
```

### Step 1 — Create the conda environment

```bash
conda create -n vae python=3.10 -y
conda activate vae
```

### Step 2 — Install PyTorch (CUDA 12.1)

```bash
conda install pytorch==2.2.2 torchvision==0.17.2 pytorch-cuda=12.1 \
    -c pytorch -c nvidia -y
```

<details>
<summary>Prefer pip?</summary>

```bash
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
```

</details>

### Step 3 — Install remaining dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` declares `torch>=2.1.0` and `torchvision>=0.16.0` as lower
bounds; pip will skip reinstalling the CUDA build you just installed.

Verify CUDA is available:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Expected: 2.2.2 True
```

### Step 4 — Download the dataset

```bash
bash scripts/download_data.sh
```

Downloads COCO 2017 train and val images (~19 GB total). Re-running is safe —
each split is skipped if its directory already exists.

> **Disk space**: ~19 GB during download, ~19 GB after extraction (zips are
> deleted automatically). Ensure you have at least 40 GB free.

Expected layout after extraction:

```
data/coco2017/
├── train2017/*.jpg   (~118,000 images, ~18 GB)
└── val2017/*.jpg     (~5,000 images,   ~1 GB)
```

### Step 5 — Run all experiments

```bash
bash scripts/run_all.sh
```

This runs four experiments in sequence. Each `src.train` call automatically
invokes `src.evaluate` after training, so one command covers:

```
train baseline  → eval baseline
train beta_0.5  → eval beta_0.5
train beta_4    → eval beta_4
train lite      → eval lite
python -m src.compare          ← generates comparison outputs
```

Total wall time is roughly **4 × (training time)** depending on your GPU.

### Step 6 — Inspect outputs

```
outputs/
├── baseline/
│   ├── best.pt                  model checkpoint (best val loss)
│   ├── last.pt                  model checkpoint (final epoch)
│   ├── metrics.json             per-epoch train/val losses + timing
│   ├── eval_metrics.json        final losses, n_params, optional FID
│   ├── samples.png              64 images sampled from N(0,I)
│   ├── reconstructions.png      8 val originals vs reconstructions
│   ├── interpolations.png       8 latent-space interpolation sequences
│   └── loss_curve.png           total / recon / KL curves
├── beta_0.5/  beta_4/  lite/   (same structure)
└── comparison/
    ├── beta_comparison.png      side-by-side for baseline/β0.5/β4
    └── summary.md               table: losses, n_params, FID for all 4 exps
```

---

## Running Individual Steps

### Train one experiment

```bash
python -m src.train --config configs/baseline.yaml
```

### Re-run evaluation from a saved checkpoint

```bash
python -m src.evaluate --config configs/baseline.yaml
```

Loads `outputs/baseline/best.pt` and regenerates all image outputs and
`eval_metrics.json`.

### Re-run comparison analysis

```bash
python -m src.compare
```

Reads `eval_metrics.json` from all available experiments. Missing experiments
are skipped with a warning.

---

## Experiment Configs at a Glance

| Config | β | base_channels | Purpose |
|---|---|---|---|
| `baseline.yaml` | 1.0 | 32 | Reference model |
| `beta_0.5.yaml` | 0.5 | 32 | Less KL → sharper reconstructions |
| `beta_4.yaml` | 4.0 | 32 | More KL → structured latent space |
| `lite.yaml` | 1.0 | 16 | Lightweight (≈¼ the parameters of full model) |

Key config fields (edit `configs/*.yaml` to change):

```yaml
epochs: 100            # total training epochs
lr: 0.001              # Adam learning rate
use_amp: true          # mixed-precision training (requires CUDA)
kl_anneal_epochs: 10   # linearly ramp beta 0→target over first 10 epochs
image_size: 256        # 256×256 resolution
batch_size: 32         # reduced vs 64px to fit 256×256 in GPU memory
latent_dim: 1024
```

---

## Dependencies

| Package | Role |
|---|---|
| `torch` / `torchvision` | Model, training, image transforms |
| `numpy` | Numerical operations |
| `pyyaml` | YAML config loading |
| `matplotlib` | Loss curves and comparison figures |
| `tqdm` | Training progress bars |
| `pillow` | Saving image grids as PNG |
| `torchmetrics` | Optional FID computation (`compute_fid: true`) |
