# VAE Generative Model Practice Project

An entry-level hands-on implementation of a Variational Autoencoder (ConvVAE) trained on
[Imagenette](https://github.com/fastai/imagenette) (a 10-class ImageNet subset).

## Project Structure

```
├── configs/          # YAML experiment configs (baseline, beta_0.5, beta_4, lite)
├── scripts/          # Data download and batch-run scripts
├── src/              # All Python source code
│   ├── config.py     # YAML config loader
│   ├── dataset.py    # DataLoader builder
│   ├── model.py      # ConvVAE model
│   ├── losses.py     # ELBO loss (recon + beta * KL)
│   ├── utils.py      # Seed, checkpointing, image saving, metrics logging
│   ├── train.py      # Training entry point (auto-runs evaluation after training)
│   ├── evaluate.py   # Generation / reconstruction / interpolation / metrics
│   └── compare.py    # Cross-experiment comparison plots and summary table
└── outputs/          # All experiment outputs (git-ignored)
```

---

## Environment Setup (GPU Server: CUDA 12.1 + Ubuntu 22.04)

### 1. Create and activate a conda environment

```bash
conda create -n vae python=3.10 -y
conda activate vae
```

### 2. Install PyTorch with CUDA 12.1 support

```bash
conda install pytorch==2.2.2 torchvision==0.17.2 pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

> If you prefer pip:
> ```bash
> pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 \
>     --index-url https://download.pytorch.org/whl/cu121
> ```

### 3. Install the remaining dependencies

```bash
pip install -r requirements.txt
```

> `requirements.txt` specifies `torch>=2.1.0` and `torchvision>=0.16.0` as lower bounds.
> Because you already installed the exact CUDA build in step 2, pip will skip reinstalling them.

### 4. Verify the installation

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Expected output: `2.2.2 True`

---

## Prepare the Dataset

```bash
bash scripts/download_data.sh
```

This downloads `imagenette2-160.tgz` (~100 MB) and extracts it to
`data/imagenette2-160/{train,val}/<class>/`.

---

## Run Experiments

### Run all 4 experiments at once

```bash
bash scripts/run_all.sh
```

This sequentially trains and evaluates: **baseline** (β=1), **beta_0.5** (β=0.5),
**beta_4** (β=4), **lite** (base_channels=16), then generates the comparison report.

### Run a single experiment

```bash
python -m src.train --config configs/baseline.yaml
```

Training automatically calls evaluation after finishing. All outputs go to
`outputs/<exp_name>/`:

| File | Description |
|---|---|
| `samples.png` | Images sampled from prior N(0,I) |
| `reconstructions.png` | Val-set originals vs. reconstructions |
| `interpolations.png` | Latent-space interpolation sequences |
| `loss_curve.png` | Train/val loss curves over epochs |
| `metrics.json` | Final losses, parameter count, optional FID |
| `last.pt` / `best.pt` | Latest and best checkpoints |

### Evaluate from a saved checkpoint

```bash
python -m src.evaluate --config configs/baseline.yaml
```

### Generate comparison report (after all experiments)

```bash
python -m src.compare
```

Outputs are written to `outputs/comparison/`.

---

## Experiment Configs

| Config | β | base_channels | Purpose |
|---|---|---|---|
| `baseline.yaml` | 1.0 | 32 | Reference model |
| `beta_0.5.yaml` | 0.5 | 32 | Lower KL weight → sharper reconstructions |
| `beta_4.yaml` | 4.0 | 32 | Higher KL weight → more structured latent space |
| `lite.yaml` | 1.0 | 16 | Half channels → fewer parameters |

---

## Three Modules

1. **Application Practice** — Train on Imagenette; generate new images from prior N(0,I),
   reconstruct unseen validation samples, and perform latent-space interpolation.
2. **Model Improvements** — Compare β values for generation quality; AMP for computational
   efficiency; `lite` config for lightweighting.
3. **Theoretical Analysis** — Discuss prior distribution, approximate posterior, and Gaussian
   likelihood assumptions based on experimental observations.
