# GAN Image Inpainting Practice Project

An entry-level hands-on implementation of a **GAN-based image inpainting** model
trained on [COCO 2017](https://cocodataset.org/) at 256×256 resolution.
Given an image with a randomly masked region, the generator fills in plausible
content; a PatchGAN discriminator drives the completions toward realistic textures.

---

## Project Modules

### Module 1 — Application Practice

Train a Context-Encoder-style generator + PatchGAN discriminator on COCO 2017
`train2017` and evaluate on the **unseen** `val2017` set:

| Output | Description |
|---|---|
| `inpainting.png` | `num_vis` rows of `[masked input | completion | ground truth]` |
| `loss_curve.png` | D loss, G adversarial loss, G recon L1, val L1 vs epoch |
| `eval_metrics.json` | Val L1, PSNR, SSIM, G/D parameter counts, optional FID |

### Module 2 — Model Improvements

Three improvements, each controlled by config (no code changes):

| Improvement | Mechanism | Configs |
|---|---|---|
| **Generation quality** | Adversarial loss ablation: full GAN vs L1 only | `baseline` vs `recon_only` |
| **Computational efficiency** | Mixed-precision AMP (`use_amp: true`) | all configs |
| **Lightweighting** | Halve G and D base channels (32→16) | `baseline` vs `lite` |

### Module 3 — Theoretical Analysis

Based on experimental results, discuss three topics:

1. **L1 blurriness vs adversarial sharpness** — compare `baseline` vs `recon_only`:
   L1 minimises expected pixel error, averaging over multi-modal completions;
   the adversarial term pushes results onto the real-image manifold.
2. **Adversarial training dynamics** — D and G loss curves; the GAN minimax game.
3. **Reconstruction loss trade-off** — higher `lambda_rec` anchors structure
   but may damp texture; lower values allow more hallucination.

---

## Repository Structure

```
├── configs/
│   ├── baseline.yaml       # Full GAN (L1 + adversarial, base_channels=32)
│   ├── recon_only.yaml     # L1 only  (lambda_adv=0, no discriminator update)
│   └── lite.yaml           # Full GAN, base_channels=16 (lightweight)
├── scripts/
│   ├── download_data.sh    # Download COCO 2017 train+val to data/coco2017/
│   └── run_all.sh          # Run all 3 experiments + comparison
├── src/
│   ├── __init__.py
│   ├── config.py           # YAML loader, --config argument parsing
│   ├── dataset.py          # CocoImageDataset, generate_mask, make_masked_image
│   ├── model.py            # Generator (Context-Encoder) + Discriminator (PatchGAN)
│   ├── losses.py           # L1 reconstruction + BCEWithLogits adversarial losses
│   ├── utils.py            # set_seed, checkpoints (G+D), image grid, MetricsLogger
│   ├── train.py            # Adversarial training loop (AMP, adv warmup, auto-eval)
│   ├── evaluate.py         # Inpainting visualisation, loss curves, metrics
│   └── compare.py          # GAN vs recon figure + summary table
├── README.md
├── requirements.txt
└── .gitignore
```

All experiment outputs go to `outputs/` (git-ignored):

```
outputs/
├── baseline/               # best.pt, last.pt, metrics.json, eval_metrics.json,
├── recon_only/             # inpainting.png, loss_curve.png
├── lite/
└── comparison/             # gan_vs_recon.png, summary.md
```

---

## Running on a GPU Server

### Step 0 — Clone (or pull) the repository

```bash
git clone <repo-url>
cd AGI_course_project_VAE
# or, if already cloned:
git pull
```

### Step 1 — Create the conda environment

```bash
conda create -n gan python=3.10 -y
conda activate gan
```

### Step 2 — Install PyTorch with CUDA

```bash
conda install pytorch==2.2.2 torchvision==0.17.2 pytorch-cuda=12.1 \
    -c pytorch -c nvidia -y
```

<details>
<summary>Prefer pip?</summary>

```bash
pip install torch==2.2.2 torchvision==0.17.2 \
    --index-url https://download.pytorch.org/whl/cu121
```

</details>

Verify CUDA is available:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Expected: 2.2.2 True
```

### Step 3 — Install remaining dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Download the dataset

```bash
bash scripts/download_data.sh
```

Downloads COCO 2017 train and val images (~19 GB total). Re-running is safe —
each split is skipped if its directory already exists.

> **Disk space**: ~40 GB free recommended (zips are deleted after extraction).

Expected layout:

```
data/coco2017/
├── train2017/*.jpg   (~118,000 images, ~18 GB)
└── val2017/*.jpg     (~5,000 images,   ~1 GB)
```

### Step 5 — Run all experiments

```bash
bash scripts/run_all.sh
```

Runs three experiments sequentially; each `src.train` call automatically invokes
`src.evaluate` after training, then `src.compare` collates all results:

```
train baseline   → eval baseline
train recon_only → eval recon_only
train lite       → eval lite
python -m src.compare
```

Monitor progress in another terminal:

```bash
tail -f outputs/baseline/train.log
```

### Step 6 — Inspect outputs

| Path | Contents |
|---|---|
| `outputs/<exp>/inpainting.png` | `num_vis` triplet rows: masked \| completed \| original |
| `outputs/<exp>/loss_curve.png` | D loss, G adv, G recon, val L1 vs epoch |
| `outputs/<exp>/eval_metrics.json` | val L1, PSNR, SSIM, param counts, FID |
| `outputs/<exp>/best.pt` | Best-val-L1 checkpoint (G + D + optimisers) |
| `outputs/comparison/gan_vs_recon.png` | Baseline vs recon_only side-by-side |
| `outputs/comparison/summary.md` | All-experiment metrics table |

---

## Running Individual Steps

```bash
# Train one experiment
python -m src.train --config configs/baseline.yaml

# Re-run evaluation from a saved checkpoint
python -m src.evaluate --config configs/baseline.yaml

# Re-run comparison analysis
python -m src.compare
```

---

## Key Config Fields

```yaml
base_channels: 32       # G and D channel multiplier; 16 for lite
bottleneck_dim: 1024    # deterministic G bottleneck size

lambda_rec: 100.0       # weight on L1 reconstruction loss
lambda_adv: 1.0         # weight on adversarial loss (0 = recon_only)
adv_warmup_epochs: 5    # epochs of recon-only warm-up before adversarial starts

mask_min_ratio: 0.25    # hole size range as fraction of image_size
mask_max_ratio: 0.5

epochs: 100
lr_g: 0.0002            # generator Adam learning rate
lr_d: 0.0002            # discriminator Adam learning rate
beta1: 0.5              # Adam beta1 (standard for GAN training)
use_amp: true           # mixed-precision training (requires CUDA)
```

---

## Architecture Summary

**Generator** (Context-Encoder style)
- Input: 4 channels `[masked_rgb (3) | mask (1)]`
- Encoder: `Conv(k=4,s=2,p=1) + BN + ReLU` × 6 (256→4, channels 4→1024)
- Bottleneck: `Linear(flat→bottleneck_dim) → Linear(bottleneck_dim→flat)` (deterministic)
- Decoder: `ConvTranspose(k=4,s=2,p=1) + BN + ReLU` × 5 + Sigmoid; no skip connections
- Output: `x_completed = x_masked + mask * G_out` (known pixels preserved exactly)

**Discriminator** (PatchGAN)
- Input: 3-channel RGB (real or completed)
- 4 × `Conv(k=4,s=2,p=1) + [InstanceNorm] + LeakyReLU(0.2)` + final Conv → 1 ch logit map
- No Sigmoid; use `BCEWithLogitsLoss`; ~70×70 receptive field at 256×256

---

## Dependencies

| Package | Role |
|---|---|
| `torch` / `torchvision` | Model, training, image transforms |
| `numpy` | Numerical operations |
| `pyyaml` | YAML config loading |
| `matplotlib` | Loss curves and comparison figures |
| `tqdm` | Training progress bars |
| `pillow` | Image loading and grid saving |
| `torchmetrics` | PSNR, SSIM, optional FID (`compute_fid: true`) |
