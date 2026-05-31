"""
Evaluation pipeline for the GAN inpainting project.

Loads outputs/{exp_name}/best.pt and writes to the same directory:

  inpainting.png    — num_vis rows of [masked_input | completed | ground_truth]
                      on validation-set images with a *fixed-seed* mask pattern
                      so all experiments are visualised on identical holes.
  loss_curve.png    — d_loss, g_adv, g_recon, val_l1 vs epoch (metrics.json).
  eval_metrics.json — val L1, PSNR, SSIM, G/D parameter counts, optional FID.

Note: prior-sample grids and latent-space interpolations are NOT produced
here — this is a deterministic conditional generator with no prior to sample
from and no latent space to interpolate in.

Usage (standalone)
------------------
    python -m src.evaluate --config configs/baseline.yaml

Called from train.py
--------------------
    from src.evaluate import main as evaluate_main
    evaluate_main(cfg)
"""

import json
import math
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend; safe on headless GPU servers
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from src.config import load_config
from src.dataset import generate_mask, get_dataloaders, make_masked_image
from src.model import Generator, build_discriminator, build_generator
from src.utils import load_checkpoint, save_image_grid


# ------------------------------------------------------------------ #
# 1. Inpainting visualisation                                          #
# ------------------------------------------------------------------ #

@torch.no_grad()
def _visualize_inpainting(
    G: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    cfg,
    device: torch.device,
    out_dir: Path,
) -> None:
    """Save a cfg.num_vis-row triplet grid to inpainting.png.

    Each row is:  [ masked input  |  G completion  |  ground truth ]

    Masks are generated with a *fixed seed* (cfg.seed) so that every
    experiment produces images with identical holes — making the visual
    comparison between experiments fair and direct.

    Args:
        G:          Generator in eval mode.
        val_loader: Validation DataLoader.
        cfg:        Config namespace.
        device:     Target device.
        out_dir:    Output directory.
    """
    print("[evaluate] generating inpainting visualisation …")
    G.eval()

    num_vis = cfg.num_vis

    # Collect num_vis images from the validation set.
    batches: list[torch.Tensor] = []
    for x_real, _ in val_loader:
        batches.append(x_real)
        if sum(t.size(0) for t in batches) >= num_vis:
            break
    x_real = torch.cat(batches, dim=0)[:num_vis]   # (num_vis, 3, H, W)

    # Fixed-seed mask generation for reproducibility across experiments.
    rng_state = random.getstate()
    random.seed(cfg.seed)
    masks = torch.stack(
        [generate_mask(cfg.image_size, cfg.mask_min_ratio, cfg.mask_max_ratio)
         for _ in range(num_vis)]
    )                                               # (num_vis, 1, H, W)
    random.setstate(rng_state)                     # restore caller's RNG state

    x_real   = x_real.to(device)
    masks    = masks.to(device)
    x_masked = make_masked_image(x_real, masks)

    G_out       = G(x_masked, masks)
    x_completed = Generator.complete(x_masked, masks, G_out)

    # Interleave triplets so make_grid with nrow=3 produces one row per image:
    #   col 0 = masked input,  col 1 = completion,  col 2 = ground truth
    triplets: list[torch.Tensor] = []
    for i in range(num_vis):
        triplets.extend([
            x_masked[i].cpu(),
            x_completed[i].cpu(),
            x_real[i].cpu(),
        ])

    # (num_vis * 3, 3, H, W) with nrow=3 → num_vis rows of 3 columns each
    save_image_grid(torch.stack(triplets), out_dir / "inpainting.png", nrow=3)
    print(f"  → {out_dir / 'inpainting.png'}")


# ------------------------------------------------------------------ #
# 2. Loss curves                                                       #
# ------------------------------------------------------------------ #

def _plot_loss_curves(out_dir: Path) -> None:
    """Read metrics.json and draw the four GAN training curves."""
    print("[evaluate] plotting loss curves …")

    log_path = out_dir / "metrics.json"
    if not log_path.exists():
        print(f"  metrics.json not found at {log_path}; skipping.")
        return

    try:
        with open(log_path) as f:
            records = json.load(f)
    except Exception as exc:
        print(f"  Could not read metrics.json ({exc!r}); skipping.")
        return

    required = {"epoch", "d_loss", "g_adv", "g_recon", "val_l1"}
    records = [r for r in records if required.issubset(r.keys())]
    if not records:
        print("  No epoch records in metrics.json; skipping.")
        return

    epochs = [r["epoch"] + 1 for r in records]

    # Three panels: D loss / G adversarial loss / G recon + val L1
    panels = [
        ("d_loss",  None,     "Discriminator loss"),
        ("g_adv",   None,     "Generator adversarial loss"),
        ("g_recon", "val_l1", "G recon L1 (train) & val L1"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (tr_key, va_key, title) in zip(axes, panels):
        ax.plot(epochs, [r[tr_key] for r in records], label=tr_key)
        if va_key is not None:
            ax.plot(epochs, [r[va_key] for r in records],
                    label=va_key, linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle(f"GAN training curves — {out_dir.name}", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curve.png", dpi=150)
    plt.close(fig)
    print(f"  → {out_dir / 'loss_curve.png'}")


# ------------------------------------------------------------------ #
# 3. Quantitative metrics                                              #
# ------------------------------------------------------------------ #

def _psnr_from_mse(mse: float) -> float:
    """PSNR in dB for [0, 1]-range images.  Returns inf when MSE == 0."""
    if mse <= 0.0:
        return float("inf")
    return -10.0 * math.log10(mse)


@torch.no_grad()
def _compute_metrics_pass(
    G: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    cfg,
) -> dict:
    """Full val-set pass; returns L1, PSNR, SSIM using torchmetrics when
    available, falling back to a simple PSNR on import failure.

    Masks are drawn randomly (same distribution as training) so the
    numbers represent an average over the full mask distribution.
    """
    G.eval()

    # Try to set up torchmetrics PSNR + SSIM.
    tm_psnr = tm_ssim = None
    try:
        from torchmetrics.image import (
            PeakSignalNoiseRatio,
            StructuralSimilarityIndexMeasure,
        )
        tm_psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
        tm_ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    except Exception:
        pass  # fall back to manual PSNR below

    sum_l1 = 0.0
    sum_mse = 0.0
    n_batches = 0

    for x_real, _ in val_loader:
        x_real = x_real.to(device)
        B = x_real.size(0)

        masks = torch.stack(
            [generate_mask(cfg.image_size, cfg.mask_min_ratio, cfg.mask_max_ratio)
             for _ in range(B)]
        ).to(device)
        x_masked    = make_masked_image(x_real, masks)
        G_out       = G(x_masked, masks)
        x_completed = Generator.complete(x_masked, masks, G_out)

        sum_l1  += F.l1_loss(x_completed, x_real).item()
        sum_mse += F.mse_loss(x_completed, x_real).item()
        n_batches += 1

        if tm_psnr is not None:
            tm_psnr.update(x_completed, x_real)
            tm_ssim.update(x_completed, x_real)

    avg_l1  = sum_l1  / n_batches
    avg_mse = sum_mse / n_batches

    if tm_psnr is not None:
        try:
            psnr = tm_psnr.compute().item()
            ssim = tm_ssim.compute().item()
            print(f"  PSNR: {psnr:.2f} dB   SSIM: {ssim:.4f}  (torchmetrics)")
        except Exception as exc:
            print(f"  torchmetrics compute failed ({exc!r}); using simple PSNR.")
            psnr = _psnr_from_mse(avg_mse)
            ssim = None
    else:
        psnr = _psnr_from_mse(avg_mse)
        ssim = None
        print(f"  PSNR: {psnr:.2f} dB  (simple, torchmetrics not available)")

    return {"val_l1": avg_l1, "psnr": psnr, "ssim": ssim}


@torch.no_grad()
def _try_fid(
    G: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    cfg,
) -> "float | None":
    """Compute FID between val originals and completions.

    Returns the score, or None on any failure (missing torchmetrics,
    OOM, etc.).  Wrapped in try/except so the pipeline never breaks.
    """
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance

        fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

        for x_real, _ in val_loader:
            x_real = x_real.to(device)
            B = x_real.size(0)
            masks = torch.stack(
                [generate_mask(cfg.image_size, cfg.mask_min_ratio, cfg.mask_max_ratio)
                 for _ in range(B)]
            ).to(device)
            x_masked    = make_masked_image(x_real, masks)
            G_out       = G(x_masked, masks)
            x_completed = Generator.complete(x_masked, masks, G_out)

            fid.update(x_real,      real=True)
            fid.update(x_completed, real=False)

        score = fid.compute().item()
        print(f"  FID: {score:.2f}")
        return score

    except Exception as exc:
        print(f"  [evaluate] FID skipped ({exc!r}).")
        return None


def _write_eval_metrics(
    G: torch.nn.Module,
    D: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    cfg,
    device: torch.device,
    out_dir: Path,
) -> None:
    """Compute all metrics and write eval_metrics.json."""
    print("[evaluate] computing final metrics …")

    n_params_g = sum(p.numel() for p in G.parameters())
    n_params_d = sum(p.numel() for p in D.parameters())

    metrics = _compute_metrics_pass(G, val_loader, device, cfg)

    summary: dict = {
        "exp_name":    cfg.exp_name,
        "base_channels": cfg.base_channels,
        "bottleneck_dim": cfg.bottleneck_dim,
        "n_params_g":  n_params_g,
        "n_params_d":  n_params_d,
        "lambda_rec":  cfg.lambda_rec,
        "lambda_adv":  cfg.lambda_adv,
        "val_l1":      round(metrics["val_l1"], 6),
        "psnr_db":     round(metrics["psnr"], 4) if metrics["psnr"] != float("inf") else None,
        "ssim":        round(metrics["ssim"], 6) if metrics["ssim"] is not None else None,
        "fid":         None,
    }

    if cfg.compute_fid:
        summary["fid"] = _try_fid(G, val_loader, device, cfg)

    out_path = out_dir / "eval_metrics.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  → {out_path}")
    print(
        f"  G params={n_params_g:,}  D params={n_params_d:,}  "
        f"val_l1={summary['val_l1']:.4f}  "
        f"PSNR={summary['psnr_db']} dB  SSIM={summary['ssim']}"
    )


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

def main(cfg=None) -> None:
    """Run all evaluation tasks for one experiment.

    Args:
        cfg: Pre-loaded config namespace.  When None (standalone usage),
             --config is parsed from sys.argv.
    """
    if cfg is None:
        cfg = load_config()

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path("outputs") / cfg.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"[evaluate] exp_name : {cfg.exp_name}")
    print(f"[evaluate] device   : {device}")
    print(f"[evaluate] out_dir  : {out_dir}")
    print(f"{'='*60}\n")

    ckpt_path = out_dir / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Train first:  python -m src.train --config <yaml>"
        )

    # Load generator from checkpoint; build discriminator for param count.
    G = build_generator(cfg).to(device)
    D = build_discriminator(cfg).to(device)

    # load_checkpoint restores G; D is optional (weights not needed for eval,
    # but we pass it so param counts come from the same architecture).
    load_checkpoint(ckpt_path, G, discriminator=D)
    G.eval()
    D.eval()

    n_params_g = sum(p.numel() for p in G.parameters())
    n_params_d = sum(p.numel() for p in D.parameters())
    print(f"[evaluate] G params: {n_params_g:,}  D params: {n_params_d:,}\n")

    _, val_loader = get_dataloaders(cfg)

    _visualize_inpainting(G, val_loader, cfg, device, out_dir)
    _plot_loss_curves(out_dir)
    _write_eval_metrics(G, D, val_loader, cfg, device, out_dir)

    print(f"\n[evaluate] Done.  All outputs in {out_dir}\n")


if __name__ == "__main__":
    main()
