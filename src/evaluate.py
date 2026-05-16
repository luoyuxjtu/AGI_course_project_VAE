"""
Evaluation pipeline for the VAE project.

Loads outputs/{exp_name}/best.pt and writes to the same directory:

  samples.png         — images sampled from the prior N(0, I)
  reconstructions.png — val-set originals (top) vs reconstructions (bottom)
  interpolations.png  — latent-space linear interpolation sequences
  loss_curve.png      — train / val loss curves read from metrics.json
  eval_metrics.json   — final losses, parameter count, optional FID

The per-epoch training log (metrics.json) is written by train.py and is
NOT modified here; eval_metrics.json is a separate flat summary used by
compare.py.

Usage (standalone)
------------------
    python -m src.evaluate --config configs/baseline.yaml

Called from train.py
--------------------
    from src.evaluate import main as evaluate_main
    evaluate_main(cfg)
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend; safe on headless GPU servers
import matplotlib.pyplot as plt
import torch

from src.config import load_config
from src.dataset import get_dataloaders
from src.losses import elbo_loss
from src.model import build_model
from src.utils import load_checkpoint, save_image_grid


# ------------------------------------------------------------------ #
# 1. Prior samples                                                     #
# ------------------------------------------------------------------ #

@torch.no_grad()
def _generate_samples(
    model: torch.nn.Module,
    cfg,
    device: torch.device,
    out_dir: Path,
) -> None:
    """Sample z ~ N(0,I), decode, and save as an image grid."""
    print("[evaluate] generating samples …")
    samples = model.sample(cfg.num_samples, device=device)  # (N, C, H, W)
    save_image_grid(samples, out_dir / "samples.png", nrow=8)
    print(f"  → {out_dir / 'samples.png'}")


# ------------------------------------------------------------------ #
# 2. Reconstructions                                                   #
# ------------------------------------------------------------------ #

@torch.no_grad()
def _generate_reconstructions(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    cfg,
    device: torch.device,
    out_dir: Path,
) -> None:
    """Show one val batch: originals (top row) vs reconstructions (bottom)."""
    print("[evaluate] generating reconstructions …")
    n_show = min(cfg.batch_size, 8)

    x, _ = next(iter(val_loader))
    x = x[:n_show].to(device)

    # Use the posterior mean for reconstruction — avoids reparameterisation
    # noise so the visual comparison is deterministic and easy to read.
    mu, _ = model.encode(x)
    x_recon = model.decode(mu)

    # cat along batch dim: first n_show = originals, next n_show = recons
    grid = torch.cat([x.cpu(), x_recon.cpu()], dim=0)
    save_image_grid(grid, out_dir / "reconstructions.png", nrow=n_show)
    print(f"  → {out_dir / 'reconstructions.png'}")


# ------------------------------------------------------------------ #
# 3. Latent-space interpolations                                       #
# ------------------------------------------------------------------ #

@torch.no_grad()
def _generate_interpolations(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    cfg,
    device: torch.device,
    out_dir: Path,
) -> None:
    """Linear interpolation between latent means of image pairs.

    Each row in the output grid is one interpolation sequence (left
    endpoint → right endpoint in n_steps steps).
    """
    print("[evaluate] generating interpolations …")
    n_steps = 8          # interpolation steps per row, including both endpoints
    n_pairs = cfg.num_interpolations

    # Collect enough images from the val set (2 images per pair)
    needed = n_pairs * 2
    collected: list[torch.Tensor] = []
    for x, _ in val_loader:
        collected.append(x)
        if sum(t.size(0) for t in collected) >= needed:
            break
    images = torch.cat(collected, dim=0)[:needed]  # (needed, C, H, W)

    rows: list[torch.Tensor] = []
    for i in range(n_pairs):
        x_a = images[2 * i].unsqueeze(0).to(device)      # (1, C, H, W)
        x_b = images[2 * i + 1].unsqueeze(0).to(device)

        # Interpolate between posterior means; using means (not samples)
        # produces smoother sequences because there is no stochastic jitter.
        mu_a, _ = model.encode(x_a)
        mu_b, _ = model.encode(x_b)

        step_imgs: list[torch.Tensor] = []
        for step in range(n_steps):
            alpha = step / (n_steps - 1)                 # 0.0 … 1.0
            z = (1.0 - alpha) * mu_a + alpha * mu_b
            step_imgs.append(model.decode(z).cpu())      # (1, C, H, W)

        rows.append(torch.cat(step_imgs, dim=0))         # (n_steps, C, H, W)

    # nrow=n_steps keeps each interpolation sequence on its own row
    all_imgs = torch.cat(rows, dim=0)                    # (n_pairs*n_steps, C, H, W)
    save_image_grid(all_imgs, out_dir / "interpolations.png", nrow=n_steps)
    print(f"  → {out_dir / 'interpolations.png'}")


# ------------------------------------------------------------------ #
# 4. Loss curves                                                       #
# ------------------------------------------------------------------ #

def _plot_loss_curves(out_dir: Path) -> None:
    """Read metrics.json and draw train / val loss curves."""
    print("[evaluate] plotting loss curves …")

    log_path = out_dir / "metrics.json"
    if not log_path.exists():
        print(f"  metrics.json not found at {log_path}; skipping loss curve.")
        return

    try:
        with open(log_path) as f:
            records = json.load(f)
    except Exception as exc:
        print(f"  Could not read metrics.json ({exc!r}); skipping loss curve.")
        return

    # Keep only per-epoch records (dicts that carry all expected keys)
    required = {"epoch", "train_total", "val_total", "train_recon",
                "val_recon", "train_kl", "val_kl"}
    records = [r for r in records if required.issubset(r.keys())]
    if not records:
        print("  No epoch records found in metrics.json; skipping loss curve.")
        return

    epochs = [r["epoch"] + 1 for r in records]   # display as 1-indexed

    panels = [
        ("train_total",  "val_total",  "Total loss"),
        ("train_recon",  "val_recon",  "Reconstruction loss"),
        ("train_kl",     "val_kl",     "KL divergence"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (tr_key, va_key, title) in zip(axes, panels):
        ax.plot(epochs, [r[tr_key] for r in records], label="train")
        ax.plot(epochs, [r[va_key] for r in records],
                label="val", linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle(f"Loss curves — {out_dir.name}", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curve.png", dpi=150)
    plt.close(fig)
    print(f"  → {out_dir / 'loss_curve.png'}")


# ------------------------------------------------------------------ #
# 5. Final metrics + FID                                              #
# ------------------------------------------------------------------ #

@torch.no_grad()
def _compute_final_val_losses(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    beta: float,
) -> tuple[float, float, float]:
    """Full validation pass; returns (avg_recon, avg_kl, avg_total)."""
    model.eval()
    sum_r = sum_k = sum_t = 0.0
    n = 0
    for x, _ in val_loader:
        x = x.to(device)
        x_recon, mu, logvar = model(x)
        recon, kl, total = elbo_loss(x_recon, x, mu, logvar, beta)
        sum_r += recon.item()
        sum_k += kl.item()
        sum_t += total.item()
        n += 1
    return sum_r / n, sum_k / n, sum_t / n


@torch.no_grad()
def _try_compute_fid(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    cfg,
    device: torch.device,
) -> "float | None":
    """Compute FID between val images and generated samples.

    Returns the FID score, or None on any failure.  All exceptions are
    caught here so that a missing torchmetrics install or an OOM on a
    small GPU never breaks the evaluation pipeline.
    """
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance

        # normalize=True: accepts float tensors in [0, 1] directly
        fid_metric = FrechetInceptionDistance(
            feature=2048, normalize=True
        ).to(device)

        # Feed the full val set as real images
        for x, _ in val_loader:
            fid_metric.update(x.to(device), real=True)

        # Generate fake images to match the size of the val set
        n_real = len(val_loader.dataset)
        generated = 0
        while generated < n_real:
            n_batch = min(cfg.batch_size, n_real - generated)
            fake = model.sample(n_batch, device=device)
            fid_metric.update(fake, real=False)
            generated += n_batch

        score = fid_metric.compute().item()
        print(f"  FID: {score:.2f}")
        return score

    except Exception as exc:
        print(f"  [evaluate] FID computation skipped ({exc!r}).")
        return None


def _write_eval_metrics(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    cfg,
    device: torch.device,
    out_dir: Path,
    n_params: int,
) -> None:
    """Compute final metrics and write eval_metrics.json."""
    print("[evaluate] computing final metrics …")

    final_recon, final_kl, final_total = _compute_final_val_losses(
        model, val_loader, device, cfg.beta
    )

    summary: dict = {
        "exp_name":      cfg.exp_name,
        "n_params":      n_params,
        "beta":          cfg.beta,
        "base_channels": cfg.base_channels,
        "latent_dim":    cfg.latent_dim,
        "image_size":    cfg.image_size,
        "final_val_recon": round(final_recon, 4),
        "final_val_kl":    round(final_kl,    4),
        "final_val_total": round(final_total,  4),
        "fid": None,
    }

    if cfg.compute_fid:
        summary["fid"] = _try_compute_fid(model, val_loader, cfg, device)

    out_path = out_dir / "eval_metrics.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  → {out_path}")
    print(
        f"  n_params={n_params:,}  "
        f"val_total={final_total:.2f}  "
        f"val_recon={final_recon:.2f}  "
        f"val_kl={final_kl:.2f}"
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            "Train first: python -m src.train --config <yaml>"
        )

    model = build_model(cfg).to(device)
    load_checkpoint(ckpt_path, model)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[evaluate] loaded best.pt  ({n_params:,} parameters)\n")

    _, val_loader = get_dataloaders(cfg)

    _generate_samples(model, cfg, device, out_dir)
    _generate_reconstructions(model, val_loader, cfg, device, out_dir)
    _generate_interpolations(model, val_loader, cfg, device, out_dir)
    _plot_loss_curves(out_dir)
    _write_eval_metrics(model, val_loader, cfg, device, out_dir, n_params)

    print(f"\n[evaluate] Done.  All outputs in {out_dir}\n")


if __name__ == "__main__":
    main()
