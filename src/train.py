"""
Training entry point.

Usage
-----
    python -m src.train --config configs/baseline.yaml

Flow
----
1. Parse --config, load YAML hyperparameters
2. Fix random seeds
3. Build DataLoaders, ConvVAE, Adam optimiser
4. Run training loop (with optional AMP and KL annealing)
5. After the final epoch, automatically call src.evaluate.main(cfg)
   so one GPU command covers train + all evaluation outputs.

All outputs (checkpoints, metrics.json) are written to
outputs/{exp_name}/.  Image outputs are produced by evaluate.py.

Dependency note
---------------
src.evaluate is imported *inside* main() (deferred import) so that
this file can be syntax-checked and imported before evaluate.py is
fully implemented.
"""

import time
from pathlib import Path

import torch
from torch.optim import Adam
from tqdm import tqdm

from src.config import load_config
from src.dataset import get_dataloaders
from src.losses import elbo_loss
from src.model import build_model
from src.utils import MetricsLogger, save_checkpoint, set_seed


# ------------------------------------------------------------------ #
# Per-epoch helpers                                                    #
# ------------------------------------------------------------------ #

def _train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: "torch.cuda.amp.GradScaler | None",
    device: torch.device,
    beta: float,
    use_amp: bool,
) -> tuple[float, float, float]:
    """Run one full training epoch.

    Returns:
        (avg_recon, avg_kl, avg_total) averaged over all batches.
    """
    model.train()
    sum_recon = sum_kl = sum_total = 0.0

    for x, _ in tqdm(loader, desc="  train", leave=False, unit="batch"):
        x = x.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # autocast speeds up compute on Tensor Cores when use_amp=True;
        # enabled=False makes it a no-op so we can share code paths.
        with torch.cuda.amp.autocast(enabled=use_amp):
            x_recon, mu, logvar = model(x)
            recon, kl, total = elbo_loss(x_recon, x, mu, logvar, beta)

        if scaler is not None:
            # AMP path: scale loss to avoid underflow, then unscale before step
            scaler.scale(total).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total.backward()
            optimizer.step()

        sum_recon += recon.item()
        sum_kl    += kl.item()
        sum_total += total.item()

    n = len(loader)
    return sum_recon / n, sum_kl / n, sum_total / n


@torch.no_grad()
def _validate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    beta: float,
    use_amp: bool,
) -> tuple[float, float, float]:
    """Evaluate on the validation set (no gradient computation).

    Returns:
        (avg_recon, avg_kl, avg_total) averaged over all batches.
    """
    model.eval()
    sum_recon = sum_kl = sum_total = 0.0

    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            x_recon, mu, logvar = model(x)
            recon, kl, total = elbo_loss(x_recon, x, mu, logvar, beta)
        sum_recon += recon.item()
        sum_kl    += kl.item()
        sum_total += total.item()

    n = len(loader)
    return sum_recon / n, sum_kl / n, sum_total / n


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

def main(cfg=None) -> None:
    """Run the full training pipeline for one experiment.

    Args:
        cfg: Pre-loaded config namespace.  When None (the default when
             invoked as __main__), --config is parsed from sys.argv.
    """
    if cfg is None:
        cfg = load_config()

    # -- Setup ---------------------------------------------------------
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Disable AMP silently when there is no GPU — GradScaler requires CUDA.
    use_amp: bool = cfg.use_amp and (device.type == "cuda")

    out_dir = Path("outputs") / cfg.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"[train] exp_name : {cfg.exp_name}")
    print(f"[train] device   : {device}")
    print(f"[train] out_dir  : {out_dir}")
    print(f"[train] use_amp  : {use_amp}")
    print(f"{'='*60}\n")

    # -- Data / model / optimiser --------------------------------------
    train_loader, val_loader = get_dataloaders(cfg)
    model = build_model(cfg).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] parameters: {n_params:,}")

    optimizer = Adam(model.parameters(), lr=cfg.lr)

    # GradScaler is the AMP companion: it dynamically scales the loss to
    # keep gradients in float16 range, then unscales before the optimiser step.
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    logger = MetricsLogger(out_dir / "metrics.json")

    # -- Training loop -------------------------------------------------
    best_val_loss = float("inf")

    for epoch in range(cfg.epochs):

        # KL annealing: ramp beta linearly from 0 → cfg.beta over the first
        # kl_anneal_epochs epochs.  Purpose: let the model learn good
        # reconstructions first before the KL term forces regularisation.
        # When kl_anneal_epochs=0 the feature is disabled and beta is constant.
        if cfg.kl_anneal_epochs > 0:
            beta_eff = cfg.beta * min(epoch / cfg.kl_anneal_epochs, 1.0)
        else:
            beta_eff = cfg.beta

        # --- train ---
        t0 = time.time()
        tr_recon, tr_kl, tr_total = _train_one_epoch(
            model, train_loader, optimizer, scaler, device, beta_eff, use_amp
        )

        # --- validate ---
        va_recon, va_kl, va_total = _validate(
            model, val_loader, device, beta_eff, use_amp
        )
        epoch_time = time.time() - t0

        # --- console summary ------------------------------------------
        print(
            f"Epoch {epoch + 1:03d}/{cfg.epochs}  "
            f"β={beta_eff:.3f}  "
            f"train[{tr_total:.1f} = {tr_recon:.1f} + {tr_kl:.1f}]  "
            f"val[{va_total:.1f} = {va_recon:.1f} + {va_kl:.1f}]  "
            f"{epoch_time:.1f}s"
        )

        # --- log to metrics.json --------------------------------------
        logger.log({
            "epoch":        epoch,
            "train_recon":  tr_recon,
            "train_kl":     tr_kl,
            "train_total":  tr_total,
            "val_recon":    va_recon,
            "val_kl":       va_kl,
            "val_total":    va_total,
            "beta":         beta_eff,
            "epoch_time_s": round(epoch_time, 2),
        })

        # --- save last.pt (always) ------------------------------------
        save_checkpoint(
            out_dir / "last.pt", model, optimizer, epoch, best_val_loss, scaler
        )

        # --- save best.pt (on improvement) ----------------------------
        if va_total < best_val_loss:
            best_val_loss = va_total
            save_checkpoint(
                out_dir / "best.pt", model, optimizer, epoch, best_val_loss, scaler
            )
            print(f"  ↑ best val loss: {best_val_loss:.4f}  → saved best.pt")

    print(f"\n[train] Training complete.  Best val loss: {best_val_loss:.4f}")

    # -- Auto-evaluate -------------------------------------------------
    # Deferred import: src.evaluate is implemented in the next step.
    # Importing inside main() (rather than at module top) prevents an
    # ImportError when evaluate.py is not yet fully written, while still
    # allowing py_compile to check this file cleanly.
    print("[train] Starting evaluation …\n")
    from src.evaluate import main as evaluate_main  # noqa: PLC0415
    evaluate_main(cfg)


if __name__ == "__main__":
    main()
