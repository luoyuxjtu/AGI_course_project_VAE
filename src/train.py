"""
Training entry point for the GAN inpainting project.

Usage
-----
    python -m src.train --config configs/baseline.yaml

Flow
----
1.  Parse --config, load YAML hyperparameters.
2.  Fix random seeds, build DataLoaders, Generator, Discriminator,
    and two Adam optimisers (opt_g / opt_d).
3.  Run training loop — adversarial warmup then G/D alternating updates.
4.  After the final epoch, automatically call src.evaluate.main(cfg)
    so one GPU command covers train + all evaluation outputs.

Adversarial warmup
------------------
For the first ``adv_warmup_epochs`` epochs the generator trains with
reconstruction (L1) only.  The discriminator is not updated and g_adv is
set to 0.  This gives the generator a stable reconstruction baseline
before the adversarial game starts, which prevents early mode collapse.

AMP
---
When ``use_amp=true`` each optimiser gets its own GradScaler.  Using
separate scalers is necessary because D and G have different loss
magnitudes and may require different scaling factors.

All outputs go to ``outputs/{exp_name}/``.

Dependency note
---------------
src.evaluate is imported *inside* main() (deferred import) so this file
can be syntax-checked before evaluate.py is fully rewritten.
"""

import time
from pathlib import Path

import torch
from torch.optim import Adam
from tqdm import tqdm

from src.config import load_config
from src.dataset import generate_mask, get_dataloaders, make_masked_image
from src.losses import (
    discriminator_loss,
    generator_adv_loss,
    generator_total_loss,
    reconstruction_loss,
)
from src.model import Generator, build_discriminator, build_generator
from src.utils import (
    MetricsLogger,
    load_checkpoint,
    save_checkpoint,
    set_seed,
)


# ------------------------------------------------------------------ #
# Per-epoch helpers                                                    #
# ------------------------------------------------------------------ #

def _train_one_epoch(
    G: torch.nn.Module,
    D: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    scaler_g: "torch.cuda.amp.GradScaler | None",
    scaler_d: "torch.cuda.amp.GradScaler | None",
    device: torch.device,
    cfg,
    adv_active: bool,
    use_amp: bool,
) -> tuple[float, float, float]:
    """Run one full training epoch; return (avg_d_loss, avg_g_adv, avg_g_recon)."""
    G.train()
    D.train()

    sum_d = sum_g_adv = sum_g_recon = 0.0

    for x_real, _ in tqdm(loader, desc="  train", leave=False, unit="batch"):
        x_real = x_real.to(device, non_blocking=True)
        B = x_real.size(0)

        # --- Masking -------------------------------------------------
        # Generate one independent random rectangular mask per image so
        # every image sees a different hole position in every epoch.
        # mask==1 marks the hole; mask==0 marks known (preserved) pixels.
        masks = torch.stack(
            [generate_mask(cfg.image_size, cfg.mask_min_ratio, cfg.mask_max_ratio)
             for _ in range(B)]
        ).to(device)                                       # (B, 1, H, W)
        x_masked = make_masked_image(x_real, masks)        # (B, 3, H, W)

        # --- Generator forward ---------------------------------------
        # Compute x_completed once; reuse in both D and G update steps.
        # D uses x_completed.detach() so G's graph is not traversed by D.
        with torch.cuda.amp.autocast(enabled=use_amp):
            G_out = G(x_masked, masks)                     # (B, 3, H, W)
            x_completed = Generator.complete(x_masked, masks, G_out)

        # --- Discriminator update ------------------------------------
        # Skip entirely during the warmup period and when lambda_adv==0.
        d_loss_val = 0.0
        if adv_active:
            opt_d.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                d_real_logits = D(x_real)
                # .detach() breaks the computation graph: gradients from
                # d_loss will only update D, never G.
                d_fake_logits = D(x_completed.detach())
                d_loss = discriminator_loss(d_real_logits, d_fake_logits)

            if scaler_d is not None:
                scaler_d.scale(d_loss).backward()
                scaler_d.step(opt_d)
                scaler_d.update()
            else:
                d_loss.backward()
                opt_d.step()

            d_loss_val = d_loss.item()

        # --- Generator update ----------------------------------------
        # opt_g.zero_grad clears G's parameter gradients from any previous
        # step without destroying the x_completed computation graph.
        opt_g.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            g_recon = reconstruction_loss(x_completed, x_real)

            if adv_active:
                # D(x_completed) without .detach(): gradients flow through
                # D and x_completed all the way back to G's parameters.
                # D's own parameters will also accumulate gradients here,
                # but opt_d is never stepped again in this iteration, so
                # those gradients are harmlessly discarded at the next
                # opt_d.zero_grad() call.
                g_adv = generator_adv_loss(D(x_completed))
            else:
                # Warmup phase or recon_only experiment: no adversarial term.
                g_adv = torch.zeros(1, device=device)

            g_total = generator_total_loss(
                g_recon, g_adv, cfg.lambda_rec, cfg.lambda_adv
            )

        if scaler_g is not None:
            scaler_g.scale(g_total).backward()
            scaler_g.step(opt_g)
            scaler_g.update()
        else:
            g_total.backward()
            opt_g.step()

        sum_d       += d_loss_val
        sum_g_adv   += g_adv.item()
        sum_g_recon += g_recon.item()

    n = len(loader)
    return sum_d / n, sum_g_adv / n, sum_g_recon / n


@torch.no_grad()
def _validate(
    G: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    cfg,
    use_amp: bool,
) -> float:
    """Compute average L1 on the validation set (generator only, no D needed).

    Returns:
        Average val L1 over all batches.
    """
    G.eval()
    sum_l1 = 0.0

    for x_real, _ in loader:
        x_real = x_real.to(device, non_blocking=True)
        B = x_real.size(0)

        masks = torch.stack(
            [generate_mask(cfg.image_size, cfg.mask_min_ratio, cfg.mask_max_ratio)
             for _ in range(B)]
        ).to(device)
        x_masked = make_masked_image(x_real, masks)

        with torch.cuda.amp.autocast(enabled=use_amp):
            G_out = G(x_masked, masks)
            x_completed = Generator.complete(x_masked, masks, G_out)
            val_l1 = reconstruction_loss(x_completed, x_real)

        sum_l1 += val_l1.item()

    return sum_l1 / len(loader)


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

def main(cfg=None) -> None:
    """Run the full GAN inpainting training pipeline for one experiment.

    Args:
        cfg: Pre-loaded config namespace.  When None (invoked as __main__),
             --config is parsed from sys.argv.
    """
    if cfg is None:
        cfg = load_config()

    # -- Setup ---------------------------------------------------------
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # GradScaler requires CUDA; disable AMP silently on CPU.
    use_amp: bool = cfg.use_amp and (device.type == "cuda")

    out_dir = Path("outputs") / cfg.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"[train] exp_name   : {cfg.exp_name}")
    print(f"[train] device     : {device}")
    print(f"[train] out_dir    : {out_dir}")
    print(f"[train] use_amp    : {use_amp}")
    print(f"[train] lambda_adv : {cfg.lambda_adv}  "
          f"(warmup {cfg.adv_warmup_epochs} epochs)")
    print(f"{'='*60}\n")

    # -- Data / models / optimisers ------------------------------------
    train_loader, val_loader = get_dataloaders(cfg)

    G = build_generator(cfg).to(device)
    D = build_discriminator(cfg).to(device)

    n_params_g = sum(p.numel() for p in G.parameters())
    n_params_d = sum(p.numel() for p in D.parameters())
    print(f"[train] G params: {n_params_g:,}  D params: {n_params_d:,}")

    # beta1=0.5 is the GAN-training standard (recommended in the DCGAN paper):
    # lower momentum means the optimiser forgets old gradient directions faster,
    # which helps during the non-stationary adversarial game.
    opt_g = Adam(G.parameters(), lr=cfg.lr_g, betas=(cfg.beta1, 0.999))
    opt_d = Adam(D.parameters(), lr=cfg.lr_d, betas=(cfg.beta1, 0.999))

    # Separate scalers so each optimiser's loss scale is independent.
    scaler_g = torch.cuda.amp.GradScaler() if use_amp else None
    scaler_d = torch.cuda.amp.GradScaler() if use_amp else None

    logger = MetricsLogger(out_dir / "metrics.json")

    # -- Checkpoint resume ---------------------------------------------
    best_val_loss = float("inf")
    start_epoch = 0

    ckpt_path = out_dir / "last.pt"
    if ckpt_path.exists():
        try:
            ckpt = load_checkpoint(
                ckpt_path, G, D, opt_g, opt_d, scaler_g, scaler_d
            )
            start_epoch   = ckpt["epoch"] + 1
            best_val_loss = ckpt["best_val_loss"]
            print(
                f"[train] Resumed from {ckpt_path}  "
                f"(epoch {ckpt['epoch'] + 1}/{cfg.epochs}, "
                f"best_val_l1={best_val_loss:.6f})"
            )
        except (KeyError, ValueError, RuntimeError) as exc:
            # Checkpoint exists but cannot be loaded — most likely it was
            # saved by the old VAE code and has an incompatible format.
            # Warn and start fresh rather than crashing.
            print(
                f"[train] WARNING: ignoring incompatible checkpoint "
                f"({type(exc).__name__}: {exc})\n"
                f"[train] Starting from epoch 1."
            )

    if start_epoch >= cfg.epochs:
        print(f"[train] Training already complete ({cfg.epochs} epochs done).")
    else:
        print(f"[train] Starting from epoch {start_epoch + 1}/{cfg.epochs}")

    # -- Training loop -------------------------------------------------
    for epoch in range(start_epoch, cfg.epochs):

        # Activate adversarial loss only after the warmup period and only
        # if the config actually wants adversarial training (lambda_adv > 0).
        # recon_only experiment: lambda_adv=0 keeps adv_active=False always.
        adv_active: bool = (cfg.lambda_adv > 0) and (epoch >= cfg.adv_warmup_epochs)

        # --- train ---
        t0 = time.time()
        avg_d, avg_g_adv, avg_g_recon = _train_one_epoch(
            G, D, train_loader, opt_g, opt_d, scaler_g, scaler_d,
            device, cfg, adv_active, use_amp,
        )

        # --- validate (G only) ---
        val_l1 = _validate(G, val_loader, device, cfg, use_amp)
        epoch_time = time.time() - t0

        # --- console summary ------------------------------------------
        mode_str = "adv+recon" if adv_active else "recon-only"
        print(
            f"Epoch {epoch + 1:03d}/{cfg.epochs}  [{mode_str}]  "
            f"d={avg_d:.4f}  g_adv={avg_g_adv:.4f}  "
            f"g_recon={avg_g_recon:.4f}  val_l1={val_l1:.4f}  "
            f"{epoch_time:.1f}s"
        )

        # --- log to metrics.json --------------------------------------
        logger.log({
            "epoch":        epoch,
            "d_loss":       round(avg_d,       6),
            "g_adv":        round(avg_g_adv,   6),
            "g_recon":      round(avg_g_recon, 6),
            "val_l1":       round(val_l1,      6),
            "adv_active":   adv_active,
            "epoch_time_s": round(epoch_time,  2),
        })

        # --- save last.pt (always) ------------------------------------
        save_checkpoint(
            out_dir / "last.pt",
            G, D, opt_g, opt_d, epoch, best_val_loss, scaler_g, scaler_d,
        )

        # --- save best.pt (on val L1 improvement) --------------------
        if val_l1 < best_val_loss:
            best_val_loss = val_l1
            save_checkpoint(
                out_dir / "best.pt",
                G, D, opt_g, opt_d, epoch, best_val_loss, scaler_g, scaler_d,
            )
            print(f"  ↑ best val L1: {best_val_loss:.6f}  → saved best.pt")

    print(f"\n[train] Training complete.  Best val L1: {best_val_loss:.6f}")

    # -- Auto-evaluate -------------------------------------------------
    # Deferred import: evaluate.py is rewritten in the next step.
    # Importing inside main() prevents an ImportError if evaluate.py is
    # temporarily in an intermediate state, while still allowing
    # py_compile to check this file cleanly.
    print("[train] Starting evaluation …\n")
    from src.evaluate import main as evaluate_main  # noqa: PLC0415
    evaluate_main(cfg)


if __name__ == "__main__":
    main()
