"""
Utility functions for the GAN inpainting project.

Contents
--------
set_seed            — fix all RNG seeds for reproducibility
save_checkpoint     — persist G + D + both optimisers + training state
load_checkpoint     — restore from a checkpoint file
save_image_grid     — write a grid of images to disk
MetricsLogger       — append per-epoch metrics to metrics.json
"""

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image


# ------------------------------------------------------------------ #
# Reproducibility                                                      #
# ------------------------------------------------------------------ #

def set_seed(seed: int) -> None:
    """Fix all relevant RNG seeds so experiments are reproducible.

    Args:
        seed: Integer seed value (e.g. 42).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make cuDNN deterministic.  This can slow down training slightly but
    # ensures identical results across runs with the same seed.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------------------------------------------------------ #
# Checkpointing                                                        #
# ------------------------------------------------------------------ #

def save_checkpoint(
    path: str | Path,
    generator: nn.Module,
    discriminator: nn.Module,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    scaler_g: Any | None = None,
    scaler_d: Any | None = None,
) -> None:
    """Save G + D weights and both optimiser states to a .pt file.

    Storing both models and both optimisers lets training resume from
    exactly the same point without a loss spike caused by stale momentum
    estimates in Adam.

    Args:
        path:          Destination file path (e.g. outputs/baseline/last.pt).
        generator:     Generator nn.Module.
        discriminator: Discriminator nn.Module.
        opt_g:         Generator Adam optimiser.
        opt_d:         Discriminator Adam optimiser.
        epoch:         Current epoch index (0-based).
        best_val_loss: Best validation L1 seen so far.
        scaler_g:      Optional AMP GradScaler for the generator.
        scaler_d:      Optional AMP GradScaler for the discriminator.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "epoch":           epoch,
        "best_val_loss":   best_val_loss,
        "generator_state": generator.state_dict(),
        "discriminator_state": discriminator.state_dict(),
        "opt_g_state":     opt_g.state_dict(),
        "opt_d_state":     opt_d.state_dict(),
    }
    if scaler_g is not None:
        payload["scaler_g_state"] = scaler_g.state_dict()
    if scaler_d is not None:
        payload["scaler_d_state"] = scaler_d.state_dict()
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    generator: nn.Module,
    discriminator: nn.Module | None = None,
    opt_g: torch.optim.Optimizer | None = None,
    opt_d: torch.optim.Optimizer | None = None,
    scaler_g: Any | None = None,
    scaler_d: Any | None = None,
) -> dict[str, Any]:
    """Load a checkpoint and restore G (and optionally D + optimisers).

    Passing only ``generator`` is enough for evaluation (no D or opts
    needed).  Pass all arguments to resume training.

    Args:
        path:          Path to the .pt checkpoint file.
        generator:     Generator whose weights are restored in-place.
        discriminator: If provided, its weights are restored.
        opt_g:         If provided, generator optimiser state is restored.
        opt_d:         If provided, discriminator optimiser state is restored.
        scaler_g:      Optional AMP GradScaler for the generator.
        scaler_d:      Optional AMP GradScaler for the discriminator.

    Returns:
        Dict with keys ``epoch`` (int) and ``best_val_loss`` (float).
    """
    payload = torch.load(path, map_location="cpu")
    generator.load_state_dict(payload["generator_state"])
    if discriminator is not None and "discriminator_state" in payload:
        discriminator.load_state_dict(payload["discriminator_state"])
    if opt_g is not None and "opt_g_state" in payload:
        opt_g.load_state_dict(payload["opt_g_state"])
    if opt_d is not None and "opt_d_state" in payload:
        opt_d.load_state_dict(payload["opt_d_state"])
    if scaler_g is not None and "scaler_g_state" in payload:
        scaler_g.load_state_dict(payload["scaler_g_state"])
    if scaler_d is not None and "scaler_d_state" in payload:
        scaler_d.load_state_dict(payload["scaler_d_state"])
    return {
        "epoch":         payload.get("epoch", 0),
        "best_val_loss": payload.get("best_val_loss", float("inf")),
    }


# ------------------------------------------------------------------ #
# Image saving                                                         #
# ------------------------------------------------------------------ #

def save_image_grid(
    tensor: torch.Tensor,
    path: str | Path,
    nrow: int = 8,
    padding: int = 2,
) -> None:
    """Arrange images in a grid and save to disk.

    Args:
        tensor:  Image batch of shape (N, C, H, W), values in [0, 1].
        path:    Output file path.  Extension determines format (.png, .jpg …).
        nrow:    Number of images per row in the grid.
        padding: Pixels of padding between images.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # Clamp to [0,1] in case of minor floating-point drift, move to CPU.
    grid = make_grid(tensor.cpu().clamp(0.0, 1.0), nrow=nrow, padding=padding)
    to_pil_image(grid).save(path)


# ------------------------------------------------------------------ #
# Metrics logging                                                      #
# ------------------------------------------------------------------ #

class MetricsLogger:
    """Append per-epoch training/validation metrics to a JSON file.

    Each call to ``log()`` appends one record (a plain dict) to an
    in-memory list and rewrites the whole JSON file atomically.  If the
    file already exists when the logger is created (e.g. after a resume),
    the existing records are loaded so the history is preserved.

    Typical record shape::

        {
            "epoch":        3,
            "d_loss":       0.45,
            "g_adv":        0.82,
            "g_recon":      0.031,
            "val_l1":       0.028,
            "epoch_time_s": 62.4
        }

    Args:
        path: Path to the JSON file (e.g. outputs/baseline/metrics.json).
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Load existing records so resumed runs extend the history.
        if self.path.exists():
            with open(self.path) as f:
                self.records: list[dict] = json.load(f)
        else:
            self.records = []

    def log(self, record: dict[str, Any]) -> None:
        """Append a record and flush to disk immediately.

        Args:
            record: Arbitrary dict; typically contains epoch index,
                    train/val loss components, and wall-clock time.
        """
        self.records.append(record)
        with open(self.path, "w") as f:
            json.dump(self.records, f, indent=2)

    def all_records(self) -> list[dict]:
        """Return the full list of logged records (read-only copy)."""
        return list(self.records)
