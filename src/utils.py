"""
Utility functions for the VAE project.

Contents
--------
set_seed            — fix all RNG seeds for reproducibility
save_checkpoint     — persist model + optimiser + training state
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
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    scaler: Any | None = None,
) -> None:
    """Save model weights and training state to a .pt file.

    Args:
        path:          Destination file path (e.g. outputs/baseline/last.pt).
        model:         The ConvVAE (or any nn.Module).
        optimizer:     Adam optimiser whose state we want to preserve so
                       training can be resumed without a loss spike.
        epoch:         Current epoch index (0-based).
        best_val_loss: Best validation loss seen so far; used to decide
                       whether to overwrite best.pt.
        scaler:        Optional GradScaler for AMP; ignored when None.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    if scaler is not None:
        payload["scaler_state"] = scaler.state_dict()
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any | None = None,
) -> dict[str, Any]:
    """Load a checkpoint and restore model (and optionally optimiser) state.

    Args:
        path:      Path to the .pt checkpoint file.
        model:     Model whose weights will be restored in-place.
        optimizer: If provided, its state is also restored (for resuming
                   training).  Pass None when loading only for evaluation.
        scaler:    Optional GradScaler; its state is restored when provided.

    Returns:
        A dict with keys ``epoch`` (int) and ``best_val_loss`` (float).
    """
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model_state"])
    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
    if scaler is not None and "scaler_state" in payload:
        scaler.load_state_dict(payload["scaler_state"])
    return {
        "epoch": payload.get("epoch", 0),
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
            "epoch": 3,
            "train_recon": 1234.5,
            "train_kl":    12.3,
            "train_total": 1246.8,
            "val_recon":   1230.1,
            "val_kl":      11.9,
            "val_total":   1242.0,
            "epoch_time_s": 45.2
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
