"""
Data loading and masking for the GAN inpainting project — COCO 2017 edition.

COCO 2017 stores images in flat directories (train2017/*.jpg,
val2017/*.jpg) with no class subdirectories.  Because GAN inpainting
training is unsupervised we don't need labels or annotations at all —
we just load every JPEG in the directory.

The custom CocoImageDataset class handles this flat layout.  It returns
(image_tensor, 0) from __getitem__; the 0 is a dummy label kept for
DataLoader API compatibility and never used by the training loop.

Image pipeline
--------------
Train:  Resize(image_size) → CenterCrop(image_size) → RandomHorizontalFlip → ToTensor
Val:    Resize(image_size) → CenterCrop(image_size) → ToTensor

Output range is [0, 1] with no further normalisation — the generator
output is Sigmoid, so both target and prediction live in [0, 1].

Masking
-------
Masks are NOT baked into the dataset; they are generated fresh each batch
inside the training / evaluation loop so that every image sees a different
hole in every epoch (better generalisation).

Mask convention (used throughout this project):
  mask == 1  →  hole / missing region  (generator must fill this)
  mask == 0  →  known pixel           (copy through as-is)

Helper functions
----------------
generate_mask(image_size, min_ratio, max_ratio)
    Returns a (1, H, W) binary float tensor with a random rectangular hole.

make_masked_image(x, mask)
    Returns x_masked = x * (1 - mask), zeroing out the hole pixels.
"""

import glob
import os
import random
from types import SimpleNamespace
from typing import Tuple

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class CocoImageDataset(Dataset):
    """Flat-directory image dataset for COCO 2017 (no class labels needed).

    COCO stores images as a flat list of JPEGs inside a single directory,
    unlike ImageFolder which expects class subdirectories.  For unsupervised
    VAE training we simply load every JPEG we find.

    Args:
        root_dir:  Path to a directory containing *.jpg images,
                   e.g. ``data/coco2017/train2017``.
        transform: torchvision transform applied to each PIL image.
    """

    def __init__(self, root_dir: str, transform=None) -> None:
        self.transform = transform

        # Collect every .jpg / .jpeg in root_dir (COCO uses .jpg)
        pattern_jpg  = os.path.join(root_dir, "*.jpg")
        pattern_jpeg = os.path.join(root_dir, "*.jpeg")
        self.image_paths: list[str] = sorted(
            glob.glob(pattern_jpg) + glob.glob(pattern_jpeg)
        )

        if not self.image_paths:
            raise FileNotFoundError(
                f"No JPEG images found in '{root_dir}'.\n"
                "Run  bash scripts/download_data.sh  to download COCO 2017."
            )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        """Load one image and return (tensor, dummy_label).

        The dummy label (0) exists only for API compatibility with DataLoader
        loops that unpack `for x, _ in loader`.  It is never used by the VAE.
        """
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, 0


def get_dataloaders(cfg: SimpleNamespace) -> Tuple[DataLoader, DataLoader]:
    """Build COCO 2017 train and val DataLoaders.

    Expected directory layout on disk:
        {cfg.data_dir}/train2017/<image>.jpg   (~118k images)
        {cfg.data_dir}/val2017/<image>.jpg     (~5k images)

    Args:
        cfg: Config namespace with fields: data_dir, image_size,
             batch_size, num_workers.

    Returns:
        (train_loader, val_loader) tuple.
    """
    # --- transforms ---------------------------------------------------
    # Train: horizontal flip for cheap augmentation.
    train_tf = transforms.Compose([
        transforms.Resize(cfg.image_size),        # shorter edge → image_size
        transforms.CenterCrop(cfg.image_size),    # square crop
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),                    # HWC uint8 → CHW float [0,1]
    ])

    # Val: deterministic — same spatial ops, no flip.
    val_tf = transforms.Compose([
        transforms.Resize(cfg.image_size),
        transforms.CenterCrop(cfg.image_size),
        transforms.ToTensor(),
    ])

    # --- datasets -----------------------------------------------------
    train_dir = os.path.join(cfg.data_dir, "train2017")
    val_dir   = os.path.join(cfg.data_dir, "val2017")

    train_dataset = CocoImageDataset(train_dir, transform=train_tf)
    val_dataset   = CocoImageDataset(val_dir,   transform=val_tf)

    # --- loaders ------------------------------------------------------
    # drop_last=True on train keeps batch sizes uniform, which avoids
    # edge-case behaviour in BatchNorm with tiny final batches.
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


# ------------------------------------------------------------------ #
# Masking utilities                                                    #
# ------------------------------------------------------------------ #

def generate_mask(
    image_size: int,
    min_ratio: float,
    max_ratio: float,
) -> Tensor:
    """Generate a single random rectangular binary mask.

    The hole side lengths are drawn uniformly from
    [min_ratio * image_size, max_ratio * image_size], and placed at a
    uniformly random valid position.

    Convention: mask == 1 marks the **missing / hole** region that the
    generator must fill in.  mask == 0 marks known pixels that are
    copied through unchanged.

    Args:
        image_size: Square image side length (pixels).
        min_ratio:  Minimum hole side length as a fraction of image_size.
        max_ratio:  Maximum hole side length as a fraction of image_size.

    Returns:
        mask: Float tensor of shape (1, image_size, image_size),
              values in {0.0, 1.0}.
    """
    mask = torch.zeros(1, image_size, image_size, dtype=torch.float32)

    min_len = max(1, int(min_ratio * image_size))
    max_len = max(min_len, int(max_ratio * image_size))

    h = random.randint(min_len, max_len)
    w = random.randint(min_len, max_len)

    # Clamp so the rectangle always fits inside the image.
    top  = random.randint(0, image_size - h)
    left = random.randint(0, image_size - w)

    # Mark the hole region as 1 (= missing, to be filled by the generator).
    mask[0, top : top + h, left : left + w] = 1.0
    return mask


def make_masked_image(x: Tensor, mask: Tensor) -> Tensor:
    """Zero out the hole region of an image using the mask.

    Computes  x_masked = x * (1 - mask).

    Known pixels (mask == 0) are kept unchanged.
    Hole pixels  (mask == 1) are set to 0.

    Works on a single image (C, H, W) or a batch (B, C, H, W) provided
    that `mask` is broadcastable (e.g. (1, H, W) or (B, 1, H, W)).

    Args:
        x:    Image tensor, values in [0, 1].
        mask: Binary mask tensor, 1 = hole, 0 = known.

    Returns:
        x_masked: Same shape as `x`, hole pixels zeroed.
    """
    return x * (1.0 - mask)
