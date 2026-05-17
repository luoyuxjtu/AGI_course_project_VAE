"""
Data loading for the VAE project — COCO 2017 edition.

COCO 2017 stores images in flat directories (train2017/*.jpg,
val2017/*.jpg) with no class subdirectories.  Because VAE training is
fully unsupervised we don't need labels or annotations at all —
we just load every JPEG in the directory.

The custom CocoImageDataset class handles this flat layout.  It returns
(image_tensor, 0) from __getitem__ so that the existing DataLoader loops
in train.py and evaluate.py (which unpack `for x, _ in loader`) work
without modification; the 0 is a dummy label that is never used.

Image pipeline
--------------
Train:  Resize(image_size) → CenterCrop(image_size) → RandomHorizontalFlip → ToTensor
Val:    Resize(image_size) → CenterCrop(image_size) → ToTensor

Output range is [0, 1] with no further normalisation — the decoder uses
Sigmoid, so both the reconstruction target and the prediction live in [0, 1].
"""

import glob
import os
from types import SimpleNamespace
from typing import Tuple

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
