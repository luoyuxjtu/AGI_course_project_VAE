"""
Data loading for the VAE project.

Imagenette is a 10-class subset of ImageNet that ships with pre-split
train/ and val/ directories, which gives us a clean sample-level
train/unseen split without any extra work.

Images are resized to cfg.image_size × cfg.image_size and converted to
[0, 1] float tensors.  We deliberately skip ImageNet-style mean/std
normalisation because the decoder uses Sigmoid and its output is already
in [0, 1] — the reconstruction target and the prediction must live in the
same range.
"""

import os
from types import SimpleNamespace
from typing import Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_dataloaders(cfg: SimpleNamespace) -> Tuple[DataLoader, DataLoader]:
    """Build train and val DataLoaders for Imagenette.

    Directory layout expected on disk:
        {cfg.data_dir}/train/<class_name>/<image>.JPEG
        {cfg.data_dir}/val/<class_name>/<image>.JPEG

    Args:
        cfg: Config namespace with fields: data_dir, image_size,
             batch_size, num_workers.

    Returns:
        (train_loader, val_loader) tuple.
    """
    # --- transforms ---------------------------------------------------
    # Train: add a horizontal flip for cheap data augmentation.
    train_tf = transforms.Compose([
        transforms.Resize(cfg.image_size),        # shorter edge → image_size
        transforms.CenterCrop(cfg.image_size),    # square crop
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),                    # HWC uint8 → CHW float [0,1]
    ])

    # Val: deterministic — no flip, same spatial ops as train.
    val_tf = transforms.Compose([
        transforms.Resize(cfg.image_size),
        transforms.CenterCrop(cfg.image_size),
        transforms.ToTensor(),
    ])

    # --- datasets -----------------------------------------------------
    train_dir = os.path.join(cfg.data_dir, "train")
    val_dir = os.path.join(cfg.data_dir, "val")

    train_dataset = datasets.ImageFolder(train_dir, transform=train_tf)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_tf)

    # --- loaders ------------------------------------------------------
    # drop_last=True on train keeps batch sizes uniform, which avoids
    # edge-case behaviour in BatchNorm with tiny final batches.
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,   # speeds up CPU→GPU transfer
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
