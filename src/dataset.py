"""
Dataset & augmentation pipeline for cucumber leaf/fruit disease classification.

Provides helpers for building PyTorch ``DataLoader`` instances and computing
inverse-frequency class weights for imbalanced datasets.

Expects an ImageFolder-style directory layout, e.g.:

    data/train/Anthracnose/*.jpg
    data/train/Bacterial_Wilt/*.jpg
    data/train/Belly_Rot/*.jpg
    data/train/Downy_Mildew/*.jpg
    data/train/Pythium_Fruit_Rot/*.jpg
    data/train/Gummy_Stem_Blight/*.jpg
    data/train/Fresh_Leaf/*.jpg
    data/train/Fresh_Cucumber/*.jpg
    data/val/<same classes>/*.jpg
    data/test/<same classes>/*.jpg

Use src/split_dataset.py to turn a single folder-of-classes download into
this train/val/test layout.
"""

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(image_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    """Returns (train_transform, eval_transform).

    Train transform applies the augmentations called for in the project
    plan: geometric (flip/rotate/crop/scale) + photometric (color jitter,
    slight noise via random erasing) while keeping lesions intact (no
    extreme crops).
    """
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.08)),
        ]
    )

    eval_tf = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_tf, eval_tf


def build_dataloaders(
    data_dir: str,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
):
    """Builds train/val/test DataLoaders from an ImageFolder layout.

    Returns: (train_loader, val_loader, test_loader, class_names)
    """
    data_dir = Path(data_dir)
    train_tf, eval_tf = build_transforms(image_size)

    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=eval_tf)
    test_ds = datasets.ImageFolder(data_dir / "test", transform=eval_tf)

    assert train_ds.classes == val_ds.classes == test_ds.classes, (
        "train/val/test class folders don't match — check your data/ layout. "
        f"train={train_ds.classes} val={val_ds.classes} test={test_ds.classes}"
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader, train_ds.classes


def class_weights(data_dir: str) -> torch.Tensor:
    """Inverse-frequency class weights from the train split, for use with
    CrossEntropyLoss(weight=...) when classes are imbalanced (very common
    with field-collected disease data — 'healthy' or one disease usually
    dominates)."""
    train_ds = datasets.ImageFolder(Path(data_dir) / "train")
    counts = torch.zeros(len(train_ds.classes))
    for _, label in train_ds.samples:
        counts[label] += 1
    weights = counts.sum() / (len(counts) * counts.clamp(min=1))
    return weights
