"""
Dataset loading utilities for the deepfake detection project.

Uses torchvision ImageFolder to load real/fake images from train/ and valid/
subdirectories and applies the required augmentation and normalization transforms.
"""

from pathlib import Path

import random

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src import PROJECT_ROOT

# ImageNet normalization constants used by pretrained EfficientNet-B0
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Extensions supported by torchvision ImageFolder
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"}


def _label_to_binary(class_index: int) -> float:
    """
    Map ImageFolder class indices to binary targets for BCEWithLogitsLoss.

    ImageFolder assigns labels alphabetically: fake=0, real=1.
    We want fake=1.0 and real=0.0 for binary cross-entropy training.
    """
    return float(1 - class_index)


def get_transforms(train: bool = True) -> transforms.Compose:
    """
    Build image transforms for training or validation.

    Training applies random horizontal flip and color jitter for augmentation.
    Validation only resizes, converts to tensor, and normalizes.
    """
    if train:
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _subset_by_class(dataset: datasets.ImageFolder, max_per_class: int) -> Subset:
    """
    Limit an ImageFolder dataset to max_per_class samples per class.

    Uses a fixed seed so fast-training runs are reproducible.
    """
    by_class: dict[int, list[int]] = {}
    for index, (_, class_index) in enumerate(dataset.samples):
        by_class.setdefault(class_index, []).append(index)

    rng = random.Random(42)
    selected: list[int] = []
    for indices in by_class.values():
        rng.shuffle(indices)
        selected.extend(indices[:max_per_class])

    return Subset(dataset, selected)


def get_dataloaders(
    data_dir: Path | str | None = None,
    batch_size: int = 32,
    num_workers: int = 0,
    max_samples_per_class: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders from the ImageFolder layout.

    Expected structure:
        data/train/{real,fake}/
        data/valid/{real,fake}/

    Returns:
        Tuple of (train_loader, val_loader).
    """
    if data_dir is None:
        data_dir = PROJECT_ROOT / "data"
    data_dir = Path(data_dir)

    train_dir = data_dir / "train"
    valid_dir = data_dir / "valid"

    if not train_dir.exists() or not valid_dir.exists():
        raise FileNotFoundError(
            f"Dataset directories not found. Expected:\n"
            f"  {train_dir}\n"
            f"  {valid_dir}\n"
            "Download the Kaggle dataset and place images in these folders."
        )

    # Verify that each class folder contains at least one image
    missing_images: list[str] = []
    for split_dir in (train_dir, valid_dir):
        for class_name in ("real", "fake"):
            class_dir = split_dir / class_name
            if not class_dir.exists():
                missing_images.append(f"{class_dir} (folder missing)")
                continue
            image_files = [
                path.name
                for path in class_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ]
            if not image_files:
                all_files = [p.name for p in class_dir.iterdir() if p.is_file()]
                hint = (
                    f" — found {all_files!r} but no .jpg/.png images"
                    if all_files
                    else " — folder is empty"
                )
                missing_images.append(f"{class_dir}{hint}")

    if missing_images:
        raise FileNotFoundError(
            "No training images found in the dataset folders.\n"
            "Each folder needs at least one image (.jpg, .png, etc.).\n"
            "Download the Kaggle dataset and add images to:\n  "
            + "\n  ".join(missing_images)
        )

    # Load datasets with class-specific transforms
    train_dataset = datasets.ImageFolder(
        root=str(train_dir),
        transform=get_transforms(train=True),
        target_transform=_label_to_binary,
    )
    val_dataset = datasets.ImageFolder(
        root=str(valid_dir),
        transform=get_transforms(train=False),
        target_transform=_label_to_binary,
    )

    # Optional subset for quick training runs
    if max_samples_per_class is not None:
        train_dataset = _subset_by_class(train_dataset, max_samples_per_class)
        val_dataset = _subset_by_class(val_dataset, max(max_samples_per_class // 5, 50))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader
