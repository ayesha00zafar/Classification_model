"""
Dataset loader for OccluzNet bite image classifier.

Expected folder structure:
    dataset/
        Normal_Class_I/
            img001.jpg
            img002.jpg
            ...
        Open_Bite/
            img001.jpg
            ...
        Crossbite/
            img001.jpg
            ...

Any common image format works: jpg, jpeg, png, bmp, tiff.
"""

import os
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ──────────────────────────────────────────
# Class mapping — must match folder names
# ──────────────────────────────────────────
CLASS_TO_IDX = {
    "Normal_Class_I": 0,
    "Open_Bite":      1,
    "Crossbite":      2,
}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}
DISPLAY_NAMES = {
    0: "Normal / Class I",
    1: "Open Bite",
    2: "Crossbite",
}

IMAGE_SIZE = 224  # EfficientNet-B0 native input size


# ──────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────
def get_train_transforms():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.RandomGrayscale(p=0.05),   # simulate X-ray / grayscale images
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ──────────────────────────────────────────
# Dataset class
# ──────────────────────────────────────────
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


class BiteDataset(Dataset):
    def __init__(self, root_dir, transform=None, split="all"):
        """
        Args:
            root_dir: path to dataset/ folder containing class subfolders
            transform: torchvision transforms to apply
            split: 'all' | 'train' | 'val' — if train/val, use BiteDatasetSplit instead
        """
        self.root_dir  = Path(root_dir)
        self.transform = transform
        self.samples   = []   # list of (image_path, label_idx)
        self.class_counts = {i: 0 for i in range(len(CLASS_TO_IDX))}

        for class_name, idx in CLASS_TO_IDX.items():
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                print(f"[WARN] Class folder not found: {class_dir}")
                continue
            for f in class_dir.iterdir():
                if f.suffix.lower() in VALID_EXTENSIONS:
                    self.samples.append((str(f), idx))
                    self.class_counts[idx] += 1

        print(f"[INFO] Loaded {len(self.samples)} images from {root_dir}")
        for idx, count in self.class_counts.items():
            print(f"       {DISPLAY_NAMES[idx]}: {count} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_class_weights(self):
        """Compute inverse-frequency weights for CrossEntropyLoss."""
        import torch
        total = len(self.samples)
        weights = []
        for i in range(len(CLASS_TO_IDX)):
            count = self.class_counts[i]
            weights.append(total / (len(CLASS_TO_IDX) * max(count, 1)))
        return torch.tensor(weights, dtype=torch.float32)


# ──────────────────────────────────────────
# DataLoader factory
# ──────────────────────────────────────────
def get_dataloaders(dataset_root, batch_size=16, val_split=0.2, num_workers=2):
    """
    Splits dataset into train/val, applies correct transforms, returns DataLoaders.
    """
    import torch
    from torch.utils.data import random_split

    full_dataset = BiteDataset(dataset_root, transform=None)

    val_size   = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    train_subset, val_subset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    # Apply different transforms to each split
    train_subset.dataset = _TransformWrapper(train_subset, get_train_transforms())
    val_subset.dataset   = _TransformWrapper(val_subset,   get_val_transforms())

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"[INFO] Train: {train_size} | Val: {val_size}")
    return train_loader, val_loader, full_dataset.get_class_weights()


class _TransformWrapper:
    """Wraps a Subset to apply a specific transform without affecting the original dataset."""
    def __init__(self, subset, transform):
        self.subset    = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image_pil, label = self.subset.dataset.samples[self.subset.indices[idx]]
        image_pil = Image.open(image_pil).convert("RGB")
        return self.transform(image_pil), label
