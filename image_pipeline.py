"""
image_pipeline.py
-----------------
A production-ready PyTorch image data pipeline with custom augmentations.
Designed for scalable Deep Learning / Computer Vision workflows.

Author: Your AGI Company
"""

import os
import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from pathlib import Path
from typing import List, Tuple, Optional, Callable, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF


# ─────────────────────────────────────────────
# CUSTOM AUGMENTATIONS
# ─────────────────────────────────────────────

class RandomGaussianNoise:
    """Add random Gaussian noise to a PIL image."""

    def __init__(self, mean: float = 0.0, std: float = 0.05, p: float = 0.5):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        np_img = np.array(img).astype(np.float32) / 255.0
        noise = np.random.normal(self.mean, self.std, np_img.shape)
        np_img = np.clip(np_img + noise, 0.0, 1.0)
        return Image.fromarray((np_img * 255).astype(np.uint8))

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std}, p={self.p})"


class RandomCutout:
    """Randomly mask out square patches from an image (Cutout augmentation)."""

    def __init__(self, num_holes: int = 1, hole_size: int = 32, fill_value: int = 0, p: float = 0.5):
        self.num_holes = num_holes
        self.hole_size = hole_size
        self.fill_value = fill_value
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        np_img = np.array(img).copy()
        h, w = np_img.shape[:2]
        for _ in range(self.num_holes):
            cx = random.randint(0, w)
            cy = random.randint(0, h)
            x1 = max(0, cx - self.hole_size // 2)
            x2 = min(w, cx + self.hole_size // 2)
            y1 = max(0, cy - self.hole_size // 2)
            y2 = min(h, cy + self.hole_size // 2)
            np_img[y1:y2, x1:x2] = self.fill_value
        return Image.fromarray(np_img)

    def __repr__(self):
        return (f"{self.__class__.__name__}(num_holes={self.num_holes}, "
                f"hole_size={self.hole_size}, p={self.p})")


class RandomSharpen:
    """Randomly apply sharpening filter to an image."""

    def __init__(self, factor_range: Tuple[float, float] = (1.0, 3.0), p: float = 0.3):
        self.factor_range = factor_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        factor = random.uniform(*self.factor_range)
        enhancer = ImageEnhance.Sharpness(img)
        return enhancer.enhance(factor)

    def __repr__(self):
        return f"{self.__class__.__name__}(factor_range={self.factor_range}, p={self.p})"


class RandomBlur:
    """Randomly apply Gaussian blur."""

    def __init__(self, radius_range: Tuple[float, float] = (0.5, 2.0), p: float = 0.3):
        self.radius_range = radius_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        radius = random.uniform(*self.radius_range)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))

    def __repr__(self):
        return f"{self.__class__.__name__}(radius_range={self.radius_range}, p={self.p})"


class MixUp:
    """
    Apply MixUp augmentation at the batch level.
    Usage: call after getting a batch from DataLoader.
    """

    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha

    def __call__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 1.0
        batch_size = images.size(0)
        index = torch.randperm(batch_size)
        mixed_images = lam * images + (1 - lam) * images[index]
        labels_a, labels_b = labels, labels[index]
        return mixed_images, labels_a, labels_b, lam

    def __repr__(self):
        return f"{self.__class__.__name__}(alpha={self.alpha})"


# ─────────────────────────────────────────────
# TRANSFORM PRESETS
# ─────────────────────────────────────────────

class TransformPresets:
    """
    Ready-made transform pipelines for common use cases.
    """

    @staticmethod
    def train(
        image_size: int = 224,
        mean: Tuple = (0.485, 0.456, 0.406),
        std: Tuple = (0.229, 0.224, 0.225),
        extra_augments: Optional[List[Callable]] = None
    ) -> T.Compose:
        transforms = [
            T.Resize((image_size + 32, image_size + 32)),
            T.RandomCrop(image_size),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.1),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            T.RandomGrayscale(p=0.05),
            T.RandomRotation(degrees=15),
            RandomGaussianNoise(std=0.03, p=0.4),
            RandomCutout(num_holes=1, hole_size=32, p=0.4),
            RandomSharpen(p=0.3),
            RandomBlur(p=0.2),
        ]
        if extra_augments:
            transforms.extend(extra_augments)
        transforms += [
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
        return T.Compose(transforms)

    @staticmethod
    def val(
        image_size: int = 224,
        mean: Tuple = (0.485, 0.456, 0.406),
        std: Tuple = (0.229, 0.224, 0.225),
    ) -> T.Compose:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    @staticmethod
    def test(
        image_size: int = 224,
        mean: Tuple = (0.485, 0.456, 0.406),
        std: Tuple = (0.229, 0.224, 0.225),
    ) -> T.Compose:
        return TransformPresets.val(image_size, mean, std)


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────

class ImageFolderDataset(Dataset):
    """
    A flexible image dataset that loads images from a directory.

    Expected folder structure:
        root/
            class_a/
                img1.jpg
                img2.png
            class_b/
                img3.jpg
            ...

    Or flat structure with a label list passed directly.
    """

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
    ):
        self.root = Path(root)
        self.transform = transform
        self.target_transform = target_transform

        self.samples, self.class_to_idx = self._load_samples(class_to_idx)
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        if len(self.samples) == 0:
            raise RuntimeError(f"No valid images found in: {root}")

    def _load_samples(
        self, class_to_idx: Optional[Dict[str, int]]
    ) -> Tuple[List[Tuple[Path, int]], Dict[str, int]]:
        samples = []
        classes = sorted([
            d.name for d in self.root.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])

        if class_to_idx is None:
            class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

        for cls_name, cls_idx in class_to_idx.items():
            cls_dir = self.root / cls_name
            if not cls_dir.exists():
                continue
            for img_path in sorted(cls_dir.iterdir()):
                if img_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                    samples.append((img_path, cls_idx))

        return samples, class_to_idx

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)

        return img, label

    def get_class_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {cls: 0 for cls in self.class_to_idx}
        for _, label in self.samples:
            cls = self.idx_to_class[label]
            counts[cls] += 1
        return counts

    def __repr__(self):
        return (
            f"ImageFolderDataset(\n"
            f"  root={self.root},\n"
            f"  num_samples={len(self.samples)},\n"
            f"  num_classes={len(self.class_to_idx)},\n"
            f"  classes={list(self.class_to_idx.keys())}\n"
            f")"
        )


# ─────────────────────────────────────────────
# PIPELINE BUILDER
# ─────────────────────────────────────────────

class ImagePipeline:
    """
    High-level builder for train/val/test DataLoaders.

    Example usage:
        pipeline = ImagePipeline(
            train_dir="data/train",
            val_dir="data/val",
            image_size=224,
            batch_size=32,
            num_workers=4,
        )
        train_loader, val_loader = pipeline.get_loaders()
    """

    def __init__(
        self,
        train_dir: str,
        val_dir: Optional[str] = None,
        test_dir: Optional[str] = None,
        image_size: int = 224,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        mean: Tuple = (0.485, 0.456, 0.406),
        std: Tuple = (0.229, 0.224, 0.225),
        extra_train_augments: Optional[List[Callable]] = None,
        use_mixup: bool = False,
        mixup_alpha: float = 0.2,
        seed: int = 42,
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.use_mixup = use_mixup
        self.mixup = MixUp(alpha=mixup_alpha) if use_mixup else None
        self.seed = seed

        train_transform = TransformPresets.train(image_size, mean, std, extra_train_augments)
        val_transform = TransformPresets.val(image_size, mean, std)
        test_transform = TransformPresets.test(image_size, mean, std)

        self.train_dataset = ImageFolderDataset(train_dir, transform=train_transform)
        class_to_idx = self.train_dataset.class_to_idx

        self.val_dataset = (
            ImageFolderDataset(val_dir, transform=val_transform, class_to_idx=class_to_idx)
            if val_dir else None
        )
        self.test_dataset = (
            ImageFolderDataset(test_dir, transform=test_transform, class_to_idx=class_to_idx)
            if test_dir else None
        )

    def _make_loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            generator=generator if shuffle else None,
            drop_last=shuffle,
        )

    def get_loaders(self) -> Dict[str, Optional[DataLoader]]:
        loaders = {
            "train": self._make_loader(self.train_dataset, shuffle=True),
            "val": self._make_loader(self.val_dataset, shuffle=False) if self.val_dataset else None,
            "test": self._make_loader(self.test_dataset, shuffle=False) if self.test_dataset else None,
        }
        return loaders

    def summary(self):
        print("=" * 50)
        print("ImagePipeline Summary")
        print("=" * 50)
        print(f"  Train samples : {len(self.train_dataset)}")
        if self.val_dataset:
            print(f"  Val samples   : {len(self.val_dataset)}")
        if self.test_dataset:
            print(f"  Test samples  : {len(self.test_dataset)}")
        print(f"  Classes       : {list(self.train_dataset.class_to_idx.keys())}")
        print(f"  Batch size    : {self.batch_size}")
        print(f"  Num workers   : {self.num_workers}")
        print(f"  MixUp         : {self.use_mixup}")
        print("=" * 50)


# ─────────────────────────────────────────────
# EXAMPLE USAGE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Example: Build a pipeline and iterate one batch

    pipeline = ImagePipeline(
        train_dir="data/train",
        val_dir="data/val",
        image_size=224,
        batch_size=32,
        num_workers=0,       # set >0 in production
        use_mixup=True,
        mixup_alpha=0.2,
        seed=42,
    )

    pipeline.summary()
    loaders = pipeline.get_loaders()
    train_loader = loaders["train"]

    for images, labels in train_loader:
        print(f"Batch images shape : {images.shape}")   # [B, 3, 224, 224]
        print(f"Batch labels shape : {labels.shape}")   # [B]

        # Apply MixUp at batch level
        if pipeline.use_mixup:
            images, labels_a, labels_b, lam = pipeline.mixup(images, labels)
            print(f"MixUp lambda       : {lam:.4f}")

        break  # just test one batch