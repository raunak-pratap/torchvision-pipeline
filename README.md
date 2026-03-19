# torchvision-pipeline

A production-ready PyTorch image data pipeline with custom augmentations, built for scalable Computer Vision and Deep Learning workflows.

## Features

- **Custom augmentations** — `RandomGaussianNoise`, `RandomCutout`, `RandomSharpen`, `RandomBlur`
- **MixUp** — batch-level label blending for better model calibration
- **Transform presets** — separate train / val / test augmentation chains, no data leakage
- **Reproducible** — fixed seed on DataLoader generator for consistent experiments
- **Flexible dataset** — supports `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`
- **Class imbalance detection** — built-in `get_class_counts()` utility

## Folder structure

```
data/
    train/
        cat/
            img001.jpg
        dog/
            img002.jpg
    val/
        cat/
            img003.jpg
        dog/
            img004.jpg
```

## Quick start

```python
from image_pipeline import ImagePipeline

pipeline = ImagePipeline(
    train_dir="data/train",
    val_dir="data/val",
    image_size=224,
    batch_size=32,
    num_workers=4,
    use_mixup=True,
    mixup_alpha=0.2,
    seed=42,
)

pipeline.summary()
loaders = pipeline.get_loaders()

for images, labels in loaders["train"]:
    print(images.shape)  # [32, 3, 224, 224]
    break
```

## Installation

```bash
pip install torch torchvision Pillow numpy
```

## Components

| Class | Role |
|---|---|
| `ImageFolderDataset` | Scans class folders, maps labels, loads images |
| `TransformPresets` | Ready-made augmentation chains for train / val / test |
| `RandomGaussianNoise` | Adds pixel-level Gaussian noise |
| `RandomCutout` | Masks random square patches |
| `RandomSharpen` | Randomly sharpens edges |
| `RandomBlur` | Applies random Gaussian blur |
| `MixUp` | Batch-level image and label blending |
| `ImagePipeline` | High-level builder — returns DataLoaders in one call |

## Part of the AGI CV stack

This pipeline is Layer 1 of a full computer vision stack being built for AGI research. Coming next: model backbone, training engine, experiment tracking, evaluation suite, and inference serving.

## License

MIT
