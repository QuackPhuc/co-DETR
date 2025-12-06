"""Data loading and processing utilities for Co-DETR.

This package provides data loading, transforms, and batching utilities
for detection datasets, with support for YOLOv5 format data.

Modules:
    datasets: Dataset classes for different formats
    transforms: Image and bbox transforms
    dataloader: DataLoader utilities with detection collation

Example:
    >>> from codetr.data import YOLODataset, build_dataloader
    >>> from codetr.data.transforms import make_coco_transforms
    >>> 
    >>> dataset = YOLODataset(
    ...     data_root="/path/to/data",
    ...     split="train",
    ...     transforms=make_coco_transforms("train"),
    ... )
    >>> dataloader = build_dataloader(dataset, batch_size=2)
"""

from .datasets import YOLODataset
from .transforms import (
    Compose,
    ToTensor,
    Normalize,
    Resize,
    RandomHorizontalFlip,
    Pad,
)
from .transforms.transforms import make_coco_transforms
from .dataloader import collate_fn, build_dataloader, build_val_dataloader

__all__ = [
    # Datasets
    "YOLODataset",
    # Transforms
    "Compose",
    "ToTensor",
    "Normalize",
    "Resize",
    "RandomHorizontalFlip",
    "Pad",
    "make_coco_transforms",
    # DataLoader
    "collate_fn",
    "build_dataloader",
    "build_val_dataloader",
]
