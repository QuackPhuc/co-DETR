"""YOLOv5 format dataset loader.

This module implements a PyTorch Dataset for loading YOLOv5 format datasets
with proper bounding box parsing and coordinate handling.

YOLOv5 Format:
    - images/train/ and images/val/ directories containing images
    - labels/train/ and labels/val/ directories containing .txt files
    - Each .txt file has lines: class_id cx cy w h (normalized 0-1)
    - Label file name matches image file name (e.g., image001.jpg -> image001.txt)

Example:
    >>> dataset = YOLODataset(
    ...     data_root="/path/to/data",
    ...     split="train",
    ...     transforms=make_coco_transforms("train"),
    ... )
    >>> image, target = dataset[0]
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import os

import torch
from torch import Tensor
from torch.utils.data import Dataset
from PIL import Image


# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


class YOLODataset(Dataset):
    """Dataset for YOLOv5 format data.
    
    Loads images and annotations in YOLOv5 format, with boxes in
    normalized cxcywh format.
    
    Args:
        data_root: Root directory containing images/ and labels/ folders.
        split: Dataset split, either 'train' or 'val'.
        transforms: Optional transform pipeline.
        class_names: Optional list of class names.
        
    Attributes:
        data_root: Root data directory.
        split: Dataset split.
        transforms: Transform pipeline.
        class_names: List of class names.
        image_paths: List of image file paths.
        
    Example:
        >>> dataset = YOLODataset(
        ...     data_root="/data/coco_yolo",
        ...     split="train",
        ...     transforms=make_coco_transforms("train"),
        ... )
        >>> image, target = dataset[0]
        >>> print(image.shape)  # (3, H, W)
        >>> print(target["boxes"].shape)  # (N, 4)
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        transforms: Optional[Callable] = None,
        class_names: Optional[List[str]] = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split
        self.transforms = transforms
        self.class_names = class_names
        
        # Setup paths
        self.image_dir = self.data_root / "images" / split
        self.label_dir = self.data_root / "labels" / split
        
        # Collect image paths
        self.image_paths = self._collect_images()
        
        if len(self.image_paths) == 0:
            raise ValueError(
                f"No images found in {self.image_dir}. "
                f"Expected YOLOv5 format with images/{split}/ directory."
            )
    
    def _collect_images(self) -> List[Path]:
        """Collect all image paths from the image directory.
        
        Returns:
            Sorted list of image paths.
        """
        image_paths = set()  # Use set to avoid duplicates
        
        if not self.image_dir.exists():
            return []
        
        for ext in IMAGE_EXTENSIONS:
            # Collect both cases but deduplicate using set
            image_paths.update(self.image_dir.glob(f"*{ext}"))
            image_paths.update(self.image_dir.glob(f"*{ext.upper()}"))
        
        return sorted(image_paths)
    
    def _get_label_path(self, image_path: Path) -> Path:
        """Get label file path for an image.
        
        Args:
            image_path: Path to image file.
            
        Returns:
            Path to corresponding label file.
        """
        label_name = image_path.stem + ".txt"
        return self.label_dir / label_name
    
    def _load_annotations(
        self,
        label_path: Path,
        img_w: int,
        img_h: int,
    ) -> Dict[str, Tensor]:
        """Load annotations from label file.
        
        Args:
            label_path: Path to .txt label file.
            img_w: Image width (for validation).
            img_h: Image height (for validation).
            
        Returns:
            Dict with 'boxes' (N, 4) and 'labels' (N,).
        """
        boxes = []
        labels = []
        
        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    
                    class_id = int(parts[0])
                    cx = float(parts[1])
                    cy = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                    
                    # Validate normalized coordinates
                    cx = max(0.0, min(1.0, cx))
                    cy = max(0.0, min(1.0, cy))
                    w = max(0.0, min(1.0, w))
                    h = max(0.0, min(1.0, h))
                    
                    # Skip invalid boxes
                    if w <= 0 or h <= 0:
                        continue
                    
                    boxes.append([cx, cy, w, h])
                    labels.append(class_id)
        
        # Convert to tensors
        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        
        return {"boxes": boxes, "labels": labels}
    
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Get a single sample.
        
        Args:
            idx: Sample index.
            
        Returns:
            Tuple of (image, target) where:
                - image: Tensor (3, H, W)
                - target: Dict with keys:
                    - 'boxes': (N, 4) cxcywh normalized
                    - 'labels': (N,) class indices
                    - 'image_id': int
                    - 'orig_size': (2,) original (H, W)
                    - 'size': (2,) current (H, W)
        """
        image_path = self.image_paths[idx]
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        img_w, img_h = image.size
        
        # Load annotations
        label_path = self._get_label_path(image_path)
        target = self._load_annotations(label_path, img_w, img_h)
        
        # Add metadata
        target["image_id"] = idx
        target["orig_size"] = torch.tensor([img_h, img_w])
        target["size"] = torch.tensor([img_h, img_w])
        
        # Apply transforms
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        
        return image, target
    
    def get_image_path(self, idx: int) -> Path:
        """Get image path for a given index.
        
        Args:
            idx: Sample index.
            
        Returns:
            Path to image file.
        """
        return self.image_paths[idx]
    
    @property
    def num_classes(self) -> Optional[int]:
        """Return number of classes if class_names is provided."""
        if self.class_names is not None:
            return len(self.class_names)
        return None
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"data_root={self.data_root}, "
            f"split={self.split}, "
            f"num_samples={len(self)}"
            f")"
        )


class YOLODatasetFromList(Dataset):
    """Dataset loading from a list of image paths.
    
    Alternative to folder-based loading when a list file is provided.
    
    Args:
        image_list_file: Path to file containing image paths (one per line).
        label_dir: Directory containing label files.
        transforms: Optional transform pipeline.
        class_names: Optional list of class names.
    """
    
    def __init__(
        self,
        image_list_file: str,
        label_dir: str,
        transforms: Optional[Callable] = None,
        class_names: Optional[List[str]] = None,
    ) -> None:
        self.label_dir = Path(label_dir)
        self.transforms = transforms
        self.class_names = class_names
        
        # Load image paths from list file
        self.image_paths = []
        with open(image_list_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and Path(line).exists():
                    self.image_paths.append(Path(line))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No valid images found in {image_list_file}")
    
    def _get_label_path(self, image_path: Path) -> Path:
        """Get label file path for an image."""
        label_name = image_path.stem + ".txt"
        return self.label_dir / label_name
    
    def _load_annotations(
        self,
        label_path: Path,
        img_w: int,
        img_h: int,
    ) -> Dict[str, Tensor]:
        """Load annotations from label file."""
        boxes = []
        labels = []
        
        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    
                    class_id = int(parts[0])
                    cx = max(0.0, min(1.0, float(parts[1])))
                    cy = max(0.0, min(1.0, float(parts[2])))
                    w = max(0.0, min(1.0, float(parts[3])))
                    h = max(0.0, min(1.0, float(parts[4])))
                    
                    if w > 0 and h > 0:
                        boxes.append([cx, cy, w, h])
                        labels.append(class_id)
        
        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        
        return {"boxes": boxes, "labels": labels}
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Get a single sample."""
        image_path = self.image_paths[idx]
        
        image = Image.open(image_path).convert("RGB")
        img_w, img_h = image.size
        
        label_path = self._get_label_path(image_path)
        target = self._load_annotations(label_path, img_w, img_h)
        
        target["image_id"] = idx
        target["orig_size"] = torch.tensor([img_h, img_w])
        target["size"] = torch.tensor([img_h, img_w])
        
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        
        return image, target
