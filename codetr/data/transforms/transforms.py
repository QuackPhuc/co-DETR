"""Data transforms for detection with bounding box coordinate tracking.

This module implements transforms that properly handle both images and
bounding box annotations, ensuring coordinate consistency throughout
the transformation pipeline.

Key features:
    - All transforms handle both image and target dict
    - Bounding boxes tracked through resize, flip, and pad operations
    - ImageNet normalization with standard mean/std
    - Aspect-ratio preserving resize
    - Padding to ensure divisibility for CNN processing

Example:
    >>> transform = Compose([
    ...     ToTensor(),
    ...     Resize(min_size=800, max_size=1333),
    ...     RandomHorizontalFlip(p=0.5),
    ...     Normalize(),
    ...     Pad(divisor=32),
    ... ])
    >>> image, target = transform(image, target)
"""

from typing import Dict, List, Optional, Tuple, Union
import random

import torch
from torch import Tensor
from PIL import Image
import torchvision.transforms.functional as F


# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class Compose:
    """Compose multiple transforms together.
    
    Args:
        transforms: List of transforms to apply sequentially.
        
    Example:
        >>> transform = Compose([ToTensor(), Normalize()])
        >>> image, target = transform(image, target)
    """
    
    def __init__(self, transforms: List) -> None:
        self.transforms = transforms
    
    def __call__(
        self,
        image: Union[Image.Image, Tensor],
        target: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        """Apply all transforms sequentially.
        
        Args:
            image: Input image (PIL or Tensor).
            target: Optional target dict with 'boxes' and 'labels'.
            
        Returns:
            Transformed image and target.
        """
        for t in self.transforms:
            image, target = t(image, target)
        return image, target
    
    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += f"\n    {t}"
        format_string += "\n)"
        return format_string


class ToTensor:
    """Convert PIL Image to tensor.
    
    Converts image from [0, 255] to [0.0, 1.0] range.
    """
    
    def __call__(
        self,
        image: Union[Image.Image, Tensor],
        target: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        """Convert image to tensor.
        
        Args:
            image: PIL Image or Tensor.
            target: Optional target dict (passed through unchanged).
            
        Returns:
            Image tensor (C, H, W) and target.
        """
        if isinstance(image, Tensor):
            return image, target
        image = F.to_tensor(image)
        return image, target
    
    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"


class Normalize:
    """Normalize image with ImageNet mean and std.
    
    Args:
        mean: Sequence of means for each channel.
        std: Sequence of stds for each channel.
    """
    
    def __init__(
        self,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
    ) -> None:
        self.mean = mean if mean is not None else IMAGENET_MEAN
        self.std = std if std is not None else IMAGENET_STD
    
    def __call__(
        self,
        image: Tensor,
        target: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        """Normalize image tensor.
        
        Args:
            image: Image tensor (C, H, W).
            target: Optional target dict (passed through unchanged).
            
        Returns:
            Normalized image and target.
        """
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


class Resize:
    """Resize image keeping aspect ratio.
    
    Scales the image so that the shorter side is at least min_size
    and the longer side is at most max_size.
    
    Args:
        min_size: Minimum size of shorter side.
        max_size: Maximum size of longer side.
        
    Note:
        Bounding boxes remain in normalized [0, 1] coordinates,
        so no box transformation is needed.
    """
    
    def __init__(
        self,
        min_size: int = 800,
        max_size: int = 1333,
    ) -> None:
        self.min_size = min_size
        self.max_size = max_size
    
    def _get_size(self, image_size: Tuple[int, int]) -> Tuple[int, int]:
        """Compute new size preserving aspect ratio.
        
        Args:
            image_size: Original (width, height).
            
        Returns:
            New (width, height).
        """
        w, h = image_size
        
        # Scale to min_size
        min_orig = min(w, h)
        max_orig = max(w, h)
        
        scale = self.min_size / min_orig
        
        # Check if max_size constraint is violated
        if max_orig * scale > self.max_size:
            scale = self.max_size / max_orig
        
        new_w = int(w * scale + 0.5)
        new_h = int(h * scale + 0.5)
        
        return new_w, new_h
    
    def __call__(
        self,
        image: Union[Image.Image, Tensor],
        target: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Union[Image.Image, Tensor], Optional[Dict[str, Tensor]]]:
        """Resize image and update target size.
        
        Args:
            image: PIL Image or Tensor.
            target: Optional target dict with 'size' key.
            
        Returns:
            Resized image and updated target.
        """
        if isinstance(image, Tensor):
            _, h, w = image.shape
            orig_size = (w, h)
        else:
            orig_size = image.size  # (width, height)
        
        new_size = self._get_size(orig_size)
        
        # Resize image
        if isinstance(image, Tensor):
            # F.resize expects (H, W) for size parameter
            image = F.resize(image, size=(new_size[1], new_size[0]))
        else:
            image = F.resize(image, size=(new_size[1], new_size[0]))
        
        # Update target size if present
        if target is not None:
            target = target.copy()
            target["size"] = torch.tensor([new_size[1], new_size[0]])  # (H, W)
        
        return image, target
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(min_size={self.min_size}, max_size={self.max_size})"


class RandomHorizontalFlip:
    """Randomly flip image horizontally.
    
    Also flips bounding box x-coordinates.
    
    Args:
        p: Probability of flipping.
        
    Note:
        For normalized boxes (cx, cy, w, h) in [0, 1]:
        new_cx = 1.0 - cx
    """
    
    def __init__(self, p: float = 0.5) -> None:
        self.p = p
    
    def __call__(
        self,
        image: Union[Image.Image, Tensor],
        target: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Union[Image.Image, Tensor], Optional[Dict[str, Tensor]]]:
        """Randomly flip image and boxes.
        
        Args:
            image: PIL Image or Tensor.
            target: Optional target dict with 'boxes' key.
            
        Returns:
            Possibly flipped image and target.
        """
        if random.random() < self.p:
            image = F.hflip(image)
            
            if target is not None and "boxes" in target:
                target = target.copy()
                boxes = target["boxes"].clone()
                
                # Flip x-center: new_cx = 1.0 - cx
                # boxes format: (cx, cy, w, h) normalized
                boxes[:, 0] = 1.0 - boxes[:, 0]
                
                target["boxes"] = boxes
        
        return image, target
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


class Pad:
    """Pad image to be divisible by given number.
    
    Pads to bottom-right, updates target size accordingly.
    
    Args:
        divisor: Image dimensions will be padded to be divisible by this.
        pad_value: Value to fill padding with.
    """
    
    def __init__(
        self,
        divisor: int = 32,
        pad_value: float = 0.0,
    ) -> None:
        self.divisor = divisor
        self.pad_value = pad_value
    
    def __call__(
        self,
        image: Tensor,
        target: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        """Pad image to divisible size.
        
        Args:
            image: Image tensor (C, H, W).
            target: Optional target dict.
            
        Returns:
            Padded image and target.
        """
        _, h, w = image.shape
        
        # Compute padding needed
        pad_h = (self.divisor - h % self.divisor) % self.divisor
        pad_w = (self.divisor - w % self.divisor) % self.divisor
        
        if pad_h > 0 or pad_w > 0:
            # Pad bottom and right
            image = torch.nn.functional.pad(
                image,
                (0, pad_w, 0, pad_h),  # (left, right, top, bottom)
                mode="constant",
                value=self.pad_value,
            )
        
        # Note: We don't update target['size'] here as boxes are normalized
        # and we need the original valid region for mask generation
        
        return image, target
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(divisor={self.divisor})"


def make_coco_transforms(
    image_set: str,
    min_size: int = 800,
    max_size: int = 1333,
) -> Compose:
    """Create standard COCO-style transforms.
    
    Args:
        image_set: 'train' or 'val'.
        min_size: Minimum size for Resize.
        max_size: Maximum size for Resize.
        
    Returns:
        Composed transform pipeline.
    """
    normalize = Compose([
        ToTensor(),
        Normalize(),
    ])
    
    if image_set == "train":
        return Compose([
            RandomHorizontalFlip(),
            normalize,
            Resize(min_size=min_size, max_size=max_size),
            Pad(divisor=32),
        ])
    
    if image_set == "val":
        return Compose([
            normalize,
            Resize(min_size=min_size, max_size=max_size),
            Pad(divisor=32),
        ])
    
    raise ValueError(f"Unknown image_set: {image_set}")
