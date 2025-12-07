"""Data transforms for detection datasets.

This module provides comprehensive image augmentation for object detection
with proper bounding box coordinate tracking.

Structure:
    _base: Core utilities (Compose, ToTensor, Normalize, Pad)
    geometric/: Spatial transforms (flip, rotation, crop, scale, affine)
    photometric/: Color/intensity transforms (jitter, blur, enhance)
    advanced/: Modern augmentations (CutOut, MixUp, CutMix, Mosaic)

Example:
    >>> from codetr.data.transforms import (
    ...     Compose, ToTensor, Normalize,
    ...     RandomHorizontalFlip, ColorJitter, CutOut
    ... )
    >>> transform = Compose([
    ...     ToTensor(),
    ...     RandomHorizontalFlip(p=0.5),
    ...     ColorJitter(brightness=0.4),
    ...     CutOut(p=0.5),
    ...     Normalize(),
    ... ])
"""

# Base utilities
from ._base import (
    Compose,
    ToTensor,
    Normalize,
    Pad,
    IMAGENET_MEAN,
    IMAGENET_STD,
)

# Geometric augmentations
from .geometric import (
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomRotation90,
    RandomCrop,
    Resize,
    RandomScale,
    RandomAffine,
)

# Photometric augmentations
from .photometric import (
    ColorJitter,
    RandomGrayscale,
    RandomChannelShuffle,
    GaussianBlur,
    RandomSharpness,
    RandomEqualize,
    RandomPosterize,
    RandomSolarize,
    RandomAutocontrast,
)

# Advanced augmentations
from .advanced import (
    CutOut,
    MixUp,
    CutMix,
    Mosaic,
    RandomSelect,
    OneOf,
)


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
    """
    normalize = Compose([ToTensor(), Normalize()])
    
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


__all__ = [
    # Base
    "Compose",
    "ToTensor",
    "Normalize",
    "Pad",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "make_coco_transforms",
    # Geometric
    "RandomHorizontalFlip",
    "RandomVerticalFlip",
    "RandomRotation90",
    "RandomCrop",
    "Resize",
    "RandomScale",
    "RandomAffine",
    # Photometric
    "ColorJitter",
    "RandomGrayscale",
    "RandomChannelShuffle",
    "GaussianBlur",
    "RandomSharpness",
    "RandomEqualize",
    "RandomPosterize",
    "RandomSolarize",
    "RandomAutocontrast",
    # Advanced
    "CutOut",
    "MixUp",
    "CutMix",
    "Mosaic",
    "RandomSelect",
    "OneOf",
]
