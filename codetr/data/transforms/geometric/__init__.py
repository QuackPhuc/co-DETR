"""Geometric augmentations for object detection.

Transforms that modify image geometry with proper bounding box tracking.
"""

from .flip import RandomHorizontalFlip, RandomVerticalFlip
from .rotation import RandomRotation90
from .crop import RandomCrop
from .scale import Resize, RandomScale
from .affine import RandomAffine

__all__ = [
    "RandomHorizontalFlip",
    "RandomVerticalFlip",
    "RandomRotation90",
    "RandomCrop",
    "Resize",
    "RandomScale",
    "RandomAffine",
]
