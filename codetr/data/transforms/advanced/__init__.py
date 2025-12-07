"""Advanced augmentation strategies for object detection.

Modern augmentation techniques requiring access to multiple samples.
"""

from .cutout import CutOut
from .mixup import MixUp
from .cutmix import CutMix
from .mosaic import Mosaic
from .combinators import RandomSelect, OneOf

__all__ = [
    "CutOut",
    "MixUp",
    "CutMix",
    "Mosaic",
    "RandomSelect",
    "OneOf",
]
