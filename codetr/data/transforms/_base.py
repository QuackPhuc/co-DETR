"""Base utilities and common functionality for transforms.

This module contains shared code used across all transform modules:
- Compose class for chaining transforms
- ToTensor and Normalize for basic preprocessing
- Pad for making dimensions divisible
- ImageNet constants
"""

from typing import Dict, List, Optional, Tuple, Union

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
    """
    
    def __init__(self, transforms: List) -> None:
        self.transforms = transforms
    
    def __call__(
        self,
        image: Union[Image.Image, Tensor],
        target: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        """Apply all transforms sequentially."""
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
        """Convert image to tensor."""
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
        """Normalize image tensor."""
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


class Pad:
    """Pad image to be divisible by given number.
    
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
        """Pad image to divisible size."""
        _, h, w = image.shape
        
        pad_h = (self.divisor - h % self.divisor) % self.divisor
        pad_w = (self.divisor - w % self.divisor) % self.divisor
        
        if pad_h > 0 or pad_w > 0:
            image = torch.nn.functional.pad(
                image,
                (0, pad_w, 0, pad_h),
                mode="constant",
                value=self.pad_value,
            )
        
        return image, target
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(divisor={self.divisor})"
