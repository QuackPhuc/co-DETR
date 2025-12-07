"""Blur and sharpness transforms for object detection."""

from typing import Dict, Optional, Tuple, Union
import random

import torch
from torch import Tensor
from PIL import Image, ImageFilter
import torchvision.transforms.functional as F


class GaussianBlur:
    """Apply Gaussian blur with random kernel size.
    
    Args:
        kernel_size_range: Range of kernel sizes (must be odd).
        sigma_range: Range of sigma values.
        p: Probability of applying blur.
    """
    
    def __init__(
        self,
        kernel_size_range: Tuple[int, int] = (3, 7),
        sigma_range: Tuple[float, float] = (0.1, 2.0),
        p: float = 0.5,
    ) -> None:
        self.kernel_size_range = kernel_size_range
        self.sigma_range = sigma_range
        self.p = p
    
    def __call__(
        self,
        image: Union[Image.Image, Tensor],
        target: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Union[Image.Image, Tensor], Optional[Dict[str, Tensor]]]:
        if random.random() >= self.p:
            return image, target
        
        k = random.randrange(self.kernel_size_range[0], self.kernel_size_range[1] + 1, 2)
        if k % 2 == 0:
            k += 1
        sigma = random.uniform(*self.sigma_range)
        
        if isinstance(image, Tensor):
            image = F.gaussian_blur(image, kernel_size=[k, k], sigma=[sigma, sigma])
        else:
            image = image.filter(ImageFilter.GaussianBlur(radius=sigma))
        
        return image, target
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(kernel_size_range={self.kernel_size_range}, p={self.p})"


class RandomSharpness:
    """Randomly adjust image sharpness.
    
    Args:
        sharpness_range: (min, max) factor. 0=blur, 1=original, 2=sharp.
        p: Probability of applying adjustment.
    """
    
    def __init__(
        self,
        sharpness_range: Tuple[float, float] = (0.5, 2.0),
        p: float = 0.5,
    ) -> None:
        self.sharpness_range = sharpness_range
        self.p = p
    
    def __call__(
        self,
        image: Union[Image.Image, Tensor],
        target: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Union[Image.Image, Tensor], Optional[Dict[str, Tensor]]]:
        if random.random() >= self.p:
            return image, target
        
        factor = random.uniform(*self.sharpness_range)
        image = F.adjust_sharpness(image, factor)
        
        return image, target
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(sharpness_range={self.sharpness_range}, p={self.p})"
