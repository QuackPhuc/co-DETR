"""Enhancement transforms for object detection."""

from typing import Dict, Optional, Tuple, Union
import random

import torch
from torch import Tensor
from PIL import Image, ImageOps
import torchvision.transforms.functional as F


class RandomEqualize:
    """Apply histogram equalization randomly.
    
    Args:
        p: Probability of applying equalization.
    """
    
    def __init__(self, p: float = 0.5) -> None:
        self.p = p
    
    def __call__(
        self,
        image: Union[Image.Image, Tensor],
        target: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Union[Image.Image, Tensor], Optional[Dict[str, Tensor]]]:
        if random.random() >= self.p:
            return image, target
        
        if isinstance(image, Tensor):
            if image.dtype == torch.float32:
                img_u8 = (image * 255).to(torch.uint8)
                image = F.equalize(img_u8).float() / 255.0
            else:
                image = F.equalize(image)
        else:
            image = ImageOps.equalize(image)
        
        return image, target
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


class RandomPosterize:
    """Reduce color bit depth randomly.
    
    Args:
        bits_range: Range of bits to keep (min, max).
        p: Probability of applying posterization.
    """
    
    def __init__(self, bits_range: Tuple[int, int] = (4, 8), p: float = 0.5) -> None:
        self.bits_range = bits_range
        self.p = p
    
    def __call__(
        self,
        image: Union[Image.Image, Tensor],
        target: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Union[Image.Image, Tensor], Optional[Dict[str, Tensor]]]:
        if random.random() >= self.p:
            return image, target
        
        bits = random.randint(*self.bits_range)
        
        if isinstance(image, Tensor):
            if image.dtype == torch.float32:
                img_u8 = (image * 255).to(torch.uint8)
                image = F.posterize(img_u8, bits).float() / 255.0
            else:
                image = F.posterize(image, bits)
        else:
            image = ImageOps.posterize(image, bits)
        
        return image, target
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(bits_range={self.bits_range}, p={self.p})"


class RandomSolarize:
    """Invert pixels above a threshold.
    
    Args:
        threshold_range: Range of thresholds in [0, 1].
        p: Probability of applying solarization.
    """
    
    def __init__(self, threshold_range: Tuple[float, float] = (0.5, 1.0), p: float = 0.5) -> None:
        self.threshold_range = threshold_range
        self.p = p
    
    def __call__(
        self,
        image: Union[Image.Image, Tensor],
        target: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Union[Image.Image, Tensor], Optional[Dict[str, Tensor]]]:
        if random.random() >= self.p:
            return image, target
        
        threshold = random.uniform(*self.threshold_range)
        
        if isinstance(image, Tensor):
            if image.dtype == torch.float32:
                thresh_u8 = int(threshold * 255)
                img_u8 = (image * 255).to(torch.uint8)
                image = F.solarize(img_u8, thresh_u8).float() / 255.0
            else:
                image = F.solarize(image, int(threshold * 255))
        else:
            image = ImageOps.solarize(image, int(threshold * 255))
        
        return image, target
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(threshold_range={self.threshold_range}, p={self.p})"


class RandomAutocontrast:
    """Apply autocontrast to maximize image contrast.
    
    Args:
        p: Probability of applying autocontrast.
    """
    
    def __init__(self, p: float = 0.5) -> None:
        self.p = p
    
    def __call__(
        self,
        image: Union[Image.Image, Tensor],
        target: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Union[Image.Image, Tensor], Optional[Dict[str, Tensor]]]:
        if random.random() >= self.p:
            return image, target
        
        if isinstance(image, Tensor):
            if image.dtype == torch.float32:
                img_u8 = (image * 255).to(torch.uint8)
                image = F.autocontrast(img_u8).float() / 255.0
            else:
                image = F.autocontrast(image)
        else:
            image = ImageOps.autocontrast(image)
        
        return image, target
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"
