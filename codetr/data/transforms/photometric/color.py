"""Color transforms for object detection."""

from typing import Dict, Optional, Tuple, Union
import random

import torch
from torch import Tensor
from PIL import Image
import torchvision.transforms.functional as F


class ColorJitter:
    """Randomly change brightness, contrast, saturation, and hue.
    
    Args:
        brightness: Factor range for brightness.
        contrast: Factor range for contrast.
        saturation: Factor range for saturation.
        hue: Factor range for hue in [-0.5, 0.5].
        p: Probability of applying the transform.
    """
    
    def __init__(
        self,
        brightness: Union[float, Tuple[float, float]] = 0.4,
        contrast: Union[float, Tuple[float, float]] = 0.4,
        saturation: Union[float, Tuple[float, float]] = 0.4,
        hue: Union[float, Tuple[float, float]] = 0.1,
        p: float = 0.8,
    ) -> None:
        self.brightness = self._to_range(brightness, 1.0)
        self.contrast = self._to_range(contrast, 1.0)
        self.saturation = self._to_range(saturation, 1.0)
        self.hue = self._to_range(hue, 0.0, (-0.5, 0.5))
        self.p = p
    
    def _to_range(self, val, center, bound=(0, float("inf"))):
        if isinstance(val, (int, float)):
            return (max(center - val, bound[0]), min(center + val, bound[1]))
        return (float(val[0]), float(val[1]))
    
    def __call__(
        self,
        image: Union[Image.Image, Tensor],
        target: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Union[Image.Image, Tensor], Optional[Dict[str, Tensor]]]:
        if random.random() >= self.p:
            return image, target
        
        transforms = []
        if self.brightness[0] != self.brightness[1]:
            b = random.uniform(*self.brightness)
            transforms.append(lambda img: F.adjust_brightness(img, b))
        if self.contrast[0] != self.contrast[1]:
            c = random.uniform(*self.contrast)
            transforms.append(lambda img: F.adjust_contrast(img, c))
        if self.saturation[0] != self.saturation[1]:
            s = random.uniform(*self.saturation)
            transforms.append(lambda img: F.adjust_saturation(img, s))
        if self.hue[0] != self.hue[1]:
            h = random.uniform(*self.hue)
            transforms.append(lambda img: F.adjust_hue(img, h))
        
        random.shuffle(transforms)
        for t in transforms:
            image = t(image)
        
        return image, target
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(brightness={self.brightness}, p={self.p})"


class RandomGrayscale:
    """Randomly convert image to grayscale.
    
    Args:
        p: Probability of converting to grayscale.
    """
    
    def __init__(self, p: float = 0.1) -> None:
        self.p = p
    
    def __call__(
        self,
        image: Union[Image.Image, Tensor],
        target: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Union[Image.Image, Tensor], Optional[Dict[str, Tensor]]]:
        if random.random() >= self.p:
            return image, target
        
        if isinstance(image, Tensor):
            image = F.rgb_to_grayscale(image, num_output_channels=image.shape[0])
        else:
            image = F.to_grayscale(image, num_output_channels=3)
        
        return image, target
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


class RandomChannelShuffle:
    """Randomly shuffle RGB channels.
    
    Args:
        p: Probability of shuffling channels.
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
        
        channels = [0, 1, 2]
        random.shuffle(channels)
        
        if isinstance(image, Tensor):
            image = image[channels, :, :]
        else:
            r, g, b = image.split()
            channel_list = [r, g, b]
            image = Image.merge("RGB", [channel_list[i] for i in channels])
        
        return image, target
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"
