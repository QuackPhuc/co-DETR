"""Flip transforms for object detection."""

from typing import Dict, Optional, Tuple, Union
import random

import torch
from torch import Tensor
from PIL import Image
import torchvision.transforms.functional as F


class RandomHorizontalFlip:
    """Randomly flip image horizontally.
    
    Also flips bounding box x-coordinates: cx -> 1 - cx
    
    Args:
        p: Probability of flipping.
    """
    
    def __init__(self, p: float = 0.5) -> None:
        self.p = p
    
    def __call__(
        self,
        image: Union[Image.Image, Tensor],
        target: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Union[Image.Image, Tensor], Optional[Dict[str, Tensor]]]:
        if random.random() < self.p:
            image = F.hflip(image)
            
            if target is not None and "boxes" in target:
                target = target.copy()
                boxes = target["boxes"].clone()
                boxes[:, 0] = 1.0 - boxes[:, 0]
                target["boxes"] = boxes
        
        return image, target
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


class RandomVerticalFlip:
    """Randomly flip image vertically.
    
    Flips bounding box y-coordinates: cy -> 1 - cy
    
    Args:
        p: Probability of flipping.
    """
    
    def __init__(self, p: float = 0.5) -> None:
        self.p = p
    
    def __call__(
        self,
        image: Union[Image.Image, Tensor],
        target: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Union[Image.Image, Tensor], Optional[Dict[str, Tensor]]]:
        if random.random() < self.p:
            image = F.vflip(image)
            
            if target is not None and "boxes" in target:
                target = target.copy()
                boxes = target["boxes"].clone()
                boxes[:, 1] = 1.0 - boxes[:, 1]
                target["boxes"] = boxes
        
        return image, target
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"
