"""Scale and resize transforms for object detection."""

from typing import Dict, Optional, Tuple, Union
import random

import torch
from torch import Tensor
from PIL import Image
import torchvision.transforms.functional as F


class Resize:
    """Resize image keeping aspect ratio.
    
    Scales so shorter side is at least min_size and longer side 
    is at most max_size.
    
    Args:
        min_size: Minimum size of shorter side.
        max_size: Maximum size of longer side.
    """
    
    def __init__(
        self,
        min_size: int = 800,
        max_size: int = 1333,
    ) -> None:
        self.min_size = min_size
        self.max_size = max_size
    
    def _get_size(self, image_size: Tuple[int, int]) -> Tuple[int, int]:
        w, h = image_size
        min_orig = min(w, h)
        max_orig = max(w, h)
        
        scale = self.min_size / min_orig
        if max_orig * scale > self.max_size:
            scale = self.max_size / max_orig
        
        return int(w * scale + 0.5), int(h * scale + 0.5)
    
    def __call__(
        self,
        image: Union[Image.Image, Tensor],
        target: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Union[Image.Image, Tensor], Optional[Dict[str, Tensor]]]:
        if isinstance(image, Tensor):
            _, h, w = image.shape
            orig_size = (w, h)
        else:
            orig_size = image.size
        
        new_size = self._get_size(orig_size)
        image = F.resize(image, size=(new_size[1], new_size[0]))
        
        if target is not None:
            target = target.copy()
            target["size"] = torch.tensor([new_size[1], new_size[0]])
        
        return image, target
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(min_size={self.min_size}, max_size={self.max_size})"


class RandomScale:
    """Randomly scale image within a range.
    
    Args:
        scale_range: (min_scale, max_scale) relative to original.
        keep_aspect_ratio: If True, scale uniformly.
    """
    
    def __init__(
        self,
        scale_range: Tuple[float, float] = (0.8, 1.2),
        keep_aspect_ratio: bool = True,
    ) -> None:
        self.scale_range = scale_range
        self.keep_aspect_ratio = keep_aspect_ratio
    
    def __call__(
        self,
        image: Union[Image.Image, Tensor],
        target: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Union[Image.Image, Tensor], Optional[Dict[str, Tensor]]]:
        if isinstance(image, Tensor):
            _, h, w = image.shape
        else:
            w, h = image.size
        
        if self.keep_aspect_ratio:
            scale = random.uniform(self.scale_range[0], self.scale_range[1])
            scale_x = scale_y = scale
        else:
            scale_x = random.uniform(self.scale_range[0], self.scale_range[1])
            scale_y = random.uniform(self.scale_range[0], self.scale_range[1])
        
        new_w = max(1, int(w * scale_x + 0.5))
        new_h = max(1, int(h * scale_y + 0.5))
        
        image = F.resize(image, size=(new_h, new_w))
        
        if target is not None:
            target = target.copy()
            target["size"] = torch.tensor([new_h, new_w])
        
        return image, target
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(scale_range={self.scale_range})"
