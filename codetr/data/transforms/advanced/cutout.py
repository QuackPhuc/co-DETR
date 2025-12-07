"""CutOut augmentation for object detection."""

from typing import Dict, Optional, Tuple, Union
import random
import math

import torch
from torch import Tensor
from PIL import Image
import torchvision.transforms.functional as F


class CutOut:
    """Randomly erase rectangular patches from the image.
    
    Args:
        num_patches: Number of patches to erase.
        size_range: Range of patch size as fraction of image.
        aspect_ratio_range: Range of aspect ratios.
        fill_value: Value to fill (float, 'random', or 'mean').
        avoid_boxes: If True, avoid erasing box centers.
        p: Probability of applying CutOut.
    """
    
    def __init__(
        self,
        num_patches: int = 1,
        size_range: Tuple[float, float] = (0.02, 0.2),
        aspect_ratio_range: Tuple[float, float] = (0.3, 3.3),
        fill_value: Union[float, str] = 0.0,
        avoid_boxes: bool = False,
        p: float = 0.5,
    ) -> None:
        self.num_patches = num_patches
        self.size_range = size_range
        self.aspect_ratio_range = aspect_ratio_range
        self.fill_value = fill_value
        self.avoid_boxes = avoid_boxes
        self.p = p
    
    def _get_patch(self, h, w, boxes=None):
        for _ in range(10):
            area = h * w * random.uniform(*self.size_range)
            ratio = random.uniform(*self.aspect_ratio_range)
            
            ph = int(math.sqrt(area / ratio))
            pw = int(ph * ratio)
            
            if ph >= h or pw >= w:
                continue
            
            top = random.randint(0, h - ph)
            left = random.randint(0, w - pw)
            
            if self.avoid_boxes and boxes is not None and len(boxes) > 0:
                px1, py1 = left / w, top / h
                px2, py2 = (left + pw) / w, (top + ph) / h
                cx, cy = boxes[:, 0], boxes[:, 1]
                if ((cx >= px1) & (cx <= px2) & (cy >= py1) & (cy <= py2)).any():
                    continue
            
            return top, left, ph, pw
        
        return 0, 0, min(10, h), min(10, w)
    
    def __call__(
        self,
        image: Union[Image.Image, Tensor],
        target: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Union[Image.Image, Tensor], Optional[Dict[str, Tensor]]]:
        if random.random() >= self.p:
            return image, target
        
        if isinstance(image, Tensor):
            c, h, w = image.shape
            image = image.clone()
            convert_back = False
        else:
            w, h = image.size
            image = F.to_tensor(image)
            c = image.shape[0]
            convert_back = True
        
        boxes = target.get("boxes") if target else None
        
        for _ in range(self.num_patches):
            top, left, ph, pw = self._get_patch(h, w, boxes)
            
            if self.fill_value == 'random':
                fill = torch.rand(c, ph, pw)
            elif self.fill_value == 'mean':
                fill = image.mean()
            else:
                fill = float(self.fill_value)
            
            image[:, top:top + ph, left:left + pw] = fill
        
        if convert_back:
            image = F.to_pil_image(image)
        
        return image, target
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_patches={self.num_patches}, p={self.p})"
