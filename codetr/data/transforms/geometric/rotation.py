"""Rotation transforms for object detection."""

from typing import Dict, List, Optional, Tuple, Union
import random

import torch
from torch import Tensor
from PIL import Image


class RandomRotation90:
    """Randomly rotate image by 90, 180, or 270 degrees.
    
    Only 90-degree rotations preserve axis-aligned bounding boxes.
    
    Args:
        angles: List of possible angles. Default [0, 90, 180, 270].
        p: Probability of applying rotation.
    
    Box transformations:
        90° CW:  (cx, cy, w, h) -> (1 - cy, cx, h, w)
        180°:    (cx, cy, w, h) -> (1 - cx, 1 - cy, w, h)
        270° CW: (cx, cy, w, h) -> (cy, 1 - cx, h, w)
    """
    
    def __init__(
        self,
        angles: Optional[List[int]] = None,
        p: float = 0.5,
    ) -> None:
        self.angles = angles if angles is not None else [0, 90, 180, 270]
        self.p = p
        
        for angle in self.angles:
            if angle not in [0, 90, 180, 270]:
                raise ValueError(f"Only 0, 90, 180, 270 supported, got {angle}")
    
    def __call__(
        self,
        image: Union[Image.Image, Tensor],
        target: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Union[Image.Image, Tensor], Optional[Dict[str, Tensor]]]:
        if random.random() >= self.p:
            return image, target
        
        angle = random.choice(self.angles)
        
        if angle == 0:
            return image, target
        
        # Rotate image
        if isinstance(image, Tensor):
            k = angle // 90
            image = torch.rot90(image, k=k, dims=[1, 2])
        else:
            image = image.rotate(-angle, expand=True)
        
        # Adjust bounding boxes
        if target is not None and "boxes" in target:
            target = target.copy()
            boxes = target["boxes"].clone()
            
            cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            
            if angle == 90:
                new_cx, new_cy, new_w, new_h = 1 - cy, cx, h, w
            elif angle == 180:
                new_cx, new_cy, new_w, new_h = 1 - cx, 1 - cy, w, h
            elif angle == 270:
                new_cx, new_cy, new_w, new_h = cy, 1 - cx, h, w
            else:
                new_cx, new_cy, new_w, new_h = cx, cy, w, h
            
            target["boxes"] = torch.stack([new_cx, new_cy, new_w, new_h], dim=1)
            
            if "size" in target and angle in [90, 270]:
                size = target["size"]
                target["size"] = torch.tensor([size[1], size[0]])
        
        return image, target
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(angles={self.angles}, p={self.p})"
