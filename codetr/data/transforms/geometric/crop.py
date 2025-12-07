"""Crop transforms for object detection."""

from typing import Dict, Optional, Tuple, Union
import random

import torch
from torch import Tensor
from PIL import Image
import torchvision.transforms.functional as F


class RandomCrop:
    """Random crop with bounding box handling.
    
    Args:
        crop_size: Target (height, width). If None, uses scales.
        scales: Random scale range relative to image size.
        min_box_visibility: Minimum visible fraction to keep box.
    """
    
    def __init__(
        self,
        crop_size: Optional[Tuple[int, int]] = None,
        scales: Tuple[float, float] = (0.5, 1.0),
        min_box_visibility: float = 0.3,
    ) -> None:
        self.crop_size = crop_size
        self.scales = scales
        self.min_box_visibility = min_box_visibility
    
    def __call__(
        self,
        image: Union[Image.Image, Tensor],
        target: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Union[Image.Image, Tensor], Optional[Dict[str, Tensor]]]:
        # Get image dimensions
        if isinstance(image, Tensor):
            _, img_h, img_w = image.shape
        else:
            img_w, img_h = image.size
        
        # Determine crop size
        if self.crop_size is not None:
            crop_h, crop_w = self.crop_size
        else:
            scale = random.uniform(self.scales[0], self.scales[1])
            crop_h = int(img_h * scale)
            crop_w = int(img_w * scale)
        
        crop_h = min(crop_h, img_h)
        crop_w = min(crop_w, img_w)
        
        # Random crop position
        top = random.randint(0, img_h - crop_h)
        left = random.randint(0, img_w - crop_w)
        
        # Crop image
        if isinstance(image, Tensor):
            image = image[:, top:top + crop_h, left:left + crop_w]
        else:
            image = F.crop(image, top, left, crop_h, crop_w)
        
        # Transform boxes
        if target is not None and "boxes" in target and len(target["boxes"]) > 0:
            target = target.copy()
            boxes = target["boxes"].clone()
            labels = target["labels"].clone() if "labels" in target else None
            
            # Store original areas
            original_areas = boxes[:, 2] * boxes[:, 3]
            
            # Convert to absolute xyxy
            cx, cy, w, h = boxes.unbind(-1)
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
            
            # Translate to crop coordinates
            x1 = x1 - left
            y1 = y1 - top
            x2 = x2 - left
            y2 = y2 - top
            
            # Clip to crop region
            x1 = x1.clamp(min=0, max=crop_w)
            y1 = y1.clamp(min=0, max=crop_h)
            x2 = x2.clamp(min=0, max=crop_w)
            y2 = y2.clamp(min=0, max=crop_h)
            
            # Calculate visibility
            clipped_w = x2 - x1
            clipped_h = y2 - y1
            clipped_areas = clipped_w * clipped_h
            abs_original = original_areas * img_w * img_h
            visibility = clipped_areas / (abs_original + 1e-6)
            
            keep = (visibility >= self.min_box_visibility) & (clipped_areas > 0)
            
            if keep.any():
                x1, y1, x2, y2 = x1[keep], y1[keep], x2[keep], y2[keep]
                
                # Normalize and convert to cxcywh
                new_cx = (x1 + x2) / 2 / crop_w
                new_cy = (y1 + y2) / 2 / crop_h
                new_w = (x2 - x1) / crop_w
                new_h = (y2 - y1) / crop_h
                
                target["boxes"] = torch.stack([new_cx, new_cy, new_w, new_h], dim=1).clamp(0, 1)
                if labels is not None:
                    target["labels"] = labels[keep]
            else:
                target["boxes"] = torch.zeros((0, 4), dtype=boxes.dtype)
                if labels is not None:
                    target["labels"] = torch.zeros((0,), dtype=labels.dtype)
            
            if "size" in target:
                target["size"] = torch.tensor([crop_h, crop_w])
        
        return image, target
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(scales={self.scales})"
