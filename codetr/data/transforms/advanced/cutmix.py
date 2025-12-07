"""CutMix augmentation for object detection."""

from typing import Callable, Dict, Optional, Tuple
import random
import math

import torch
from torch import Tensor
import torchvision.transforms.functional as F


class CutMix:
    """CutMix augmentation for object detection.
    
    Replaces a rectangular patch with a patch from another image.
    
    Args:
        alpha: Beta distribution parameter for cut ratio.
        min_cut_ratio: Minimum ratio of image to cut.
        p: Probability of applying CutMix.
    """
    
    def __init__(self, alpha: float = 1.0, min_cut_ratio: float = 0.2, p: float = 0.5) -> None:
        self.alpha = alpha
        self.min_cut_ratio = min_cut_ratio
        self.p = p
    
    def _get_cut_bbox(self, h, w, lam):
        cut_ratio = math.sqrt(1.0 - lam)
        cut_w, cut_h = int(w * cut_ratio), int(h * cut_ratio)
        cx, cy = random.randint(0, w), random.randint(0, h)
        
        return (
            max(0, cx - cut_w // 2),
            max(0, cy - cut_h // 2),
            min(w, cx + cut_w // 2),
            min(h, cy + cut_h // 2),
        )
    
    def _filter_boxes(self, boxes, labels, region, inside):
        if len(boxes) == 0:
            return boxes, labels
        
        rx1, ry1, rx2, ry2 = region
        cx, cy = boxes[:, 0], boxes[:, 1]
        center_inside = (cx >= rx1) & (cx <= rx2) & (cy >= ry1) & (cy <= ry2)
        
        if inside:
            keep = center_inside
        else:
            keep = ~center_inside
        
        return boxes[keep], labels[keep]
    
    def __call__(
        self,
        image: Tensor,
        target: Optional[Dict[str, Tensor]],
        get_sample: Callable[[], Tuple[Tensor, Dict[str, Tensor]]],
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if random.random() >= self.p:
            return image, target
        
        image2, target2 = get_sample()
        
        _, h, w = image.shape
        _, h2, w2 = image2.shape
        if h != h2 or w != w2:
            image2 = F.resize(image2, size=(h, w))
        
        lam = random.betavariate(self.alpha, self.alpha)
        lam = max(self.min_cut_ratio, min(1 - self.min_cut_ratio, lam))
        
        x1, y1, x2, y2 = self._get_cut_bbox(h, w, lam)
        
        mixed = image.clone()
        mixed[:, y1:y2, x1:x2] = image2[:, y1:y2, x1:x2]
        
        if target is not None and target2 is not None:
            cut_region = (x1 / w, y1 / h, x2 / w, y2 / h)
            
            boxes1 = target.get("boxes", torch.zeros(0, 4))
            boxes2 = target2.get("boxes", torch.zeros(0, 4))
            labels1 = target.get("labels", torch.zeros(0, dtype=torch.long))
            labels2 = target2.get("labels", torch.zeros(0, dtype=torch.long))
            
            b1, l1 = self._filter_boxes(boxes1, labels1, cut_region, inside=False)
            b2, l2 = self._filter_boxes(boxes2, labels2, cut_region, inside=True)
            
            mixed_target = {
                "boxes": torch.cat([b1, b2], dim=0),
                "labels": torch.cat([l1, l2], dim=0),
            }
            for k in target:
                if k not in ["boxes", "labels"]:
                    mixed_target[k] = target[k]
        else:
            mixed_target = target
        
        return mixed, mixed_target
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(alpha={self.alpha}, p={self.p})"
