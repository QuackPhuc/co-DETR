"""MixUp augmentation for object detection."""

from typing import Callable, Dict, Optional, Tuple
import random

import torch
from torch import Tensor
import torchvision.transforms.functional as F


class MixUp:
    """MixUp augmentation for object detection.
    
    Blends two images linearly and concatenates their boxes.
    
    Args:
        alpha: Beta distribution parameter. Higher = more mixing.
        p: Probability of applying MixUp.
    """
    
    def __init__(self, alpha: float = 1.5, p: float = 0.5) -> None:
        self.alpha = alpha
        self.p = p
    
    def __call__(
        self,
        image: Tensor,
        target: Optional[Dict[str, Tensor]],
        get_sample: Callable[[], Tuple[Tensor, Dict[str, Tensor]]],
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        """
        Args:
            image: Image tensor (C, H, W).
            target: Target dict with 'boxes' and 'labels'.
            get_sample: Callable returning (image, target) from dataset.
        """
        if random.random() >= self.p:
            return image, target
        
        image2, target2 = get_sample()
        
        _, h1, w1 = image.shape
        _, h2, w2 = image2.shape
        if h1 != h2 or w1 != w2:
            image2 = F.resize(image2, size=(h1, w1))
        
        lam = random.betavariate(self.alpha, self.alpha)
        mixed = lam * image + (1 - lam) * image2
        
        if target is not None and target2 is not None:
            boxes1 = target.get("boxes", torch.zeros(0, 4))
            boxes2 = target2.get("boxes", torch.zeros(0, 4))
            labels1 = target.get("labels", torch.zeros(0, dtype=torch.long))
            labels2 = target2.get("labels", torch.zeros(0, dtype=torch.long))
            
            mixed_target = {
                "boxes": torch.cat([boxes1, boxes2], dim=0),
                "labels": torch.cat([labels1, labels2], dim=0),
                "mixup_lambda": torch.tensor(lam),
            }
            for k in target:
                if k not in ["boxes", "labels"]:
                    mixed_target[k] = target[k]
        else:
            mixed_target = target
        
        return mixed, mixed_target
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(alpha={self.alpha}, p={self.p})"
