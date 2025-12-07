"""Mosaic augmentation for object detection."""

from typing import Callable, Dict, Optional, Tuple
import random

import torch
from torch import Tensor
import torchvision.transforms.functional as F


class Mosaic:
    """Mosaic augmentation (YOLO-style).
    
    Combines four images into a 2x2 grid with a random center.
    
    Args:
        output_size: Size of output image (height, width).
        center_range: Range for mosaic center as fraction of size.
        fill_value: Fill value for empty regions.
        p: Probability of applying Mosaic.
    """
    
    def __init__(
        self,
        output_size: Tuple[int, int] = (640, 640),
        center_range: Tuple[float, float] = (0.25, 0.75),
        fill_value: float = 0.5,
        p: float = 0.5,
    ) -> None:
        self.output_size = output_size
        self.center_range = center_range
        self.fill_value = fill_value
        self.p = p
    
    def _place_image(self, canvas, image, region):
        dx1, dy1, dx2, dy2 = region
        dh, dw = dy2 - dy1, dx2 - dx1
        if dh <= 0 or dw <= 0:
            return canvas
        canvas[:, dy1:dy2, dx1:dx2] = F.resize(image, size=(dh, dw))
        return canvas
    
    def _transform_boxes(self, boxes, src_size, dst_region, out_size):
        if len(boxes) == 0:
            return boxes
        
        dx1, dy1, dx2, dy2 = dst_region
        dh, dw = dy2 - dy1, dx2 - dx1
        out_h, out_w = out_size
        
        cx = boxes[:, 0] * dw + dx1
        cy = boxes[:, 1] * dh + dy1
        w = boxes[:, 2] * dw
        h = boxes[:, 3] * dh
        
        return torch.stack([
            cx / out_w,
            cy / out_h,
            w / out_w,
            h / out_h,
        ], dim=1).clamp(0, 1)
    
    def __call__(
        self,
        image: Tensor,
        target: Optional[Dict[str, Tensor]],
        get_sample: Callable[[], Tuple[Tensor, Dict[str, Tensor]]],
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if random.random() >= self.p:
            return image, target
        
        out_h, out_w = self.output_size
        c = image.shape[0]
        
        canvas = torch.full((c, out_h, out_w), self.fill_value, dtype=image.dtype)
        
        cx = int(random.uniform(*self.center_range) * out_w)
        cy = int(random.uniform(*self.center_range) * out_h)
        
        regions = [
            (0, 0, cx, cy),
            (cx, 0, out_w, cy),
            (0, cy, cx, out_h),
            (cx, cy, out_w, out_h),
        ]
        
        images = [image]
        targets = [target]
        for _ in range(3):
            img, tgt = get_sample()
            images.append(img)
            targets.append(tgt)
        
        all_boxes, all_labels = [], []
        
        for img, tgt, region in zip(images, targets, regions):
            _, src_h, src_w = img.shape
            canvas = self._place_image(canvas, img, region)
            
            if tgt is not None and "boxes" in tgt and len(tgt["boxes"]) > 0:
                boxes = self._transform_boxes(tgt["boxes"], (src_h, src_w), region, (out_h, out_w))
                all_boxes.append(boxes)
                all_labels.append(tgt.get("labels", torch.zeros(len(boxes), dtype=torch.long)))
        
        if all_boxes:
            mosaic_boxes = torch.cat(all_boxes, dim=0)
            mosaic_labels = torch.cat(all_labels, dim=0)
            valid = (mosaic_boxes[:, 2] > 0.005) & (mosaic_boxes[:, 3] > 0.005)
            mosaic_boxes = mosaic_boxes[valid]
            mosaic_labels = mosaic_labels[valid]
        else:
            mosaic_boxes = torch.zeros(0, 4)
            mosaic_labels = torch.zeros(0, dtype=torch.long)
        
        return canvas, {
            "boxes": mosaic_boxes,
            "labels": mosaic_labels,
            "size": torch.tensor([out_h, out_w]),
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(output_size={self.output_size}, p={self.p})"
