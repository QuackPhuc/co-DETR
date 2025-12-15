"""Affine transforms for object detection."""

from typing import Dict, Optional, Tuple, Union
import random
import math

import torch
from torch import Tensor
from PIL import Image
import torchvision.transforms.functional as F


class RandomAffine:
    """Random affine transformation.

    Args:
        degrees: Rotation angle range (-degrees, +degrees).
        translate: Max translation as fraction of image size.
        scale: Scale range (min_scale, max_scale).
        shear: Shear range in degrees.
        fill: Fill value for areas outside image.
    """

    def __init__(
        self,
        degrees: float = 10.0,
        translate: Optional[Tuple[float, float]] = None,
        scale: Optional[Tuple[float, float]] = None,
        shear: Optional[float] = None,
        fill: float = 0.0,
    ) -> None:
        self.degrees = degrees
        self.translate = translate if translate is not None else (0.1, 0.1)
        self.scale = scale if scale is not None else (0.9, 1.1)
        self.shear = shear if shear is not None else 5.0
        self.fill = fill

    def _get_params(self, w: int, h: int):
        angle = random.uniform(-self.degrees, self.degrees)
        tx = random.uniform(-self.translate[0] * w, self.translate[0] * w)
        ty = random.uniform(-self.translate[1] * h, self.translate[1] * h)
        scale = random.uniform(self.scale[0], self.scale[1])
        shear_x = random.uniform(-self.shear, self.shear)
        shear_y = random.uniform(-self.shear, self.shear)
        return angle, (int(tx), int(ty)), scale, (shear_x, shear_y)

    def _transform_boxes(self, boxes, w, h, angle, translate, scale, shear):
        if len(boxes) == 0:
            return boxes

        cx, cy, bw, bh = boxes.unbind(-1)

        # To absolute corners
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h

        corners = torch.stack(
            [
                torch.stack([x1, y1], dim=1),
                torch.stack([x2, y1], dim=1),
                torch.stack([x2, y2], dim=1),
                torch.stack([x1, y2], dim=1),
            ],
            dim=1,
        )  # [N, 4, 2]

        # Transform corners
        center_x, center_y = w / 2, h / 2
        angle_rad = math.radians(angle)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)

        m00 = scale * cos_a
        m01 = scale * (-sin_a + cos_a * math.tan(math.radians(shear[0])))
        m10 = scale * (sin_a + cos_a * math.tan(math.radians(shear[1])))
        m11 = scale * cos_a

        # Center, transform, uncenter
        corners_c = corners.clone()
        corners_c[:, :, 0] -= center_x
        corners_c[:, :, 1] -= center_y

        new_x = (
            m00 * corners_c[:, :, 0]
            + m01 * corners_c[:, :, 1]
            + center_x
            + translate[0]
        )
        new_y = (
            m10 * corners_c[:, :, 0]
            + m11 * corners_c[:, :, 1]
            + center_y
            + translate[1]
        )

        # Bounding box of transformed corners
        x_min = new_x.min(dim=1)[0].clamp(0, w)
        x_max = new_x.max(dim=1)[0].clamp(0, w)
        y_min = new_y.min(dim=1)[0].clamp(0, h)
        y_max = new_y.max(dim=1)[0].clamp(0, h)

        # Back to normalized cxcywh
        new_cx = (x_min + x_max) / 2 / w
        new_cy = (y_min + y_max) / 2 / h
        new_w = (x_max - x_min) / w
        new_h = (y_max - y_min) / h

        return torch.stack([new_cx, new_cy, new_w, new_h], dim=1)

    def __call__(
        self,
        image: Union[Image.Image, Tensor],
        target: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Union[Image.Image, Tensor], Optional[Dict[str, Tensor]]]:
        if isinstance(image, Tensor):
            _, h, w = image.shape
        else:
            w, h = image.size

        angle, translate, scale, shear = self._get_params(w, h)

        # Apply to image
        if isinstance(image, Tensor):
            pil_img = F.to_pil_image(image)
            transformed = F.affine(
                pil_img, angle, translate, scale, list(shear), fill=self.fill
            )
            image = F.to_tensor(transformed)
        else:
            image = F.affine(
                image, angle, translate, scale, list(shear), fill=self.fill
            )

        # Transform boxes
        if target is not None and "boxes" in target and len(target["boxes"]) > 0:
            target = target.copy()
            boxes = self._transform_boxes(
                target["boxes"].clone(), w, h, angle, translate, scale, shear
            )

            valid = (boxes[:, 2] > 0.01) & (boxes[:, 3] > 0.01)
            target["boxes"] = boxes[valid] if valid.any() else torch.zeros((0, 4))
            if "labels" in target:
                target["labels"] = (
                    target["labels"][valid]
                    if valid.any()
                    else torch.zeros((0,), dtype=torch.long)
                )

        return image, target

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(degrees={self.degrees})"
