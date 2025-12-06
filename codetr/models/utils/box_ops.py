"""Bounding box operations and utilities.

This module provides essential bounding box operations including coordinate
transformations, IoU computation, and GIoU calculation for object detection.
"""

from typing import Tuple

import torch
from torch import Tensor


def bbox_xyxy_to_cxcywh(boxes: Tensor) -> Tensor:
    """Convert bounding boxes from (x1, y1, x2, y2) to (cx, cy, w, h) format.

    Args:
        boxes: Bounding boxes in (x1, y1, x2, y2) format.
            Shape: (N, 4) where N is the number of boxes.
            (x1, y1) is top-left corner, (x2, y2) is bottom-right corner.

    Returns:
        Bounding boxes in (cx, cy, w, h) format.
        Shape: (N, 4) where (cx, cy) is center and (w, h) is width/height.

    Examples:
        >>> boxes = torch.tensor([[0, 0, 10, 10], [5, 5, 15, 15]])
        >>> bbox_xyxy_to_cxcywh(boxes)
        tensor([[ 5.,  5., 10., 10.],
                [10., 10., 10., 10.]])
    """
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack([cx, cy, w, h], dim=-1)


def bbox_cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    """Convert bounding boxes from (cx, cy, w, h) to (x1, y1, x2, y2) format.

    Args:
        boxes: Bounding boxes in (cx, cy, w, h) format.
            Shape: (N, 4) where (cx, cy) is center and (w, h) is width/height.

    Returns:
        Bounding boxes in (x1, y1, x2, y2) format.
        Shape: (N, 4) where (x1, y1) is top-left, (x2, y2) is bottom-right.

    Examples:
        >>> boxes = torch.tensor([[5, 5, 10, 10], [10, 10, 10, 10]])
        >>> bbox_cxcywh_to_xyxy(boxes)
        tensor([[ 0.,  0., 10., 10.],
                [ 5.,  5., 15., 15.]])
    """
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_area(boxes: Tensor) -> Tensor:
    """Compute the area of bounding boxes.

    Args:
        boxes: Bounding boxes in (x1, y1, x2, y2) format.
            Shape: (N, 4).

    Returns:
        Area of each box. Shape: (N,).

    Examples:
        >>> boxes = torch.tensor([[0, 0, 10, 10], [5, 5, 15, 15]])
        >>> box_area(boxes)
        tensor([100., 100.])
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tuple[Tensor, Tensor]:
    """Compute IoU (Intersection over Union) between two sets of boxes.

    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.

    Args:
        boxes1: First set of boxes. Shape: (N, 4).
        boxes2: Second set of boxes. Shape: (M, 4).

    Returns:
        Tuple of (iou, union):
            - iou: IoU matrix. Shape: (N, M).
            - union: Union area matrix. Shape: (N, M).

    Examples:
        >>> boxes1 = torch.tensor([[0, 0, 10, 10]])
        >>> boxes2 = torch.tensor([[5, 5, 15, 15], [0, 0, 10, 10]])
        >>> iou, union = box_iou(boxes1, boxes2)
        >>> iou
        tensor([[0.1429, 1.0000]])
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    # Compute intersection
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # (N, M, 2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # (N, M, 2)

    wh = (rb - lt).clamp(min=0)  # (N, M, 2)
    inter = wh[:, :, 0] * wh[:, :, 1]  # (N, M)

    # Compute union
    union = area1[:, None] + area2 - inter

    # Compute IoU
    iou = inter / union

    return iou, union


def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Compute Generalized IoU (GIoU) between two sets of boxes.

    GIoU is an extension of IoU that provides a better gradient signal for
    non-overlapping boxes. It considers the smallest enclosing box.

    Reference: https://giou.stanford.edu/

    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.

    Args:
        boxes1: First set of boxes. Shape: (N, 4).
        boxes2: Second set of boxes. Shape: (M, 4).

    Returns:
        GIoU matrix. Shape: (N, M).
        Values range from -1 to 1, where 1 means perfect overlap.

    Examples:
        >>> boxes1 = torch.tensor([[0, 0, 10, 10]])
        >>> boxes2 = torch.tensor([[5, 5, 15, 15], [0, 0, 10, 10]])
        >>> generalized_box_iou(boxes1, boxes2)
        tensor([[-0.3571,  1.0000]])
    """
    # Compute IoU and union
    iou, union = box_iou(boxes1, boxes2)

    # Compute smallest enclosing box
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])  # (N, M, 2)
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])  # (N, M, 2)

    wh = (rb - lt).clamp(min=0)  # (N, M, 2)
    area_enclosing = wh[:, :, 0] * wh[:, :, 1]  # (N, M)

    # Compute GIoU
    giou = iou - (area_enclosing - union) / area_enclosing

    return giou


def inverse_sigmoid(x: Tensor, eps: float = 1e-5) -> Tensor:
    """Compute inverse sigmoid (logit) function.

    This function is the inverse of the sigmoid function:
        inverse_sigmoid(sigmoid(x)) ≈ x

    Args:
        x: Input tensor with values in (0, 1). Shape: any.
        eps: Small epsilon to avoid numerical instability. Default: 1e-5.

    Returns:
        Inverse sigmoid of input. Shape: same as input.

    Examples:
        >>> x = torch.tensor([0.1, 0.5, 0.9])
        >>> inverse_sigmoid(x)
        tensor([-2.1972,  0.0000,  2.1972])
        >>> torch.sigmoid(inverse_sigmoid(x))
        tensor([0.1000, 0.5000, 0.9000])
    """
    x = x.clamp(min=eps, max=1 - eps)
    return torch.log(x / (1 - x))
