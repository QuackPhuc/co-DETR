"""
Loss Functions for Object Detection.

This module provides loss functions for DETR-based object detection:
- Focal Loss: Classification with focal mechanism for class imbalance
- L1 Loss: Bounding box coordinate regression
- GIoU Loss: IoU-based loss for better box overlap optimization
- DIoU Loss: Distance-IoU for faster convergence

All losses support multiple reduction modes and integration with DETR pipeline.
"""

from codetr.models.losses.focal_loss import (
    FocalLoss,
    SigmoidFocalLoss,
)
from codetr.models.losses.l1_loss import (
    L1Loss,
    SmoothL1Loss,
    l1_loss,
    smooth_l1_loss,
)
from codetr.models.losses.giou_loss import (
    GIoULoss,
    DIoULoss,
    giou_loss,
    diou_loss,
)

__all__ = [
    # Focal Loss
    "FocalLoss",
    "SigmoidFocalLoss",
    # L1 Loss
    "L1Loss",
    "SmoothL1Loss",
    "l1_loss",
    "smooth_l1_loss",
    # GIoU Loss
    "GIoULoss",
    "DIoULoss",
    "giou_loss",
    "diou_loss",
]
