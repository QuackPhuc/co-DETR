"""
Generalized IoU Loss for Bounding Box Regression.

This module implements GIoU (Generalized Intersection over Union) loss for
bounding box regression. GIoU addresses limitations of standard IoU by
penalizing boxes based on their relative position and size.

Paper: Generalized Intersection over Union (Rezatofighi et al., 2019)
https://arxiv.org/abs/1902.09630

Example:
    >>> import torch
    >>> from codetr.models.losses.giou_loss import GIoULoss
    >>>
    >>> criterion = GIoULoss(reduction='mean')
    >>> pred_boxes = torch.rand(8, 100, 4)  # (batch, queries, 4)
    >>> target_boxes = torch.rand(8, 100, 4)
    >>> loss = criterion(pred_boxes, target_boxes)
"""

import torch
import torch.nn as nn
from typing import Optional

from codetr.models.utils.box_ops import generalized_box_iou, box_area


class GIoULoss(nn.Module):
    """
    Generalized IoU Loss for bounding box regression.

    GIoU extends standard IoU by incorporating the area of the smallest enclosing
    box. This provides meaningful gradients even when boxes don't overlap and
    better handles the relative position of boxes.

    GIoU formula:
        GIoU = IoU - |C - (A ∪ B)| / |C|
    where:
        - IoU is standard Intersection over Union
        - C is the smallest enclosing box
        - A, B are the predicted and ground truth boxes

    Loss is defined as: L_GIoU = 1 - GIoU

    Attributes:
        reduction (str): Specifies reduction to apply: 'none', 'mean', 'sum'.
        eps (float): Small epsilon for numerical stability.

    Example:
        >>> criterion = GIoULoss(reduction='mean')
        >>> # Box format: [x1, y1, x2, y2] in absolute coordinates
        >>> pred = torch.tensor([[10, 10, 50, 50], [20, 20, 60, 60]])
        >>> target = torch.tensor([[15, 15, 55, 55], [25, 25, 65, 65]])
        >>> loss = criterion(pred, target)
        >>> print(f"GIoU Loss: {loss.item():.4f}")
    """

    def __init__(self, reduction: str = "mean", eps: float = 1e-7) -> None:
        """
        Initialize GIoU Loss.

        Args:
            reduction: Reduction method - 'none', 'mean', or 'sum'.
                'none': no reduction, returns per-box loss.
                'mean': returns mean loss over all boxes.
                'sum': returns sum of all losses.
            eps: Small value for numerical stability in division.

        Raises:
            ValueError: If reduction not in ['none', 'mean', 'sum'].
        """
        super().__init__()

        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(
                f"Reduction must be 'none', 'mean', or 'sum', got {reduction}"
            )

        self.reduction = reduction
        self.eps = eps

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        num_boxes: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute GIoU loss between predicted and target boxes.

        Args:
            inputs: Predicted bounding boxes of shape (N, 4) or (batch, num_queries, 4).
                Box format should be [x1, y1, x2, y2] in absolute coordinates.
                If boxes are in [cx, cy, w, h] format, convert them first using
                box_ops.bbox_cxcywh_to_xyxy().
            targets: Ground truth boxes of same shape as inputs.
                Must use the same coordinate format as inputs.
            mask: Optional binary mask of shape (batch, num_queries).
                If provided, only computes loss on positions where mask=True.
                Useful for ignoring padding or invalid boxes.
            num_boxes: Optional normalization factor. If provided and reduction='mean',
                loss is divided by num_boxes instead of number of elements.

        Returns:
            Scalar loss tensor if reduction is 'mean' or 'sum'.
            Tensor of shape (N,) or (batch, num_queries) if reduction is 'none'.

        Shape:
            - inputs: (N, 4) or (B, Q, 4) where N=num_boxes, B=batch, Q=queries
            - targets: same as inputs
            - mask: (B, Q) optional
            - output: scalar or (N,) or (B, Q)

        Note:
            Boxes must be in [x1, y1, x2, y2] format where x1 < x2 and y1 < y2.
            GIoU is invariant to box size scaling but not to coordinate format.

        Example:
            >>> criterion = GIoULoss()
            >>> # Boxes in xyxy format
            >>> pred = torch.tensor([[10, 10, 50, 50], [60, 60, 100, 100]])
            >>> target = torch.tensor([[15, 15, 55, 55], [65, 65, 105, 105]])
            >>> loss = criterion(pred, target)
            >>>
            >>> # With mask for batched input
            >>> pred = torch.rand(2, 100, 4) * 100  # (batch, queries, 4)
            >>> target = torch.rand(2, 100, 4) * 100
            >>> mask = torch.randint(0, 2, (2, 100)).bool()
            >>> loss = criterion(pred, target, mask=mask, num_boxes=50)
        """
        # Store original shape for reshaping
        original_shape = inputs.shape

        # Flatten to (N, 4) for easier processing
        if inputs.dim() == 3:
            batch_size, num_queries = inputs.shape[0], inputs.shape[1]
            inputs_flat = inputs.view(-1, 4)
            targets_flat = targets.view(-1, 4)
            if mask is not None:
                mask_flat = mask.view(-1)
        else:
            inputs_flat = inputs
            targets_flat = targets
            mask_flat = mask

        # Compute GIoU using the utility function
        # Returns tensor of shape (N,) with values in [-1, 1]
        giou = generalized_box_iou(inputs_flat, targets_flat)

        # GIoU loss: 1 - GIoU
        # Range: [0, 2] where 0 is perfect match, 2 is worst case
        loss = 1.0 - torch.diag(giou)

        # Reshape back if necessary
        if len(original_shape) == 3:
            loss = loss.view(batch_size, num_queries)

        # Apply mask if provided
        if mask is not None:
            if len(original_shape) == 3:
                loss = loss * mask.float()
            else:
                loss = loss * mask_flat.float()

        # Apply reduction
        if self.reduction == "none":
            return loss
        elif self.reduction == "sum":
            return loss.sum()
        else:  # mean
            if num_boxes is not None and num_boxes > 0:
                # Normalize by number of boxes
                return loss.sum() / num_boxes
            elif mask is not None:
                # Normalize by number of valid boxes
                num_valid = mask.sum() if mask.dim() == 2 else mask_flat.sum()
                if num_valid > 0:
                    return loss.sum() / num_valid
                else:
                    return loss.sum() * 0.0  # Return zero if no valid boxes
            else:
                return loss.mean()

    def __repr__(self) -> str:
        """Return string representation."""
        return f"{self.__class__.__name__}(reduction={self.reduction})"


class DIoULoss(nn.Module):
    """
    Distance IoU Loss for bounding box regression.

    DIoU (Distance-IoU) extends GIoU by directly minimizing the normalized
    distance between predicted and target box centers. This provides faster
    convergence than GIoU.

    DIoU formula:
        DIoU = IoU - (d^2 / c^2)
    where:
        - d is the Euclidean distance between box centers
        - c is the diagonal length of the smallest enclosing box

    Paper: Distance-IoU Loss (Zheng et al., 2020)
    https://arxiv.org/abs/1911.08287

    Attributes:
        reduction (str): Reduction method.
        eps (float): Numerical stability constant.

    Example:
        >>> criterion = DIoULoss(reduction='mean')
        >>> pred = torch.tensor([[10, 10, 50, 50]])
        >>> target = torch.tensor([[15, 15, 55, 55]])
        >>> loss = criterion(pred, target)
    """

    def __init__(self, reduction: str = "mean", eps: float = 1e-7) -> None:
        """
        Initialize DIoU Loss.

        Args:
            reduction: 'none', 'mean', or 'sum'.
            eps: Small value for numerical stability.
        """
        super().__init__()

        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(
                f"Reduction must be 'none', 'mean', or 'sum', got {reduction}"
            )

        self.reduction = reduction
        self.eps = eps

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        num_boxes: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute DIoU loss.

        Args:
            inputs: Predicted boxes (N, 4) or (B, Q, 4) in [x1, y1, x2, y2] format.
            targets: Ground truth boxes, same shape as inputs.
            mask: Optional binary mask (B, Q).
            num_boxes: Optional normalization factor.

        Returns:
            Scalar or per-box loss tensor.
        """
        # Store original shape
        original_shape = inputs.shape

        # Flatten to (N, 4)
        if inputs.dim() == 3:
            batch_size, num_queries = inputs.shape[0], inputs.shape[1]
            inputs_flat = inputs.view(-1, 4)
            targets_flat = targets.view(-1, 4)
            if mask is not None:
                mask_flat = mask.view(-1)
        else:
            inputs_flat = inputs
            targets_flat = targets
            mask_flat = mask

        # Compute standard IoU using GIoU function
        giou_matrix = generalized_box_iou(inputs_flat, targets_flat)
        iou = torch.diag(giou_matrix)

        # Compute center points
        pred_centers = (inputs_flat[:, :2] + inputs_flat[:, 2:]) / 2
        target_centers = (targets_flat[:, :2] + targets_flat[:, 2:]) / 2

        # Distance between centers (squared)
        center_distance_sq = torch.sum((pred_centers - target_centers) ** 2, dim=1)

        # Compute diagonal of smallest enclosing box
        # Enclosing box coordinates
        enclose_x1 = torch.min(inputs_flat[:, 0], targets_flat[:, 0])
        enclose_y1 = torch.min(inputs_flat[:, 1], targets_flat[:, 1])
        enclose_x2 = torch.max(inputs_flat[:, 2], targets_flat[:, 2])
        enclose_y2 = torch.max(inputs_flat[:, 3], targets_flat[:, 3])

        # Diagonal length squared
        diagonal_sq = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2

        # DIoU = IoU - (d^2 / c^2)
        diou = iou - (center_distance_sq / (diagonal_sq + self.eps))

        # DIoU loss: 1 - DIoU
        loss = 1.0 - diou

        # Reshape if necessary
        if len(original_shape) == 3:
            loss = loss.view(batch_size, num_queries)

        # Apply mask
        if mask is not None:
            if len(original_shape) == 3:
                loss = loss * mask.float()
            else:
                loss = loss * mask_flat.float()

        # Apply reduction
        if self.reduction == "none":
            return loss
        elif self.reduction == "sum":
            return loss.sum()
        else:  # mean
            if num_boxes is not None and num_boxes > 0:
                return loss.sum() / num_boxes
            elif mask is not None:
                num_valid = mask.sum() if mask.dim() == 2 else mask_flat.sum()
                if num_valid > 0:
                    return loss.sum() / num_valid
                else:
                    return loss.sum() * 0.0
            else:
                return loss.mean()

    def __repr__(self) -> str:
        """Return string representation."""
        return f"{self.__class__.__name__}(reduction={self.reduction})"


def giou_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = "mean",
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Functional interface for GIoU loss.

    Args:
        inputs: Predicted boxes in [x1, y1, x2, y2] format.
        targets: Ground truth boxes in same format.
        reduction: 'none', 'mean', or 'sum'.
        mask: Optional binary mask.

    Returns:
        Loss tensor.

    Example:
        >>> pred = torch.rand(100, 4) * 100
        >>> target = torch.rand(100, 4) * 100
        >>> loss = giou_loss(pred, target, reduction='mean')
    """
    criterion = GIoULoss(reduction=reduction)
    return criterion(inputs, targets, mask=mask)


def diou_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = "mean",
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Functional interface for DIoU loss.

    Args:
        inputs: Predicted boxes in [x1, y1, x2, y2] format.
        targets: Ground truth boxes in same format.
        reduction: 'none', 'mean', or 'sum'.
        mask: Optional binary mask.

    Returns:
        Loss tensor.

    Example:
        >>> pred = torch.rand(100, 4) * 100
        >>> target = torch.rand(100, 4) * 100
        >>> loss = diou_loss(pred, target, reduction='mean')
    """
    criterion = DIoULoss(reduction=reduction)
    return criterion(inputs, targets, mask=mask)
