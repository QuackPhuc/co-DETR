"""
L1 Loss for Bounding Box Regression.

This module implements standard L1 and Smooth L1 losses for bounding box
regression in object detection tasks. These losses are commonly used in
DETR-based architectures for predicting box coordinates.

Example:
    >>> import torch
    >>> from codetr.models.losses.l1_loss import L1Loss, SmoothL1Loss
    >>>
    >>> criterion = L1Loss(reduction='mean')
    >>> pred_boxes = torch.rand(8, 100, 4)  # (batch, queries, 4)
    >>> target_boxes = torch.rand(8, 100, 4)
    >>> loss = criterion(pred_boxes, target_boxes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class L1Loss(nn.Module):
    """
    Standard L1 (Mean Absolute Error) loss for bounding box regression.

    Computes the element-wise absolute difference between predictions and targets.
    L1 loss is less sensitive to outliers compared to L2 loss and provides
    stable gradients for bounding box regression.

    Loss formula: L1(x, y) = |x - y|

    Attributes:
        reduction (str): Specifies reduction to apply: 'none', 'mean', 'sum'.
        eps (float): Small epsilon for numerical stability.

    Example:
        >>> criterion = L1Loss(reduction='mean')
        >>> predictions = torch.tensor([[0.5, 0.5, 0.3, 0.3]])
        >>> targets = torch.tensor([[0.6, 0.4, 0.3, 0.4]])
        >>> loss = criterion(predictions, targets)
        >>> print(f"L1 Loss: {loss.item():.4f}")
    """

    def __init__(self, reduction: str = "mean", eps: float = 1e-8) -> None:
        """
        Initialize L1 Loss.

        Args:
            reduction: Reduction method - 'none', 'mean', or 'sum'.
                'none': no reduction, returns per-element loss.
                'mean': returns mean of all elements.
                'sum': returns sum of all elements.
            eps: Small value for numerical stability.

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
        Compute L1 loss between predictions and targets.

        Args:
            inputs: Predicted values of shape (batch_size, num_queries, 4).
                Box format is typically [cx, cy, w, h] normalized to [0, 1].
            targets: Ground truth values of same shape as inputs.
            mask: Optional binary mask of shape (batch_size, num_queries).
                If provided, only computes loss on positions where mask=True.
                Useful for ignoring padding or invalid boxes.
            num_boxes: Optional normalization factor. If provided and reduction='mean',
                loss is divided by num_boxes instead of number of elements.

        Returns:
            Scalar loss tensor if reduction is 'mean' or 'sum'.
            Tensor of same shape as input if reduction is 'none'.

        Shape:
            - inputs: (N, Q, 4) where N=batch, Q=queries
            - targets: (N, Q, 4)
            - mask: (N, Q) optional
            - output: scalar or (N, Q, 4)

        Example:
            >>> criterion = L1Loss()
            >>> pred = torch.rand(2, 100, 4)
            >>> target = torch.rand(2, 100, 4)
            >>> mask = torch.randint(0, 2, (2, 100)).bool()
            >>> loss = criterion(pred, target, mask=mask, num_boxes=50)
        """
        # Compute absolute difference
        loss = torch.abs(inputs - targets)  # (N, Q, 4)

        # Apply mask if provided
        if mask is not None:
            # Expand mask to match box dimensions: (N, Q) -> (N, Q, 1)
            mask_expanded = mask.unsqueeze(-1)
            loss = loss * mask_expanded.float()

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
                # Normalize by number of valid elements
                num_valid = mask.sum() * inputs.shape[-1]  # mask count * 4 coords
                if num_valid > 0:
                    return loss.sum() / num_valid
                else:
                    return loss.sum() * 0.0  # Return zero if no valid boxes
            else:
                return loss.mean()

    def __repr__(self) -> str:
        """Return string representation."""
        return f"{self.__class__.__name__}(reduction={self.reduction})"


class SmoothL1Loss(nn.Module):
    """
    Smooth L1 Loss (Huber Loss) for bounding box regression.

    Smooth L1 combines the benefits of L1 and L2 losses:
    - For small errors (|x| < beta): uses L2 loss (smooth gradients)
    - For large errors (|x| >= beta): uses L1 loss (robust to outliers)

    Loss formula:
        smooth_l1(x) = 0.5 * x^2 / beta,        if |x| < beta
                       |x| - 0.5 * beta,        otherwise

    Attributes:
        beta (float): Threshold for switching between L1 and L2.
            Smaller beta makes the loss more similar to L1.
        reduction (str): Reduction method.

    Example:
        >>> criterion = SmoothL1Loss(beta=1.0, reduction='mean')
        >>> pred = torch.randn(2, 100, 4, requires_grad=True)
        >>> target = torch.randn(2, 100, 4)
        >>> loss = criterion(pred, target)
        >>> loss.backward()
    """

    def __init__(self, beta: float = 1.0, reduction: str = "mean") -> None:
        """
        Initialize Smooth L1 Loss.

        Args:
            beta: Threshold for switching between L1 and L2 behavior.
                Typical values: 1.0 for general use, 1/9 for Faster R-CNN.
            reduction: Reduction method - 'none', 'mean', or 'sum'.

        Raises:
            ValueError: If beta <= 0 or reduction invalid.
        """
        super().__init__()

        if beta <= 0:
            raise ValueError(f"Beta must be positive, got {beta}")
        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(
                f"Reduction must be 'none', 'mean', or 'sum', got {reduction}"
            )

        self.beta = beta
        self.reduction = reduction

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        num_boxes: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute Smooth L1 loss.

        Args:
            inputs: Predicted values (N, Q, 4).
            targets: Ground truth values (N, Q, 4).
            mask: Optional binary mask (N, Q).
            num_boxes: Optional normalization factor.

        Returns:
            Scalar or per-element loss tensor.

        Example:
            >>> criterion = SmoothL1Loss(beta=1.0)
            >>> pred = torch.tensor([[0.5, 0.5, 0.3, 0.3]])
            >>> target = torch.tensor([[0.6, 0.4, 0.3, 0.4]])
            >>> loss = criterion(pred, target)
        """
        # Compute absolute difference
        diff = torch.abs(inputs - targets)

        # Compute smooth L1 loss
        # For |diff| < beta: 0.5 * diff^2 / beta
        # For |diff| >= beta: |diff| - 0.5 * beta
        loss = torch.where(
            diff < self.beta, 0.5 * diff**2 / self.beta, diff - 0.5 * self.beta
        )

        # Apply mask if provided
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)
            loss = loss * mask_expanded.float()

        # Apply reduction
        if self.reduction == "none":
            return loss
        elif self.reduction == "sum":
            return loss.sum()
        else:  # mean
            if num_boxes is not None and num_boxes > 0:
                return loss.sum() / num_boxes
            elif mask is not None:
                num_valid = mask.sum() * inputs.shape[-1]
                if num_valid > 0:
                    return loss.sum() / num_valid
                else:
                    return loss.sum() * 0.0
            else:
                return loss.mean()

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"{self.__class__.__name__}("
            f"beta={self.beta}, "
            f"reduction={self.reduction})"
        )


def l1_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = "mean",
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Functional interface for L1 loss.

    Args:
        inputs: Predicted values.
        targets: Ground truth values.
        reduction: 'none', 'mean', or 'sum'.
        mask: Optional binary mask.

    Returns:
        Loss tensor.

    Example:
        >>> pred = torch.rand(2, 100, 4)
        >>> target = torch.rand(2, 100, 4)
        >>> loss = l1_loss(pred, target, reduction='mean')
    """
    criterion = L1Loss(reduction=reduction)
    return criterion(inputs, targets, mask=mask)


def smooth_l1_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    beta: float = 1.0,
    reduction: str = "mean",
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Functional interface for Smooth L1 loss.

    Args:
        inputs: Predicted values.
        targets: Ground truth values.
        beta: Smoothness threshold.
        reduction: 'none', 'mean', or 'sum'.
        mask: Optional binary mask.

    Returns:
        Loss tensor.

    Example:
        >>> pred = torch.rand(2, 100, 4)
        >>> target = torch.rand(2, 100, 4)
        >>> loss = smooth_l1_loss(pred, target, beta=1.0)
    """
    criterion = SmoothL1Loss(beta=beta, reduction=reduction)
    return criterion(inputs, targets, mask=mask)
