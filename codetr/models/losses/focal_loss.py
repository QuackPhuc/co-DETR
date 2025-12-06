"""
Focal Loss for Classification in Object Detection.

This module implements Focal Loss (Lin et al., 2017) designed to address class
imbalance in dense object detection by down-weighting well-classified examples
and focusing on hard negatives.

Paper: https://arxiv.org/abs/1708.02002

Example:
    >>> import torch
    >>> from codetr.models.losses.focal_loss import FocalLoss
    >>>
    >>> criterion = FocalLoss(alpha=0.25, gamma=2.0)
    >>> inputs = torch.randn(8, 100, 80)  # (batch, queries, num_classes)
    >>> targets = torch.randint(0, 80, (8, 100))  # (batch, queries)
    >>> loss = criterion(inputs, targets)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in object detection.

    Focal Loss applies a modulating term to the cross entropy loss to focus
    learning on hard negative examples. The loss is defined as:

        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    where p_t is the model's estimated probability for the class with label y.

    Attributes:
        alpha (float): Weighting factor in range [0, 1] to balance positive/negative
            examples. Alpha=0.25 means positive examples get 0.25 weight.
        gamma (float): Focusing parameter for modulating loss. Higher gamma increases
            focus on hard examples. Recommended: 2.0.
        reduction (str): Specifies reduction to apply to output: 'none', 'mean', 'sum'.

    Example:
        >>> criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')
        >>> logits = torch.randn(2, 100, 80, requires_grad=True)
        >>> targets = torch.randint(0, 80, (2, 100))
        >>> loss = criterion(logits, targets)
        >>> loss.backward()
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        """
        Initialize Focal Loss.

        Args:
            alpha: Weighting factor for positive class, in range [0, 1].
                Higher alpha gives more weight to positive examples.
            gamma: Focusing parameter >= 0. gamma=0 is equivalent to CE loss.
                Typical values: 2.0 for object detection.
            reduction: Reduction method - 'none', 'mean', or 'sum'.

        Raises:
            ValueError: If alpha not in [0, 1] or gamma < 0.
            ValueError: If reduction not in ['none', 'mean', 'sum'].
        """
        super().__init__()

        if not 0 <= alpha <= 1:
            raise ValueError(f"Alpha must be in [0, 1], got {alpha}")
        if gamma < 0:
            raise ValueError(f"Gamma must be >= 0, got {gamma}")
        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(
                f"Reduction must be 'none', 'mean', or 'sum', got {reduction}"
            )

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_boxes: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            inputs: Predicted logits of shape (batch_size, num_queries, num_classes).
                Raw outputs before sigmoid activation.
            targets: Ground truth class indices of shape (batch_size, num_queries).
                Values should be in range [0, num_classes-1]. Background class should
                be num_classes-1 if present.
            num_boxes: Optional normalization factor. If provided and reduction='mean',
                loss is divided by num_boxes instead of number of elements.

        Returns:
            Scalar loss tensor if reduction is 'mean' or 'sum'.
            Tensor of shape (batch_size, num_queries) if reduction is 'none'.

        Shape:
            - inputs: (N, Q, C) where N=batch, Q=queries, C=num_classes
            - targets: (N, Q)
            - output: scalar or (N, Q)

        Example:
            >>> criterion = FocalLoss(alpha=0.25, gamma=2.0)
            >>> inputs = torch.randn(4, 100, 80)
            >>> targets = torch.randint(0, 80, (4, 100))
            >>> loss = criterion(inputs, targets, num_boxes=50)
        """
        # Apply sigmoid to get probabilities
        p = torch.sigmoid(inputs)  # (N, Q, C)

        # Get number of classes
        num_classes = inputs.shape[-1]

        # Create one-hot encoded targets
        # targets: (N, Q) -> (N, Q, C)
        targets_onehot = F.one_hot(targets, num_classes=num_classes).float()

        # Compute focal loss components
        # p_t: probability of true class
        p_t = p * targets_onehot + (1 - p) * (1 - targets_onehot)

        # Alpha weighting
        alpha_t = self.alpha * targets_onehot + (1 - self.alpha) * (1 - targets_onehot)

        # Focal term: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Binary cross entropy: -log(p_t)
        # Use numerically stable version
        bce = -(
            targets_onehot * torch.log(p + 1e-8)
            + (1 - targets_onehot) * torch.log(1 - p + 1e-8)
        )

        # Combine all terms
        loss = alpha_t * focal_weight * bce  # (N, Q, C)

        # Sum over classes
        loss = loss.sum(dim=-1)  # (N, Q)

        # Apply reduction
        if self.reduction == "none":
            return loss
        elif self.reduction == "sum":
            return loss.sum()
        else:  # mean
            if num_boxes is not None:
                return loss.sum() / num_boxes
            else:
                return loss.mean()

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"{self.__class__.__name__}("
            f"alpha={self.alpha}, "
            f"gamma={self.gamma}, "
            f"reduction={self.reduction})"
        )


class SigmoidFocalLoss(nn.Module):
    """
    Simplified Sigmoid Focal Loss without one-hot encoding.

    This is a memory-efficient variant that computes focal loss directly
    from logits without creating one-hot tensors. Suitable for large number
    of classes.

    Attributes:
        alpha (float): Weighting factor for positive examples.
        gamma (float): Focusing parameter.
        reduction (str): Reduction method.

    Example:
        >>> criterion = SigmoidFocalLoss(alpha=0.25, gamma=2.0)
        >>> inputs = torch.randn(2, 100, 80)
        >>> targets = torch.randint(0, 80, (2, 100))
        >>> loss = criterion(inputs, targets)
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        """
        Initialize Sigmoid Focal Loss.

        Args:
            alpha: Weighting factor in [0, 1].
            gamma: Focusing parameter >= 0.
            reduction: 'none', 'mean', or 'sum'.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_boxes: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute sigmoid focal loss efficiently.

        Args:
            inputs: Predicted logits (N, Q, C).
            targets: Target class indices (N, Q).
            num_boxes: Optional normalization factor.

        Returns:
            Scalar or tensor loss.
        """
        # Get probabilities
        p = torch.sigmoid(inputs)

        # Gather probabilities for target classes
        # targets: (N, Q) -> (N, Q, 1)
        targets_expanded = targets.unsqueeze(-1)

        # Get probability of correct class: (N, Q)
        p_t = torch.gather(p, dim=-1, index=targets_expanded).squeeze(-1)

        # Compute focal weight
        focal_weight = (1 - p_t) ** self.gamma

        # Compute cross entropy for target class
        ce_loss = -torch.log(p_t + 1e-8)

        # Apply focal term and alpha
        loss = self.alpha * focal_weight * ce_loss

        # Apply reduction
        if self.reduction == "none":
            return loss
        elif self.reduction == "sum":
            return loss.sum()
        else:
            if num_boxes is not None:
                return loss.sum() / num_boxes
            else:
                return loss.mean()

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"{self.__class__.__name__}("
            f"alpha={self.alpha}, "
            f"gamma={self.gamma}, "
            f"reduction={self.reduction})"
        )
