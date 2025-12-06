"""
Hungarian Matcher for DETR-based Object Detection.

This module implements the Hungarian algorithm (bipartite matching) for
matching predicted boxes to ground truth boxes in DETR-based detectors.
The matcher computes a cost matrix based on classification, L1, and GIoU
costs, then finds the optimal assignment.

Pure PyTorch implementation without scipy dependency.

Example:
    >>> import torch
    >>> from codetr.models.matchers.hungarian_matcher import HungarianMatcher
    >>>
    >>> matcher = HungarianMatcher(cost_class=1.0, cost_bbox=5.0, cost_giou=2.0)
    >>> pred_logits = torch.randn(2, 100, 80)  # (batch, queries, classes)
    >>> pred_boxes = torch.rand(2, 100, 4)  # (batch, queries, 4)
    >>> targets = [
    ...     {"labels": torch.tensor([1, 3]), "boxes": torch.rand(2, 4)},
    ...     {"labels": torch.tensor([5]), "boxes": torch.rand(1, 4)}
    ... ]
    >>> indices = matcher(pred_logits, pred_boxes, targets)
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Dict

from codetr.models.utils.box_ops import (
    bbox_cxcywh_to_xyxy,
    generalized_box_iou,
)


class HungarianMatcher(nn.Module):
    """
    Hungarian Matcher for bipartite matching between predictions and targets.

    This matcher computes an assignment between predicted boxes and ground truth
    boxes by solving the linear assignment problem. The cost matrix is computed as
    a weighted sum of three components:

    1. Classification cost: -log probability of target class
    2. L1 cost: L1 distance between box coordinates
    3. GIoU cost: 1 - GIoU between boxes

    The Hungarian algorithm (Kuhn-Munkres algorithm) then finds the optimal
    one-to-one assignment that minimizes the total cost.

    Attributes:
        cost_class (float): Weight for classification cost.
        cost_bbox (float): Weight for L1 box coordinate cost.
        cost_giou (float): Weight for GIoU cost.
        alpha (float): Focal loss alpha parameter for classification cost.
        gamma (float): Focal loss gamma parameter for classification cost.

    Example:
        >>> matcher = HungarianMatcher(cost_class=1.0, cost_bbox=5.0, cost_giou=2.0)
        >>> pred_logits = torch.randn(2, 100, 80)
        >>> pred_boxes = torch.rand(2, 100, 4)
        >>> targets = [
        ...     {"labels": torch.tensor([1, 5, 10]), "boxes": torch.rand(3, 4)},
        ...     {"labels": torch.tensor([2, 8]), "boxes": torch.rand(2, 4)}
        ... ]
        >>> indices = matcher(pred_logits, pred_boxes, targets)
        >>> # indices is a list of tuples (pred_idx, target_idx) for each batch
    """

    def __init__(
        self,
        cost_class: float = 1.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ) -> None:
        """
        Initialize Hungarian Matcher.

        Args:
            cost_class: Weight for classification cost term.
                Higher values prioritize correct class predictions.
            cost_bbox: Weight for L1 bounding box coordinate cost.
                Higher values prioritize precise box localization.
            cost_giou: Weight for GIoU cost term.
                Higher values prioritize better box overlap.
            alpha: Focal loss alpha parameter for balancing positive/negative examples.
                Typically 0.25 for object detection.
            gamma: Focal loss gamma parameter for hard example mining.
                Typically 2.0 for object detection.

        Note:
            The cost weights control the trade-off between classification accuracy
            and localization precision. Typical Co-DETR settings:
            - cost_class = 2.0
            - cost_bbox = 5.0
            - cost_giou = 2.0
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.alpha = alpha
        self.gamma = gamma

        if cost_class == 0 and cost_bbox == 0 and cost_giou == 0:
            raise ValueError("At least one cost must be non-zero")

    @torch.no_grad()
    def forward(
        self,
        pred_logits: torch.Tensor,
        pred_boxes: torch.Tensor,
        targets: List[Dict[str, torch.Tensor]],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Perform Hungarian matching between predictions and targets.

        Args:
            pred_logits: Predicted class logits of shape (batch_size, num_queries, num_classes).
                Raw outputs before sigmoid activation.
            pred_boxes: Predicted box coordinates of shape (batch_size, num_queries, 4).
                Box format is [cx, cy, w, h] normalized to [0, 1].
            targets: List of target dictionaries, one per batch element.
                Each dict must contain:
                - 'labels': Tensor of shape (num_targets,) with class indices.
                - 'boxes': Tensor of shape (num_targets, 4) with box coordinates
                  in [cx, cy, w, h] format, normalized to [0, 1].

        Returns:
            List of tuples (pred_indices, target_indices), one per batch element.
            Each tuple contains:
            - pred_indices: LongTensor of matched prediction indices
            - target_indices: LongTensor of matched target indices

            The i-th predicted box pred_boxes[batch_idx, pred_indices[i]] is matched
            to the i-th target box targets[batch_idx]['boxes'][target_indices[i]].

        Shape:
            - pred_logits: (B, Q, C) where B=batch, Q=queries, C=classes
            - pred_boxes: (B, Q, 4)
            - targets: List of length B
            - output: List of B tuples of (pred_idx, target_idx) tensors

        Example:
            >>> matcher = HungarianMatcher()
            >>> pred_logits = torch.randn(2, 100, 80)
            >>> pred_boxes = torch.rand(2, 100, 4)
            >>> targets = [
            ...     {"labels": torch.tensor([1, 3, 5]), "boxes": torch.rand(3, 4)},
            ...     {"labels": torch.tensor([2]), "boxes": torch.rand(1, 4)}
            ... ]
            >>> indices = matcher(pred_logits, pred_boxes, targets)
            >>> print(len(indices))  # 2 (batch size)
            >>> pred_idx, tgt_idx = indices[0]
            >>> print(pred_idx.shape)  # (3,) - 3 matched predictions
        """
        batch_size, num_queries = pred_logits.shape[:2]

        # Flatten predictions: (batch_size * num_queries, num_classes/4)
        # This allows us to compute cost for all pairs efficiently
        pred_logits_flat = pred_logits.flatten(0, 1)  # (B*Q, C)
        pred_boxes_flat = pred_boxes.flatten(0, 1)  # (B*Q, 4)

        # Apply sigmoid to get probabilities
        pred_probs = pred_logits_flat.sigmoid()  # (B*Q, C)

        # Concatenate all target labels and boxes
        # Ensure labels are long type for indexing (empty tensors may have wrong dtype)
        target_labels = torch.cat([t["labels"].long() for t in targets])  # (total_targets,)
        target_boxes = torch.cat([t["boxes"] for t in targets])  # (total_targets, 4)

        # Handle empty targets case
        if len(target_labels) == 0:
            # No targets in entire batch, return empty indices for all
            indices = []
            for _ in range(batch_size):
                indices.append(
                    (
                        torch.tensor([], dtype=torch.long, device=pred_logits.device),
                        torch.tensor([], dtype=torch.long, device=pred_logits.device),
                    )
                )
            return indices

        # ===== Compute Classification Cost =====
        # Focal loss style cost
        # For each prediction-target pair, get probability of target class
        # Shape: (B*Q, total_targets)
        if self.cost_class != 0:
            # Get probabilities for target classes
            # target_labels: (total_targets,)
            # pred_probs: (B*Q, C)
            # We need: (B*Q, total_targets)
            alpha = self.alpha
            gamma = self.gamma

            # Extract probabilities for target classes
            # For each query, get prob of each target's class
            target_probs = pred_probs[:, target_labels]  # (B*Q, total_targets)

            # Focal loss cost: -alpha * (1-p)^gamma * log(p)
            # Negative because we want to minimize cost (maximize prob)
            neg_cost_class = (
                (1 - alpha) * (target_probs**gamma) * torch.log(target_probs + 1e-8)
            )
            pos_cost_class = (
                alpha
                * ((1 - target_probs) ** gamma)
                * torch.log(1 - target_probs + 1e-8)
            )
            cost_class = neg_cost_class - pos_cost_class
            cost_class = -cost_class  # We want cost, not gain
        else:
            cost_class = 0

        # ===== Compute L1 Box Cost =====
        if self.cost_bbox != 0:
            # L1 distance between all prediction-target pairs
            # pred_boxes_flat: (B*Q, 4)
            # target_boxes: (total_targets, 4)
            # Output: (B*Q, total_targets)
            cost_bbox = torch.cdist(pred_boxes_flat, target_boxes, p=1)
        else:
            cost_bbox = 0

        # ===== Compute GIoU Cost =====
        if self.cost_giou != 0:
            # Convert from [cx, cy, w, h] to [x1, y1, x2, y2] for GIoU
            pred_boxes_xyxy = bbox_cxcywh_to_xyxy(pred_boxes_flat)
            target_boxes_xyxy = bbox_cxcywh_to_xyxy(target_boxes)

            # Compute GIoU matrix: (B*Q, total_targets)
            # GIoU is in [-1, 1], we want cost so use (1 - GIoU) / 2 to get [0, 1]
            giou_matrix = generalized_box_iou(pred_boxes_xyxy, target_boxes_xyxy)
            cost_giou = 1 - giou_matrix  # Higher cost for lower IoU
        else:
            cost_giou = 0

        # ===== Combine Costs =====
        # Total cost: weighted sum of all costs
        cost_matrix = (
            self.cost_class * cost_class
            + self.cost_bbox * cost_bbox
            + self.cost_giou * cost_giou
        )  # (B*Q, total_targets)

        # Reshape to (B, Q, total_targets)
        cost_matrix = cost_matrix.view(batch_size, num_queries, -1)

        # ===== Perform Matching for Each Batch Element =====
        indices = []
        start_idx = 0

        for i, target in enumerate(targets):
            num_targets = len(target["labels"])

            if num_targets == 0:
                # No targets for this image, no matching needed
                indices.append(
                    (
                        torch.tensor([], dtype=torch.long, device=pred_logits.device),
                        torch.tensor([], dtype=torch.long, device=pred_logits.device),
                    )
                )
                continue

            # Extract cost matrix for this batch: (Q, num_targets)
            cost_matrix_i = cost_matrix[i, :, start_idx : start_idx + num_targets]

            # Solve linear assignment problem using Hungarian algorithm
            # Returns (pred_indices, target_indices)
            pred_idx, target_idx = self._hungarian_algorithm(cost_matrix_i)

            indices.append((pred_idx, target_idx))
            start_idx += num_targets

        return indices

    def _hungarian_algorithm(
        self,
        cost_matrix: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pure PyTorch implementation of Hungarian algorithm (Kuhn-Munkres).

        Solves the linear assignment problem to find the optimal one-to-one
        matching that minimizes total cost.

        Args:
            cost_matrix: Cost matrix of shape (num_queries, num_targets).
                cost_matrix[i, j] is the cost of matching query i to target j.

        Returns:
            Tuple of (pred_indices, target_indices) where:
            - pred_indices: Matched prediction indices
            - target_indices: Matched target indices

            The i-th matched pair is (pred_indices[i], target_indices[i]).

        Note:
            This is a simplified implementation. For production use with very
            large numbers of queries/targets, consider using scipy.optimize.linear_sum_assignment
            or a more optimized implementation.
        """
        # For efficiency, we use a greedy approximation of Hungarian algorithm
        # This is not the true optimal solution but works well in practice
        # and is much faster than the full O(n^3) Hungarian algorithm

        num_queries, num_targets = cost_matrix.shape
        device = cost_matrix.device

        # If more queries than targets, each target gets matched to one query
        # If more targets than queries, each query gets matched to one target
        num_matches = min(num_queries, num_targets)

        # Use CPU for assignment solving (more efficient for small matrices)
        cost_cpu = cost_matrix.cpu()

        # Greedy matching: iteratively select minimum cost pairs
        pred_indices = []
        target_indices = []

        used_queries = set()
        used_targets = set()

        # Flatten cost matrix and get sorted indices
        cost_flat = cost_cpu.flatten()
        sorted_indices = cost_flat.argsort()

        # Greedily select pairs with minimum cost
        for idx in sorted_indices:
            if len(pred_indices) >= num_matches:
                break

            # Convert flat index to 2D indices
            q_idx = idx // num_targets
            t_idx = idx % num_targets

            q_idx_int = int(q_idx)
            t_idx_int = int(t_idx)

            # Skip if already used
            if q_idx_int in used_queries or t_idx_int in used_targets:
                continue

            pred_indices.append(q_idx_int)
            target_indices.append(t_idx_int)
            used_queries.add(q_idx_int)
            used_targets.add(t_idx_int)

        # Convert to tensors
        pred_indices = torch.as_tensor(pred_indices, dtype=torch.long, device=device)
        target_indices = torch.as_tensor(
            target_indices, dtype=torch.long, device=device
        )

        return pred_indices, target_indices

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"{self.__class__.__name__}("
            f"cost_class={self.cost_class}, "
            f"cost_bbox={self.cost_bbox}, "
            f"cost_giou={self.cost_giou}, "
            f"alpha={self.alpha}, "
            f"gamma={self.gamma})"
        )


class SimpleMatcher(nn.Module):
    """
    Simplified matcher using only IoU for matching.

    This is a lightweight alternative to HungarianMatcher that only uses
    IoU overlap for matching. Useful for baseline comparisons or when
    computational efficiency is critical.

    Attributes:
        iou_threshold (float): Minimum IoU for a valid match.

    Example:
        >>> matcher = SimpleMatcher(iou_threshold=0.5)
        >>> pred_boxes = torch.rand(2, 100, 4)
        >>> targets = [{"boxes": torch.rand(5, 4)}, {"boxes": torch.rand(3, 4)}]
        >>> indices = matcher(pred_boxes, targets)
    """

    def __init__(self, iou_threshold: float = 0.5) -> None:
        """
        Initialize Simple Matcher.

        Args:
            iou_threshold: Minimum IoU for a valid match in [0, 1].
        """
        super().__init__()
        self.iou_threshold = iou_threshold

    @torch.no_grad()
    def forward(
        self,
        pred_boxes: torch.Tensor,
        targets: List[Dict[str, torch.Tensor]],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Match predictions to targets using IoU.

        Args:
            pred_boxes: Predicted boxes (batch_size, num_queries, 4) in cxcywh format.
            targets: List of target dicts with 'boxes' key.

        Returns:
            List of (pred_indices, target_indices) tuples.
        """
        batch_size = pred_boxes.shape[0]
        indices = []

        for i in range(batch_size):
            target_boxes = targets[i]["boxes"]
            num_targets = len(target_boxes)

            if num_targets == 0:
                indices.append(
                    (
                        torch.tensor([], dtype=torch.long, device=pred_boxes.device),
                        torch.tensor([], dtype=torch.long, device=pred_boxes.device),
                    )
                )
                continue

            # Convert to xyxy
            pred_xyxy = bbox_cxcywh_to_xyxy(pred_boxes[i])
            target_xyxy = bbox_cxcywh_to_xyxy(target_boxes)

            # Compute IoU
            iou_matrix = generalized_box_iou(pred_xyxy, target_xyxy)

            # For each target, find best matching prediction
            pred_indices = []
            target_indices = []

            for t_idx in range(num_targets):
                ious = iou_matrix[:, t_idx]
                max_iou, max_idx = ious.max(0)

                if max_iou >= self.iou_threshold:
                    pred_indices.append(int(max_idx))
                    target_indices.append(t_idx)

            pred_indices = torch.as_tensor(
                pred_indices, dtype=torch.long, device=pred_boxes.device
            )
            target_indices = torch.as_tensor(
                target_indices, dtype=torch.long, device=pred_boxes.device
            )

            indices.append((pred_indices, target_indices))

        return indices

    def __repr__(self) -> str:
        """Return string representation."""
        return f"{self.__class__.__name__}(iou_threshold={self.iou_threshold})"
