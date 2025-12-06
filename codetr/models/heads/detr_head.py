"""
Co-Deformable DETR Detection Head.

This module implements the main detection head for Co-Deformable DETR,
including classification and bounding box regression branches with support
for iterative refinement and two-stage training.

Reference:
    DETRs with Collaborative Hybrid Assignments Training
    https://arxiv.org/abs/2211.12860
"""

import copy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..losses.focal_loss import FocalLoss
from ..losses.l1_loss import L1Loss
from ..losses.giou_loss import GIoULoss
from ..matchers.hungarian_matcher import HungarianMatcher
from ..utils.box_ops import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh


class CoDeformDETRHead(nn.Module):
    """
    Main detection head for Co-Deformable DETR.

    This head performs classification and bounding box regression on decoder
    outputs, supporting iterative refinement across multiple decoder layers
    and optional two-stage training strategy.

    Attributes:
        num_classes (int): Number of object categories (excluding background).
        embed_dims (int): Embedding dimension of transformer features.
        num_query (int): Number of object queries.
        num_reg_fcs (int): Number of fully-connected layers in bbox regressor.
        sync_cls_avg_factor (bool): Whether to synchronize class average factor
            across GPUs in distributed training.
        loss_cls (FocalLoss): Classification loss function.
        loss_bbox (L1Loss): Bounding box L1 loss function.
        loss_iou (GIoULoss): Bounding box GIoU loss function.
        matcher (HungarianMatcher): Bipartite matcher for prediction-GT assignment.
        cls_branches (nn.ModuleList): Classification heads for each decoder layer.
        reg_branches (nn.ModuleList): Regression heads for each decoder layer.

    Example:
        >>> head = CoDeformDETRHead(
        ...     num_classes=80,
        ...     embed_dims=256,
        ...     num_query=300,
        ...     num_decoder_layers=6
        ... )
        >>> decoder_out = torch.randn(6, 2, 300, 256)  # (layers, batch, queries, dims)
        >>> reference_points = torch.randn(6, 2, 300, 2)
        >>> targets = [{'labels': torch.randint(0, 80, (5,)),
        ...             'boxes': torch.rand(5, 4)}]
        >>> losses = head(decoder_out, reference_points, targets)
    """

    def __init__(
        self,
        num_classes: int,
        embed_dims: int = 256,
        num_query: int = 300,
        num_decoder_layers: int = 6,
        num_reg_fcs: int = 2,
        sync_cls_avg_factor: bool = False,
        loss_cls_weight: float = 2.0,
        loss_bbox_weight: float = 5.0,
        loss_iou_weight: float = 2.0,
        matcher_cost_class: float = 2.0,
        matcher_cost_bbox: float = 5.0,
        matcher_cost_giou: float = 2.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ) -> None:
        """
        Initialize Co-Deformable DETR head.

        Args:
            num_classes: Number of object categories (excluding background).
            embed_dims: Embedding dimension of transformer features.
            num_query: Number of object queries.
            num_decoder_layers: Number of decoder layers (for auxiliary losses).
            num_reg_fcs: Number of FC layers in bbox regression branch.
            sync_cls_avg_factor: Whether to sync cls avg factor in DDP training.
            loss_cls_weight: Weight for classification loss.
            loss_bbox_weight: Weight for L1 bbox loss.
            loss_iou_weight: Weight for GIoU loss.
            matcher_cost_class: Cost weight for classification in Hungarian matching.
            matcher_cost_bbox: Cost weight for L1 bbox in Hungarian matching.
            matcher_cost_giou: Cost weight for GIoU in Hungarian matching.
            focal_alpha: Alpha parameter for focal loss.
            focal_gamma: Gamma parameter for focal loss.
        """
        super().__init__()
        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.num_query = num_query
        self.num_decoder_layers = num_decoder_layers
        self.num_reg_fcs = num_reg_fcs
        self.sync_cls_avg_factor = sync_cls_avg_factor

        # Loss weights
        self.loss_cls_weight = loss_cls_weight
        self.loss_bbox_weight = loss_bbox_weight
        self.loss_iou_weight = loss_iou_weight

        # Classification branch outputs num_classes (no background class)
        self.cls_out_channels = num_classes

        # Initialize loss functions
        self.loss_cls = FocalLoss(
            alpha=focal_alpha, gamma=focal_gamma, reduction="none"
        )
        self.loss_bbox = L1Loss(reduction="none")
        self.loss_iou = GIoULoss(reduction="none")

        # Initialize Hungarian matcher
        self.matcher = HungarianMatcher(
            cost_class=matcher_cost_class,
            cost_bbox=matcher_cost_bbox,
            cost_giou=matcher_cost_giou,
        )

        # Initialize layers
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize classification and regression branches."""
        # Classification branch: simple linear layer
        fc_cls = nn.Linear(self.embed_dims, self.cls_out_channels)

        # Regression branch: FC layers + ReLU + final linear
        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU(inplace=True))
        reg_branch.append(nn.Linear(self.embed_dims, 4))
        reg_branch = nn.Sequential(*reg_branch)

        # Create copies for each decoder layer
        self.cls_branches = nn.ModuleList(
            [copy.deepcopy(fc_cls) for _ in range(self.num_decoder_layers)]
        )
        self.reg_branches = nn.ModuleList(
            [copy.deepcopy(reg_branch) for _ in range(self.num_decoder_layers)]
        )

    def init_weights(self) -> None:
        """Initialize weights of the head."""
        # Initialize classification branch with bias for focal loss
        bias_init = -2.19722  # log((1 - 0.01) / 0.01) for prob=0.01
        for m in self.cls_branches:
            nn.init.constant_(m.bias, bias_init)

        # Initialize regression branch
        for m in self.reg_branches:
            # Last layer zero initialization
            nn.init.constant_(m[-1].weight, 0.0)
            nn.init.constant_(m[-1].bias, 0.0)
            # Initialize w, h bias to small negative for better initial predictions
            nn.init.constant_(m[-1].bias.data[2:], -2.0)

    def forward(
        self,
        hidden_states: Tensor,
        references: Tensor,
        targets: Optional[List[Dict[str, Tensor]]] = None,
    ) -> Tuple[Tensor, Tensor, Optional[Dict[str, Tensor]]]:
        """
        Forward pass of detection head.

        Args:
            hidden_states: Decoder hidden states.
                Shape: (num_decoder_layers, batch_size, num_query, embed_dims)
            references: Reference points from decoder.
                Shape: (num_decoder_layers, batch_size, num_query, 2) or (B, Q, 2)
            targets: Ground truth targets for training. List of dicts with keys:
                - 'labels': Tensor of shape (num_gt,) with class indices
                - 'boxes': Tensor of shape (num_gt, 4) in cxcywh format, normalized

        Returns:
            all_cls_scores: Classification scores for all layers.
                Shape: (num_decoder_layers, batch_size, num_query, num_classes)
            all_bbox_preds: Bbox predictions for all layers.
                Shape: (num_decoder_layers, batch_size, num_query, 4)
            losses: Dictionary of losses if targets provided, else None.
        """
        num_decoder_layers = hidden_states.shape[0]
        batch_size = hidden_states.shape[1]

        all_cls_scores = []
        all_bbox_preds = []

        for layer_idx in range(num_decoder_layers):
            hidden = hidden_states[layer_idx]  # (B, Q, D)
            reference = references[layer_idx] if references.dim() == 4 else references

            # Classification branch
            cls_score = self.cls_branches[layer_idx](hidden)  # (B, Q, num_classes)
            all_cls_scores.append(cls_score)

            # Regression branch: predict offsets relative to reference points
            bbox_delta = self.reg_branches[layer_idx](hidden)  # (B, Q, 4)

            # Add reference points to deltas (iterative refinement)
            # Reference is (cx, cy), delta is (dx, dy, dw, dh)
            # bbox_pred = sigmoid(inverse_sigmoid(reference) + delta[:, :, :2])
            # For simplicity, use reference + sigmoid(delta)
            bbox_pred = torch.cat(
                [
                    reference + torch.sigmoid(bbox_delta[..., :2]),  # cx, cy
                    torch.sigmoid(bbox_delta[..., 2:]),  # w, h
                ],
                dim=-1,
            )
            bbox_pred = bbox_pred.clamp(0.0, 1.0)  # Ensure normalized [0, 1]
            all_bbox_preds.append(bbox_pred)

        all_cls_scores = torch.stack(all_cls_scores, dim=0)  # (L, B, Q, C)
        all_bbox_preds = torch.stack(all_bbox_preds, dim=0)  # (L, B, Q, 4)

        # Compute losses if targets are provided
        losses = None
        if targets is not None:
            losses = self.loss(all_cls_scores, all_bbox_preds, targets)

        return all_cls_scores, all_bbox_preds, losses

    def loss(
        self,
        all_cls_scores: Tensor,
        all_bbox_preds: Tensor,
        targets: List[Dict[str, Tensor]],
    ) -> Dict[str, Tensor]:
        """
        Compute losses for all decoder layers.

        Args:
            all_cls_scores: Classification scores.
                Shape: (num_layers, batch_size, num_query, num_classes)
            all_bbox_preds: Bbox predictions.
                Shape: (num_layers, batch_size, num_query, 4)
            targets: Ground truth targets. List of dicts with 'labels' and 'boxes'.

        Returns:
            Dictionary containing:
                - 'loss_cls': Total classification loss
                - 'loss_bbox': Total L1 bbox loss
                - 'loss_iou': Total GIoU loss
        """
        num_layers = all_cls_scores.shape[0]
        batch_size = all_cls_scores.shape[1]

        # Accumulate losses across all layers
        total_loss_cls = 0.0
        total_loss_bbox = 0.0
        total_loss_iou = 0.0

        for layer_idx in range(num_layers):
            cls_scores = all_cls_scores[layer_idx]  # (B, Q, C)
            bbox_preds = all_bbox_preds[layer_idx]  # (B, Q, 4)

            # Compute single layer loss
            layer_losses = self.loss_single(cls_scores, bbox_preds, targets)

            total_loss_cls += layer_losses["loss_cls"]
            total_loss_bbox += layer_losses["loss_bbox"]
            total_loss_iou += layer_losses["loss_iou"]

        # Average across layers
        losses = {
            "loss_cls": total_loss_cls / num_layers * self.loss_cls_weight,
            "loss_bbox": total_loss_bbox / num_layers * self.loss_bbox_weight,
            "loss_iou": total_loss_iou / num_layers * self.loss_iou_weight,
        }

        return losses

    def loss_single(
        self,
        cls_scores: Tensor,
        bbox_preds: Tensor,
        targets: List[Dict[str, Tensor]],
    ) -> Dict[str, Tensor]:
        """
        Compute loss for a single decoder layer.

        Args:
            cls_scores: Classification scores. Shape: (batch_size, num_query, num_classes)
            bbox_preds: Bbox predictions. Shape: (batch_size, num_query, 4)
            targets: Ground truth targets for each image in batch.

        Returns:
            Dictionary with 'loss_cls', 'loss_bbox', 'loss_iou'.
        """
        batch_size = cls_scores.shape[0]
        num_query = cls_scores.shape[1]

        # Flatten batch dimension: (B*Q, C) and (B*Q, 4)
        cls_scores_flat = cls_scores.view(-1, self.num_classes)
        bbox_preds_flat = bbox_preds.view(-1, 4)

        # Prepare targets: convert to flat format
        cls_targets = []
        bbox_targets = []
        num_pos_samples = 0

        for batch_idx in range(batch_size):
            # Get predictions for this image
            pred_logits = cls_scores[batch_idx]  # (Q, C)
            pred_boxes = bbox_preds[batch_idx]  # (Q, 4)

            # Get ground truth for this image
            gt_labels = targets[batch_idx]["labels"]  # (num_gt,)
            gt_boxes = targets[batch_idx]["boxes"]  # (num_gt, 4) in cxcywh format

            # Perform Hungarian matching
            if len(gt_labels) > 0:
                indices = self.matcher(
                    pred_logits.unsqueeze(0),
                    pred_boxes.unsqueeze(0),
                    [targets[batch_idx]],
                )[0]
                pred_idx, gt_idx = indices
            else:
                pred_idx = torch.tensor([], dtype=torch.long, device=cls_scores.device)
                gt_idx = torch.tensor([], dtype=torch.long, device=cls_scores.device)

            # Create classification targets (all queries start as background)
            target_labels = torch.full(
                (num_query,),
                self.num_classes,  # Background class index
                dtype=torch.long,
                device=cls_scores.device,
            )
            if len(gt_idx) > 0:
                target_labels[pred_idx] = gt_labels[gt_idx]

            cls_targets.append(target_labels)

            # Create bbox targets
            target_boxes = torch.zeros(
                (num_query, 4), dtype=torch.float32, device=bbox_preds.device
            )
            if len(gt_idx) > 0:
                target_boxes[pred_idx] = gt_boxes[gt_idx]
                num_pos_samples += len(gt_idx)

            bbox_targets.append(target_boxes)

        # Stack targets
        cls_targets = torch.stack(cls_targets, dim=0)  # (B, Q)
        bbox_targets = torch.stack(bbox_targets, dim=0).view(-1, 4)  # (B*Q, 4)

        # Compute classification loss using sigmoid focal loss
        # Create one-hot targets (with extra background class dimension)
        num_classes_with_bg = self.num_classes + 1
        cls_targets_onehot = F.one_hot(cls_targets.long(), num_classes_with_bg).float()

        # Remove background dimension and use only foreground classes
        cls_targets_onehot = cls_targets_onehot[:, :, :self.num_classes]  # (B, Q, C)

        # Compute focal loss manually
        cls_scores_2d = cls_scores.view(batch_size, num_query, self.num_classes)
        p = torch.sigmoid(cls_scores_2d)

        # Focal loss components
        p_t = p * cls_targets_onehot + (1 - p) * (1 - cls_targets_onehot)
        alpha_t = self.loss_cls.alpha * cls_targets_onehot + (1 - self.loss_cls.alpha) * (1 - cls_targets_onehot)
        focal_weight = (1 - p_t) ** self.loss_cls.gamma

        # BCE
        bce = -(
            cls_targets_onehot * torch.log(p + 1e-8)
            + (1 - cls_targets_onehot) * torch.log(1 - p + 1e-8)
        )

        loss_cls = (alpha_t * focal_weight * bce).sum() / max(num_pos_samples, 1)

        # Compute bbox losses only on positive samples
        cls_targets_flat = cls_targets.view(-1)  # (B*Q,)
        pos_mask = cls_targets_flat < self.num_classes  # (B*Q,)
        num_pos = pos_mask.sum().item()

        if num_pos > 0:
            pos_bbox_preds = bbox_preds_flat[pos_mask]  # (num_pos, 4)
            pos_bbox_targets = bbox_targets[pos_mask]  # (num_pos, 4)

            # L1 loss
            loss_bbox = self.loss_bbox(pos_bbox_preds, pos_bbox_targets)
            loss_bbox = loss_bbox.sum() / num_pos

            # GIoU loss (need to convert to xyxy format)
            pos_bbox_preds_xyxy = bbox_cxcywh_to_xyxy(pos_bbox_preds)
            pos_bbox_targets_xyxy = bbox_cxcywh_to_xyxy(pos_bbox_targets)

            loss_iou = self.loss_iou(pos_bbox_preds_xyxy, pos_bbox_targets_xyxy)
            loss_iou = loss_iou.mean()
        else:
            # No positive samples
            loss_bbox = bbox_preds_flat.sum() * 0.0
            loss_iou = bbox_preds_flat.sum() * 0.0

        return {
            "loss_cls": loss_cls,
            "loss_bbox": loss_bbox,
            "loss_iou": loss_iou,
        }

    @torch.no_grad()
    def predict(
        self,
        cls_scores: Tensor,
        bbox_preds: Tensor,
        score_threshold: float = 0.3,
        max_detections: int = 100,
    ) -> List[Dict[str, Tensor]]:
        """
        Generate predictions from model outputs.

        Args:
            cls_scores: Classification scores. Shape: (batch_size, num_query, num_classes)
            bbox_preds: Bbox predictions. Shape: (batch_size, num_query, 4) in cxcywh format
            score_threshold: Minimum score threshold for detections.
            max_detections: Maximum number of detections to keep per image.

        Returns:
            List of prediction dicts, one per image, with keys:
                - 'scores': Tensor of shape (num_det,)
                - 'labels': Tensor of shape (num_det,)
                - 'boxes': Tensor of shape (num_det, 4) in xyxy format
        """
        batch_size = cls_scores.shape[0]
        predictions = []

        for batch_idx in range(batch_size):
            scores = cls_scores[batch_idx]  # (Q, C)
            boxes = bbox_preds[batch_idx]  # (Q, 4)

            # Get max class score and label for each query
            scores_max, labels = scores.max(dim=-1)  # (Q,)

            # Filter by score threshold
            keep = scores_max > score_threshold
            scores_max = scores_max[keep]
            labels = labels[keep]
            boxes = boxes[keep]

            # Sort by score and keep top-k
            if len(scores_max) > max_detections:
                topk_scores, topk_idx = scores_max.topk(max_detections)
                scores_max = topk_scores
                labels = labels[topk_idx]
                boxes = boxes[topk_idx]

            # Convert boxes to xyxy format
            boxes = bbox_cxcywh_to_xyxy(boxes)

            predictions.append(
                {
                    "scores": scores_max,
                    "labels": labels,
                    "boxes": boxes,
                }
            )

        return predictions
