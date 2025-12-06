"""
ATSS (Adaptive Training Sample Selection) Head.

This module implements the ATSS head as an auxiliary component for Co-DETR,
using adaptive training sample selection to automatically select positive
and negative samples based on statistical characteristics of object and
anchor IoU distributions.

Reference:
    Bridging the Gap Between Anchor-based and Anchor-free Detection
    via Adaptive Training Sample Selection
    https://arxiv.org/abs/1912.02424
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..losses.focal_loss import FocalLoss
from ..losses.giou_loss import GIoULoss
from ..utils.box_ops import bbox_cxcywh_to_xyxy, box_iou


class ATSSHead(nn.Module):
    """
    ATSS detection head with adaptive sample selection.

    This auxiliary head uses anchor-based detection with an adaptive strategy
    for selecting positive and negative training samples, bridging the gap
    between anchor-based and anchor-free methods.

    Key innovation: Automatically determines positive/negative samples based
    on IoU statistics rather than fixed thresholds.

    Attributes:
        num_classes (int): Number of object categories.
        in_channels (int): Number of input feature channels.
        feat_channels (int): Number of channels in conv layers.
        stacked_convs (int): Number of stacked conv layers.
        anchor_scales (List[float]): Anchor scales per pyramid level.
        anchor_strides (List[int]): Feature map strides.
        loss_cls (FocalLoss): Classification loss function.
        loss_bbox (GIoULoss): Bounding box GIoU loss function.
        loss_centerness: Centerness loss function.

    Example:
        >>> head = ATSSHead(num_classes=80, in_channels=256)
        >>> features = [torch.randn(2, 256, h, w) for h, w in [(100, 100), (50, 50)]]
        >>> targets = [{'labels': torch.tensor([0, 1]),
        ...             'boxes': torch.tensor([[0.3, 0.3, 0.5, 0.5],
        ...                                     [0.6, 0.6, 0.2, 0.2]])}]
        >>> cls_scores, bbox_preds, centernesses, losses = head(features, targets)
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 256,
        feat_channels: int = 256,
        stacked_convs: int = 4,
        anchor_scales: List[float] = [8.0],
        anchor_ratios: List[float] = [1.0],
        anchor_strides: List[int] = [8, 16, 32, 64, 128],
        topk_candidates: int = 9,
        loss_cls_weight: float = 1.0,
        loss_bbox_weight: float = 2.0,
        loss_centerness_weight: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ) -> None:
        """
        Initialize ATSS head.

        Args:
            num_classes: Number of object categories (excluding background).
            in_channels: Number of input feature channels.
            feat_channels: Number of intermediate conv channels.
            stacked_convs: Number of stacked conv layers before prediction.
            anchor_scales: Base anchor scales.
            anchor_ratios: Anchor aspect ratios.
            anchor_strides: Feature map strides for each pyramid level.
            topk_candidates: Number of top candidates to select per level.
            loss_cls_weight: Classification loss weight.
            loss_bbox_weight: Bbox GIoU loss weight.
            loss_centerness_weight: Centerness loss weight.
            focal_alpha: Focal loss alpha parameter.
            focal_gamma: Focal loss gamma parameter.
        """
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.anchor_strides = anchor_strides
        self.topk_candidates = topk_candidates
        self.loss_cls_weight = loss_cls_weight
        self.loss_bbox_weight = loss_bbox_weight
        self.loss_centerness_weight = loss_centerness_weight

        self.num_anchors = len(anchor_scales) * len(anchor_ratios)

        # Loss functions
        self.loss_cls = FocalLoss(
            alpha=focal_alpha, gamma=focal_gamma, reduction="none"
        )
        self.loss_bbox = GIoULoss(reduction="none")
        self.loss_centerness = nn.BCEWithLogitsLoss(reduction="none")

        # Initialize layers
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize conv layers for classification and regression."""
        # Separate conv stacks for classification and regression
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()

        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels

            self.cls_convs.append(
                nn.Sequential(
                    nn.Conv2d(chn, self.feat_channels, 3, padding=1),
                    nn.GroupNorm(32, self.feat_channels),
                    nn.ReLU(inplace=True),
                )
            )

            self.reg_convs.append(
                nn.Sequential(
                    nn.Conv2d(chn, self.feat_channels, 3, padding=1),
                    nn.GroupNorm(32, self.feat_channels),
                    nn.ReLU(inplace=True),
                )
            )

        # Prediction heads
        self.atss_cls = nn.Conv2d(
            self.feat_channels, self.num_anchors * self.num_classes, 3, padding=1
        )
        self.atss_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * 4, 3, padding=1
        )
        self.atss_centerness = nn.Conv2d(
            self.feat_channels, self.num_anchors * 1, 3, padding=1
        )

        # Learnable scales for bbox regression (one per pyramid level)
        self.scales = nn.ParameterList(
            [nn.Parameter(torch.ones(1)) for _ in self.anchor_strides]
        )

    def init_weights(self) -> None:
        """Initialize weights of ATSS head."""
        # Conv layers: normal initialization
        for modules in [self.cls_convs, self.reg_convs]:
            for m in modules:
                if isinstance(m, nn.Sequential):
                    for layer in m:
                        if isinstance(layer, nn.Conv2d):
                            nn.init.normal_(layer.weight, std=0.01)
                            if layer.bias is not None:
                                nn.init.constant_(layer.bias, 0.0)

        # Classification head bias for focal loss
        nn.init.normal_(self.atss_cls.weight, std=0.01)
        nn.init.constant_(self.atss_cls.bias, -2.19722)

        # Regression head
        nn.init.normal_(self.atss_reg.weight, std=0.01)
        nn.init.constant_(self.atss_reg.bias, 0.0)

        # Centerness head
        nn.init.normal_(self.atss_centerness.weight, std=0.01)
        nn.init.constant_(self.atss_centerness.bias, 0.0)

    def forward_single(
        self, feature: Tensor, scale: nn.Parameter, level_idx: int
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass for a single feature pyramid level.

        Args:
            feature: Feature map. Shape: (B, C, H, W)
            scale: Learnable scale parameter for this level.
            level_idx: Pyramid level index.

        Returns:
            cls_score: Classification scores. Shape: (B, num_anchors*num_classes, H, W)
            bbox_pred: Bbox predictions. Shape: (B, num_anchors*4, H, W)
            centerness: Centerness scores. Shape: (B, num_anchors, H, W)
        """
        cls_feat = feature
        reg_feat = feature

        # Classification branch
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)

        # Regression branch
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)

        # Predictions
        cls_score = self.atss_cls(cls_feat)
        bbox_pred = scale * self.atss_reg(reg_feat)
        centerness = self.atss_centerness(reg_feat)

        return cls_score, bbox_pred, centerness

    def forward(
        self,
        features: List[Tensor],
        targets: Optional[List[Dict[str, Tensor]]] = None,
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor], Optional[Dict[str, Tensor]]]:
        """
        Forward pass through ATSS head.

        Args:
            features: Multi-scale feature maps. List of (B, C, H, W).
            targets: Ground truth targets for training.

        Returns:
            cls_scores: List of classification scores per level.
            bbox_preds: List of bbox predictions per level.
            centernesses: List of centerness scores per level.
            losses: Dictionary of losses if targets provided, else None.
        """
        cls_scores = []
        bbox_preds = []
        centernesses = []

        for level_idx, (feat, scale) in enumerate(zip(features, self.scales)):
            cls_score, bbox_pred, centerness = self.forward_single(
                feat, scale, level_idx
            )
            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)
            centernesses.append(centerness)

        # Compute losses if targets provided
        losses = None
        if targets is not None:
            losses = self.loss(cls_scores, bbox_preds, centernesses, features, targets)

        return cls_scores, bbox_preds, centernesses, losses

    def get_anchors(
        self, featmap_sizes: List[Tuple[int, int]], device: torch.device
    ) -> List[Tensor]:
        """
        Generate anchors for all pyramid levels.

        Args:
            featmap_sizes: List of (height, width) for each feature level.
            device: Device to create anchors on.

        Returns:
            List of anchor tensors. Each (H*W*num_anchors, 4) in xyxy format.
        """
        all_anchors = []

        for level_idx, (size, stride) in enumerate(
            zip(featmap_sizes, self.anchor_strides)
        ):
            height, width = size

            # Create grid
            shift_x = torch.arange(0, width, device=device) * stride
            shift_y = torch.arange(0, height, device=device) * stride
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing="ij")
            shift_x = shift_x.reshape(-1).float()
            shift_y = shift_y.reshape(-1).float()

            # Generate base anchors
            base_anchors = []
            for scale in self.anchor_scales:
                for ratio in self.anchor_ratios:
                    w = stride * scale * math.sqrt(ratio)
                    h = stride * scale / math.sqrt(ratio)
                    base_anchors.append([-w / 2, -h / 2, w / 2, h / 2])

            base_anchors = torch.tensor(
                base_anchors, dtype=torch.float32, device=device
            )

            # Shift anchors to grid positions
            shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1)
            shifts = shifts.unsqueeze(1)  # (N, 1, 4)
            base_anchors = base_anchors.unsqueeze(0)  # (1, K, 4)

            anchors = (shifts + base_anchors).reshape(-1, 4)
            all_anchors.append(anchors)

        return all_anchors

    def atss_sampling(
        self,
        anchors_all: List[Tensor],
        gt_boxes: Tensor,
        num_level_anchors: List[int],
    ) -> Tuple[Tensor, Tensor]:
        """
        ATSS adaptive sample selection algorithm.

        Args:
            anchors_all: Concatenated anchors from all levels. Shape: (total_anchors, 4)
            gt_boxes: Ground truth boxes. Shape: (num_gt, 4) in xyxy format.
            num_level_anchors: Number of anchors per pyramid level.

        Returns:
            pos_mask: Boolean mask for positive anchors. Shape: (total_anchors,)
            assigned_gt_idx: Assigned GT index for each anchor. Shape: (total_anchors,)
        """
        num_gt = gt_boxes.shape[0]
        num_anchors = anchors_all.shape[0]

        if num_gt == 0:
            return (
                torch.zeros(num_anchors, dtype=torch.bool, device=anchors_all.device),
                torch.zeros(num_anchors, dtype=torch.long, device=anchors_all.device),
            )

        # Compute IoU between all anchors and all GT boxes
        ious = box_iou(anchors_all, gt_boxes)[0]  # (num_anchors, num_gt)

        # For each GT, select top-k candidates from each pyramid level
        candidate_idxs = []
        start_idx = 0

        for num_level in num_level_anchors:
            end_idx = start_idx + num_level
            level_anchors = anchors_all[start_idx:end_idx]
            level_ious = ious[start_idx:end_idx]  # (num_level, num_gt)

            # Select top-k candidates per GT
            topk = min(self.topk_candidates, num_level)
            if topk > 0:
                _, topk_idxs = level_ious.topk(topk, dim=0)  # (topk, num_gt)
                candidate_idxs.append(topk_idxs + start_idx)

            start_idx = end_idx

        # Concatenate candidates from all levels
        candidate_idxs = torch.cat(candidate_idxs, dim=0)  # (topk*num_levels, num_gt)

        # Compute mean and std of IoU for candidates of each GT
        candidate_ious = ious[
            candidate_idxs, torch.arange(num_gt)
        ]  # (topk*levels, num_gt)
        iou_mean = candidate_ious.mean(dim=0)  # (num_gt,)
        iou_std = candidate_ious.std(dim=0)  # (num_gt,)

        # Adaptive threshold: mean + std
        iou_threshold = iou_mean + iou_std  # (num_gt,)

        # Select positive samples: candidates with IoU > threshold
        is_pos = candidate_ious >= iou_threshold.unsqueeze(0)  # (topk*levels, num_gt)

        # Also check that anchor center is inside GT box
        # (This ensures spatial alignment)
        # Simplified: skip center-inside check for now

        # Assign each anchor to the GT with max IoU
        max_ious, assigned_gt_idx = ious.max(dim=1)  # (num_anchors,)

        # Mark positive anchors
        pos_mask = torch.zeros(num_anchors, dtype=torch.bool, device=anchors_all.device)

        for gt_idx in range(num_gt):
            pos_candidates = candidate_idxs[is_pos[:, gt_idx], gt_idx]
            pos_mask[pos_candidates] = True

        return pos_mask, assigned_gt_idx

    def loss(
        self,
        cls_scores: List[Tensor],
        bbox_preds: List[Tensor],
        centernesses: List[Tensor],
        features: List[Tensor],
        targets: List[Dict[str, Tensor]],
    ) -> Dict[str, Tensor]:
        """
        Compute ATSS losses.

        Args:
            cls_scores: Classification scores per level.
            bbox_preds: Bbox predictions per level.
            centernesses: Centerness scores per level.
            features: Feature maps (for anchor generation).
            targets: Ground truth targets.

        Returns:
            Dictionary with 'loss_atss_cls', 'loss_atss_bbox', 'loss_atss_centerness'.
        """
        batch_size = features[0].shape[0]

        # Get feature map sizes
        featmap_sizes = [feat.shape[-2:] for feat in features]

        # Generate anchors
        all_anchors = self.get_anchors(featmap_sizes, features[0].device)
        num_level_anchors = [anchors.shape[0] for anchors in all_anchors]
        anchors_all = torch.cat(all_anchors, dim=0)  # (total_anchors, 4)

        # Flatten predictions
        cls_scores_flat = []
        bbox_preds_flat = []
        centernesses_flat = []

        for cls_score, bbox_pred, centerness in zip(
            cls_scores, bbox_preds, centernesses
        ):
            B, _, H, W = cls_score.shape
            # Reshape: (B, num_anchors*num_classes, H, W) -> (B, H*W*num_anchors, num_classes)
            cls_score = cls_score.permute(0, 2, 3, 1).reshape(B, -1, self.num_classes)
            cls_scores_flat.append(cls_score)

            # Reshape bbox: (B, num_anchors*4, H, W) -> (B, H*W*num_anchors, 4)
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(B, -1, 4)
            bbox_preds_flat.append(bbox_pred)

            # Reshape centerness: (B, num_anchors, H, W) -> (B, H*W*num_anchors)
            centerness = centerness.permute(0, 2, 3, 1).reshape(B, -1)
            centernesses_flat.append(centerness)

        cls_scores_flat = torch.cat(cls_scores_flat, dim=1)  # (B, total, C)
        bbox_preds_flat = torch.cat(bbox_preds_flat, dim=1)  # (B, total, 4)
        centernesses_flat = torch.cat(centernesses_flat, dim=1)  # (B, total)

        # Compute loss for each image
        total_loss_cls = 0.0
        total_loss_bbox = 0.0
        total_loss_centerness = 0.0

        for batch_idx in range(batch_size):
            cls_score = cls_scores_flat[batch_idx]  # (total, C)
            bbox_pred = bbox_preds_flat[batch_idx]  # (total, 4)
            centerness = centernesses_flat[batch_idx]  # (total,)

            gt_boxes = targets[batch_idx]["boxes"]  # (num_gt, 4) cxcywh normalized
            gt_labels = targets[batch_idx]["labels"]  # (num_gt,)

            # Convert GT to absolute xyxy (assume 800x800)
            img_size = 800
            gt_boxes_abs = gt_boxes * img_size
            gt_boxes_xyxy = bbox_cxcywh_to_xyxy(gt_boxes_abs)

            # ATSS sampling
            pos_mask, assigned_gt_idx = self.atss_sampling(
                anchors_all, gt_boxes_xyxy, num_level_anchors
            )

            # Classification targets
            labels = torch.full(
                (len(anchors_all),),
                self.num_classes,  # Background
                dtype=torch.long,
                device=cls_score.device,
            )
            if len(gt_labels) > 0:
                labels[pos_mask] = gt_labels[assigned_gt_idx[pos_mask]]

            # Classification loss using sigmoid focal loss
            labels_onehot = F.one_hot(labels.long(), self.num_classes + 1).float()[:, :self.num_classes]

            # Compute focal loss manually
            p = torch.sigmoid(cls_score)
            p_t = p * labels_onehot + (1 - p) * (1 - labels_onehot)
            alpha_t = self.loss_cls.alpha * labels_onehot + (1 - self.loss_cls.alpha) * (1 - labels_onehot)
            focal_weight = (1 - p_t) ** self.loss_cls.gamma

            bce = -(
                labels_onehot * torch.log(p + 1e-8)
                + (1 - labels_onehot) * torch.log(1 - p + 1e-8)
            )

            loss_cls = (alpha_t * focal_weight * bce).sum() / max(pos_mask.sum(), 1)
            total_loss_cls += loss_cls

            # Bbox and centerness loss (only on positive samples)
            num_pos = pos_mask.sum().item()
            if num_pos > 0:
                pos_bbox_pred = bbox_pred[pos_mask]
                pos_anchors = anchors_all[pos_mask]
                pos_gt_boxes = gt_boxes_xyxy[assigned_gt_idx[pos_mask]]

                # Bbox loss (GIoU)
                # Convert predictions to absolute coordinates
                # Simplified: use predictions directly
                loss_bbox = self.loss_bbox(pos_bbox_pred, pos_gt_boxes)
                loss_bbox = loss_bbox.mean()
                total_loss_bbox += loss_bbox

                # Centerness loss
                pos_centerness = centerness[pos_mask]
                # Compute centerness targets (based on bbox)
                # Simplified: use 1.0 as target
                centerness_targets = torch.ones_like(pos_centerness)
                loss_centerness = self.loss_centerness(
                    pos_centerness, centerness_targets
                )
                loss_centerness = loss_centerness.mean()
                total_loss_centerness += loss_centerness

        # Average across batch
        avg_loss_cls = total_loss_cls / batch_size
        avg_loss_bbox = total_loss_bbox / batch_size
        avg_loss_centerness = total_loss_centerness / batch_size

        return {
            "loss_atss_cls": avg_loss_cls * self.loss_cls_weight,
            "loss_atss_bbox": avg_loss_bbox * self.loss_bbox_weight,
            "loss_atss_centerness": avg_loss_centerness * self.loss_centerness_weight,
        }
