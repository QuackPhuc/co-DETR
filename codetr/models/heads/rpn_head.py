"""
Region Proposal Network (RPN) Head.

This module implements the RPN head as an auxiliary component for Co-DETR,
generating object proposals through anchor-based detection with classification
and bounding box regression.

Reference:
    Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks
    https://arxiv.org/abs/1506.01497
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import nms

from ..losses.focal_loss import FocalLoss
from ..losses.l1_loss import SmoothL1Loss
from ..utils.box_ops import bbox_xyxy_to_cxcywh, bbox_cxcywh_to_xyxy, box_iou


class AnchorGenerator:
    """
    Generate anchors for RPN.

    Creates anchor boxes at multiple scales and aspect ratios for each
    spatial location in the feature map.

    Attributes:
        anchor_scales (List[float]): Anchor scales relative to stride.
        anchor_ratios (List[float]): Anchor aspect ratios.
        anchor_strides (List[int]): Feature map strides.

    Example:
        >>> generator = AnchorGenerator(
        ...     anchor_scales=[8, 16, 32],
        ...     anchor_ratios=[0.5, 1.0, 2.0],
        ...     anchor_strides=[8, 16, 32, 64]
        ... )
        >>> featmap_sizes = [(100, 100), (50, 50), (25, 25), (13, 13)]
        >>> anchors = generator.grid_anchors(featmap_sizes, device='cuda')
    """

    def __init__(
        self,
        anchor_scales: List[float] = [8.0, 16.0, 32.0],
        anchor_ratios: List[float] = [0.5, 1.0, 2.0],
        anchor_strides: List[int] = [8, 16, 32, 64],
    ) -> None:
        """
        Initialize anchor generator.

        Args:
            anchor_scales: Base scales for anchors (in pixels relative to stride).
            anchor_ratios: Aspect ratios for anchors (width/height).
            anchor_strides: Strides of feature maps at each pyramid level.
        """
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.anchor_strides = anchor_strides
        self.num_anchors_per_loc = len(anchor_scales) * len(anchor_ratios)

    def generate_base_anchors(
        self, stride: int, device: torch.device
    ) -> Tensor:
        """
        Generate base anchor boxes for a single pyramid level.

        Args:
            stride: Feature map stride.
            device: Device to create anchors on.

        Returns:
            Base anchors. Shape: (num_anchors, 4) in xyxy format.
        """
        base_anchors = []

        for scale in self.anchor_scales:
            for ratio in self.anchor_ratios:
                # Compute anchor width and height
                w = scale * stride * math.sqrt(ratio)
                h = scale * stride / math.sqrt(ratio)

                # Create anchor centered at origin
                x1 = -w / 2
                y1 = -h / 2
                x2 = w / 2
                y2 = h / 2

                base_anchors.append([x1, y1, x2, y2])

        return torch.tensor(base_anchors, dtype=torch.float32, device=device)

    def grid_anchors(
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

        for size, stride in zip(featmap_sizes, self.anchor_strides):
            height, width = size

            # Generate base anchors
            base_anchors = self.generate_base_anchors(stride, device)

            # Create spatial grid
            shift_x = torch.arange(0, width, device=device) * stride + stride // 2
            shift_y = torch.arange(0, height, device=device) * stride + stride // 2
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing="ij")

            # Flatten shifts
            shift_x = shift_x.reshape(-1).float()
            shift_y = shift_y.reshape(-1).float()
            shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1)

            # Generate all anchors by combining shifts and base anchors
            # shifts: (H*W, 4), base_anchors: (K, 4)
            # Result: (H*W*K, 4)
            shifts = shifts.unsqueeze(1)  # (H*W, 1, 4)
            base_anchors = base_anchors.unsqueeze(0)  # (1, K, 4)
            anchors = (shifts + base_anchors).reshape(-1, 4)

            all_anchors.append(anchors)

        return all_anchors


class RPNHead(nn.Module):
    """
    Region Proposal Network head.

    This auxiliary head generates region proposals for two-stage detectors
    through anchor-based classification (objectness) and bounding box regression.

    Attributes:
        in_channels (int): Number of input feature channels.
        feat_channels (int): Number of intermediate conv channels.
        num_anchors (int): Number of anchors per spatial location.
        anchor_generator (AnchorGenerator): Anchor generator.
        loss_cls (FocalLoss): Classification loss function.
        loss_bbox (SmoothL1Loss): Bounding box regression loss.

    Example:
        >>> head = RPNHead(in_channels=256)
        >>> features = [torch.randn(2, 256, h, w) for h, w in [(100, 100), (50, 50)]]
        >>> targets = [{'labels': torch.tensor([0]),
        ...             'boxes': torch.tensor([[0.5, 0.5, 0.3, 0.3]])}]
        >>> proposals, losses = head(features, targets)
    """

    def __init__(
        self,
        in_channels: int = 256,
        feat_channels: int = 256,
        anchor_scales: List[float] = [8.0, 16.0, 32.0],
        anchor_ratios: List[float] = [0.5, 1.0, 2.0],
        anchor_strides: List[int] = [8, 16, 32, 64],
        nms_threshold: float = 0.7,
        score_threshold: float = 0.05,
        pre_nms_top_n: int = 2000,
        post_nms_top_n: int = 1000,
        loss_cls_weight: float = 1.0,
        loss_bbox_weight: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        pos_iou_threshold: float = 0.7,
        neg_iou_threshold: float = 0.3,
    ) -> None:
        """
        Initialize RPN head.

        Args:
            in_channels: Number of input feature channels.
            feat_channels: Number of intermediate conv channels.
            anchor_scales: Anchor scales for anchor generator.
            anchor_ratios: Anchor aspect ratios.
            anchor_strides: Feature map strides.
            nms_threshold: IoU threshold for NMS.
            score_threshold: Minimum objectness score.
            pre_nms_top_n: Top-k proposals before NMS.
            post_nms_top_n: Top-k proposals after NMS.
            loss_cls_weight: Classification loss weight.
            loss_bbox_weight: Bbox regression loss weight.
            focal_alpha: Focal loss alpha parameter.
            focal_gamma: Focal loss gamma parameter.
            pos_iou_threshold: IoU threshold for positive anchors.
            neg_iou_threshold: IoU threshold for negative anchors.
        """
        super().__init__()
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.nms_threshold = nms_threshold
        self.score_threshold = score_threshold
        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.loss_cls_weight = loss_cls_weight
        self.loss_bbox_weight = loss_bbox_weight
        self.pos_iou_threshold = pos_iou_threshold
        self.neg_iou_threshold = neg_iou_threshold

        # Initialize anchor generator
        self.anchor_generator = AnchorGenerator(
            anchor_scales=anchor_scales,
            anchor_ratios=anchor_ratios,
            anchor_strides=anchor_strides,
        )
        self.num_anchors = self.anchor_generator.num_anchors_per_loc

        # Loss functions
        self.loss_cls = FocalLoss(
            alpha=focal_alpha, gamma=focal_gamma, reduction="none"
        )
        self.loss_bbox = SmoothL1Loss(beta=1.0 / 9.0, reduction="none")

        # Initialize layers
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize conv layers for RPN."""
        # Shared 3x3 conv layer
        self.rpn_conv = nn.Conv2d(
            self.in_channels, self.feat_channels, kernel_size=3, padding=1
        )

        # Classification head (objectness)
        self.rpn_cls = nn.Conv2d(
            self.feat_channels, self.num_anchors * 1, kernel_size=1
        )

        # Regression head (bbox offsets)
        self.rpn_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * 4, kernel_size=1
        )

    def init_weights(self) -> None:
        """Initialize weights of RPN head."""
        # Conv layer
        nn.init.normal_(self.rpn_conv.weight, std=0.01)
        nn.init.constant_(self.rpn_conv.bias, 0.0)

        # Classification head bias for focal loss
        nn.init.normal_(self.rpn_cls.weight, std=0.01)
        nn.init.constant_(self.rpn_cls.bias, -2.19722)

        # Regression head
        nn.init.normal_(self.rpn_reg.weight, std=0.01)
        nn.init.constant_(self.rpn_reg.bias, 0.0)

    def forward_single(self, feature: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass for a single feature pyramid level.

        Args:
            feature: Feature map. Shape: (B, C, H, W)

        Returns:
            cls_score: Objectness scores. Shape: (B, num_anchors, H, W)
            bbox_pred: Bbox predictions. Shape: (B, num_anchors*4, H, W)
        """
        # Shared conv
        x = F.relu(self.rpn_conv(feature))

        # Predictions
        cls_score = self.rpn_cls(x)
        bbox_pred = self.rpn_reg(x)

        return cls_score, bbox_pred

    def forward(
        self,
        features: List[Tensor],
        targets: Optional[List[Dict[str, Tensor]]] = None,
    ) -> Tuple[List[Tensor], Optional[Dict[str, Tensor]]]:
        """
        Forward pass through RPN head.

        Args:
            features: Multi-scale feature maps. List of (B, C, H, W).
            targets: Ground truth targets for training.

        Returns:
            proposals: List of proposal boxes per image. Each (num_proposals, 4) in xyxy.
            losses: Dictionary of losses if targets provided, else None.
        """
        batch_size = features[0].shape[0]

        # Forward through each level
        cls_scores = []
        bbox_preds = []

        for feature in features:
            cls_score, bbox_pred = self.forward_single(feature)
            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)

        # Generate anchors
        device = features[0].device
        featmap_sizes = [feat.shape[-2:] for feat in features]
        all_anchors = self.anchor_generator.grid_anchors(featmap_sizes, device)

        # Generate proposals
        proposals = self.get_proposals(
            cls_scores, bbox_preds, all_anchors, batch_size
        )

        # Compute losses if targets provided
        losses = None
        if targets is not None:
            losses = self.loss(cls_scores, bbox_preds, all_anchors, targets)

        return proposals, losses

    def decode_boxes(self, anchors: Tensor, deltas: Tensor) -> Tensor:
        """
        Decode bbox predictions from anchors and deltas.

        Args:
            anchors: Anchor boxes. Shape: (N, 4) in xyxy format.
            deltas: Predicted deltas. Shape: (N, 4) as (dx, dy, dw, dh).

        Returns:
            Decoded boxes. Shape: (N, 4) in xyxy format.
        """
        # Convert anchors to cxcywh
        anchors_cxcywh = bbox_xyxy_to_cxcywh(anchors)
        anchor_cx, anchor_cy, anchor_w, anchor_h = anchors_cxcywh.unbind(-1)

        # Apply deltas
        dx, dy, dw, dh = deltas.unbind(-1)

        pred_cx = anchor_cx + dx * anchor_w
        pred_cy = anchor_cy + dy * anchor_h
        pred_w = anchor_w * torch.exp(dw.clamp(max=math.log(1000.0 / 16.0)))
        pred_h = anchor_h * torch.exp(dh.clamp(max=math.log(1000.0 / 16.0)))

        # Convert back to xyxy
        pred_boxes = torch.stack(
            [pred_cx, pred_cy, pred_w, pred_h], dim=-1
        )
        pred_boxes_xyxy = bbox_cxcywh_to_xyxy(pred_boxes)

        return pred_boxes_xyxy

    def encode_boxes(self, anchors: Tensor, gt_boxes: Tensor) -> Tensor:
        """
        Encode ground truth boxes relative to anchors.

        Args:
            anchors: Anchor boxes. Shape: (N, 4) in xyxy format.
            gt_boxes: Ground truth boxes. Shape: (N, 4) in xyxy format.

        Returns:
            Encoded targets. Shape: (N, 4) as (dx, dy, dw, dh).
        """
        # Convert to cxcywh
        anchors_cxcywh = bbox_xyxy_to_cxcywh(anchors)
        gt_cxcywh = bbox_xyxy_to_cxcywh(gt_boxes)

        anchor_cx, anchor_cy, anchor_w, anchor_h = anchors_cxcywh.unbind(-1)
        gt_cx, gt_cy, gt_w, gt_h = gt_cxcywh.unbind(-1)

        # Compute deltas
        dx = (gt_cx - anchor_cx) / anchor_w
        dy = (gt_cy - anchor_cy) / anchor_h
        dw = torch.log(gt_w / anchor_w)
        dh = torch.log(gt_h / anchor_h)

        return torch.stack([dx, dy, dw, dh], dim=-1)

    def get_proposals(
        self,
        cls_scores: List[Tensor],
        bbox_preds: List[Tensor],
        all_anchors: List[Tensor],
        batch_size: int,
    ) -> List[Tensor]:
        """
        Generate proposals from predictions.

        Args:
            cls_scores: Classification scores per level.
            bbox_preds: Bbox predictions per level.
            all_anchors: Anchor boxes per level.
            batch_size: Batch size.

        Returns:
            List of proposal boxes per image. Each (num_proposals, 4) in xyxy.
        """
        proposals_list = []

        for batch_idx in range(batch_size):
            proposals_per_img = []
            scores_per_img = []

            for cls_score, bbox_pred, anchors in zip(
                cls_scores, bbox_preds, all_anchors
            ):
                # Extract predictions for this image
                cls_score_img = cls_score[batch_idx]  # (num_anchors, H, W)
                bbox_pred_img = bbox_pred[batch_idx]  # (num_anchors*4, H, W)

                # Reshape
                H, W = cls_score_img.shape[-2:]
                cls_score_img = cls_score_img.view(-1)  # (num_anchors*H*W,)
                bbox_pred_img = bbox_pred_img.permute(1, 2, 0).reshape(-1, 4)

                # Apply sigmoid to get objectness scores
                scores = torch.sigmoid(cls_score_img)

                # Decode boxes
                boxes = self.decode_boxes(anchors, bbox_pred_img)

                proposals_per_img.append(boxes)
                scores_per_img.append(scores)

            # Concatenate all levels
            proposals_per_img = torch.cat(proposals_per_img, dim=0)
            scores_per_img = torch.cat(scores_per_img, dim=0)

            # Filter by score threshold
            keep = scores_per_img > self.score_threshold
            proposals_per_img = proposals_per_img[keep]
            scores_per_img = scores_per_img[keep]

            # Sort by score and keep top-k before NMS
            if len(scores_per_img) > self.pre_nms_top_n:
                topk_scores, topk_idx = scores_per_img.topk(self.pre_nms_top_n)
                proposals_per_img = proposals_per_img[topk_idx]
                scores_per_img = topk_scores

            # Apply NMS
            if len(proposals_per_img) > 0:
                keep_nms = nms(proposals_per_img, scores_per_img, self.nms_threshold)
                keep_nms = keep_nms[: self.post_nms_top_n]
                proposals_per_img = proposals_per_img[keep_nms]

            proposals_list.append(proposals_per_img)

        return proposals_list

    def loss(
        self,
        cls_scores: List[Tensor],
        bbox_preds: List[Tensor],
        all_anchors: List[Tensor],
        targets: List[Dict[str, Tensor]],
    ) -> Dict[str, Tensor]:
        """
        Compute RPN losses.

        Args:
            cls_scores: Classification scores per level.
            bbox_preds: Bbox predictions per level.
            all_anchors: Anchor boxes per level.
            targets: Ground truth targets.

        Returns:
            Dictionary with 'loss_rpn_cls' and 'loss_rpn_bbox'.
        """
        batch_size = cls_scores[0].shape[0]

        # Concatenate all levels
        anchors_all = torch.cat(all_anchors, dim=0)  # (total_anchors, 4)

        # Flatten predictions
        cls_scores_flat = []
        bbox_preds_flat = []

        for cls_score, bbox_pred in zip(cls_scores, bbox_preds):
            B, _, H, W = cls_score.shape
            # Reshape: (B, num_anchors, H, W) -> (B, H*W*num_anchors)
            cls_score = cls_score.permute(0, 2, 3, 1).reshape(B, -1)
            cls_scores_flat.append(cls_score)

            # Reshape: (B, num_anchors*4, H, W) -> (B, H*W*num_anchors, 4)
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(B, -1, 4)
            bbox_preds_flat.append(bbox_pred)

        cls_scores_flat = torch.cat(cls_scores_flat, dim=1)  # (B, total)
        bbox_preds_flat = torch.cat(bbox_preds_flat, dim=1)  # (B, total, 4)

        # Compute loss for each image
        total_loss_cls = 0.0
        total_loss_bbox = 0.0

        for batch_idx in range(batch_size):
            cls_score = cls_scores_flat[batch_idx]  # (total,)
            bbox_pred = bbox_preds_flat[batch_idx]  # (total, 4)

            gt_boxes = targets[batch_idx]["boxes"]  # (num_gt, 4) cxcywh normalized
            gt_labels = targets[batch_idx]["labels"]  # (num_gt,)

            # Convert GT to absolute xyxy (assume 800x800)
            img_size = 800
            gt_boxes_abs = gt_boxes * img_size
            gt_boxes_xyxy = bbox_cxcywh_to_xyxy(gt_boxes_abs)

            # Assign anchors to GT boxes based on IoU
            if len(gt_boxes_xyxy) > 0:
                ious = box_iou(anchors_all, gt_boxes_xyxy)[0]  # (total, num_gt)
                max_ious, max_idx = ious.max(dim=1)  # (total,)
            else:
                max_ious = torch.zeros(len(anchors_all), device=anchors_all.device)
                max_idx = torch.zeros(
                    len(anchors_all), dtype=torch.long, device=anchors_all.device
                )

            # Label anchors: positive (IoU >= 0.7), negative (IoU < 0.3)
            labels = torch.zeros(len(anchors_all), device=cls_score.device)
            labels[max_ious >= self.pos_iou_threshold] = 1.0
            labels[max_ious < self.neg_iou_threshold] = 0.0

            # Ignore anchors with intermediate IoU (0.3 <= IoU < 0.7)
            ignore_mask = (max_ious >= self.neg_iou_threshold) & (
                max_ious < self.pos_iou_threshold
            )

            # Classification loss using binary cross-entropy with focal
            p = torch.sigmoid(cls_score)
            p_t = p * labels + (1 - p) * (1 - labels)
            focal_weight = (1 - p_t) ** self.loss_cls.gamma
            bce = -(labels * torch.log(p + 1e-8) + (1 - labels) * torch.log(1 - p + 1e-8))

            loss_cls = (focal_weight * bce * ~ignore_mask).sum() / max(
                (~ignore_mask).sum(), 1
            )
            total_loss_cls += loss_cls

            # Bbox regression loss (only on positive anchors)
            pos_mask = labels == 1.0
            num_pos = pos_mask.sum().item()

            if num_pos > 0:
                pos_anchors = anchors_all[pos_mask]
                pos_bbox_pred = bbox_pred[pos_mask]
                pos_gt_boxes = gt_boxes_xyxy[max_idx[pos_mask]]

                # Encode GT boxes
                bbox_targets = self.encode_boxes(pos_anchors, pos_gt_boxes)

                # Smooth L1 loss
                loss_bbox = self.loss_bbox(pos_bbox_pred, bbox_targets)
                loss_bbox = loss_bbox.sum() / num_pos
                total_loss_bbox += loss_bbox
            else:
                total_loss_bbox += bbox_pred.sum() * 0.0

        # Average across batch
        avg_loss_cls = total_loss_cls / batch_size
        avg_loss_bbox = total_loss_bbox / batch_size

        return {
            "loss_rpn_cls": avg_loss_cls * self.loss_cls_weight,
            "loss_rpn_bbox": avg_loss_bbox * self.loss_bbox_weight,
        }
