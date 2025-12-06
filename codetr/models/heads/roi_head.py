"""
Region of Interest (RoI) Head.

This module implements the RoI head as an auxiliary component for Co-DETR,
performing classification and bounding box regression on RoI-aligned features
extracted from region proposals.

Reference:
    Mask R-CNN
    https://arxiv.org/abs/1703.06870
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import roi_align

from ..losses.focal_loss import FocalLoss
from ..losses.l1_loss import L1Loss
from ..utils.box_ops import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh, box_iou


class RoIHead(nn.Module):
    """
    RoI Head for region-based object detection.

    This auxiliary head extracts fixed-size features from region proposals
    using RoI Align, then performs classification and bounding box refinement
    through fully-connected layers.

    Attributes:
        in_channels (int): Number of input feature channels.
        roi_feat_size (int): Spatial size of RoI features after pooling.
        num_classes (int): Number of object categories.
        fc_out_channels (int): Number of output channels from FC layers.
        loss_cls (FocalLoss): Classification loss function.
        loss_bbox (L1Loss): Bounding box regression loss function.

    Example:
        >>> head = RoIHead(num_classes=80, in_channels=256)
        >>> features = [torch.randn(2, 256, 100, 100)]
        >>> proposals = [torch.rand(10, 4) * 100 for _ in range(2)]
        >>> targets = [{'labels': torch.tensor([0, 1]),
        ...             'boxes': torch.tensor([[0.3, 0.3, 0.5, 0.5],
        ...                                     [0.6, 0.6, 0.2, 0.2]])}]
        >>> cls_scores, bbox_preds, losses = head(features, proposals, targets)
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 256,
        roi_feat_size: int = 7,
        fc_out_channels: int = 1024,
        num_fc_layers: int = 2,
        roi_output_size: Tuple[int, int] = (7, 7),
        spatial_scale: float = 0.25,
        sampling_ratio: int = 2,
        loss_cls_weight: float = 1.0,
        loss_bbox_weight: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        pos_iou_threshold: float = 0.5,
        neg_iou_threshold: float = 0.5,
        num_sample_rois: int = 512,
        pos_fraction: float = 0.25,
    ) -> None:
        """
        Initialize RoI head.

        Args:
            num_classes: Number of object categories (excluding background).
            in_channels: Number of input feature channels.
            roi_feat_size: Spatial size of RoI features.
            fc_out_channels: Number of FC layer output channels.
            num_fc_layers: Number of fully-connected layers.
            roi_output_size: Output size of RoI Align (height, width).
            spatial_scale: Spatial scale factor for mapping proposals to features.
            sampling_ratio: Number of sampling points in RoI Align.
            loss_cls_weight: Classification loss weight.
            loss_bbox_weight: Bbox regression loss weight.
            focal_alpha: Focal loss alpha parameter.
            focal_gamma: Focal loss gamma parameter.
            pos_iou_threshold: IoU threshold for positive samples.
            neg_iou_threshold: IoU threshold for negative samples.
            num_sample_rois: Number of RoIs to sample per image.
            pos_fraction: Fraction of positive samples.
        """
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.roi_feat_size = roi_feat_size
        self.fc_out_channels = fc_out_channels
        self.num_fc_layers = num_fc_layers
        self.roi_output_size = roi_output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
        self.loss_cls_weight = loss_cls_weight
        self.loss_bbox_weight = loss_bbox_weight
        self.pos_iou_threshold = pos_iou_threshold
        self.neg_iou_threshold = neg_iou_threshold
        self.num_sample_rois = num_sample_rois
        self.pos_fraction = pos_fraction

        # Loss functions
        self.loss_cls = FocalLoss(
            alpha=focal_alpha, gamma=focal_gamma, reduction="none"
        )
        self.loss_bbox = L1Loss(reduction="none")

        # Initialize layers
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize FC layers for classification and regression."""
        # Shared FC layers
        fc_in_channels = (
            self.in_channels * self.roi_output_size[0] * self.roi_output_size[1]
        )

        self.shared_fcs = nn.ModuleList()
        in_channels = fc_in_channels
        for _ in range(self.num_fc_layers):
            self.shared_fcs.append(nn.Linear(in_channels, self.fc_out_channels))
            in_channels = self.fc_out_channels

        # Classification head
        self.fc_cls = nn.Linear(self.fc_out_channels, self.num_classes)

        # Regression head
        self.fc_reg = nn.Linear(self.fc_out_channels, self.num_classes * 4)

    def init_weights(self) -> None:
        """Initialize weights of RoI head."""
        # Xavier initialization for FC layers
        for m in self.shared_fcs:
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.0)

        # Classification head bias for focal loss
        nn.init.xavier_uniform_(self.fc_cls.weight)
        nn.init.constant_(self.fc_cls.bias, -2.19722)  # log((1-0.01)/0.01)

        # Regression head
        nn.init.xavier_uniform_(self.fc_reg.weight)
        nn.init.constant_(self.fc_reg.bias, 0.0)

    def extract_roi_features(
        self, features: List[Tensor], rois: Tensor, spatial_scale: float
    ) -> Tensor:
        """
        Extract RoI-aligned features from feature maps.

        Args:
            features: List of feature maps. Typically use only the first level.
                Each with shape (B, C, H, W).
            rois: Region of interest boxes in xyxy format.
                Shape: (num_rois, 5) where first column is batch index.
            spatial_scale: Spatial scale for mapping RoIs to feature map.

        Returns:
            RoI features. Shape: (num_rois, C, roi_h, roi_w)
        """
        # Use the first feature level (typically P4 with stride 16)
        feature = features[0]

        # Apply RoI Align
        roi_feats = roi_align(
            feature,
            rois,
            output_size=self.roi_output_size,
            spatial_scale=spatial_scale,
            sampling_ratio=self.sampling_ratio,
            aligned=True,
        )

        return roi_feats

    def sample_rois(
        self,
        proposals: List[Tensor],
        targets: List[Dict[str, Tensor]],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Sample RoIs for training by assigning proposals to ground truth.

        Args:
            proposals: List of proposal boxes per image. Each (num_proposals, 4) in xyxy.
            targets: Ground truth targets per image.

        Returns:
            sampled_rois: Sampled RoI boxes with batch indices.
                Shape: (total_sampled, 5) where first column is batch index.
            sampled_labels: Class labels for sampled RoIs. Shape: (total_sampled,)
            sampled_bbox_targets: Bbox regression targets. Shape: (total_sampled, 4)
        """
        batch_size = len(proposals)

        all_sampled_rois = []
        all_sampled_labels = []
        all_sampled_bbox_targets = []

        for batch_idx in range(batch_size):
            props = proposals[batch_idx]  # (num_proposals, 4) in xyxy
            gt_boxes = targets[batch_idx]["boxes"]  # (num_gt, 4) in cxcywh, normalized
            gt_labels = targets[batch_idx]["labels"]  # (num_gt,)

            if len(props) == 0:
                continue

            # Convert GT boxes to xyxy absolute coordinates (assume 800x800 image)
            img_size = 800
            gt_boxes_abs = gt_boxes * img_size
            gt_boxes_xyxy = bbox_cxcywh_to_xyxy(gt_boxes_abs)

            # Compute IoU between proposals and GT boxes
            if len(gt_boxes_xyxy) > 0:
                ious = box_iou(props, gt_boxes_xyxy)[0]  # (num_proposals, num_gt)
                max_ious, max_idx = ious.max(dim=1)  # (num_proposals,)
            else:
                max_ious = torch.zeros(len(props), device=props.device)
                max_idx = torch.zeros(len(props), dtype=torch.long, device=props.device)

            # Assign labels: positive (IoU >= 0.5), negative (IoU < 0.5)
            pos_mask = max_ious >= self.pos_iou_threshold
            neg_mask = max_ious < self.neg_iou_threshold

            # Sample positive and negative RoIs
            num_pos = int(self.num_sample_rois * self.pos_fraction)
            num_neg = self.num_sample_rois - num_pos

            pos_indices = torch.where(pos_mask)[0]
            neg_indices = torch.where(neg_mask)[0]

            # Random sampling
            if len(pos_indices) > num_pos:
                perm = torch.randperm(len(pos_indices), device=props.device)[:num_pos]
                pos_indices = pos_indices[perm]

            if len(neg_indices) > num_neg:
                perm = torch.randperm(len(neg_indices), device=props.device)[:num_neg]
                neg_indices = neg_indices[perm]

            sampled_indices = torch.cat([pos_indices, neg_indices], dim=0)
            sampled_rois = props[sampled_indices]  # (num_sampled, 4)

            # Add batch index as first column
            batch_indices = torch.full(
                (len(sampled_rois), 1),
                batch_idx,
                dtype=sampled_rois.dtype,
                device=sampled_rois.device,
            )
            sampled_rois = torch.cat(
                [batch_indices, sampled_rois], dim=1
            )  # (num_sampled, 5)

            # Assign labels
            sampled_labels = torch.full(
                (len(sampled_indices),),
                self.num_classes,  # Background class
                dtype=torch.long,
                device=props.device,
            )
            if len(pos_indices) > 0:
                sampled_labels[: len(pos_indices)] = gt_labels[max_idx[pos_indices]]

            # Compute bbox targets (for positive samples, use matched GT boxes)
            sampled_bbox_targets = torch.zeros(
                (len(sampled_indices), 4), dtype=torch.float32, device=props.device
            )
            if len(pos_indices) > 0 and len(gt_boxes_xyxy) > 0:
                matched_gt_boxes = gt_boxes_xyxy[max_idx[pos_indices]]
                sampled_bbox_targets[: len(pos_indices)] = matched_gt_boxes

            all_sampled_rois.append(sampled_rois)
            all_sampled_labels.append(sampled_labels)
            all_sampled_bbox_targets.append(sampled_bbox_targets)

        # Concatenate all samples
        if len(all_sampled_rois) > 0:
            sampled_rois = torch.cat(all_sampled_rois, dim=0)
            sampled_labels = torch.cat(all_sampled_labels, dim=0)
            sampled_bbox_targets = torch.cat(all_sampled_bbox_targets, dim=0)
        else:
            device = proposals[0].device
            sampled_rois = torch.zeros((0, 5), device=device)
            sampled_labels = torch.zeros((0,), dtype=torch.long, device=device)
            sampled_bbox_targets = torch.zeros((0, 4), device=device)

        return sampled_rois, sampled_labels, sampled_bbox_targets

    def forward(
        self,
        features: List[Tensor],
        proposals: List[Tensor],
        targets: Optional[List[Dict[str, Tensor]]] = None,
    ) -> Tuple[Tensor, Tensor, Optional[Dict[str, Tensor]]]:
        """
        Forward pass through RoI head.

        Args:
            features: Multi-scale feature maps. List of (B, C, H, W).
            proposals: Region proposals per image. List of (num_proposals, 4) in xyxy.
            targets: Ground truth targets for training.

        Returns:
            cls_scores: Classification scores. Shape: (num_rois, num_classes)
            bbox_preds: Bbox predictions. Shape: (num_rois, num_classes*4)
            losses: Dictionary of losses if targets provided, else None.
        """
        # Sample RoIs during training
        if self.training and targets is not None:
            rois, labels, bbox_targets = self.sample_rois(proposals, targets)
        else:
            # During inference, use all proposals
            batch_size = len(proposals)
            rois_list = []
            for batch_idx, props in enumerate(proposals):
                batch_indices = torch.full(
                    (len(props), 1),
                    batch_idx,
                    dtype=props.dtype,
                    device=props.device,
                )
                rois_with_batch = torch.cat([batch_indices, props], dim=1)
                rois_list.append(rois_with_batch)
            rois = (
                torch.cat(rois_list, dim=0)
                if len(rois_list) > 0
                else torch.zeros((0, 5))
            )
            labels = None
            bbox_targets = None

        if len(rois) == 0:
            # No RoIs to process
            device = features[0].device
            return (
                torch.zeros((0, self.num_classes), device=device),
                torch.zeros((0, self.num_classes * 4), device=device),
                None,
            )

        # Extract RoI features
        roi_feats = self.extract_roi_features(features, rois, self.spatial_scale)

        # Flatten RoI features
        roi_feats_flat = roi_feats.flatten(1)  # (num_rois, C*H*W)

        # Pass through shared FC layers
        x = roi_feats_flat
        for fc in self.shared_fcs:
            x = F.relu(fc(x))

        # Classification and regression branches
        cls_scores = self.fc_cls(x)  # (num_rois, num_classes)
        bbox_preds = self.fc_reg(x)  # (num_rois, num_classes*4)

        # Compute losses if targets provided
        losses = None
        if self.training and targets is not None and labels is not None:
            losses = self.loss(cls_scores, bbox_preds, labels, bbox_targets)

        return cls_scores, bbox_preds, losses

    def loss(
        self,
        cls_scores: Tensor,
        bbox_preds: Tensor,
        labels: Tensor,
        bbox_targets: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Compute RoI head losses.

        Args:
            cls_scores: Classification scores. Shape: (num_rois, num_classes)
            bbox_preds: Bbox predictions. Shape: (num_rois, num_classes*4)
            labels: Ground truth labels. Shape: (num_rois,)
            bbox_targets: Ground truth boxes. Shape: (num_rois, 4)

        Returns:
            Dictionary with 'loss_roi_cls' and 'loss_roi_bbox'.
        """
        # Classification loss using sigmoid focal loss
        num_classes_with_bg = self.num_classes + 1
        labels_onehot = F.one_hot(labels.long(), num_classes_with_bg).float()[:, :self.num_classes]

        # Compute focal loss manually
        p = torch.sigmoid(cls_scores)
        p_t = p * labels_onehot + (1 - p) * (1 - labels_onehot)
        alpha_t = self.loss_cls.alpha * labels_onehot + (1 - self.loss_cls.alpha) * (1 - labels_onehot)
        focal_weight = (1 - p_t) ** self.loss_cls.gamma

        bce = -(
            labels_onehot * torch.log(p + 1e-8)
            + (1 - labels_onehot) * torch.log(1 - p + 1e-8)
        )

        loss_cls = (alpha_t * focal_weight * bce).sum() / max(len(labels), 1)

        # Bbox regression loss (only on positive samples)
        pos_mask = labels < self.num_classes
        num_pos = pos_mask.sum().item()

        if num_pos > 0:
            # Select bbox predictions for the correct class
            pos_labels = labels[pos_mask]
            pos_bbox_preds = bbox_preds[pos_mask].reshape(-1, self.num_classes, 4)
            pos_bbox_preds = pos_bbox_preds[
                torch.arange(num_pos), pos_labels
            ]  # (num_pos, 4)

            pos_bbox_targets = bbox_targets[pos_mask]

            loss_bbox = self.loss_bbox(pos_bbox_preds, pos_bbox_targets)
            loss_bbox = loss_bbox.mean()
        else:
            loss_bbox = bbox_preds.sum() * 0.0

        return {
            "loss_roi_cls": loss_cls * self.loss_cls_weight,
            "loss_roi_bbox": loss_bbox * self.loss_bbox_weight,
        }

    @torch.no_grad()
    def predict(
        self,
        cls_scores: Tensor,
        bbox_preds: Tensor,
        rois: Tensor,
        score_threshold: float = 0.3,
    ) -> List[Dict[str, Tensor]]:
        """
        Generate predictions from RoI head outputs.

        Args:
            cls_scores: Classification scores. Shape: (num_rois, num_classes)
            bbox_preds: Bbox predictions. Shape: (num_rois, num_classes*4)
            rois: RoI boxes with batch indices. Shape: (num_rois, 5)
            score_threshold: Minimum score threshold.

        Returns:
            List of prediction dicts per image with 'scores', 'labels', 'boxes'.
        """
        # Apply softmax to get class probabilities
        probs = F.softmax(cls_scores, dim=-1)  # (num_rois, num_classes)
        max_probs, labels = probs.max(dim=-1)  # (num_rois,)

        # Filter by score threshold
        keep = max_probs > score_threshold
        labels = labels[keep]
        max_probs = max_probs[keep]
        rois_keep = rois[keep]

        # Get bbox predictions for selected class
        bbox_preds_keep = bbox_preds[keep].reshape(-1, self.num_classes, 4)
        num_keep = len(labels)
        if num_keep > 0:
            boxes = bbox_preds_keep[torch.arange(num_keep), labels]
        else:
            boxes = torch.zeros((0, 4), device=bbox_preds.device)

        # Group by batch index
        batch_indices = rois_keep[:, 0].long()
        unique_batches = batch_indices.unique()

        predictions = []
        for batch_idx in unique_batches:
            mask = batch_indices == batch_idx
            predictions.append(
                {
                    "scores": max_probs[mask],
                    "labels": labels[mask],
                    "boxes": boxes[mask],
                }
            )

        return predictions
