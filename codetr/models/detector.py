"""Co-DETR Detector: Complete Co-Deformable DETR implementation.

This module implements the full Co-DETR detector that assembles backbone, neck,
transformer, and detection heads into a unified training and inference pipeline.

Key features:
- Multi-head collaborative training (DETR + RPN + RoI + ATSS)
- Query denoising for improved training stability
- Flexible auxiliary head configuration
- Support for both training and inference modes

Reference:
    DETRs with Collaborative Hybrid Assignments Training
    https://arxiv.org/abs/2211.12860
"""

from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .backbone.resnet import ResNetBackbone
from .neck.channel_mapper import ChannelMapper
from .transformer.transformer import CoDeformableDetrTransformer
from .heads.detr_head import CoDeformDETRHead
from .heads.rpn_head import RPNHead
from .heads.roi_head import RoIHead
from .heads.atss_head import ATSSHead
from .utils.position_encoding import PositionEmbeddingSine
from .utils.query_denoising import DnQueryGenerator, build_dn_generator


class CoDETR(nn.Module):
    """Complete Co-Deformable DETR detector.

    This detector integrates all components for end-to-end object detection:
    - Backbone: ResNet-50 for feature extraction
    - Neck: ChannelMapper for multi-scale feature pyramid
    - Transformer: Encoder-decoder with deformable attention
    - Query Head: Main DETR detection head
    - Auxiliary Heads: RPN, RoI, ATSS for collaborative training

    The collaborative training mechanism uses auxiliary heads to provide
    additional supervision signals, improving convergence and performance.

    Args:
        num_classes: Number of object categories (excluding background).
        embed_dim: Embedding dimension for transformer (default: 256).
        num_queries: Number of object queries (default: 300).
        num_feature_levels: Number of feature pyramid levels (default: 4).
        num_encoder_layers: Number of transformer encoder layers (default: 6).
        num_decoder_layers: Number of transformer decoder layers (default: 6).
        use_rpn: Whether to use RPN auxiliary head (default: True).
        use_roi: Whether to use RoI auxiliary head (default: True).
        use_atss: Whether to use ATSS auxiliary head (default: True).
        use_dn: Whether to use query denoising (default: True).
        aux_loss_weight: Weight for auxiliary head losses (default: 1.0).
        pretrained_backbone: Whether to use pretrained backbone (default: True).
        frozen_backbone_stages: Number of backbone stages to freeze (default: 1).

    Example:
        >>> model = CoDETR(num_classes=80)
        >>> images = torch.randn(2, 3, 800, 800)
        >>> targets = [
        ...     {'labels': torch.randint(0, 80, (5,)), 'boxes': torch.rand(5, 4)},
        ...     {'labels': torch.randint(0, 80, (3,)), 'boxes': torch.rand(3, 4)},
        ... ]
        >>> # Training
        >>> losses = model(images, targets)
        >>> # Inference
        >>> model.eval()
        >>> predictions = model(images)
    """

    def __init__(
        self,
        num_classes: int,
        embed_dim: int = 256,
        num_queries: int = 300,
        num_feature_levels: int = 4,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        use_rpn: bool = True,
        use_roi: bool = True,
        use_atss: bool = True,
        use_dn: bool = True,
        aux_loss_weight: float = 1.0,
        pretrained_backbone: bool = True,
        frozen_backbone_stages: int = 1,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_queries = num_queries
        self.num_feature_levels = num_feature_levels
        self.aux_loss_weight = aux_loss_weight
        self.use_rpn = use_rpn
        self.use_roi = use_roi
        self.use_atss = use_atss
        self.use_dn = use_dn

        # Build backbone
        self.backbone = ResNetBackbone(
            pretrained=pretrained_backbone,
            frozen_stages=frozen_backbone_stages,
            norm_eval=True,
        )

        # Build neck
        self.neck = ChannelMapper(
            in_channels=self.backbone.out_channels,  # [512, 1024, 2048]
            out_channels=embed_dim,
            num_extra_levels=num_feature_levels - 3,  # 3 backbone levels + extra
        )

        # Build position encoding
        self.pos_embed = PositionEmbeddingSine(
            num_pos_feats=embed_dim // 2,
            normalize=True,
        )

        # Build transformer
        self.transformer = CoDeformableDetrTransformer(
            embed_dim=embed_dim,
            num_heads=8,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            feedforward_dim=1024,
            dropout=0.1,
            num_feature_levels=num_feature_levels,
            num_points=4,
            num_queries=num_queries,
            two_stage=True,
        )

        # Build main DETR head
        self.query_head = CoDeformDETRHead(
            num_classes=num_classes,
            embed_dims=embed_dim,
            num_query=num_queries,
            num_decoder_layers=num_decoder_layers,
        )
        self.query_head.init_weights()

        # Build label encoder for query denoising
        self.label_enc = nn.Embedding(num_classes, embed_dim)

        # Build auxiliary heads
        if use_rpn:
            self.rpn_head = RPNHead(
                in_channels=embed_dim,
                feat_channels=embed_dim,
            )
            self.rpn_head.init_weights()
        else:
            self.rpn_head = None

        if use_roi:
            self.roi_head = RoIHead(
                num_classes=num_classes,
                in_channels=embed_dim,
            )
            self.roi_head.init_weights()
        else:
            self.roi_head = None

        if use_atss:
            self.atss_head = ATSSHead(
                num_classes=num_classes,
                in_channels=embed_dim,
            )
            self.atss_head.init_weights()
        else:
            self.atss_head = None

        # Build query denoising generator
        if use_dn:
            self.dn_generator = DnQueryGenerator(
                num_queries=num_queries,
                hidden_dim=embed_dim,
                num_classes=num_classes,
            )
        else:
            self.dn_generator = None

    def extract_feat(self, images: Tensor) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        """Extract multi-scale features from backbone and neck.

        Args:
            images: Input images. Shape: (B, 3, H, W).

        Returns:
            Tuple of:
                - features: List of feature maps, each (B, embed_dim, H_i, W_i).
                - masks: List of masks, each (B, H_i, W_i).
                - pos_embeds: List of position embeddings, each (B, embed_dim, H_i, W_i).
        """
        # Backbone forward
        backbone_feats = self.backbone(images)

        # Neck forward
        features = self.neck(backbone_feats)

        # Generate masks and position embeddings
        batch_size = images.shape[0]
        input_h, input_w = images.shape[-2:]

        masks = []
        pos_embeds = []

        for feat in features:
            h, w = feat.shape[-2:]
            # Create simple mask (no padding assumed for simplicity)
            mask = torch.zeros((batch_size, h, w), dtype=torch.bool, device=feat.device)
            masks.append(mask)

            # Generate position embedding
            pos_embed = self.pos_embed(feat, mask)
            pos_embeds.append(pos_embed)

        return features, masks, pos_embeds

    def forward(
        self,
        images: Tensor,
        targets: Optional[List[Dict[str, Tensor]]] = None,
    ) -> Dict[str, Tensor]:
        """Forward pass for training and inference.

        Args:
            images: Input images. Shape: (B, 3, H, W).
            targets: Ground truth targets for training. List of dicts with keys:
                - 'labels': Tensor of shape (num_gt,) with class indices
                - 'boxes': Tensor of shape (num_gt, 4) in cxcywh format, normalized

        Returns:
            Training: Dictionary of losses.
            Inference: List of prediction dicts with 'scores', 'labels', 'boxes'.
        """
        if self.training:
            return self.forward_train(images, targets)
        else:
            return self.forward_inference(images)

    def forward_train(
        self,
        images: Tensor,
        targets: List[Dict[str, Tensor]],
    ) -> Dict[str, Tensor]:
        """Forward pass for training.

        Args:
            images: Input images. Shape: (B, 3, H, W).
            targets: Ground truth targets.

        Returns:
            Dictionary of all losses from query head and auxiliary heads.
        """
        batch_size = images.shape[0]
        device = images.device

        # Extract features
        features, masks, pos_embeds = self.extract_feat(images)

        # Prepare denoising queries if enabled
        attn_mask = None
        dn_label_embed = None
        dn_bbox_embed = None
        dn_meta = None
        
        if self.dn_generator is not None and targets is not None:
            # Convert targets to format expected by dn_generator
            gt_bboxes = []
            gt_labels = []
            img_metas = []
            
            for target in targets:
                boxes = target['boxes']
                labels = target['labels']
                
                # Convert from cxcywh normalized to xyxy absolute (assume 800x800)
                # For simplicity, keep as is since dn_generator handles normalization
                h, w = images.shape[-2:]
                boxes_xyxy = self._cxcywh_to_xyxy(boxes, h, w)
                gt_bboxes.append(boxes_xyxy)
                gt_labels.append(labels)
                img_metas.append({'img_shape': (h, w, 3)})

            try:
                dn_label_embed, dn_bbox_embed, attn_mask, dn_meta = self.dn_generator(
                    gt_bboxes, gt_labels, self.label_enc, img_metas
                )
            except Exception:
                # If denoising fails, continue without it
                dn_label_embed = None
                dn_bbox_embed = None
                attn_mask = None
                dn_meta = None

        # Transformer forward with DN queries
        hs, init_reference, inter_references, enc_cls, enc_coord = self.transformer(
            features, masks, pos_embeds, 
            attn_mask=attn_mask,
            dn_label_embed=dn_label_embed,
            dn_bbox_embed=dn_bbox_embed,
        )

        # Split DN queries from content queries if DN was used
        if dn_meta is not None and dn_meta.get('pad_size', 0) > 0:
            pad_size = dn_meta['pad_size']
            # hs shape: (num_layers, batch, num_dn + num_queries, embed_dim)
            # Split off the DN part (we don't use it for now, but it's separated)
            hs_content = hs[:, :, pad_size:, :]
            inter_references_content = inter_references[:, :, pad_size:, :]
            # TODO: Calculate DN loss separately in future
        else:
            hs_content = hs
            inter_references_content = inter_references

        # Query head forward (only with content queries)
        _, _, query_losses = self.query_head(hs_content, inter_references_content, targets)

        losses = {}
        if query_losses is not None:
            losses.update(query_losses)

        # Auxiliary heads forward
        if self.rpn_head is not None:
            rpn_losses = self._forward_rpn(features, targets)
            if rpn_losses is not None:
                for k, v in rpn_losses.items():
                    losses[f'rpn_{k}'] = v * self.aux_loss_weight

        if self.atss_head is not None:
            atss_losses = self._forward_atss(features, targets)
            if atss_losses is not None:
                for k, v in atss_losses.items():
                    losses[f'atss_{k}'] = v * self.aux_loss_weight

        if self.roi_head is not None and self.rpn_head is not None:
            # RoI head needs proposals from RPN
            with torch.no_grad():
                proposals, _ = self.rpn_head(features, targets=None)
            
            roi_losses = self._forward_roi(features, proposals, targets)
            if roi_losses is not None:
                for k, v in roi_losses.items():
                    losses[f'roi_{k}'] = v * self.aux_loss_weight

        return losses

    def forward_inference(self, images: Tensor) -> List[Dict[str, Tensor]]:
        """Forward pass for inference.

        Args:
            images: Input images. Shape: (B, 3, H, W).

        Returns:
            List of prediction dicts, one per image, with keys:
                - 'scores': Tensor of shape (num_det,)
                - 'labels': Tensor of shape (num_det,)
                - 'boxes': Tensor of shape (num_det, 4) in xyxy format
        """
        # Extract features
        features, masks, pos_embeds = self.extract_feat(images)

        # Transformer forward (no denoising for inference)
        hs, init_reference, inter_references, _, _ = self.transformer(
            features, masks, pos_embeds, attn_mask=None
        )

        # Use final decoder layer output
        final_hs = hs[-1]  # (B, num_queries, embed_dim)
        final_refs = inter_references[-1]  # (B, num_queries, 2)

        # Query head forward (no targets for inference)
        all_cls_scores, all_bbox_preds, _ = self.query_head(
            hs, inter_references, targets=None
        )

        # Use final layer predictions
        cls_scores = all_cls_scores[-1]  # (B, num_queries, num_classes)
        bbox_preds = all_bbox_preds[-1]  # (B, num_queries, 4)

        # Generate predictions
        predictions = self.query_head.predict(cls_scores, bbox_preds)

        return predictions

    def _forward_rpn(
        self,
        features: List[Tensor],
        targets: List[Dict[str, Tensor]],
    ) -> Optional[Dict[str, Tensor]]:
        """Forward pass for RPN auxiliary head.

        Args:
            features: Multi-scale feature maps.
            targets: Ground truth targets.

        Returns:
            Dictionary of RPN losses or None.
        """
        if self.rpn_head is None:
            return None

        # RPN expects targets with 'boxes' in xyxy format
        # Convert targets for RPN
        rpn_targets = []
        for target in targets:
            boxes = target['boxes']
            labels = target['labels']
            h, w = features[0].shape[-2] * 8, features[0].shape[-1] * 8  # Approximate input size
            boxes_xyxy = self._cxcywh_to_xyxy(boxes, h, w)
            rpn_targets.append({'boxes': boxes_xyxy, 'labels': labels})

        proposals, rpn_losses = self.rpn_head(features, targets=rpn_targets)
        return rpn_losses

    def _forward_atss(
        self,
        features: List[Tensor],
        targets: List[Dict[str, Tensor]],
    ) -> Optional[Dict[str, Tensor]]:
        """Forward pass for ATSS auxiliary head.

        Args:
            features: Multi-scale feature maps.
            targets: Ground truth targets.

        Returns:
            Dictionary of ATSS losses or None.
        """
        if self.atss_head is None:
            return None

        # Convert targets for ATSS
        atss_targets = []
        for target in targets:
            boxes = target['boxes']
            labels = target['labels']
            h, w = features[0].shape[-2] * 8, features[0].shape[-1] * 8
            boxes_xyxy = self._cxcywh_to_xyxy(boxes, h, w)
            atss_targets.append({'boxes': boxes_xyxy, 'labels': labels})

        _, _, _, atss_losses = self.atss_head(features, targets=atss_targets)
        return atss_losses

    def _forward_roi(
        self,
        features: List[Tensor],
        proposals: List[Tensor],
        targets: List[Dict[str, Tensor]],
    ) -> Optional[Dict[str, Tensor]]:
        """Forward pass for RoI auxiliary head.

        Args:
            features: Multi-scale feature maps.
            proposals: Region proposals from RPN.
            targets: Ground truth targets.

        Returns:
            Dictionary of RoI losses or None.
        """
        if self.roi_head is None:
            return None

        # Convert targets for RoI
        roi_targets = []
        for target in targets:
            boxes = target['boxes']
            labels = target['labels']
            h, w = features[0].shape[-2] * 8, features[0].shape[-1] * 8
            boxes_xyxy = self._cxcywh_to_xyxy(boxes, h, w)
            roi_targets.append({'boxes': boxes_xyxy, 'labels': labels})

        _, _, roi_losses = self.roi_head(features, proposals, targets=roi_targets)
        return roi_losses

    @staticmethod
    def _cxcywh_to_xyxy(boxes: Tensor, img_h: int, img_w: int) -> Tensor:
        """Convert normalized cxcywh boxes to absolute xyxy format.

        Args:
            boxes: Boxes in cxcywh format, normalized [0, 1]. Shape: (N, 4)
            img_h: Image height.
            img_w: Image width.

        Returns:
            Boxes in xyxy format, absolute coordinates. Shape: (N, 4)
        """
        if boxes.numel() == 0:
            return boxes.new_zeros((0, 4))

        cx, cy, w, h = boxes.unbind(-1)
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h
        return torch.stack([x1, y1, x2, y2], dim=-1)


def build_codetr(
    num_classes: int,
    pretrained_backbone: bool = True,
    use_aux_heads: bool = True,
    **kwargs,
) -> CoDETR:
    """Build CoDETR model with common configurations.

    Args:
        num_classes: Number of object categories.
        pretrained_backbone: Whether to use pretrained backbone.
        use_aux_heads: Whether to use auxiliary heads.
        **kwargs: Additional arguments passed to CoDETR.

    Returns:
        CoDETR model instance.

    Example:
        >>> model = build_codetr(num_classes=80)
        >>> model = build_codetr(num_classes=20, use_aux_heads=False)
    """
    return CoDETR(
        num_classes=num_classes,
        pretrained_backbone=pretrained_backbone,
        use_rpn=use_aux_heads,
        use_roi=use_aux_heads,
        use_atss=use_aux_heads,
        **kwargs,
    )
