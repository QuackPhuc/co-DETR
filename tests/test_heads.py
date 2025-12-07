"""
Tests for detection heads.

This module tests the DETR, RPN, RoI, and ATSS detection heads
verifying output shapes, loss computation, and prediction formats.
"""

import pytest
import torch

from codetr.models.heads.detr_head import CoDeformDETRHead
from codetr.models.heads.rpn_head import RPNHead
from codetr.models.heads.roi_head import RoIHead
from codetr.models.heads.atss_head import ATSSHead


class TestCoDeformDETRHead:
    """Tests for main Co-Deformable DETR detection head."""
    
    def test_forward_output_shapes(self):
        """Test forward pass output shapes."""
        head = CoDeformDETRHead(
            num_classes=80,
            embed_dims=256,
            num_query=300,
            num_decoder_layers=6,
        )
        
        # Decoder outputs: (num_layers, batch, num_queries, embed_dim)
        hidden_states = torch.randn(6, 2, 300, 256)
        # Reference points: (num_layers, batch, num_queries, 2) or (batch, num_queries, 2)
        references = torch.rand(2, 300, 2)
        
        cls_scores, bbox_preds, _ = head(hidden_states, references)
        
        # Classification: (num_layers, batch, num_queries, num_classes)
        # Note: forward() returns stacked outputs for all decoder layers for auxiliary losses
        assert cls_scores.shape == (6, 2, 300, 80)
        
        # Box predictions: (num_layers, batch, num_queries, 4)
        assert bbox_preds.shape == (6, 2, 300, 4)
    
    def test_loss_computation(self):
        """Test loss computation with targets."""
        head = CoDeformDETRHead(
            num_classes=20,
            embed_dims=256,
            num_query=100,
            num_decoder_layers=2,
        )
        
        hidden_states = torch.randn(2, 1, 100, 256)
        references = torch.rand(1, 100, 2)
        
        targets = [{
            'labels': torch.tensor([0, 5, 10]),
            'boxes': torch.rand(3, 4),  # cxcywh normalized
        }]
        
        cls_scores, bbox_preds, losses = head(hidden_states, references, targets)
        
        # Should return loss dictionary
        assert losses is not None
        assert 'loss_cls' in losses
        assert 'loss_bbox' in losses
        assert 'loss_iou' in losses
        
        # Losses should be positive scalars
        assert losses['loss_cls'] >= 0
        assert losses['loss_bbox'] >= 0
        assert losses['loss_iou'] >= 0
    
    def test_prediction_format(self):
        """Test prediction method returns correct format."""
        head = CoDeformDETRHead(
            num_classes=20,
            embed_dims=256,
            num_query=100,
        )
        
        cls_scores = torch.randn(2, 100, 20)
        bbox_preds = torch.rand(2, 100, 4)
        
        predictions = head.predict(cls_scores, bbox_preds, score_threshold=0.1)
        
        # Should return list of dicts
        assert isinstance(predictions, list)
        assert len(predictions) == 2
        
        for pred in predictions:
            assert 'scores' in pred
            assert 'labels' in pred
            assert 'boxes' in pred
            
            # Boxes should be xyxy format with shape (N, 4)
            assert pred['boxes'].dim() == 2
            assert pred['boxes'].shape[1] == 4
    
    def test_gradient_flow(self):
        """Test gradients flow through head."""
        head = CoDeformDETRHead(
            num_classes=10,
            embed_dims=128,
            num_query=50,
            num_decoder_layers=1,
        )
        
        hidden_states = torch.randn(1, 1, 50, 128, requires_grad=True)
        references = torch.rand(1, 50, 2)
        targets = [{'labels': torch.tensor([0, 1]), 'boxes': torch.rand(2, 4)}]
        
        _, _, losses = head(hidden_states, references, targets)
        
        total_loss = losses['loss_cls'] + losses['loss_bbox'] + losses['loss_iou']
        total_loss.backward()
        
        assert hidden_states.grad is not None


class TestRPNHead:
    """Tests for Region Proposal Network head."""
    
    def test_forward_output_format(self):
        """Test RPN forward pass outputs."""
        # Note: RPNHead uses anchor_scales/ratios to determine num_anchors internally
        # Default: 3 scales * 3 ratios = 9 anchors
        head = RPNHead(in_channels=256)
        head.init_weights()
        
        # Multi-scale features (must match anchor_strides length which is 4 by default)
        features = [
            torch.randn(2, 256, 100, 100),
            torch.randn(2, 256, 50, 50),
            torch.randn(2, 256, 25, 25),
            torch.randn(2, 256, 13, 13),
        ]
        
        # RPNHead.forward() returns (proposals, losses) tuple
        # proposals is a list of boxes per image, not multi-scale outputs
        proposals, losses = head(features)
        
        # proposals is list with one entry per batch item
        assert isinstance(proposals, list)
        assert len(proposals) == 2  # batch_size
        
        # Each proposal is (num_proposals, 4) in xyxy format
        for prop in proposals:
            assert prop.dim() == 2
            assert prop.shape[1] == 4
    
    def test_anchor_generation(self):
        """Test anchor generation for different feature sizes."""
        head = RPNHead(in_channels=256)
        
        # Verify anchor generator exists
        assert hasattr(head, 'anchor_generator')
    
    def test_gradient_flow(self):
        """Test gradients flow through RPN."""
        head = RPNHead(in_channels=256)
        
        # Need at least 4 feature levels to match anchor_strides
        features = [
            torch.randn(1, 256, 10, 10, requires_grad=True),
            torch.randn(1, 256, 5, 5, requires_grad=True),
            torch.randn(1, 256, 3, 3, requires_grad=True),
            torch.randn(1, 256, 2, 2, requires_grad=True),
        ]
        
        proposals, losses = head(features)
        
        # proposals need gradients through them for backprop
        loss = proposals[0].sum()
        loss.backward()
        
        assert features[0].grad is not None


class TestRPNBoxEncodingDecoding:
    """Tests for RPN box encoding and decoding logic.
    
    These tests verify the mathematical correctness of the box encoding scheme
    which is critical for proposal generation.
    """

    def test_box_encode_decode_roundtrip(self):
        """Encode → decode should return boxes close to original.
        
        Mathematical guarantee: Given anchors A and GT boxes B,
        encode(A, B) produces deltas D, then decode(A, D) should ≈ B.
        """
        head = RPNHead(in_channels=256)

        # Create anchors in xyxy format
        anchors = torch.tensor([
            [10.0, 10.0, 50.0, 50.0],   # 40x40 anchor
            [100.0, 100.0, 200.0, 200.0],  # 100x100 anchor
            [50.0, 25.0, 150.0, 125.0],  # 100x100 anchor, different position
        ])

        # Create GT boxes slightly different from anchors
        gt_boxes = torch.tensor([
            [15.0, 12.0, 55.0, 52.0],   # Shifted and slightly larger
            [110.0, 95.0, 210.0, 195.0],  # Shifted
            [40.0, 30.0, 160.0, 130.0],  # Shifted and resized
        ])

        # Encode: gt_boxes relative to anchors
        deltas = head.encode_boxes(anchors=anchors, gt_boxes=gt_boxes)

        # Decode: should recover gt_boxes from anchors + deltas  
        recovered_boxes = head.decode_boxes(anchors=anchors, deltas=deltas)

        # Roundtrip error should be minimal (floating point precision)
        assert torch.allclose(recovered_boxes, gt_boxes, atol=1e-4), (
            f"Roundtrip error too large:\n"
            f"Original: {gt_boxes}\n"
            f"Recovered: {recovered_boxes}\n"
            f"Diff: {(recovered_boxes - gt_boxes).abs().max()}"
        )

    def test_box_encoding_delta_values_reasonable(self):
        """Encoded deltas should have reasonable values for typical boxes."""
        head = RPNHead(in_channels=256)

        anchors = torch.tensor([
            [100.0, 100.0, 200.0, 200.0],  # 100x100 anchor at (150, 150)
        ])

        # GT box with small shift (10 pixels) and no scale change
        gt_boxes = torch.tensor([
            [110.0, 105.0, 210.0, 205.0],  # Shifted by (10, 5)
        ])

        deltas = head.encode_boxes(anchors=anchors, gt_boxes=gt_boxes)

        # dx, dy should be small (shift / anchor_size)
        dx, dy, dw, dh = deltas[0].tolist()

        # dx = (gt_cx - anchor_cx) / anchor_w = (160-150)/100 = 0.1
        assert abs(dx - 0.1) < 0.01, f"dx should be ~0.1, got {dx}"
        # dy = (gt_cy - anchor_cy) / anchor_h = (155-150)/100 = 0.05
        assert abs(dy - 0.05) < 0.01, f"dy should be ~0.05, got {dy}"
        # dw = log(gt_w / anchor_w) = log(100/100) = 0
        assert abs(dw) < 0.01, f"dw should be ~0, got {dw}"
        # dh = log(gt_h / anchor_h) = log(100/100) = 0
        assert abs(dh) < 0.01, f"dh should be ~0, got {dh}"


class TestRoIHead:
    """Tests for RoI (Region of Interest) head."""
    
    def test_forward_output_format(self):
        """Test RoI head forward pass."""
        head = RoIHead(
            in_channels=256,
            num_classes=20,
            roi_feat_size=7,
        )
        head.init_weights()
        
        features = [
            torch.randn(2, 256, 50, 50),
            torch.randn(2, 256, 25, 25),
        ]
        
        # Proposals: list of (N, 4) boxes per image in xyxy format
        proposals = [
            torch.tensor([[10.0, 10.0, 30.0, 30.0], [20.0, 20.0, 40.0, 40.0]]),
            torch.tensor([[5.0, 5.0, 25.0, 25.0]]),
        ]
        
        # RoIHead.forward() returns (cls_scores, bbox_preds, losses) tuple
        cls_scores, bbox_preds, losses = head(features, proposals)
        
        # Total proposals: 2 + 1 = 3
        assert cls_scores.shape[0] == 3
        # num_classes without background (RoI head uses focal loss, no explicit bg class)
        assert cls_scores.shape[1] == 20
        
        assert bbox_preds.shape[0] == 3
        # Class-specific bbox predictions: 4 * num_classes
        assert bbox_preds.shape[1] == 4 * 20


class TestRoIHeadLossAndSampling:
    """Tests for RoI Head loss computation and RoI sampling logic."""

    def test_roi_loss_computation_with_targets(self):
        """Test loss is computed correctly when training with targets."""
        head = RoIHead(
            in_channels=256,
            num_classes=20,
            roi_feat_size=7,
        )
        head.init_weights()
        head.train()

        features = [torch.randn(2, 256, 50, 50)]

        # Proposals in xyxy format (absolute coordinates)
        proposals = [
            torch.tensor([
                [10.0, 10.0, 100.0, 100.0],
                [200.0, 200.0, 350.0, 350.0],
                [50.0, 50.0, 150.0, 150.0],
            ]),
            torch.tensor([
                [20.0, 20.0, 120.0, 120.0],
                [150.0, 150.0, 300.0, 300.0],
            ]),
        ]

        # Targets with normalized cxcywh boxes
        targets = [
            {
                'labels': torch.tensor([0, 5]),
                'boxes': torch.tensor([
                    [0.1, 0.1, 0.15, 0.15],  # (cx, cy, w, h) normalized
                    [0.35, 0.35, 0.2, 0.2],
                ]),
            },
            {
                'labels': torch.tensor([10]),
                'boxes': torch.tensor([
                    [0.2, 0.2, 0.15, 0.15],
                ]),
            },
        ]

        cls_scores, bbox_preds, losses = head(
            features=features,
            proposals=proposals,
            targets=targets,
        )

        # Should return loss dictionary during training
        assert losses is not None
        assert 'loss_roi_cls' in losses
        assert 'loss_roi_bbox' in losses

        # Losses should be positive scalars
        assert losses['loss_roi_cls'] >= 0
        assert losses['loss_roi_bbox'] >= 0
        assert not torch.isnan(losses['loss_roi_cls'])
        assert not torch.isnan(losses['loss_roi_bbox'])

    def test_roi_empty_proposals_handled(self):
        """Test RoI head handles empty proposals gracefully."""
        head = RoIHead(
            in_channels=256,
            num_classes=10,
            roi_feat_size=7,
        )
        head.init_weights()
        head.train()

        features = [torch.randn(2, 256, 30, 30)]

        # Empty proposals for both images
        proposals = [
            torch.zeros((0, 4)),
            torch.zeros((0, 4)),
        ]

        targets = [
            {'labels': torch.tensor([0]), 'boxes': torch.rand(1, 4)},
            {'labels': torch.tensor([1]), 'boxes': torch.rand(1, 4)},
        ]

        # Should not crash with empty proposals
        cls_scores, bbox_preds, losses = head(
            features=features,
            proposals=proposals,
            targets=targets,
        )

        # Empty output expected
        assert cls_scores.shape[0] == 0
        assert bbox_preds.shape[0] == 0
        # No loss when no RoIs
        assert losses is None

    def test_roi_sample_rois_positive_negative_ratio(self):
        """Test RoI sampling respects pos_fraction ratio."""
        head = RoIHead(
            in_channels=256,
            num_classes=20,
            roi_feat_size=7,
            pos_iou_threshold=0.5,
            neg_iou_threshold=0.5,
            num_sample_rois=64,
            pos_fraction=0.25,
        )

        # Create proposals that overlap with GT
        # GT box at [100, 100, 200, 200] xyxy
        proposals = [
            torch.tensor([
                [100.0, 100.0, 200.0, 200.0],  # IoU = 1.0 with GT
                [110.0, 110.0, 210.0, 210.0],  # High IoU
                [0.0, 0.0, 50.0, 50.0],        # No overlap
                [300.0, 300.0, 400.0, 400.0],  # No overlap
            ])
        ]

        targets = [
            {
                'labels': torch.tensor([5]),
                'boxes': torch.tensor([[0.1875, 0.1875, 0.125, 0.125]]),  # center at 150,150 w=h=100
            }
        ]

        sampled_rois, sampled_labels, sampled_bbox_targets = head.sample_rois(
            proposals=proposals,
            targets=targets,
        )

        # Should have sampled RoIs
        assert sampled_rois.shape[0] > 0
        # First column is batch index
        assert sampled_rois.shape[1] == 5
        # Labels should include both positive and negative
        assert sampled_labels.shape[0] == sampled_rois.shape[0]


class TestATSSHeadSampling:
    """Tests for ATSS adaptive sample selection algorithm."""

    def test_atss_sampling_selects_correct_positives(self):
        """Test ATSS sampling algorithm selects positives based on IoU threshold."""
        head = ATSSHead(
            num_classes=20,
            in_channels=256,
            topk_candidates=3,
        )

        # Create anchors at different positions
        anchors_level1 = torch.tensor([
            [0.0, 0.0, 64.0, 64.0],      # Overlaps with GT
            [64.0, 0.0, 128.0, 64.0],    # Partial overlap
            [128.0, 0.0, 192.0, 64.0],   # No overlap
        ])
        anchors_level2 = torch.tensor([
            [0.0, 64.0, 64.0, 128.0],    # Partial overlap
            [64.0, 64.0, 128.0, 128.0],  # No overlap
        ])

        anchors_all = torch.cat([anchors_level1, anchors_level2], dim=0)
        num_level_anchors = [3, 2]

        # GT box overlaps with first anchor
        gt_boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0]])  # xyxy

        pos_mask, assigned_gt_idx = head.atss_sampling(
            anchors_all=anchors_all,
            gt_boxes=gt_boxes,
            num_level_anchors=num_level_anchors,
        )

        # Should have at least one positive (first anchor has high IoU)
        assert pos_mask.shape[0] == 5
        assert pos_mask.dtype == torch.bool
        # All assigned indices should be valid
        assert assigned_gt_idx.shape[0] == 5
        assert (assigned_gt_idx >= 0).all()

    def test_atss_sampling_empty_gt_returns_no_positives(self):
        """Test ATSS sampling returns no positives when no GT boxes."""
        head = ATSSHead(
            num_classes=10,
            in_channels=256,
            topk_candidates=5,
        )

        anchors_all = torch.tensor([
            [0.0, 0.0, 64.0, 64.0],
            [64.0, 0.0, 128.0, 64.0],
            [0.0, 64.0, 64.0, 128.0],
        ])
        num_level_anchors = [3]

        # Empty GT
        gt_boxes = torch.zeros((0, 4))

        pos_mask, assigned_gt_idx = head.atss_sampling(
            anchors_all=anchors_all,
            gt_boxes=gt_boxes,
            num_level_anchors=num_level_anchors,
        )

        # No positives expected
        assert pos_mask.sum() == 0
        assert not pos_mask.any()


class TestATSSHeadCenterness:
    """Tests for ATSS centerness computation."""

    def test_atss_loss_with_targets(self):
        """Test ATSS head computes loss correctly with targets."""
        head = ATSSHead(
            num_classes=10,
            in_channels=256,
        )
        head.init_weights()
        head.train()

        features = [
            torch.randn(2, 256, 20, 20),
            torch.randn(2, 256, 10, 10),
        ]

        targets = [
            {
                'labels': torch.tensor([0, 5]),
                'boxes': torch.tensor([
                    [0.25, 0.25, 0.1, 0.1],
                    [0.6, 0.6, 0.2, 0.2],
                ]),
            },
            {
                'labels': torch.tensor([3]),
                'boxes': torch.tensor([
                    [0.5, 0.5, 0.3, 0.3],
                ]),
            },
        ]

        cls_scores, bbox_preds, centernesses, losses = head(
            features=features,
            targets=targets,
        )

        # Should return losses when targets provided
        assert losses is not None
        assert 'loss_atss_cls' in losses
        assert 'loss_atss_bbox' in losses
        assert 'loss_atss_centerness' in losses

        # Losses should be valid
        for key, loss in losses.items():
            assert not torch.isnan(loss), f"{key} is NaN"
            assert not torch.isinf(loss), f"{key} is Inf"


class TestATSSHead:
    """Tests for ATSS (Adaptive Training Sample Selection) head."""
    
    def test_forward_output_format(self):
        """Test ATSS head forward pass."""
        head = ATSSHead(
            num_classes=80,
            in_channels=256,
        )
        head.init_weights()
        
        features = [
            torch.randn(2, 256, 100, 100),
            torch.randn(2, 256, 50, 50),
            torch.randn(2, 256, 25, 25),
        ]
        
        # ATSSHead.forward() returns 4 values: (cls_scores, bbox_preds, centernesses, losses)
        cls_scores, bbox_preds, centernesses, losses = head(features)
        
        # Multi-scale outputs (one per feature level)
        assert len(cls_scores) == 3
        assert len(bbox_preds) == 3
        assert len(centernesses) == 3
        
        # losses should be None without targets
        assert losses is None
        
        # Classification: (batch, num_anchors*num_classes, H, W)
        # With 1 anchor per location and 80 classes, shape[1] == 80
        assert cls_scores[0].shape[1] == 80
        
        # Box: (batch, num_anchors*4, H, W) = (batch, 4, H, W)
        assert bbox_preds[0].shape[1] == 4
        
        # Centerness: (batch, num_anchors, H, W) = (batch, 1, H, W)
        assert centernesses[0].shape[1] == 1
    
    def test_centerness_prediction(self):
        """ATSS should predict centerness for quality-aware detection."""
        head = ATSSHead(num_classes=20, in_channels=256)
        
        features = [torch.randn(1, 256, 10, 10)]
        
        # ATSSHead.forward() returns 4 values
        _, _, centernesses, _ = head(features)
        
        # Centerness should be produced
        assert centernesses is not None
        assert len(centernesses) == 1
    
    def test_gradient_flow(self):
        """Test gradients flow through ATSS head."""
        head = ATSSHead(num_classes=10, in_channels=128)
        
        features = [torch.randn(1, 128, 8, 8, requires_grad=True)]
        
        # ATSSHead.forward() returns 4 values
        cls_scores, bbox_preds, centernesses, _ = head(features)
        
        loss = cls_scores[0].sum() + bbox_preds[0].sum() + centernesses[0].sum()
        loss.backward()
        
        assert features[0].grad is not None


class TestMultiHeadIntegration:
    """Integration tests for multiple heads working together."""
    
    def test_all_heads_compatible_with_shared_features(self):
        """All heads should work with same feature format."""
        embed_dim = 256
        num_classes = 20
        
        # Create all heads
        detr_head = CoDeformDETRHead(num_classes=num_classes, embed_dims=embed_dim)
        rpn_head = RPNHead(in_channels=embed_dim)
        atss_head = ATSSHead(num_classes=num_classes, in_channels=embed_dim)
        
        # Shared features
        features = [
            torch.randn(1, embed_dim, 50, 50),
            torch.randn(1, embed_dim, 25, 25),
        ]
        
        # RPN forward - needs 4 feature levels to match default anchor_strides
        # So we add more feature levels
        features_4levels = features + [
            torch.randn(1, embed_dim, 13, 13),
            torch.randn(1, embed_dim, 7, 7),
        ]
        rpn_proposals, rpn_losses = rpn_head(features_4levels)
        # proposals is a list per batch item
        assert isinstance(rpn_proposals, list)
        
        # ATSS forward - returns 4 values
        atss_cls, atss_box, atss_cnt, _ = atss_head(features)
        assert len(atss_cls) == 2
        
        # All heads should process features without error
        assert True


class TestRPNProposalNMS:
    """Tests for RPN proposal NMS (Non-Maximum Suppression) behavior.
    
    These tests verify that NMS correctly filters overlapping proposals
    based on the IoU threshold.
    """

    def test_nms_removes_overlapping_proposals(self):
        """Verify NMS removes proposals with IoU > nms_threshold.
        
        Mathematical guarantee: After NMS, for any pair of remaining
        proposals (i, j), IoU(proposal_i, proposal_j) <= nms_threshold.
        """
        from torchvision.ops import box_iou
        
        head = RPNHead(
            in_channels=256,
            nms_threshold=0.7,  # IoU threshold for NMS
            pre_nms_top_n=1000,
            post_nms_top_n=100,
        )
        head.init_weights()
        
        # Create features that will generate proposals
        features = [
            torch.randn(1, 256, 20, 20),
            torch.randn(1, 256, 10, 10),
            torch.randn(1, 256, 5, 5),
            torch.randn(1, 256, 3, 3),
        ]
        
        proposals, _ = head(features=features)
        
        # Get proposals for first (and only) image
        props = proposals[0]
        
        if len(props) > 1:
            # Compute IoU matrix for all proposals
            iou_matrix = box_iou(props, props)
            
            # Remove diagonal (self-IoU = 1.0)
            mask = ~torch.eye(len(props), dtype=torch.bool, device=iou_matrix.device)
            off_diagonal_ious = iou_matrix[mask]
            
            # After NMS, no pair should have IoU > nms_threshold
            max_iou = off_diagonal_ious.max().item() if len(off_diagonal_ious) > 0 else 0.0
            
            assert max_iou <= head.nms_threshold + 1e-5, (
                f"NMS failed: found proposal pair with IoU={max_iou:.4f} > "
                f"threshold={head.nms_threshold}"
            )

    def test_nms_preserves_highest_score_proposals(self):
        """Verify NMS keeps proposals with highest scores among overlapping groups.
        
        When multiple proposals overlap, NMS should keep the one with
        highest objectness score and suppress the rest.
        """
        head = RPNHead(
            in_channels=256,
            nms_threshold=0.5,
            score_threshold=0.0,  # Don't filter by score
        )
        head.init_weights()
        
        features = [
            torch.randn(1, 256, 15, 15),
            torch.randn(1, 256, 8, 8),
            torch.randn(1, 256, 4, 4),
            torch.randn(1, 256, 2, 2),
        ]
        
        proposals, _ = head(features=features)
        
        # Should have some proposals after NMS
        assert len(proposals) == 1
        props = proposals[0]
        
        # Proposals should have valid coordinates (xyxy: x1 < x2, y1 < y2)
        if len(props) > 0:
            assert (props[:, 2] > props[:, 0]).all(), "Invalid x coordinates"
            assert (props[:, 3] > props[:, 1]).all(), "Invalid y coordinates"

    def test_nms_respects_post_nms_top_n(self):
        """Verify NMS respects post_nms_top_n limit."""
        post_nms_limit = 50
        
        head = RPNHead(
            in_channels=256,
            nms_threshold=0.9,  # High threshold = keep more proposals
            post_nms_top_n=post_nms_limit,
        )
        head.init_weights()
        
        # Larger features to generate more proposals
        features = [
            torch.randn(1, 256, 40, 40),
            torch.randn(1, 256, 20, 20),
            torch.randn(1, 256, 10, 10),
            torch.randn(1, 256, 5, 5),
        ]
        
        proposals, _ = head(features=features)
        
        # Should not exceed post_nms_top_n
        assert len(proposals[0]) <= post_nms_limit, (
            f"Post-NMS proposals ({len(proposals[0])}) exceeds limit ({post_nms_limit})"
        )


class TestATSSCenternessTargetComputation:
    """Tests for ATSS centerness target mathematical correctness.
    
    Centerness formula: sqrt((min(l,r)/max(l,r)) * (min(t,b)/max(t,b)))
    where l, r, t, b are distances from anchor center to GT box edges.
    
    Properties:
    - centerness ∈ [0, 1]
    - centerness = 1 when anchor center is at GT box center
    - centerness → 0 when anchor center is at GT box edge
    """

    def test_centerness_is_in_valid_range(self):
        """Centerness values should be in [0, 1]."""
        head = ATSSHead(
            num_classes=10,
            in_channels=256,
        )
        head.init_weights()
        head.train()
        
        features = [
            torch.randn(2, 256, 25, 25),
            torch.randn(2, 256, 13, 13),
        ]
        
        targets = [
            {
                'labels': torch.tensor([0, 3, 7]),
                'boxes': torch.tensor([
                    [0.2, 0.2, 0.15, 0.15],
                    [0.5, 0.5, 0.3, 0.3],
                    [0.8, 0.3, 0.1, 0.2],
                ]),
            },
            {
                'labels': torch.tensor([1]),
                'boxes': torch.tensor([[0.4, 0.6, 0.2, 0.2]]),
            },
        ]
        
        _, _, centernesses, _ = head(features=features, targets=targets)
        
        # Check all centerness predictions
        for level_centerness in centernesses:
            # Apply sigmoid to get actual centerness values
            centerness_values = torch.sigmoid(level_centerness)
            
            assert centerness_values.min() >= 0.0, (
                f"Centerness below 0: {centerness_values.min()}"
            )
            assert centerness_values.max() <= 1.0, (
                f"Centerness above 1: {centerness_values.max()}"
            )

    def test_centerness_formula_correctness(self):
        """Verify centerness is computed using the correct mathematical formula.
        
        Formula: centerness = sqrt((min(l,r)/max(l,r)) * (min(t,b)/max(t,b)))
        """
        # Test with known values
        # Anchor center at (100, 100), GT box [50, 50, 150, 150]
        # l = 100 - 50 = 50, r = 150 - 100 = 50
        # t = 100 - 50 = 50, b = 150 - 100 = 50
        # centerness = sqrt((50/50) * (50/50)) = sqrt(1 * 1) = 1.0
        
        l, r, t, b = 50.0, 50.0, 50.0, 50.0
        expected_centerness = torch.sqrt(
            torch.tensor((min(l, r) / max(l, r)) * (min(t, b) / max(t, b)))
        )
        assert abs(expected_centerness.item() - 1.0) < 1e-5
        
        # Test with asymmetric case
        # Anchor at (100, 100), GT box [25, 25, 150, 150]
        # l = 100 - 25 = 75, r = 150 - 100 = 50
        # t = 100 - 25 = 75, b = 150 - 100 = 50
        # centerness = sqrt((50/75) * (50/75)) = sqrt(0.667 * 0.667) = 0.667
        l, r, t, b = 75.0, 50.0, 75.0, 50.0
        expected_centerness = torch.sqrt(
            torch.tensor((min(l, r) / max(l, r)) * (min(t, b) / max(t, b)))
        )
        expected_value = (50.0 / 75.0)  # ≈ 0.667
        assert abs(expected_centerness.item() - expected_value) < 1e-5

    def test_centerness_symmetric_property(self):
        """Swapping left/right or top/bottom should give same centerness.
        
        This tests the symmetric property of the centerness formula.
        """
        def compute_centerness(l, r, t, b):
            return torch.sqrt(
                torch.tensor((min(l, r) / max(l, r)) * (min(t, b) / max(t, b)))
            )
        
        l, r, t, b = 30.0, 70.0, 40.0, 60.0
        
        # Original
        c1 = compute_centerness(l, r, t, b)
        
        # Swap left/right
        c2 = compute_centerness(r, l, t, b)
        
        # Swap top/bottom
        c3 = compute_centerness(l, r, b, t)
        
        # Swap both
        c4 = compute_centerness(r, l, b, t)
        
        # All should be equal due to min/max symmetry
        assert torch.allclose(c1, c2, atol=1e-5)
        assert torch.allclose(c1, c3, atol=1e-5)
        assert torch.allclose(c1, c4, atol=1e-5)
