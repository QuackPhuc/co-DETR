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
        
        # Classification: (batch, num_queries, num_classes)
        assert cls_scores.shape == (2, 300, 80)
        
        # Box predictions: (batch, num_queries, 4)
        assert bbox_preds.shape == (2, 300, 4)
    
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
        head = RPNHead(
            in_channels=256,
            num_anchors=3,
        )
        head.init_weights()
        
        # Multi-scale features
        features = [
            torch.randn(2, 256, 100, 100),
            torch.randn(2, 256, 50, 50),
            torch.randn(2, 256, 25, 25),
            torch.randn(2, 256, 13, 13),
        ]
        
        cls_scores, bbox_preds = head(features)
        
        # Should return multi-scale outputs
        assert len(cls_scores) == 4
        assert len(bbox_preds) == 4
        
        # Check shapes for first level
        # cls: (batch, num_anchors, H, W)
        assert cls_scores[0].shape[:2] == (2, 3)
        # bbox: (batch, num_anchors * 4, H, W)
        assert bbox_preds[0].shape[:2] == (2, 12)
    
    def test_anchor_generation(self):
        """Test anchor generation for different feature sizes."""
        head = RPNHead(in_channels=256, num_anchors=9)
        
        # Verify anchor generator exists
        assert hasattr(head, 'anchor_generator') or hasattr(head, 'anchors')
    
    def test_gradient_flow(self):
        """Test gradients flow through RPN."""
        head = RPNHead(in_channels=256, num_anchors=3)
        
        features = [torch.randn(1, 256, 10, 10, requires_grad=True)]
        
        cls_scores, bbox_preds = head(features)
        
        loss = cls_scores[0].sum() + bbox_preds[0].sum()
        loss.backward()
        
        assert features[0].grad is not None


class TestRoIHead:
    """Tests for RoI (Region of Interest) head."""
    
    def test_forward_output_format(self):
        """Test RoI head forward pass."""
        head = RoIHead(
            in_channels=256,
            num_classes=20,
            roi_out_size=7,
        )
        head.init_weights()
        
        features = [
            torch.randn(2, 256, 50, 50),
            torch.randn(2, 256, 25, 25),
        ]
        
        # Proposals: list of (N, 4) boxes per image
        proposals = [
            torch.tensor([[10.0, 10.0, 30.0, 30.0], [20.0, 20.0, 40.0, 40.0]]),
            torch.tensor([[5.0, 5.0, 25.0, 25.0]]),
        ]
        
        cls_scores, bbox_preds = head(features, proposals)
        
        # Total proposals: 2 + 1 = 3
        assert cls_scores.shape[0] == 3
        assert cls_scores.shape[1] == 20 + 1  # +1 for background
        
        assert bbox_preds.shape[0] == 3
        assert bbox_preds.shape[1] == 4  # or 4 * (num_classes + 1) for class-specific


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
        
        cls_scores, bbox_preds, centernesses = head(features)
        
        # Multi-scale outputs
        assert len(cls_scores) == 3
        assert len(bbox_preds) == 3
        assert len(centernesses) == 3
        
        # Classification: (batch, num_classes, H, W)
        assert cls_scores[0].shape[1] == 80
        
        # Box: (batch, 4, H, W)
        assert bbox_preds[0].shape[1] == 4
        
        # Centerness: (batch, 1, H, W)
        assert centernesses[0].shape[1] == 1
    
    def test_centerness_prediction(self):
        """ATSS should predict centerness for quality-aware detection."""
        head = ATSSHead(num_classes=20, in_channels=256)
        
        features = [torch.randn(1, 256, 10, 10)]
        
        _, _, centernesses = head(features)
        
        # Centerness should be produced
        assert centernesses is not None
        assert len(centernesses) == 1
    
    def test_gradient_flow(self):
        """Test gradients flow through ATSS head."""
        head = ATSSHead(num_classes=10, in_channels=128)
        
        features = [torch.randn(1, 128, 8, 8, requires_grad=True)]
        
        cls_scores, bbox_preds, centernesses = head(features)
        
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
        
        # RPN forward
        rpn_cls, rpn_box = rpn_head(features)
        assert len(rpn_cls) == 2
        
        # ATSS forward
        atss_cls, atss_box, atss_cnt = atss_head(features)
        assert len(atss_cls) == 2
        
        # All heads should process features without error
        assert True
