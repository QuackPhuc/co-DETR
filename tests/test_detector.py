"""
Tests for CoDETR detector.

This module tests the complete Co-Deformable DETR detector including
training and inference modes, loss aggregation, and auxiliary heads.
"""

import pytest
import torch
from typing import List, Dict

from codetr.models.detector import CoDETR, build_codetr


class TestCoDETRForwardPass:
    """Tests for CoDETR forward pass in training and inference modes."""
    
    def test_forward_train_returns_losses(self):
        """Training mode should return loss dictionary."""
        model = CoDETR(
            num_classes=20,
            embed_dim=256,
            num_queries=100,
            num_encoder_layers=1,
            num_decoder_layers=1,
            use_rpn=False,
            use_roi=False,
            use_atss=False,
            use_dn=False,
            pretrained_backbone=False,
        )
        model.train()
        
        images = torch.randn(2, 3, 256, 256)
        targets = [
            {'labels': torch.tensor([0, 1, 2]), 'boxes': torch.rand(3, 4)},
            {'labels': torch.tensor([5, 10]), 'boxes': torch.rand(2, 4)},
        ]
        
        outputs = model(images, targets)
        
        # Training mode returns losses
        assert isinstance(outputs, dict)
        assert 'loss_cls' in outputs or 'query_head' in str(outputs.keys())
    
    def test_forward_inference_returns_predictions(self):
        """Inference mode should return list of prediction dicts."""
        model = CoDETR(
            num_classes=20,
            embed_dim=256,
            num_queries=50,
            num_encoder_layers=1,
            num_decoder_layers=1,
            use_rpn=False,
            use_roi=False,
            use_atss=False,
            use_dn=False,
            pretrained_backbone=False,
        )
        model.eval()
        
        images = torch.randn(2, 3, 256, 256)
        
        with torch.no_grad():
            outputs = model(images)
        
        # Inference mode returns list of predictions
        assert isinstance(outputs, list)
        assert len(outputs) == 2  # One per batch item
        
        for pred in outputs:
            assert 'scores' in pred
            assert 'labels' in pred
            assert 'boxes' in pred


class TestCoDETROutputShapes:
    """Tests for intermediate output shapes."""
    
    def test_extract_feat_output_shapes(self):
        """Test feature extraction produces correct shapes."""
        model = CoDETR(
            num_classes=20,
            embed_dim=256,
            num_feature_levels=4,
            pretrained_backbone=False,
        )
        
        images = torch.randn(2, 3, 256, 256)
        
        features, masks, pos_embeds = model.extract_feat(images)
        
        # Should return 4 feature levels
        assert len(features) == 4
        assert len(masks) == 4
        assert len(pos_embeds) == 4
        
        # All features should have embed_dim channels
        for feat in features:
            assert feat.shape[1] == 256
        
        # Masks should match spatial dimensions
        for feat, mask in zip(features, masks):
            assert mask.shape[1:] == feat.shape[2:]


class TestCoDETRAuxiliaryHeads:
    """Tests for auxiliary detection heads."""
    
    def test_with_rpn_head(self):
        """Test model with RPN auxiliary head."""
        model = CoDETR(
            num_classes=20,
            embed_dim=256,
            num_queries=50,
            num_encoder_layers=1,
            num_decoder_layers=1,
            use_rpn=True,
            use_roi=False,
            use_atss=False,
            pretrained_backbone=False,
        )
        model.train()
        
        assert model.rpn_head is not None
        
        images = torch.randn(1, 3, 256, 256)
        targets = [{'labels': torch.tensor([0]), 'boxes': torch.rand(1, 4)}]
        
        outputs = model(images, targets)
        
        # Should include RPN losses
        assert isinstance(outputs, dict)
    
    def test_with_atss_head(self):
        """Test model with ATSS auxiliary head."""
        model = CoDETR(
            num_classes=20,
            embed_dim=256,
            num_queries=50,
            num_encoder_layers=1,
            num_decoder_layers=1,
            use_rpn=False,
            use_roi=False,
            use_atss=True,
            pretrained_backbone=False,
        )
        model.train()
        
        assert model.atss_head is not None
    
    def test_all_aux_heads_disabled(self):
        """Test model with no auxiliary heads."""
        model = CoDETR(
            num_classes=20,
            embed_dim=256,
            use_rpn=False,
            use_roi=False,
            use_atss=False,
            pretrained_backbone=False,
        )
        
        assert model.rpn_head is None
        assert model.roi_head is None
        assert model.atss_head is None


class TestCoDETRGradientFlow:
    """Tests for gradient flow through entire model."""
    
    def test_gradient_flow_main_head(self):
        """Test gradients flow through main detection head."""
        model = CoDETR(
            num_classes=10,
            embed_dim=128,
            num_queries=20,
            num_encoder_layers=1,
            num_decoder_layers=1,
            use_rpn=False,
            use_roi=False,
            use_atss=False,
            use_dn=False,
            pretrained_backbone=False,
            frozen_backbone_stages=0,
        )
        model.train()
        
        images = torch.randn(1, 3, 128, 128, requires_grad=True)
        targets = [{'labels': torch.tensor([0, 1]), 'boxes': torch.rand(2, 4)}]
        
        outputs = model(images, targets)
        
        # Sum all losses
        total_loss = sum(v for v in outputs.values() if isinstance(v, torch.Tensor))
        total_loss.backward()
        
        # Input should have gradient
        assert images.grad is not None
        assert not torch.isnan(images.grad).any()
    
    def test_no_nan_in_outputs(self):
        """Model outputs should not contain NaN values."""
        model = CoDETR(
            num_classes=20,
            embed_dim=256,
            num_queries=50,
            pretrained_backbone=False,
        )
        model.eval()
        
        images = torch.randn(1, 3, 256, 256)
        
        with torch.no_grad():
            outputs = model(images)
        
        for pred in outputs:
            assert not torch.isnan(pred['scores']).any()
            assert not torch.isnan(pred['boxes']).any()


class TestBuildCoDETR:
    """Tests for build_codetr factory function."""
    
    def test_build_with_aux_heads(self):
        """Test building model with auxiliary heads enabled."""
        model = build_codetr(
            num_classes=80,
            pretrained_backbone=False,
            use_aux_heads=True,
        )
        
        assert isinstance(model, CoDETR)
        assert model.rpn_head is not None or model.atss_head is not None
    
    def test_build_without_aux_heads(self):
        """Test building model without auxiliary heads."""
        model = build_codetr(
            num_classes=80,
            pretrained_backbone=False,
            use_aux_heads=False,
        )
        
        assert isinstance(model, CoDETR)


class TestCoDETRMemoryEfficiency:
    """Tests for memory-related behavior."""
    
    def test_model_parameter_count(self):
        """Test model has reasonable parameter count."""
        model = CoDETR(
            num_classes=80,
            embed_dim=256,
            num_queries=300,
            num_encoder_layers=6,
            num_decoder_layers=6,
            use_rpn=False,
            use_roi=False,
            use_atss=False,
            pretrained_backbone=False,
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Model should have params (rough estimate: 30-50M for DETR variant)
        assert total_params > 1_000_000  # At least 1M params
        assert trainable_params > 0
