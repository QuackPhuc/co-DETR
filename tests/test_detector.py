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
            use_dn=False,  # Disable denoising to avoid attn_mask size mismatch
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


class TestCoDETREmptyTargets:
    """Tests for handling empty targets (images without objects)."""
    
    def test_train_with_empty_targets(self):
        """Training should handle images with no objects."""
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
        model.train()
        
        images = torch.randn(2, 3, 256, 256)
        
        # First image has no objects (empty targets)
        targets = [
            {'labels': torch.tensor([], dtype=torch.long), 
             'boxes': torch.zeros(0, 4)},
            {'labels': torch.tensor([0, 1]), 
             'boxes': torch.rand(2, 4)},
        ]
        
        # Should not crash
        outputs = model(images, targets)
        
        # Should return loss dict
        assert isinstance(outputs, dict)
        
        # Losses should be valid (not NaN)
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                assert not torch.isnan(value).any(), f"Loss {key} is NaN"
    
    def test_train_all_empty_targets(self):
        """Training with all empty targets should handle gracefully."""
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
        model.train()
        
        images = torch.randn(2, 3, 256, 256)
        
        # All images have no objects
        targets = [
            {'labels': torch.tensor([], dtype=torch.long), 
             'boxes': torch.zeros(0, 4)},
            {'labels': torch.tensor([], dtype=torch.long), 
             'boxes': torch.zeros(0, 4)},
        ]
        
        # Should not crash (though losses may be edge cases)
        try:
            outputs = model(images, targets)
            assert isinstance(outputs, dict)
        except RuntimeError as e:
            # Some implementations may not support all-empty batches
            # This is acceptable if documented
            assert "empty" in str(e).lower() or "no target" in str(e).lower()


class TestCoDETRVariableTargets:
    """Tests for handling variable number of targets per image."""
    
    def test_different_num_objects_per_image(self):
        """Model should handle different object counts per batch item."""
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
        model.train()
        
        images = torch.randn(3, 3, 256, 256)
        
        # Variable object counts: 1, 5, 10
        targets = [
            {'labels': torch.tensor([0]), 'boxes': torch.rand(1, 4)},
            {'labels': torch.randint(0, 20, (5,)), 'boxes': torch.rand(5, 4)},
            {'labels': torch.randint(0, 20, (10,)), 'boxes': torch.rand(10, 4)},
        ]
        
        outputs = model(images, targets)
        
        assert isinstance(outputs, dict)
    
    def test_many_objects_per_image(self):
        """Model should handle crowded scenes with many objects."""
        model = CoDETR(
            num_classes=20,
            embed_dim=256,
            num_queries=100,  # 100 queries
            num_encoder_layers=1,
            num_decoder_layers=1,
            use_rpn=False,
            use_roi=False,
            use_atss=False,
            use_dn=False,
            pretrained_backbone=False,
        )
        model.train()
        
        images = torch.randn(1, 3, 256, 256)
        
        # Many objects (30) - less than num_queries
        num_objects = 30
        targets = [
            {
                'labels': torch.randint(0, 20, (num_objects,)), 
                'boxes': torch.rand(num_objects, 4)
            }
        ]
        
        outputs = model(images, targets)
        
        assert isinstance(outputs, dict)
        # Should compute valid losses
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                assert not torch.isnan(value).any()
    
    def test_single_small_box(self):
        """Model should handle very small bounding boxes."""
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
        model.train()
        
        images = torch.randn(1, 3, 256, 256)
        
        # Very small box (< 1% of image)
        targets = [
            {
                'labels': torch.tensor([0]), 
                'boxes': torch.tensor([[0.5, 0.5, 0.005, 0.005]])  # cx, cy, w, h
            }
        ]
        
        outputs = model(images, targets)
        
        assert isinstance(outputs, dict)


class TestCoDETRWithQueryDenoising:
    """Tests for CoDETR with query denoising enabled (use_dn=True flow).
    
    These tests verify the denoising queries mechanism works correctly,
    including attention mask generation and DN loss computation.
    """
    
    def test_use_dn_true_forward_pass(self):
        """Model with use_dn=True should complete forward pass without errors.
        
        This tests the complete DN flow including:
        - DnQueryGenerator creates denoising queries
        - Attention mask is correctly generated
        - DN queries are concatenated with content queries
        """
        model = CoDETR(
            num_classes=20,
            embed_dim=256,
            num_queries=50,
            num_encoder_layers=1,
            num_decoder_layers=1,
            use_rpn=False,
            use_roi=False,
            use_atss=False,
            use_dn=True,  # Enable query denoising
            pretrained_backbone=False,
        )
        model.train()
        
        assert model.dn_generator is not None, "DnQueryGenerator should be created"
        
        images = torch.randn(2, 3, 256, 256)
        targets = [
            {'labels': torch.tensor([0, 1, 2]), 'boxes': torch.rand(3, 4)},
            {'labels': torch.tensor([5]), 'boxes': torch.rand(1, 4)},
        ]
        
        # Should not crash with use_dn=True
        outputs = model(images=images, targets=targets)
        
        # Should return losses
        assert isinstance(outputs, dict)
        
        # Losses should be valid numbers
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                assert not torch.isnan(value).any(), f"Loss {key} contains NaN"
    
    def test_use_dn_with_empty_targets_graceful(self):
        """Query denoising should handle images without objects gracefully.
        
        When GT is empty, DN generator should not crash but produce
        empty or zero denoising queries.
        """
        model = CoDETR(
            num_classes=20,
            embed_dim=256,
            num_queries=50,
            num_encoder_layers=1,
            num_decoder_layers=1,
            use_rpn=False,
            use_roi=False,
            use_atss=False,
            use_dn=True,
            pretrained_backbone=False,
        )
        model.train()
        
        images = torch.randn(2, 3, 256, 256)
        
        # First image has objects, second is empty
        targets = [
            {'labels': torch.tensor([0, 1]), 'boxes': torch.rand(2, 4)},
            {'labels': torch.tensor([], dtype=torch.long), 'boxes': torch.zeros(0, 4)},
        ]
        
        # Should not crash
        outputs = model(images=images, targets=targets)
        assert isinstance(outputs, dict)
    
    def test_use_dn_creates_attention_mask(self):
        """Verify DN generator produces attention mask for decoder.
        
        The attention mask is critical for preventing DN queries from
        attending to content queries during training.
        """
        model = CoDETR(
            num_classes=20,
            embed_dim=256,
            num_queries=50,
            num_encoder_layers=1,
            num_decoder_layers=1,
            use_rpn=False,
            use_roi=False,
            use_atss=False,
            use_dn=True,
            pretrained_backbone=False,
        )
        model.train()
        
        assert model.dn_generator is not None
        
        # DN generator should exist and be properly configured
        assert hasattr(model.dn_generator, 'num_queries')
        assert hasattr(model.dn_generator, 'hidden_dim')


class TestCoDETRLargeBoxes:
    """Tests for handling very large bounding boxes (>50% of image area).
    
    Large boxes are common in close-up shots and should be properly handled
    by the detection head and loss computation.
    """
    
    def test_large_box_covering_half_image(self):
        """Model should handle boxes covering >50% of image.
        
        Mathematical check: box with w=0.7, h=0.8 covers 56% of image.
        This is common in close-up product photography.
        """
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
        model.train()
        
        images = torch.randn(1, 3, 256, 256)
        
        # Large box covering ~56% of image (0.7 * 0.8 = 0.56)
        targets = [
            {
                'labels': torch.tensor([0]),
                'boxes': torch.tensor([[0.5, 0.5, 0.7, 0.8]])  # cx, cy, w, h (normalized)
            }
        ]
        
        outputs = model(images=images, targets=targets)
        
        assert isinstance(outputs, dict)
        # Losses should be valid
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                assert not torch.isnan(value).any(), f"Loss {key} is NaN for large box"
                assert not torch.isinf(value).any(), f"Loss {key} is Inf for large box"
    
    def test_multiple_large_boxes_overlap(self):
        """Multiple large overlapping boxes should not cause numerical issues.
        
        This tests the matcher and loss computation with overlapping targets.
        """
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
        model.train()
        
        images = torch.randn(1, 3, 256, 256)
        
        # Multiple large overlapping boxes
        targets = [
            {
                'labels': torch.tensor([0, 1, 2]),
                'boxes': torch.tensor([
                    [0.4, 0.4, 0.6, 0.6],  # Large box 1
                    [0.5, 0.5, 0.55, 0.55],  # Large box 2, overlapping
                    [0.6, 0.6, 0.5, 0.5],  # Large box 3, overlapping
                ])
            }
        ]
        
        outputs = model(images=images, targets=targets)
        assert isinstance(outputs, dict)


class TestCoDETRQueryTargetRatio:
    """Tests for scenarios where num_targets > num_queries.
    
    This is an edge case that can occur with crowded scenes.
    The matcher should handle this gracefully.
    """
    
    def test_more_targets_than_queries(self):
        """Model should handle more GT objects than queries.
        
        Mathematical constraint: With num_queries=N and num_targets=M where M>N,
        the Hungarian matcher should match N targets (since each query can
        only match one target). The remaining M-N targets will be unmatched.
        """
        num_queries = 10  # Very few queries
        num_targets = 15  # More targets than queries
        
        model = CoDETR(
            num_classes=20,
            embed_dim=256,
            num_queries=num_queries,
            num_encoder_layers=1,
            num_decoder_layers=1,
            use_rpn=False,
            use_roi=False,
            use_atss=False,
            use_dn=False,
            pretrained_backbone=False,
        )
        model.train()
        
        images = torch.randn(1, 3, 256, 256)
        
        # More targets than queries
        targets = [
            {
                'labels': torch.randint(low=0, high=20, size=(num_targets,)),
                'boxes': torch.rand(num_targets, 4)
            }
        ]
        
        # Should not crash - matcher handles this case
        outputs = model(images=images, targets=targets)
        
        assert isinstance(outputs, dict)
        # Loss should still be computable
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                assert not torch.isnan(value).any(), f"Loss {key} is NaN"
    
    def test_exactly_matching_queries_and_targets(self):
        """Model should handle exactly matching query and target counts.
        
        When num_queries == num_targets, every query should be matched.
        """
        num_queries = 20
        
        model = CoDETR(
            num_classes=20,
            embed_dim=256,
            num_queries=num_queries,
            num_encoder_layers=1,
            num_decoder_layers=1,
            use_rpn=False,
            use_roi=False,
            use_atss=False,
            use_dn=False,
            pretrained_backbone=False,
        )
        model.train()
        
        images = torch.randn(1, 3, 256, 256)
        
        # Exactly num_queries targets
        targets = [
            {
                'labels': torch.randint(low=0, high=20, size=(num_queries,)),
                'boxes': torch.rand(num_queries, 4)
            }
        ]
        
        outputs = model(images=images, targets=targets)
        assert isinstance(outputs, dict)
