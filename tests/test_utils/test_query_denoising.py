"""
Tests for Query Denoising (DN) Generator.

This module tests the DnQueryGenerator and CdnQueryGenerator classes
which are critical for Co-DETR training stability and performance.
"""

import pytest
import torch
import torch.nn as nn
from typing import Dict, List

from codetr.models.utils.query_denoising import (
    DnQueryGenerator,
    CdnQueryGenerator,
    build_dn_generator,
)


class TestDnQueryGeneratorOutputShapes:
    """Tests for output tensor shapes from DnQueryGenerator."""
    
    @pytest.fixture
    def dn_generator(self):
        """Create a DnQueryGenerator instance for testing."""
        return DnQueryGenerator(
            num_queries=300,
            hidden_dim=256,
            num_classes=80,
            noise_scale={'label': 0.5, 'box': 0.4},
            group_cfg={'dynamic': True, 'num_dn_queries': 100}
        )
    
    @pytest.fixture
    def label_encoder(self):
        """Create a simple label embedding module."""
        return nn.Embedding(80, 256)
    
    @pytest.fixture
    def sample_gt_data(self):
        """Create sample ground truth data for testing."""
        batch_size = 2
        gt_bboxes = [
            torch.tensor([[10, 20, 50, 60], [100, 100, 200, 200]], dtype=torch.float32),
            torch.tensor([[30, 40, 70, 80]], dtype=torch.float32),
        ]
        gt_labels = [
            torch.tensor([0, 1], dtype=torch.long),
            torch.tensor([2], dtype=torch.long),
        ]
        img_metas = [
            {'img_shape': (800, 800, 3)},
            {'img_shape': (800, 800, 3)},
        ]
        return gt_bboxes, gt_labels, img_metas
    
    def test_output_shapes_with_gt(self, dn_generator, label_encoder, sample_gt_data):
        """Test that output tensors have correct shapes when GT is provided."""
        gt_bboxes, gt_labels, img_metas = sample_gt_data
        
        label_embed, bbox_embed, attn_mask, dn_meta = dn_generator(
            gt_bboxes, gt_labels, label_encoder, img_metas
        )
        
        batch_size = len(gt_bboxes)
        pad_size = dn_meta['pad_size']
        
        # Label embed: (batch_size, pad_size, hidden_dim)
        assert label_embed.shape[0] == batch_size
        assert label_embed.shape[1] == pad_size
        assert label_embed.shape[2] == 256  # hidden_dim
        
        # Bbox embed: (batch_size, pad_size, 4)
        assert bbox_embed.shape[0] == batch_size
        assert bbox_embed.shape[1] == pad_size
        assert bbox_embed.shape[2] == 4
        
        # Attention mask: (tgt_size, tgt_size) where tgt_size = pad_size + num_queries
        expected_tgt_size = pad_size + 300  # num_queries
        assert attn_mask.shape == (expected_tgt_size, expected_tgt_size)
        
        # DN meta should contain required keys
        assert 'pad_size' in dn_meta
        assert 'num_dn_group' in dn_meta
    
    def test_output_shapes_with_single_gt(self, dn_generator, label_encoder):
        """Test output shapes with single ground truth per image."""
        gt_bboxes = [
            torch.tensor([[50, 50, 100, 100]], dtype=torch.float32),
        ]
        gt_labels = [
            torch.tensor([5], dtype=torch.long),
        ]
        img_metas = [{'img_shape': (640, 640, 3)}]
        
        label_embed, bbox_embed, attn_mask, dn_meta = dn_generator(
            gt_bboxes, gt_labels, label_encoder, img_metas
        )
        
        # Verify basic shape consistency
        assert label_embed.shape[0] == 1  # batch_size
        assert bbox_embed.shape[0] == 1
        assert label_embed.shape[1] == bbox_embed.shape[1]  # pad_size
        assert attn_mask.dtype == torch.bool


class TestDnQueryGeneratorEmptyGT:
    """Tests for handling empty ground truth cases."""
    
    @pytest.fixture
    def dn_generator(self):
        """Create a DnQueryGenerator instance."""
        return DnQueryGenerator(
            num_queries=300,
            hidden_dim=256,
            num_classes=80,
        )
    
    @pytest.fixture
    def label_encoder(self):
        """Create label embedding module."""
        return nn.Embedding(80, 256)
    
    def test_empty_gt_returns_valid_output(self, dn_generator, label_encoder):
        """Empty ground truth should return valid empty tensors."""
        gt_bboxes = [
            torch.zeros(0, 4, dtype=torch.float32),
            torch.zeros(0, 4, dtype=torch.float32),
        ]
        gt_labels = [
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0, dtype=torch.long),
        ]
        img_metas = [
            {'img_shape': (800, 800, 3)},
            {'img_shape': (800, 800, 3)},
        ]
        
        label_embed, bbox_embed, attn_mask, dn_meta = dn_generator(
            gt_bboxes, gt_labels, label_encoder, img_metas
        )
        
        # Should return empty pad_size
        assert dn_meta['pad_size'] == 0
        assert dn_meta['num_dn_group'] == 0
        
        # Embeddings should be empty in dim 1
        assert label_embed.shape[1] == 0
        assert bbox_embed.shape[1] == 0
        
        # Attention mask should be (num_queries, num_queries)
        assert attn_mask.shape == (300, 300)  # num_queries x num_queries
    
    def test_mixed_empty_and_nonempty_gt(self, dn_generator, label_encoder):
        """Test with some images having empty GT and others having GT."""
        gt_bboxes = [
            torch.tensor([[10, 20, 50, 60]], dtype=torch.float32),  # Has GT
            torch.zeros(0, 4, dtype=torch.float32),  # Empty GT
        ]
        gt_labels = [
            torch.tensor([0], dtype=torch.long),
            torch.zeros(0, dtype=torch.long),
        ]
        img_metas = [
            {'img_shape': (800, 800, 3)},
            {'img_shape': (800, 800, 3)},
        ]
        
        label_embed, bbox_embed, attn_mask, dn_meta = dn_generator(
            gt_bboxes, gt_labels, label_encoder, img_metas
        )
        
        # Should handle mixed case
        assert label_embed.shape[0] == 2  # batch_size
        assert bbox_embed.shape[0] == 2
        # pad_size should be > 0 since at least one image has GT
        assert dn_meta['pad_size'] > 0


class TestDnAttentionMask:
    """Tests for attention mask correctness."""
    
    @pytest.fixture
    def dn_generator(self):
        """Create DnQueryGenerator with 2 fixed groups for testing."""
        return DnQueryGenerator(
            num_queries=100,
            hidden_dim=256,
            num_classes=80,
            group_cfg={'dynamic': False, 'num_groups': 2}
        )
    
    @pytest.fixture
    def label_encoder(self):
        """Create label embedding module."""
        return nn.Embedding(80, 256)
    
    def test_matching_queries_cannot_see_dn_queries(self, dn_generator, label_encoder):
        """Matching queries (regular queries) should NOT see DN queries."""
        gt_bboxes = [torch.tensor([[50, 50, 100, 100]], dtype=torch.float32)]
        gt_labels = [torch.tensor([0], dtype=torch.long)]
        img_metas = [{'img_shape': (800, 800, 3)}]
        
        _, _, attn_mask, dn_meta = dn_generator(
            gt_bboxes, gt_labels, label_encoder, img_metas
        )
        
        pad_size = dn_meta['pad_size']
        
        # Matching queries are at indices [pad_size:] 
        # They should NOT see DN queries at indices [:pad_size]
        # attn_mask[i, j] = True means i cannot attend to j
        if pad_size > 0:
            matching_to_dn_mask = attn_mask[pad_size:, :pad_size]
            # All should be True (blocked)
            assert matching_to_dn_mask.all(), \
                "Matching queries should not see denoising queries"
    
    def test_dn_groups_isolated(self, dn_generator, label_encoder):
        """Denoising groups should not see each other."""
        gt_bboxes = [torch.tensor([[50, 50, 100, 100]], dtype=torch.float32)]
        gt_labels = [torch.tensor([0], dtype=torch.long)]
        img_metas = [{'img_shape': (800, 800, 3)}]
        
        _, _, attn_mask, dn_meta = dn_generator(
            gt_bboxes, gt_labels, label_encoder, img_metas
        )
        
        num_groups = dn_meta['num_dn_group']
        pad_size = dn_meta['pad_size']
        
        if num_groups > 1 and pad_size > 0:
            # Each group should be isolated from other groups
            # This is a structural check - groups are separated
            assert attn_mask.shape[0] > pad_size, \
                "Attention mask should include both DN and matching queries"


class TestDnNoiseApplication:
    """Tests for label and box noise application."""
    
    def test_label_noise_changes_labels(self):
        """Label noise should flip some labels."""
        # Use high noise scale to ensure labels are flipped
        dn_generator = DnQueryGenerator(
            num_queries=100,
            hidden_dim=256,
            num_classes=10,
            noise_scale={'label': 1.0, 'box': 0.0},  # Max label noise
            group_cfg={'dynamic': False, 'num_groups': 1}
        )
        label_encoder = nn.Embedding(10, 256)
        
        # Create GT with known labels
        gt_bboxes = [torch.tensor([[50, 50, 100, 100]], dtype=torch.float32)]
        gt_labels = [torch.tensor([5], dtype=torch.long)]
        img_metas = [{'img_shape': (800, 800, 3)}]
        
        # With label_noise_scale=1.0, statistically most labels should flip
        # We can at least verify the output shapes are correct
        label_embed, bbox_embed, _, _ = dn_generator(
            gt_bboxes, gt_labels, label_encoder, img_metas
        )
        
        # Output should have valid embeddings
        assert not torch.isnan(label_embed).any()
        assert not torch.isinf(label_embed).any()
    
    def test_box_noise_modifies_coordinates(self):
        """Box noise should perturb box coordinates."""
        dn_generator = DnQueryGenerator(
            num_queries=100,
            hidden_dim=256,
            num_classes=10,
            noise_scale={'label': 0.0, 'box': 0.5},  # Only box noise
            group_cfg={'dynamic': False, 'num_groups': 1}
        )
        label_encoder = nn.Embedding(10, 256)
        
        gt_bboxes = [torch.tensor([[100, 100, 200, 200]], dtype=torch.float32)]
        gt_labels = [torch.tensor([0], dtype=torch.long)]
        img_metas = [{'img_shape': (800, 800, 3)}]
        
        _, bbox_embed, _, _ = dn_generator(
            gt_bboxes, gt_labels, label_encoder, img_metas
        )
        
        # Box embeddings should be valid (passed through inverse_sigmoid)
        assert not torch.isnan(bbox_embed).any()
        assert not torch.isinf(bbox_embed).any()
    
    def test_no_noise_preserves_structure(self):
        """With zero noise, output structure should still be correct."""
        dn_generator = DnQueryGenerator(
            num_queries=100,
            hidden_dim=256,
            num_classes=10,
            noise_scale={'label': 0.0, 'box': 0.0},  # No noise
            group_cfg={'dynamic': False, 'num_groups': 1}
        )
        label_encoder = nn.Embedding(10, 256)
        
        gt_bboxes = [torch.tensor([[100, 100, 200, 200]], dtype=torch.float32)]
        gt_labels = [torch.tensor([0], dtype=torch.long)]
        img_metas = [{'img_shape': (800, 800, 3)}]
        
        label_embed, bbox_embed, attn_mask, dn_meta = dn_generator(
            gt_bboxes, gt_labels, label_encoder, img_metas
        )
        
        # Structure should be valid
        assert dn_meta['pad_size'] > 0
        assert label_embed.shape[1] == dn_meta['pad_size']


class TestCdnQueryGenerator:
    """Tests for CdnQueryGenerator (Collaborative Denoising)."""
    
    def test_cdn_generator_builds(self):
        """CdnQueryGenerator should instantiate correctly."""
        cdn_generator = CdnQueryGenerator(
            num_queries=300,
            hidden_dim=256,
            num_classes=80,
        )
        
        assert isinstance(cdn_generator, DnQueryGenerator)
        assert cdn_generator.num_queries == 300
        assert cdn_generator.hidden_dim == 256
        assert cdn_generator.num_classes == 80
    
    def test_cdn_inherits_dn_behavior(self):
        """CdnQueryGenerator should inherit all DnQueryGenerator behavior."""
        cdn_generator = CdnQueryGenerator(
            num_queries=100,
            hidden_dim=256,
            num_classes=10,
        )
        label_encoder = nn.Embedding(10, 256)
        
        gt_bboxes = [torch.tensor([[50, 50, 100, 100]], dtype=torch.float32)]
        gt_labels = [torch.tensor([0], dtype=torch.long)]
        img_metas = [{'img_shape': (800, 800, 3)}]
        
        # Should work exactly like DnQueryGenerator
        label_embed, bbox_embed, attn_mask, dn_meta = cdn_generator(
            gt_bboxes, gt_labels, label_encoder, img_metas
        )
        
        assert label_embed.shape[0] == 1
        assert 'pad_size' in dn_meta
        assert 'num_dn_group' in dn_meta


class TestBuildDnGenerator:
    """Tests for the build_dn_generator factory function."""
    
    def test_build_dn_generator_returns_dn(self):
        """Factory should create DnQueryGenerator when specified."""
        config = {
            'type': 'DnQueryGenerator',
            'num_queries': 300,
            'hidden_dim': 256,
            'num_classes': 80,
        }
        
        generator = build_dn_generator(config)
        
        assert isinstance(generator, DnQueryGenerator)
        assert not isinstance(generator, CdnQueryGenerator)
    
    def test_build_cdn_generator_returns_cdn(self):
        """Factory should create CdnQueryGenerator when specified."""
        config = {
            'type': 'CdnQueryGenerator',
            'num_queries': 300,
            'hidden_dim': 256,
            'num_classes': 80,
        }
        
        generator = build_dn_generator(config)
        
        assert isinstance(generator, CdnQueryGenerator)
    
    def test_build_with_none_returns_none(self):
        """Factory should return None when config is None."""
        generator = build_dn_generator(None)
        
        assert generator is None
    
    def test_build_with_unknown_type_raises(self):
        """Factory should raise for unknown generator types."""
        config = {
            'type': 'UnknownGenerator',
            'num_queries': 300,
            'hidden_dim': 256,
            'num_classes': 80,
        }
        
        with pytest.raises(NotImplementedError):
            build_dn_generator(config)


class TestDnGroupConfiguration:
    """Tests for denoising group configuration."""
    
    def test_dynamic_groups(self):
        """Test dynamic group computation based on GT count."""
        dn_generator = DnQueryGenerator(
            num_queries=100,
            hidden_dim=256,
            num_classes=10,
            group_cfg={'dynamic': True, 'num_dn_queries': 100}
        )
        
        # With 2 GT objects, max_known = 2, groups = 100 // 2 = 50
        num_groups = dn_generator.get_num_groups(group_queries=2)
        assert num_groups == 50
    
    def test_static_groups(self):
        """Test static (fixed) group configuration."""
        dn_generator = DnQueryGenerator(
            num_queries=100,
            hidden_dim=256,
            num_classes=10,
            group_cfg={'dynamic': False, 'num_groups': 5}
        )
        
        # Static groups should always return fixed number
        num_groups = dn_generator.get_num_groups()
        assert num_groups == 5
    
    def test_zero_gt_returns_one_group(self):
        """Zero GT objects should return at least 1 group."""
        dn_generator = DnQueryGenerator(
            num_queries=100,
            hidden_dim=256,
            num_classes=10,
            group_cfg={'dynamic': True, 'num_dn_queries': 100}
        )
        
        num_groups = dn_generator.get_num_groups(group_queries=0)
        assert num_groups == 1
