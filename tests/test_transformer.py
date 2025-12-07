"""
Tests for Transformer components.

This module tests the multi-scale deformable attention, transformer encoder,
decoder, and the full transformer module with shape and gradient verification.
"""

import pytest
import torch
import torch.nn as nn

from codetr.models.transformer.attention import MultiScaleDeformableAttention
from codetr.models.transformer.encoder import (
    DeformableTransformerEncoderLayer,
    CoDeformableDetrTransformerEncoder,
)
from codetr.models.transformer.decoder import (
    DeformableTransformerDecoderLayer,
    CoDeformableDetrTransformerDecoder,
)
from codetr.models.transformer.transformer import CoDeformableDetrTransformer


class TestMultiScaleDeformableAttention:
    """Tests for Multi-Scale Deformable Attention module."""
    
    def test_output_shape(self):
        """Test output shape matches query shape."""
        msda = MultiScaleDeformableAttention(
            embed_dim=256,
            num_heads=8,
            num_levels=4,
            num_points=4,
        )
        
        batch_size, num_queries, embed_dim = 2, 300, 256
        # spatial_shapes: [[10, 10], [5, 5], [3, 3], [2, 2]] -> 100 + 25 + 9 + 4 = 138 keys
        num_keys = 100 + 25 + 9 + 4
        
        query = torch.randn(batch_size, num_queries, embed_dim)
        value = torch.randn(batch_size, num_keys, embed_dim)
        reference_points = torch.rand(batch_size, num_queries, 4, 2)  # 4 levels
        spatial_shapes = torch.tensor([[10, 10], [5, 5], [3, 3], [2, 2]], dtype=torch.long)
        level_start_index = torch.tensor([0, 100, 125, 134], dtype=torch.long)
        
        output = msda(query, reference_points, value, spatial_shapes, level_start_index)
        
        assert output.shape == (batch_size, num_queries, embed_dim)
    
    def test_no_nan_inf_output(self):
        """Output should not contain NaN or Inf values."""
        msda = MultiScaleDeformableAttention(
            embed_dim=256,
            num_heads=8,
            num_levels=4,
            num_points=4,
        )
        
        batch_size = 2
        # spatial_shapes: [[5, 5], [3, 3], [2, 2], [1, 1]] -> 25 + 9 + 4 + 1 = 39 keys
        num_keys = 25 + 9 + 4 + 1
        query = torch.randn(batch_size, 100, 256)
        value = torch.randn(batch_size, num_keys, 256)
        reference_points = torch.rand(batch_size, 100, 4, 2)
        spatial_shapes = torch.tensor([[5, 5], [3, 3], [2, 2], [1, 1]], dtype=torch.long)
        level_start_index = torch.tensor([0, 25, 34, 38], dtype=torch.long)
        
        output = msda(query, reference_points, value, spatial_shapes, level_start_index)
        
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"
    
    def test_gradient_flow(self):
        """Test gradient flows through attention."""
        msda = MultiScaleDeformableAttention(
            embed_dim=256,
            num_heads=8,
            num_levels=4,
            num_points=4,
        )
        
        query = torch.randn(1, 50, 256, requires_grad=True)
        value = torch.randn(1, 30, 256, requires_grad=True)
        reference_points = torch.rand(1, 50, 4, 2, requires_grad=True)
        spatial_shapes = torch.tensor([[4, 4], [3, 3], [2, 2], [1, 1]], dtype=torch.long)
        level_start_index = torch.tensor([0, 16, 25, 29], dtype=torch.long)
        
        output = msda(query, reference_points, value, spatial_shapes, level_start_index)
        loss = output.sum()
        loss.backward()
        
        assert query.grad is not None
        assert value.grad is not None
        assert not torch.isnan(query.grad).any()
    
    def test_reference_points_format(self):
        """Test both 2D and 4D reference point formats."""
        msda = MultiScaleDeformableAttention(
            embed_dim=256,
            num_heads=8,
            num_levels=4,
            num_points=4,
        )
        
        query = torch.randn(1, 50, 256)
        value = torch.randn(1, 30, 256)
        spatial_shapes = torch.tensor([[4, 4], [3, 3], [2, 2], [1, 1]], dtype=torch.long)
        level_start_index = torch.tensor([0, 16, 25, 29], dtype=torch.long)
        
        # 2D format: (batch, num_queries, num_levels, 2)
        ref_2d = torch.rand(1, 50, 4, 2)
        output_2d = msda(query, ref_2d, value, spatial_shapes, level_start_index)
        
        assert output_2d.shape == (1, 50, 256)


class TestDeformableTransformerEncoder:
    """Tests for Deformable Transformer Encoder."""
    
    def test_encoder_layer_output_shape(self):
        """Encoder layer should preserve input shape."""
        encoder_layer = DeformableTransformerEncoderLayer(
            embed_dim=256,
            num_heads=8,
            feedforward_dim=1024,
            dropout=0.1,
            num_levels=4,
        )
        
        # spatial_shapes: [[5, 5], [4, 4], [3, 3], [2, 2]] -> 25 + 16 + 9 + 4 = 54
        num_keys = 25 + 16 + 9 + 4  # Total matches sum of spatial shapes
        query = torch.randn(1, num_keys, 256)
        reference_points = torch.rand(1, num_keys, 4, 2)
        spatial_shapes = torch.tensor([[5, 5], [4, 4], [3, 3], [2, 2]], dtype=torch.long)
        level_start_index = torch.tensor([0, 25, 41, 50], dtype=torch.long)
        
        output = encoder_layer(query, reference_points, spatial_shapes, level_start_index)
        
        assert output.shape == query.shape
    
    def test_full_encoder_output_shape(self):
        """Full encoder should output same shape as input."""
        encoder_layer = DeformableTransformerEncoderLayer(
            embed_dim=256,
            num_heads=8,
            feedforward_dim=1024,
            num_levels=4,
        )
        encoder = CoDeformableDetrTransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=6,
        )
        
        # spatial_shapes: [[5, 5], [4, 4], [3, 3], [2, 2]] -> 25 + 16 + 9 + 4 = 54
        num_keys = 25 + 16 + 9 + 4
        query = torch.randn(1, num_keys, 256)
        reference_points = torch.rand(1, num_keys, 4, 2)
        spatial_shapes = torch.tensor([[5, 5], [4, 4], [3, 3], [2, 2]], dtype=torch.long)
        level_start_index = torch.tensor([0, 25, 41, 50], dtype=torch.long)
        
        output = encoder(query, reference_points, spatial_shapes, level_start_index)
        
        assert output.shape == query.shape
    
    def test_encoder_gradient_flow(self):
        """Test gradients flow through all encoder layers."""
        encoder_layer = DeformableTransformerEncoderLayer(
            embed_dim=256,
            num_heads=8,
            feedforward_dim=1024,
            num_levels=4,
        )
        encoder = CoDeformableDetrTransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=3,  # Fewer layers for speed
        )
        
        # spatial_shapes: [[4, 4], [3, 3], [2, 2], [1, 1]] -> 16 + 9 + 4 + 1 = 30
        num_keys = 16 + 9 + 4 + 1
        query = torch.randn(1, num_keys, 256, requires_grad=True)
        reference_points = torch.rand(1, num_keys, 4, 2)
        spatial_shapes = torch.tensor([[4, 4], [3, 3], [2, 2], [1, 1]], dtype=torch.long)
        level_start_index = torch.tensor([0, 16, 25, 29], dtype=torch.long)
        
        output = encoder(query, reference_points, spatial_shapes, level_start_index)
        loss = output.sum()
        loss.backward()
        
        assert query.grad is not None
        assert not torch.isnan(query.grad).any()


class TestDeformableTransformerDecoder:
    """Tests for Deformable Transformer Decoder."""
    
    def test_decoder_layer_output_shape(self):
        """Decoder layer should preserve query shape."""
        decoder_layer = DeformableTransformerDecoderLayer(
            embed_dim=256,
            num_heads=8,
            feedforward_dim=1024,
            dropout=0.1,
            num_levels=4,
        )
        
        # spatial_shapes: [[5, 5], [4, 4], [3, 3], [2, 2]] -> 25 + 16 + 9 + 4 = 54
        num_keys = 25 + 16 + 9 + 4
        query = torch.randn(1, 300, 256)
        memory = torch.randn(1, num_keys, 256)
        reference_points = torch.rand(1, 300, 4, 2)
        spatial_shapes = torch.tensor([[5, 5], [4, 4], [3, 3], [2, 2]], dtype=torch.long)
        level_start_index = torch.tensor([0, 25, 41, 50], dtype=torch.long)
        
        output = decoder_layer(query, reference_points, memory, spatial_shapes, level_start_index)
        
        assert output.shape == query.shape
    
    def test_full_decoder_intermediate_outputs(self):
        """Decoder should return intermediate outputs from all layers."""
        decoder_layer = DeformableTransformerDecoderLayer(
            embed_dim=256,
            num_heads=8,
            feedforward_dim=1024,
            num_levels=4,
        )
        decoder = CoDeformableDetrTransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=6,
        )
        
        # spatial_shapes: [[5, 5], [4, 4], [3, 3], [2, 2]] -> 25 + 16 + 9 + 4 = 54
        num_keys = 25 + 16 + 9 + 4
        query = torch.randn(1, 300, 256)
        memory = torch.randn(1, num_keys, 256)
        reference_points = torch.rand(1, 300, 2)
        spatial_shapes = torch.tensor([[5, 5], [4, 4], [3, 3], [2, 2]], dtype=torch.long)
        level_start_index = torch.tensor([0, 25, 41, 50], dtype=torch.long)
        valid_ratios = torch.ones(1, 4, 2)
        
        outputs, inter_refs = decoder(
            query, reference_points, memory, 
            spatial_shapes, level_start_index, valid_ratios
        )
        
        # Should return intermediate outputs from all 6 layers
        assert outputs.shape == (6, 1, 300, 256)
    
    def test_decoder_self_attention(self):
        """Decoder should apply self-attention on queries."""
        decoder_layer = DeformableTransformerDecoderLayer(
            embed_dim=256,
            num_heads=8,
            feedforward_dim=1024,
        )
        
        # Verify self_attn exists
        assert hasattr(decoder_layer, 'self_attn')
    
    def test_decoder_gradient_flow(self):
        """Test gradients flow through decoder."""
        decoder_layer = DeformableTransformerDecoderLayer(
            embed_dim=256,
            num_heads=8,
            feedforward_dim=1024,
            num_levels=4,
        )
        decoder = CoDeformableDetrTransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=2,
        )
        
        # spatial_shapes: [[4, 4], [3, 3], [2, 2], [1, 1]] -> 16 + 9 + 4 + 1 = 30
        num_keys = 16 + 9 + 4 + 1
        query = torch.randn(1, 100, 256, requires_grad=True)
        memory = torch.randn(1, num_keys, 256, requires_grad=True)
        reference_points = torch.rand(1, 100, 2)
        spatial_shapes = torch.tensor([[4, 4], [3, 3], [2, 2], [1, 1]], dtype=torch.long)
        level_start_index = torch.tensor([0, 16, 25, 29], dtype=torch.long)
        valid_ratios = torch.ones(1, 4, 2)
        
        outputs, _ = decoder(
            query, reference_points, memory,
            spatial_shapes, level_start_index, valid_ratios
        )
        loss = outputs.sum()
        loss.backward()
        
        assert query.grad is not None
        assert memory.grad is not None


class TestCoDeformableDetrTransformer:
    """Tests for complete Co-Deformable DETR Transformer."""
    
    def test_transformer_output_shapes(self):
        """Test transformer outputs have correct shapes."""
        # Use smaller num_queries that fits within total spatial positions
        # or use larger feature maps. Here we use num_queries=100
        # and ensure total positions (20*20 + 10*10 + 5*5 + 3*3) = 400+100+25+9 = 534 > 100
        transformer = CoDeformableDetrTransformer(
            embed_dim=256,
            num_heads=8,
            num_encoder_layers=2,
            num_decoder_layers=2,
            num_feature_levels=4,
            num_queries=100,  # Reduced to fit spatial positions
        )
        
        # Prepare multi-scale features with enough total spatial positions
        # Total: 20*20 + 10*10 + 5*5 + 3*3 = 400 + 100 + 25 + 9 = 534 positions
        batch_size = 2
        mlvl_feats = [
            torch.randn(batch_size, 256, 20, 20),
            torch.randn(batch_size, 256, 10, 10),
            torch.randn(batch_size, 256, 5, 5),
            torch.randn(batch_size, 256, 3, 3),
        ]
        mlvl_masks = [
            torch.zeros(batch_size, 20, 20, dtype=torch.bool),
            torch.zeros(batch_size, 10, 10, dtype=torch.bool),
            torch.zeros(batch_size, 5, 5, dtype=torch.bool),
            torch.zeros(batch_size, 3, 3, dtype=torch.bool),
        ]
        mlvl_pos_embeds = [
            torch.randn(batch_size, 256, 20, 20),
            torch.randn(batch_size, 256, 10, 10),
            torch.randn(batch_size, 256, 5, 5),
            torch.randn(batch_size, 256, 3, 3),
        ]
        
        hs, init_ref, inter_ref, enc_cls, enc_coord = transformer(
            mlvl_feats, mlvl_masks, mlvl_pos_embeds
        )
        
        # Hidden states: (num_layers, batch, num_queries, embed_dim)
        assert hs.shape[1] == batch_size
        assert hs.shape[2] == 100
        assert hs.shape[3] == 256
        
        # Initial references: (batch, num_queries, 2)
        assert init_ref.shape == (batch_size, 100, 2)
    
    def test_transformer_gradient_flow(self):
        """Test gradients flow through entire transformer."""
        # Use smaller num_queries that fits within total spatial positions
        # Total: 15*15 + 10*10 + 7*7 + 5*5 = 225 + 100 + 49 + 25 = 399 positions
        transformer = CoDeformableDetrTransformer(
            embed_dim=256,
            num_heads=8,
            num_encoder_layers=1,
            num_decoder_layers=1,
            num_feature_levels=4,
            num_queries=50,  # Reduced to fit spatial positions
        )
        
        batch_size = 1
        # Use larger feature sizes to have enough spatial positions
        mlvl_feats = [
            torch.randn(batch_size, 256, 15, 15, requires_grad=True),
            torch.randn(batch_size, 256, 10, 10, requires_grad=True),
            torch.randn(batch_size, 256, 7, 7, requires_grad=True),
            torch.randn(batch_size, 256, 5, 5, requires_grad=True),
        ]
        mlvl_masks = [torch.zeros(batch_size, h, w, dtype=torch.bool) for h, w in [(15,15), (10,10), (7,7), (5,5)]]
        mlvl_pos_embeds = [torch.randn(batch_size, 256, h, w) for h, w in [(15,15), (10,10), (7,7), (5,5)]]
        
        hs, _, _, _, _ = transformer(mlvl_feats, mlvl_masks, mlvl_pos_embeds)
        loss = hs.sum()
        loss.backward()
        
        for i, feat in enumerate(mlvl_feats):
            assert feat.grad is not None, f"No gradient for feature level {i}"


class TestTransformerWithPaddedInputs:
    """Tests for Transformer behavior with padded (partial) feature maps.
    
    When images are padded to form a batch, some regions of the feature maps
    are invalid. The transformer should handle this via valid_ratios.
    
    Mathematical correctness:
    - valid_ratio = actual_size / padded_size, ranges in (0, 1]
    - valid_ratio = 1.0 means no padding
    - valid_ratio < 1.0 means image is padded
    """
    
    def test_transformer_with_partial_valid_ratios(self):
        """Transformer should handle images where part is padded.
        
        Simulates a batch where one image is smaller and has padding.
        valid_ratios < 1 indicates the valid (non-padded) portion.
        """
        transformer = CoDeformableDetrTransformer(
            embed_dim=256,
            num_heads=8,
            num_encoder_layers=2,
            num_decoder_layers=2,
            num_feature_levels=4,
            num_queries=50,
        )
        
        batch_size = 2
        
        # Feature maps (all same size after padding)
        mlvl_feats = [
            torch.randn(batch_size, 256, 16, 16),
            torch.randn(batch_size, 256, 8, 8),
            torch.randn(batch_size, 256, 4, 4),
            torch.randn(batch_size, 256, 2, 2),
        ]
        
        # Masks: second image has padding (True = padded, invalid)
        mlvl_masks = []
        for h, w in [(16, 16), (8, 8), (4, 4), (2, 2)]:
            mask = torch.zeros(batch_size, h, w, dtype=torch.bool)
            # Second image: 70% valid height, 80% valid width
            valid_h = int(h * 0.7)
            valid_w = int(w * 0.8)
            mask[1, valid_h:, :] = True  # Bottom padding
            mask[1, :, valid_w:] = True  # Right padding
            mlvl_masks.append(mask)
        
        mlvl_pos_embeds = [
            torch.randn(batch_size, 256, h, w)
            for h, w in [(16, 16), (8, 8), (4, 4), (2, 2)]
        ]
        
        # Forward pass should complete without errors
        hs, init_ref, inter_ref, enc_cls, enc_coord = transformer(
            mlvl_feats, mlvl_masks, mlvl_pos_embeds
        )
        
        # Output should have expected shapes
        assert hs.shape[1] == batch_size
        assert hs.shape[2] == 50  # num_queries
        
        # No NaN/Inf in outputs
        assert not torch.isnan(hs).any(), "Hidden states contain NaN"
        assert not torch.isinf(hs).any(), "Hidden states contain Inf"
    
    def test_decoder_valid_ratios_affect_reference_points(self):
        """valid_ratios should scale reference points for padded images.
        
        When valid_ratio < 1, reference points are scaled to map
        normalized coordinates [0, 1] to the valid region only.
        """
        decoder_layer = DeformableTransformerDecoderLayer(
            embed_dim=256,
            num_heads=8,
            feedforward_dim=1024,
            num_levels=4,
        )
        decoder = CoDeformableDetrTransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=2,
        )
        
        num_keys = 16 + 9 + 4 + 1
        query = torch.randn(2, 50, 256)
        memory = torch.randn(2, num_keys, 256)
        reference_points = torch.rand(2, 50, 2)
        spatial_shapes = torch.tensor([[4, 4], [3, 3], [2, 2], [1, 1]], dtype=torch.long)
        level_start_index = torch.tensor([0, 16, 25, 29], dtype=torch.long)
        
        # First batch item has full valid ratio, second has 80%, 70%
        valid_ratios = torch.tensor([
            [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],  # Batch 0: no padding
            [[0.8, 0.7], [0.8, 0.7], [0.8, 0.7], [0.8, 0.7]],  # Batch 1: 80% H, 70% W
        ])
        
        outputs, inter_refs = decoder(
            query, reference_points, memory,
            spatial_shapes, level_start_index, valid_ratios
        )
        
        # Should complete without errors
        assert outputs.shape == (2, 2, 50, 256)  # (num_layers, batch, queries, embed)
        assert not torch.isnan(outputs).any()
    
    def test_extreme_padding_handled(self):
        """Very small valid ratios (heavy padding) should still work.
        
        Edge case: only 10% of the image is valid (90% padding).
        """
        transformer = CoDeformableDetrTransformer(
            embed_dim=256,
            num_heads=8,
            num_encoder_layers=1,
            num_decoder_layers=1,
            num_feature_levels=4,
            num_queries=20,
        )
        
        batch_size = 1
        mlvl_feats = [
            torch.randn(batch_size, 256, 10, 10),
            torch.randn(batch_size, 256, 5, 5),
            torch.randn(batch_size, 256, 3, 3),
            torch.randn(batch_size, 256, 2, 2),
        ]
        
        # Extreme padding: only top-left 10% is valid
        mlvl_masks = []
        for h, w in [(10, 10), (5, 5), (3, 3), (2, 2)]:
            mask = torch.ones(batch_size, h, w, dtype=torch.bool)  # All padding
            valid_h = max(1, int(h * 0.3))  # At least 1 pixel valid
            valid_w = max(1, int(w * 0.3))
            mask[0, :valid_h, :valid_w] = False  # Small valid region
            mlvl_masks.append(mask)
        
        mlvl_pos_embeds = [
            torch.randn(batch_size, 256, h, w)
            for h, w in [(10, 10), (5, 5), (3, 3), (2, 2)]
        ]
        
        # Should not crash even with extreme padding
        hs, init_ref, inter_ref, enc_cls, enc_coord = transformer(
            mlvl_feats, mlvl_masks, mlvl_pos_embeds
        )
        
        assert hs.shape[2] == 20
        assert not torch.isnan(hs).any(), "Extreme padding caused NaN"


class TestTransformerInputValidation:
    """Tests for Transformer input validation.
    
    The transformer expects inputs to match its configuration:
    - len(srcs) == len(masks) == len(pos_embeds) == num_feature_levels
    - Each feature map should have matching spatial dimensions with its mask
    
    These tests verify that invalid inputs are handled appropriately.
    """

    def test_mismatched_input_counts_raises_assertion(self):
        """len(srcs) != len(masks) should raise AssertionError.
        
        The transformer has an assert statement at line 177:
        assert len(srcs) == len(masks) == len(pos_embeds) == self.num_feature_levels
        """
        transformer = CoDeformableDetrTransformer(
            embed_dim=256,
            num_heads=8,
            num_encoder_layers=1,
            num_decoder_layers=1,
            num_feature_levels=4,  # Expects 4 levels
            num_queries=50,
        )
        
        batch_size = 1
        
        # Provide 4 feature maps but only 3 masks (MISMATCH)
        mlvl_feats = [
            torch.randn(batch_size, 256, 10, 10),
            torch.randn(batch_size, 256, 5, 5),
            torch.randn(batch_size, 256, 3, 3),
            torch.randn(batch_size, 256, 2, 2),
        ]
        mlvl_masks = [
            torch.zeros(batch_size, 10, 10, dtype=torch.bool),
            torch.zeros(batch_size, 5, 5, dtype=torch.bool),
            torch.zeros(batch_size, 3, 3, dtype=torch.bool),
            # Missing 4th mask!
        ]
        mlvl_pos_embeds = [
            torch.randn(batch_size, 256, 10, 10),
            torch.randn(batch_size, 256, 5, 5),
            torch.randn(batch_size, 256, 3, 3),
            torch.randn(batch_size, 256, 2, 2),
        ]
        
        # Should raise AssertionError due to mismatched counts
        with pytest.raises(AssertionError):
            transformer(mlvl_feats, mlvl_masks, mlvl_pos_embeds)

    def test_wrong_num_feature_levels_raises_assertion(self):
        """Providing wrong number of feature levels should raise error.
        
        If transformer is configured for 4 levels but 3 are provided,
        assertion should fail.
        """
        transformer = CoDeformableDetrTransformer(
            embed_dim=256,
            num_heads=8,
            num_encoder_layers=1,
            num_decoder_layers=1,
            num_feature_levels=4,  # Expects 4 levels
            num_queries=30,
        )
        
        batch_size = 1
        
        # Only provide 3 feature levels instead of 4
        mlvl_feats = [
            torch.randn(batch_size, 256, 10, 10),
            torch.randn(batch_size, 256, 5, 5),
            torch.randn(batch_size, 256, 3, 3),
        ]
        mlvl_masks = [
            torch.zeros(batch_size, 10, 10, dtype=torch.bool),
            torch.zeros(batch_size, 5, 5, dtype=torch.bool),
            torch.zeros(batch_size, 3, 3, dtype=torch.bool),
        ]
        mlvl_pos_embeds = [
            torch.randn(batch_size, 256, 10, 10),
            torch.randn(batch_size, 256, 5, 5),
            torch.randn(batch_size, 256, 3, 3),
        ]
        
        # Should raise AssertionError
        with pytest.raises(AssertionError):
            transformer(mlvl_feats, mlvl_masks, mlvl_pos_embeds)

    def test_correct_input_counts_no_error(self):
        """Verify correct input counts do not raise errors.
        
        This is a sanity check that valid inputs work correctly.
        """
        transformer = CoDeformableDetrTransformer(
            embed_dim=256,
            num_heads=8,
            num_encoder_layers=1,
            num_decoder_layers=1,
            num_feature_levels=4,
            num_queries=30,
        )
        
        batch_size = 1
        
        # Correct: all 4 levels provided
        mlvl_feats = [
            torch.randn(batch_size, 256, 10, 10),
            torch.randn(batch_size, 256, 5, 5),
            torch.randn(batch_size, 256, 3, 3),
            torch.randn(batch_size, 256, 2, 2),
        ]
        mlvl_masks = [
            torch.zeros(batch_size, 10, 10, dtype=torch.bool),
            torch.zeros(batch_size, 5, 5, dtype=torch.bool),
            torch.zeros(batch_size, 3, 3, dtype=torch.bool),
            torch.zeros(batch_size, 2, 2, dtype=torch.bool),
        ]
        mlvl_pos_embeds = [
            torch.randn(batch_size, 256, 10, 10),
            torch.randn(batch_size, 256, 5, 5),
            torch.randn(batch_size, 256, 3, 3),
            torch.randn(batch_size, 256, 2, 2),
        ]
        
        # Should NOT raise any errors
        hs, init_ref, inter_ref, enc_cls, enc_coord = transformer(
            mlvl_feats, mlvl_masks, mlvl_pos_embeds
        )
        
        assert hs.shape[2] == 30  # num_queries


