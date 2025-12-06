"""
Tests for positional encoding module.

This module tests the sine-cosine positional embeddings for 2D feature maps,
verifying output shapes, determinism, and mask handling.
"""

import pytest
import torch

from codetr.models.utils.position_encoding import PositionEmbeddingSine


class TestPositionEmbeddingSine:
    """Tests for 2D sine-cosine positional embeddings."""
    
    def test_output_shape(self):
        """Test output shape is (batch, 2*num_pos_feats, H, W)."""
        pos_enc = PositionEmbeddingSine(num_pos_feats=128)
        
        batch, channels, height, width = 2, 256, 25, 25
        x = torch.randn(batch, channels, height, width)
        mask = torch.zeros(batch, height, width, dtype=torch.bool)
        
        pos = pos_enc(x, mask)
        
        # Output should be (batch, 256, 25, 25) since num_pos_feats=128 -> 2*128=256
        assert pos.shape == (batch, 256, height, width)
    
    def test_output_shape_different_num_pos_feats(self):
        """Test output shape with different num_pos_feats values."""
        for num_pos_feats in [64, 128, 256]:
            pos_enc = PositionEmbeddingSine(num_pos_feats=num_pos_feats)
            
            x = torch.randn(1, 256, 10, 10)
            pos = pos_enc(x)
            
            expected_channels = 2 * num_pos_feats
            assert pos.shape == (1, expected_channels, 10, 10)
    
    def test_deterministic_output(self):
        """Positional encoding should be deterministic for same input."""
        pos_enc = PositionEmbeddingSine(num_pos_feats=128)
        
        x = torch.randn(2, 256, 25, 25)
        mask = torch.zeros(2, 25, 25, dtype=torch.bool)
        
        pos1 = pos_enc(x, mask)
        pos2 = pos_enc(x, mask)
        
        assert torch.allclose(pos1, pos2)
    
    def test_different_positions_different_embeddings(self):
        """Different spatial positions should have different embeddings."""
        pos_enc = PositionEmbeddingSine(num_pos_feats=128)
        
        x = torch.randn(1, 256, 10, 10)
        pos = pos_enc(x)
        
        # Embeddings at (0,0) and (5,5) should be different
        pos_0_0 = pos[0, :, 0, 0]
        pos_5_5 = pos[0, :, 5, 5]
        
        assert not torch.allclose(pos_0_0, pos_5_5)
    
    def test_masked_positions_zeroed(self):
        """Masked (padding) positions should have zero embeddings."""
        pos_enc = PositionEmbeddingSine(num_pos_feats=128)
        
        x = torch.randn(1, 256, 10, 10)
        mask = torch.zeros(1, 10, 10, dtype=torch.bool)
        # Mark bottom-right quadrant as padding
        mask[0, 5:, 5:] = True
        
        pos = pos_enc(x, mask)
        
        # Masked positions should be zero
        masked_region = pos[0, :, 5:, 5:]
        assert torch.allclose(masked_region, torch.zeros_like(masked_region))
        
        # Valid positions should be non-zero
        valid_region = pos[0, :, :5, :5]
        assert not torch.allclose(valid_region, torch.zeros_like(valid_region))
    
    def test_no_mask_default(self):
        """Test forward pass without providing mask (all valid)."""
        pos_enc = PositionEmbeddingSine(num_pos_feats=128)
        
        x = torch.randn(2, 256, 10, 10)
        
        # Should not raise error
        pos = pos_enc(x)
        
        assert pos.shape == (2, 256, 10, 10)
        # All positions should be non-zero
        assert pos.abs().sum() > 0
    
    def test_value_range(self):
        """Positional embeddings should be in reasonable range (sine/cosine -> [-1, 1])."""
        pos_enc = PositionEmbeddingSine(num_pos_feats=128, normalize=True)
        
        x = torch.randn(2, 256, 50, 50)
        pos = pos_enc(x)
        
        # After sine/cosine, values should be in [-1, 1]
        assert pos.min() >= -1.0 - 1e-5
        assert pos.max() <= 1.0 + 1e-5
    
    def test_temperature_effect(self):
        """Different temperature should produce different frequency patterns."""
        pos_enc_low_temp = PositionEmbeddingSine(num_pos_feats=128, temperature=1000)
        pos_enc_high_temp = PositionEmbeddingSine(num_pos_feats=128, temperature=20000)
        
        x = torch.randn(1, 256, 20, 20)
        
        pos_low = pos_enc_low_temp(x)
        pos_high = pos_enc_high_temp(x)
        
        # Different temperatures should produce different patterns
        assert not torch.allclose(pos_low, pos_high)
    
    def test_batch_independence(self):
        """Positional encodings should be same across batch dimension."""
        pos_enc = PositionEmbeddingSine(num_pos_feats=128)
        
        x = torch.randn(3, 256, 15, 15)
        mask = torch.zeros(3, 15, 15, dtype=torch.bool)
        
        pos = pos_enc(x, mask)
        
        # All batch items should have same positional encoding
        assert torch.allclose(pos[0], pos[1])
        assert torch.allclose(pos[1], pos[2])
    
    def test_repr(self):
        """Test string representation contains key parameters."""
        pos_enc = PositionEmbeddingSine(
            num_pos_feats=128,
            temperature=10000,
            normalize=True,
        )
        
        repr_str = repr(pos_enc)
        
        assert "PositionEmbeddingSine" in repr_str
        assert "128" in repr_str  # num_pos_feats
        assert "10000" in repr_str  # temperature
