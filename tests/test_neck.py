"""
Tests for ChannelMapper neck module.

This module tests the feature pyramid construction including channel projection,
extra level generation, and normalization.
"""

import pytest
import torch
import torch.nn as nn

from codetr.models.neck.channel_mapper import ChannelMapper


class TestChannelMapperOutputShapes:
    """Tests for ChannelMapper output shapes and channels."""
    
    def test_output_channels_all_uniform(self):
        """All output feature maps should have uniform channels (256)."""
        mapper = ChannelMapper(
            in_channels=[512, 1024, 2048],
            out_channels=256,
            num_extra_levels=1,
        )
        
        features = [
            torch.randn(2, 512, 100, 100),   # C3
            torch.randn(2, 1024, 50, 50),    # C4
            torch.randn(2, 2048, 25, 25),    # C5
        ]
        
        outputs = mapper(features)
        
        # All outputs should have 256 channels
        for i, out in enumerate(outputs):
            assert out.shape[1] == 256, f"Output {i} has {out.shape[1]} channels, expected 256"
    
    def test_spatial_sizes_preserved(self):
        """Input spatial sizes should be preserved for lateral convolutions."""
        mapper = ChannelMapper(
            in_channels=[512, 1024, 2048],
            out_channels=256,
            num_extra_levels=0,  # No extra levels
        )
        
        features = [
            torch.randn(2, 512, 100, 100),
            torch.randn(2, 1024, 50, 50),
            torch.randn(2, 2048, 25, 25),
        ]
        
        outputs = mapper(features)
        
        assert len(outputs) == 3
        assert outputs[0].shape == (2, 256, 100, 100)  # P3
        assert outputs[1].shape == (2, 256, 50, 50)    # P4
        assert outputs[2].shape == (2, 256, 25, 25)    # P5
    
    def test_extra_levels_generated(self):
        """Extra levels should be generated via stride-2 downsampling."""
        mapper = ChannelMapper(
            in_channels=[512, 1024, 2048],
            out_channels=256,
            num_extra_levels=2,  # Add P6 and P7
        )
        
        features = [
            torch.randn(2, 512, 100, 100),
            torch.randn(2, 1024, 50, 50),
            torch.randn(2, 2048, 25, 25),
        ]
        
        outputs = mapper(features)
        
        assert len(outputs) == 5  # P3, P4, P5, P6, P7
        
        # P6: 25 -> floor((25+2*1-3)/2)+1 = 13
        assert outputs[3].shape[1] == 256
        # Spatial size reduced by stride-2 conv
        assert outputs[3].shape[2] < outputs[2].shape[2]
        
        # P7: further reduced
        assert outputs[4].shape[2] < outputs[3].shape[2]
    
    def test_single_extra_level(self):
        """Test with default single extra level (P6)."""
        mapper = ChannelMapper(
            in_channels=[512, 1024, 2048],
            out_channels=256,
            num_extra_levels=1,
        )
        
        features = [
            torch.randn(1, 512, 100, 100),
            torch.randn(1, 1024, 50, 50),
            torch.randn(1, 2048, 25, 25),
        ]
        
        outputs = mapper(features)
        
        assert len(outputs) == 4  # P3, P4, P5, P6
        
        # P6 should be downsampled from P5
        assert outputs[3].shape == (1, 256, 13, 13)


class TestChannelMapperNormalization:
    """Tests for normalization in ChannelMapper."""
    
    def test_groupnorm_applied(self):
        """Test GroupNorm is applied when norm_type='GN'."""
        mapper = ChannelMapper(
            in_channels=[512, 1024, 2048],
            out_channels=256,
            norm_type="GN",
            num_groups=32,
        )
        
        # Check lateral_norms exist and are GroupNorm
        assert mapper.lateral_norms is not None
        for norm in mapper.lateral_norms:
            assert isinstance(norm, nn.GroupNorm)
            assert norm.num_groups == 32
    
    def test_batchnorm_applied(self):
        """Test BatchNorm is applied when norm_type='BN'."""
        mapper = ChannelMapper(
            in_channels=[512, 1024, 2048],
            out_channels=256,
            norm_type="BN",
        )
        
        assert mapper.lateral_norms is not None
        for norm in mapper.lateral_norms:
            assert isinstance(norm, nn.BatchNorm2d)
    
    def test_no_normalization(self):
        """Test no normalization when norm_type=None."""
        mapper = ChannelMapper(
            in_channels=[512, 1024, 2048],
            out_channels=256,
            norm_type=None,
        )
        
        assert mapper.lateral_norms is None


class TestChannelMapperGradientFlow:
    """Tests for gradient flow through ChannelMapper."""
    
    def test_gradient_flow_all_paths(self):
        """Test gradients flow through all output paths."""
        mapper = ChannelMapper(
            in_channels=[512, 1024, 2048],
            out_channels=256,
            num_extra_levels=1,
        )
        
        features = [
            torch.randn(1, 512, 50, 50, requires_grad=True),
            torch.randn(1, 1024, 25, 25, requires_grad=True),
            torch.randn(1, 2048, 13, 13, requires_grad=True),
        ]
        
        outputs = mapper(features)
        
        # Sum all outputs for loss
        loss = sum(out.sum() for out in outputs)
        loss.backward()
        
        # All inputs should have gradients
        for i, feat in enumerate(features):
            assert feat.grad is not None, f"No gradient for input {i}"
            assert not torch.isnan(feat.grad).any()
        
        # All conv weights should have gradients
        for i, conv in enumerate(mapper.lateral_convs):
            assert conv.weight.grad is not None, f"No gradient for lateral_conv {i}"
    
    def test_extra_convs_gradient(self):
        """Test gradients flow through extra level convolutions."""
        mapper = ChannelMapper(
            in_channels=[512, 1024, 2048],
            out_channels=256,
            num_extra_levels=2,
        )
        
        features = [
            torch.randn(1, 512, 50, 50),
            torch.randn(1, 1024, 25, 25),
            torch.randn(1, 2048, 13, 13),
        ]
        
        outputs = mapper(features)
        
        # Use only extra levels for loss
        loss = outputs[3].sum() + outputs[4].sum()
        loss.backward()
        
        # Extra convs should have gradients
        for i, conv in enumerate(mapper.extra_convs):
            assert conv.weight.grad is not None, f"No gradient for extra_conv {i}"


class TestChannelMapperInputValidation:
    """Tests for input validation."""
    
    def test_mismatched_input_count_raises(self):
        """Should raise error if feature count doesn't match in_channels."""
        mapper = ChannelMapper(
            in_channels=[512, 1024, 2048],
            out_channels=256,
        )
        
        # Only 2 features instead of 3
        features = [
            torch.randn(1, 512, 50, 50),
            torch.randn(1, 1024, 25, 25),
        ]
        
        with pytest.raises(AssertionError):
            mapper(features)


class TestChannelMapperWeightInitialization:
    """Tests for weight initialization."""
    
    def test_xavier_initialization(self):
        """Weights should be initialized with Xavier uniform."""
        mapper = ChannelMapper(
            in_channels=[512, 1024, 2048],
            out_channels=256,
        )
        
        # Xavier uniform produces values roughly in [-sqrt(6/(fan_in+fan_out)), sqrt(...)]
        # For 1x1 conv from 512->256: fan_in=512, fan_out=256
        # Bound ≈ sqrt(6/768) ≈ 0.088
        
        for conv in mapper.lateral_convs:
            # Check weights are not all zeros or ones
            assert conv.weight.std() > 0.01
            # Check biases are zeros
            assert torch.allclose(conv.bias, torch.zeros_like(conv.bias))
