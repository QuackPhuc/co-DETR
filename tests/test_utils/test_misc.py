"""
Tests for miscellaneous utility functions.

This module tests the NestedTensor class, tensor batching utilities,
valid ratio computation, and interpolation wrapper.
"""

import pytest
import torch
from torch import Tensor

from codetr.models.utils.misc import (
    NestedTensor,
    nested_tensor_from_tensor_list,
    get_valid_ratio,
    interpolate,
)


class TestNestedTensor:
    """Tests for NestedTensor container class."""
    
    def test_nested_tensor_creation(self):
        """Test basic NestedTensor creation."""
        tensors = torch.randn(2, 3, 100, 100)
        mask = torch.zeros(2, 100, 100, dtype=torch.bool)
        
        nested = NestedTensor(tensors, mask)
        
        assert nested.tensors.shape == (2, 3, 100, 100)
        assert nested.mask.shape == (2, 100, 100)
    
    def test_nested_tensor_decompose(self):
        """Test decompose returns tensor and mask."""
        tensors = torch.randn(2, 3, 50, 50)
        mask = torch.ones(2, 50, 50, dtype=torch.bool)
        
        nested = NestedTensor(tensors, mask)
        t, m = nested.decompose()
        
        assert torch.equal(t, tensors)
        assert torch.equal(m, mask)
    
    def test_nested_tensor_to_device(self):
        """Test device transfer."""
        tensors = torch.randn(2, 3, 50, 50)
        mask = torch.zeros(2, 50, 50, dtype=torch.bool)
        
        nested = NestedTensor(tensors, mask)
        nested_cpu = nested.to(torch.device("cpu"))
        
        assert nested_cpu.tensors.device == torch.device("cpu")
        assert nested_cpu.mask.device == torch.device("cpu")
    
    def test_nested_tensor_repr(self):
        """Test string representation."""
        tensors = torch.randn(2, 3, 100, 100)
        mask = torch.zeros(2, 100, 100, dtype=torch.bool)
        
        nested = NestedTensor(tensors, mask)
        repr_str = repr(nested)
        
        assert "NestedTensor" in repr_str
        assert "100" in repr_str  # dimensions should appear


class TestNestedTensorFromTensorList:
    """Tests for creating NestedTensor from list of tensors."""
    
    def test_same_size_tensors(self):
        """Test with tensors of same size."""
        tensor_list = [
            torch.randn(3, 100, 100),
            torch.randn(3, 100, 100),
        ]
        
        nested = nested_tensor_from_tensor_list(tensor_list)
        
        assert nested.tensors.shape == (2, 3, 100, 100)
        assert nested.mask.shape == (2, 100, 100)
        # All positions should be valid (mask=False)
        assert not nested.mask.any()
    
    def test_different_size_tensors_padding(self):
        """Test tensors with different sizes are padded correctly."""
        tensor_list = [
            torch.ones(3, 100, 80),   # H=100, W=80
            torch.ones(3, 80, 100),   # H=80, W=100
        ]
        
        nested = nested_tensor_from_tensor_list(tensor_list)
        
        # Should be padded to max size: (100, 100)
        assert nested.tensors.shape == (2, 3, 100, 100)
        assert nested.mask.shape == (2, 100, 100)
    
    def test_mask_correctness_for_padded_tensors(self):
        """Test mask correctly identifies valid vs padded regions."""
        tensor_list = [
            torch.ones(3, 50, 50),   # Small image
            torch.ones(3, 100, 100), # Large image
        ]
        
        nested = nested_tensor_from_tensor_list(tensor_list)
        
        # First tensor: valid region is (50, 50), rest is padding
        # mask[0, :50, :50] should be False (valid)
        # mask[0, 50:, :] and mask[0, :, 50:] should be True (padding)
        assert not nested.mask[0, :50, :50].any()  # Valid region
        assert nested.mask[0, 50:, :].all()  # Padded height
        assert nested.mask[0, :, 50:].all()  # Padded width
        
        # Second tensor: all valid
        assert not nested.mask[1].any()
    
    def test_size_divisibility(self):
        """Test size_divisibility pads to divisible sizes."""
        tensor_list = [
            torch.randn(3, 100, 100),
            torch.randn(3, 90, 95),
        ]
        
        nested = nested_tensor_from_tensor_list(tensor_list, size_divisibility=32)
        
        # Max size is (100, 100), rounded up to divisible by 32 = (128, 128)
        assert nested.tensors.shape[2] % 32 == 0
        assert nested.tensors.shape[3] % 32 == 0
        assert nested.tensors.shape == (2, 3, 128, 128)
    
    def test_empty_list_raises_error(self):
        """Empty tensor list should raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            nested_tensor_from_tensor_list([])
    
    def test_mismatched_channels_raises_error(self):
        """Tensors with different channel counts should raise error."""
        tensor_list = [
            torch.randn(3, 100, 100),  # 3 channels
            torch.randn(4, 100, 100),  # 4 channels
        ]
        
        with pytest.raises(ValueError, match="same number of channels"):
            nested_tensor_from_tensor_list(tensor_list)
    
    def test_preserves_dtype_and_device(self):
        """Test that dtype and device are preserved."""
        tensor_list = [
            torch.randn(3, 50, 50, dtype=torch.float16),
            torch.randn(3, 60, 60, dtype=torch.float16),
        ]
        
        nested = nested_tensor_from_tensor_list(tensor_list)
        
        assert nested.tensors.dtype == torch.float16


class TestGetValidRatio:
    """Tests for valid ratio computation."""
    
    def test_full_valid_mask(self):
        """All valid positions should give ratio [1.0, 1.0]."""
        mask = torch.zeros(2, 100, 100, dtype=torch.bool)
        
        ratios = get_valid_ratio(mask)
        
        assert ratios.shape == (2, 2)
        assert torch.allclose(ratios, torch.ones(2, 2))
    
    def test_partial_height_padding(self):
        """Test with height padding."""
        mask = torch.zeros(1, 100, 100, dtype=torch.bool)
        mask[0, 80:, :] = True  # Bottom 20% is padding
        
        ratios = get_valid_ratio(mask)
        
        # Height ratio = 80/100 = 0.8, Width ratio = 1.0
        expected = torch.tensor([[0.8, 1.0]])
        assert torch.allclose(ratios, expected)
    
    def test_partial_width_padding(self):
        """Test with width padding."""
        mask = torch.zeros(1, 100, 100, dtype=torch.bool)
        mask[0, :, 70:] = True  # Right 30% is padding
        
        ratios = get_valid_ratio(mask)
        
        # Height ratio = 1.0, Width ratio = 70/100 = 0.7
        expected = torch.tensor([[1.0, 0.7]])
        assert torch.allclose(ratios, expected)
    
    def test_corner_padding(self):
        """Test with corner (both dimensions) padding."""
        mask = torch.zeros(1, 100, 100, dtype=torch.bool)
        mask[0, 80:, :] = True   # Bottom padding
        mask[0, :, 90:] = True   # Right padding
        
        ratios = get_valid_ratio(mask)
        
        # Height ratio = 80/100 = 0.8
        # Width ratio = 90/100 = 0.9 (from rows 0-79, columns 0-89 are valid)
        expected = torch.tensor([[0.8, 0.9]])
        assert torch.allclose(ratios, expected)
    
    def test_batch_independence(self):
        """Test ratio computation for batch with different padding."""
        mask = torch.zeros(2, 100, 100, dtype=torch.bool)
        mask[0, 80:, :] = True  # First: 80% valid height
        mask[1, :, 50:] = True  # Second: 50% valid width
        
        ratios = get_valid_ratio(mask)
        
        expected = torch.tensor([
            [0.8, 1.0],
            [1.0, 0.5],
        ])
        assert torch.allclose(ratios, expected)


class TestInterpolate:
    """Tests for interpolation wrapper."""
    
    def test_interpolate_upsample(self):
        """Test upsampling with interpolate."""
        x = torch.randn(1, 3, 50, 50)
        
        result = interpolate(x, size=(100, 100), mode="bilinear")
        
        assert result.shape == (1, 3, 100, 100)
    
    def test_interpolate_downsample(self):
        """Test downsampling with interpolate."""
        x = torch.randn(1, 3, 100, 100)
        
        result = interpolate(x, size=(50, 50), mode="bilinear")
        
        assert result.shape == (1, 3, 50, 50)
    
    def test_interpolate_scale_factor(self):
        """Test with scale_factor instead of size."""
        x = torch.randn(1, 3, 100, 100)
        
        result = interpolate(x, scale_factor=0.5, mode="nearest")
        
        assert result.shape == (1, 3, 50, 50)
    
    def test_interpolate_modes(self):
        """Test different interpolation modes."""
        x = torch.randn(1, 3, 50, 50)
        
        for mode in ["nearest", "bilinear", "bicubic"]:
            result = interpolate(x, size=(100, 100), mode=mode)
            assert result.shape == (1, 3, 100, 100)
