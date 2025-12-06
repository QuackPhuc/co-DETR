"""
Tests for dataloader and collation.

This module tests the custom collate function and dataloader utilities
for batching variable-size images.
"""

import pytest
import torch

from codetr.data.dataloader import collate_fn, build_dataloader
from codetr.models.utils.misc import NestedTensor


class TestCollateFn:
    """Tests for custom collate function."""
    
    def test_collate_single_sample(self):
        """Test collation with single sample."""
        samples = [
            (torch.randn(3, 100, 100), {'labels': torch.tensor([0, 1]), 'boxes': torch.rand(2, 4)})
        ]
        
        nested_tensor, targets = collate_fn(samples)
        
        assert isinstance(nested_tensor, NestedTensor)
        assert len(targets) == 1
    
    def test_collate_multiple_samples(self):
        """Test collation with multiple samples of same size."""
        samples = [
            (torch.randn(3, 100, 100), {'labels': torch.tensor([0]), 'boxes': torch.rand(1, 4)}),
            (torch.randn(3, 100, 100), {'labels': torch.tensor([1, 2]), 'boxes': torch.rand(2, 4)}),
        ]
        
        nested_tensor, targets = collate_fn(samples)
        
        # Batch dimension should be 2
        assert nested_tensor.tensors.shape[0] == 2
        assert nested_tensor.mask.shape[0] == 2
        
        # Targets list should have 2 items
        assert len(targets) == 2
    
    def test_collate_variable_sizes(self):
        """Test collation with different image sizes."""
        samples = [
            (torch.randn(3, 100, 80), {'labels': torch.tensor([0]), 'boxes': torch.rand(1, 4)}),
            (torch.randn(3, 120, 90), {'labels': torch.tensor([1]), 'boxes': torch.rand(1, 4)}),
            (torch.randn(3, 80, 110), {'labels': torch.tensor([2]), 'boxes': torch.rand(1, 4)}),
        ]
        
        nested_tensor, targets = collate_fn(samples)
        
        # Should be padded to max size
        max_h = 120
        max_w = 110
        
        assert nested_tensor.tensors.shape[0] == 3
        assert nested_tensor.tensors.shape[2] >= max_h
        assert nested_tensor.tensors.shape[3] >= max_w
    
    def test_padding_mask_correctness(self):
        """Test padding mask correctly identifies valid vs padded regions."""
        samples = [
            (torch.ones(3, 50, 50), {'labels': torch.tensor([0]), 'boxes': torch.rand(1, 4)}),
            (torch.ones(3, 100, 100), {'labels': torch.tensor([1]), 'boxes': torch.rand(1, 4)}),
        ]
        
        nested_tensor, _ = collate_fn(samples)
        
        # First sample: valid region is (50, 50)
        # mask[0, :50, :50] should be False (valid)
        assert not nested_tensor.mask[0, :50, :50].any()
        
        # mask[0, 50:, :] should be True (padding) if padded
        if nested_tensor.tensors.shape[2] > 50:
            assert nested_tensor.mask[0, 50:, :].all()
        
        # Second sample: all valid (100x100)
        if nested_tensor.tensors.shape[2] == 100 and nested_tensor.tensors.shape[3] == 100:
            assert not nested_tensor.mask[1].any()
    
    def test_targets_preserved(self):
        """Test target dictionaries are preserved correctly."""
        targets_in = [
            {'labels': torch.tensor([0, 1]), 'boxes': torch.rand(2, 4), 'image_id': 42},
            {'labels': torch.tensor([5]), 'boxes': torch.rand(1, 4), 'image_id': 43},
        ]
        samples = [
            (torch.randn(3, 100, 100), targets_in[0]),
            (torch.randn(3, 100, 100), targets_in[1]),
        ]
        
        _, targets_out = collate_fn(samples)
        
        assert len(targets_out) == 2
        assert torch.equal(targets_out[0]['labels'], targets_in[0]['labels'])
        assert torch.equal(targets_out[1]['labels'], targets_in[1]['labels'])


class TestNestedTensorOutput:
    """Tests for NestedTensor output from collation."""
    
    def test_nested_tensor_decompose(self):
        """Test NestedTensor can be decomposed."""
        samples = [
            (torch.randn(3, 100, 100), {'labels': torch.tensor([0]), 'boxes': torch.rand(1, 4)}),
        ]
        
        nested_tensor, _ = collate_fn(samples)
        
        tensors, mask = nested_tensor.decompose()
        
        assert tensors.shape[0] == 1
        assert mask.shape[0] == 1
    
    def test_nested_tensor_device_transfer(self):
        """Test NestedTensor can be moved to device."""
        samples = [
            (torch.randn(3, 50, 50), {'labels': torch.tensor([0]), 'boxes': torch.rand(1, 4)}),
        ]
        
        nested_tensor, _ = collate_fn(samples)
        
        # Move to CPU (should work regardless of GPU availability)
        nested_cpu = nested_tensor.to(torch.device('cpu'))
        
        assert nested_cpu.tensors.device == torch.device('cpu')
        assert nested_cpu.mask.device == torch.device('cpu')
