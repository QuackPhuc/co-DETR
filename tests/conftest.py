"""
Pytest fixtures and configuration for Co-DETR test suite.

This module provides shared fixtures for testing the Co-DETR PyTorch
implementation, including dummy data generators and common utilities.
"""

import pytest
import torch
from torch import Tensor
from typing import Dict, List


# ============================================================================
# Device Fixtures
# ============================================================================

@pytest.fixture
def device() -> torch.device:
    """Get available device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def cpu_device() -> torch.device:
    """Get CPU device for deterministic tests."""
    return torch.device("cpu")


# ============================================================================
# Size Fixtures
# ============================================================================

@pytest.fixture
def batch_size() -> int:
    """Default batch size for testing."""
    return 2


@pytest.fixture
def num_classes() -> int:
    """Default number of classes for testing."""
    return 80


@pytest.fixture
def num_queries() -> int:
    """Default number of object queries."""
    return 300


@pytest.fixture
def embed_dim() -> int:
    """Default embedding dimension."""
    return 256


@pytest.fixture
def image_size() -> tuple:
    """Default image size (height, width)."""
    return (800, 800)


# ============================================================================
# Dummy Data Fixtures
# ============================================================================

@pytest.fixture
def dummy_image(batch_size: int, image_size: tuple, cpu_device: torch.device) -> Tensor:
    """
    Create dummy batch of images.
    
    Returns:
        Tensor of shape (batch_size, 3, height, width) with values in [0, 1].
    """
    height, width = image_size
    return torch.rand(batch_size, 3, height, width, device=cpu_device)


@pytest.fixture
def dummy_targets(batch_size: int, num_classes: int, cpu_device: torch.device) -> List[Dict[str, Tensor]]:
    """
    Create dummy targets for training.
    
    Each target contains:
        - 'labels': Class indices, shape (num_gt,)
        - 'boxes': Normalized cxcywh boxes, shape (num_gt, 4)
    
    Returns:
        List of target dicts, one per batch item.
    """
    targets = []
    for i in range(batch_size):
        num_gt = torch.randint(1, 10, (1,)).item()  # 1-9 objects per image
        targets.append({
            'labels': torch.randint(0, num_classes, (num_gt,), device=cpu_device),
            'boxes': torch.rand(num_gt, 4, device=cpu_device),  # cxcywh normalized
        })
    return targets


@pytest.fixture
def dummy_feature_maps(batch_size: int, embed_dim: int, cpu_device: torch.device) -> List[Tensor]:
    """
    Create dummy multi-scale feature maps (P3, P4, P5, P6).
    
    Returns:
        List of 4 feature tensors with decreasing spatial sizes.
    """
    return [
        torch.rand(batch_size, embed_dim, 100, 100, device=cpu_device),  # P3
        torch.rand(batch_size, embed_dim, 50, 50, device=cpu_device),    # P4
        torch.rand(batch_size, embed_dim, 25, 25, device=cpu_device),    # P5
        torch.rand(batch_size, embed_dim, 13, 13, device=cpu_device),    # P6
    ]


@pytest.fixture
def dummy_backbone_features(batch_size: int, cpu_device: torch.device) -> List[Tensor]:
    """
    Create dummy backbone features (C3, C4, C5) from ResNet-50.
    
    Returns:
        List of 3 feature tensors with ResNet channel dimensions.
    """
    return [
        torch.rand(batch_size, 512, 100, 100, device=cpu_device),   # C3
        torch.rand(batch_size, 1024, 50, 50, device=cpu_device),    # C4
        torch.rand(batch_size, 2048, 25, 25, device=cpu_device),    # C5
    ]


@pytest.fixture
def dummy_boxes_xyxy(cpu_device: torch.device) -> Tensor:
    """
    Create dummy boxes in xyxy format for testing box operations.
    
    Returns:
        Tensor of shape (4, 4) with valid xyxy boxes.
    """
    return torch.tensor([
        [0.0, 0.0, 10.0, 10.0],   # 10x10 box at origin
        [5.0, 5.0, 15.0, 15.0],   # 10x10 box overlapping
        [20.0, 20.0, 30.0, 30.0], # 10x10 box non-overlapping
        [0.0, 0.0, 100.0, 100.0], # 100x100 box enclosing
    ], device=cpu_device)


@pytest.fixture
def dummy_boxes_cxcywh(cpu_device: torch.device) -> Tensor:
    """
    Create dummy boxes in cxcywh format for testing.
    
    Returns:
        Tensor of shape (4, 4) with valid cxcywh boxes.
    """
    return torch.tensor([
        [5.0, 5.0, 10.0, 10.0],     # Center (5,5), size 10x10
        [10.0, 10.0, 10.0, 10.0],   # Center (10,10), size 10x10
        [25.0, 25.0, 10.0, 10.0],   # Center (25,25), size 10x10
        [50.0, 50.0, 100.0, 100.0], # Center (50,50), size 100x100
    ], device=cpu_device)


# ============================================================================
# Helper Functions
# ============================================================================

def assert_tensor_shape(tensor: Tensor, expected_shape: tuple, name: str = "tensor"):
    """Assert tensor has expected shape with informative error message."""
    assert tensor.shape == expected_shape, (
        f"{name} has shape {tensor.shape}, expected {expected_shape}"
    )


def assert_no_nan_inf(tensor: Tensor, name: str = "tensor"):
    """Assert tensor has no NaN or Inf values."""
    assert not torch.isnan(tensor).any(), f"{name} contains NaN values"
    assert not torch.isinf(tensor).any(), f"{name} contains Inf values"


def assert_gradients_exist(module: torch.nn.Module, name: str = "module"):
    """Assert that all trainable parameters have gradients after backward."""
    for param_name, param in module.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, (
                f"Parameter {param_name} in {name} has no gradient"
            )
            assert not torch.isnan(param.grad).any(), (
                f"Parameter {param_name} in {name} has NaN gradients"
            )
