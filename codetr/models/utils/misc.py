"""Miscellaneous utility functions and classes.

This module provides utility classes and functions for handling nested tensors,
padding, masks, and other common operations in object detection.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


class NestedTensor:
    """Container for tensors with different spatial sizes and their masks.

    This class handles batches of images/features with different sizes by
    storing both the padded tensors and binary masks indicating valid regions.

    Args:
        tensors: Padded tensor. Shape: (batch, channels, height, width).
        mask: Binary mask. Shape: (batch, height, width).
            True indicates padding/invalid region, False indicates valid region.

    Attributes:
        tensors: The padded tensor.
        mask: The binary mask.

    Examples:
        >>> tensor = torch.randn(2, 3, 100, 100)
        >>> mask = torch.zeros(2, 100, 100, dtype=torch.bool)
        >>> mask[0, 80:, :] = True  # Mark bottom region as padding
        >>> nested = NestedTensor(tensor, mask)
        >>> nested.tensors.shape
        torch.Size([2, 3, 100, 100])
    """

    def __init__(self, tensors: Tensor, mask: Tensor) -> None:
        self.tensors = tensors
        self.mask = mask

    def to(self, device: torch.device) -> "NestedTensor":
        """Move tensors and mask to specified device.

        Args:
            device: Target device.

        Returns:
            New NestedTensor on the target device.
        """
        cast_tensor = self.tensors.to(device)
        cast_mask = self.mask.to(device)
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self) -> Tuple[Tensor, Tensor]:
        """Decompose into tensors and mask.

        Returns:
            Tuple of (tensors, mask).
        """
        return self.tensors, self.mask

    def __repr__(self) -> str:
        """String representation."""
        return f"NestedTensor(tensors={self.tensors.shape}, mask={self.mask.shape})"


def nested_tensor_from_tensor_list(
    tensor_list: List[Tensor],
    size_divisibility: int = 0,
) -> NestedTensor:
    """Create NestedTensor from a list of tensors with different sizes.

    This function pads all tensors to the same size (max size in the batch)
    and creates corresponding masks.

    Args:
        tensor_list: List of tensors with shape (channels, height, width).
            All tensors must have the same number of channels but can have
            different heights and widths.
        size_divisibility: If > 0, pad to make height and width divisible
            by this value. Useful for models requiring specific input sizes.
            Default: 0 (no divisibility constraint).

    Returns:
        NestedTensor containing padded tensors and masks.

    Examples:
        >>> tensors = [
        ...     torch.randn(3, 100, 100),
        ...     torch.randn(3, 80, 90),
        ...     torch.randn(3, 120, 85),
        ... ]
        >>> nested = nested_tensor_from_tensor_list(tensors, size_divisibility=32)
        >>> nested.tensors.shape  # Padded to (3, 3, 128, 128)
        torch.Size([3, 3, 128, 128])
        >>> nested.mask.shape
        torch.Size([3, 128, 128])
    """
    if len(tensor_list) == 0:
        raise ValueError("tensor_list cannot be empty")

    # Check that all tensors have the same number of channels
    channels = tensor_list[0].shape[0]
    for tensor in tensor_list:
        if tensor.shape[0] != channels:
            raise ValueError("All tensors must have the same number of channels")

    # Find maximum height and width
    max_height = max([tensor.shape[1] for tensor in tensor_list])
    max_width = max([tensor.shape[2] for tensor in tensor_list])

    # Adjust for size divisibility
    if size_divisibility > 0:
        max_height = (
            (max_height + size_divisibility - 1) // size_divisibility
        ) * size_divisibility
        max_width = (
            (max_width + size_divisibility - 1) // size_divisibility
        ) * size_divisibility

    batch_size = len(tensor_list)
    dtype = tensor_list[0].dtype
    device = tensor_list[0].device

    # Create padded tensor and mask
    padded_tensors = torch.zeros(
        (batch_size, channels, max_height, max_width),
        dtype=dtype,
        device=device,
    )
    masks = torch.ones(
        (batch_size, max_height, max_width),
        dtype=torch.bool,
        device=device,
    )

    # Fill in the data and update masks
    for i, tensor in enumerate(tensor_list):
        h, w = tensor.shape[1], tensor.shape[2]
        padded_tensors[i, :, :h, :w] = tensor
        masks[i, :h, :w] = False  # False indicates valid region

    return NestedTensor(padded_tensors, masks)


def get_valid_ratio(mask: Tensor) -> Tensor:
    """Compute the ratio of valid (non-padded) region in each feature map.

    This is useful for attention mechanisms to understand the actual spatial
    extent of features before padding.

    Args:
        mask: Binary mask. Shape: (batch, height, width).
            True indicates padding/invalid, False indicates valid.

    Returns:
        Valid ratios. Shape: (batch, 2) where ratios[:, 0] is height ratio
        and ratios[:, 1] is width ratio.

    Examples:
        >>> mask = torch.zeros(2, 100, 100, dtype=torch.bool)
        >>> mask[0, 80:, :] = True  # 80% valid height
        >>> mask[1, :, 90:] = True  # 90% valid width
        >>> get_valid_ratio(mask)
        tensor([[0.8000, 1.0000],
                [1.0000, 0.9000]])
    """
    batch_size, height, width = mask.shape

    # Invert mask: True -> valid, False -> invalid
    valid_mask = ~mask

    # Compute valid height and width for each sample
    valid_height = valid_mask.sum(dim=1, dtype=torch.float32)  # (batch, width)
    valid_width = valid_mask.sum(dim=2, dtype=torch.float32)  # (batch, height)

    # Get maximum valid height and width for each sample
    valid_height = valid_height.max(dim=1)[0]  # (batch,)
    valid_width = valid_width.max(dim=1)[0]  # (batch,)

    # Compute ratios
    valid_ratio_h = valid_height / height
    valid_ratio_w = valid_width / width

    # Stack into (batch, 2)
    valid_ratios = torch.stack([valid_ratio_h, valid_ratio_w], dim=1)

    return valid_ratios


def interpolate(
    input: Tensor,
    size: Optional[Tuple[int, int]] = None,
    scale_factor: Optional[float] = None,
    mode: str = "nearest",
    align_corners: Optional[bool] = None,
) -> Tensor:
    """Wrapper around torch.nn.functional.interpolate with better defaults.

    Args:
        input: Input tensor. Shape: (batch, channels, height, width).
        size: Target spatial size (height, width).
        scale_factor: Multiplier for spatial size.
        mode: Interpolation mode. Options: 'nearest', 'linear', 'bilinear',
            'bicubic', 'trilinear', 'area'. Default: 'nearest'.
        align_corners: If True, align corners of input and output. Only used
            for 'linear', 'bilinear', 'bicubic', 'trilinear'. Default: None.

    Returns:
        Interpolated tensor.
    """
    if mode in ("linear", "bilinear", "bicubic", "trilinear") and align_corners is None:
        align_corners = False

    return F.interpolate(
        input,
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
    )
