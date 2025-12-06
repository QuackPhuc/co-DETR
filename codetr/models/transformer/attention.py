"""Multi-Scale Deformable Attention Module for Co-Deformable DETR.

This module implements the core deformable attention mechanism that enables
efficient multi-scale feature aggregation in the transformer architecture.
Uses PyTorch's grid_sample for deformable sampling (no custom CUDA required).

References:
    Deformable DETR: https://arxiv.org/abs/2010.04159
    Co-DETR: https://arxiv.org/abs/2211.12860
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MultiScaleDeformableAttention(nn.Module):
    """Multi-scale deformable attention module.

    This module performs attention over multi-scale feature maps with learnable
    sampling offsets and attention weights. Each query can attend to multiple
    reference points across different feature scales.

    Args:
        embed_dim: Total dimension of the model (default: 256).
        num_heads: Number of attention heads (default: 8).
        num_levels: Number of feature pyramid levels (default: 4).
        num_points: Number of sampling points per attention head per level (default: 4).
        dropout: Dropout rate for attention weights (default: 0.1).

    Shape:
        - Input query: (batch, num_queries, embed_dim)
        - Input reference_points: (batch, num_queries, num_levels, 2) in [0, 1]
        - Input value: (batch, num_keys, embed_dim)
        - Input spatial_shapes: (num_levels, 2) - (H, W) for each level
        - Input level_start_index: (num_levels,) - Starting index for each level
        - Output: (batch, num_queries, embed_dim)

    Example:
        >>> attn = MultiScaleDeformableAttention(embed_dim=256, num_heads=8)
        >>> query = torch.randn(2, 300, 256)
        >>> reference_points = torch.rand(2, 300, 4, 2)
        >>> value = torch.randn(2, 12000, 256)
        >>> spatial_shapes = torch.tensor([[100, 100], [50, 50], [25, 25], [13, 13]])
        >>> level_start_index = torch.tensor([0, 10000, 12500, 13125])
        >>> output = attn(query, reference_points, value, spatial_shapes, level_start_index)
        >>> output.shape
        torch.Size([2, 300, 256])
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim must be divisible by num_heads, "
                f"got embed_dim={embed_dim} and num_heads={num_heads}"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.total_points = num_heads * num_levels * num_points
        self.head_dim = embed_dim // num_heads

        self.sampling_offsets = nn.Linear(
            embed_dim, num_heads * num_levels * num_points * 2
        )
        self.attention_weights = nn.Linear(
            embed_dim, num_heads * num_levels * num_points
        )
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize parameters with Xavier uniform and zeros."""
        nn.init.constant_(self.sampling_offsets.weight, 0.0)

        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * torch.pi / self.num_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.num_heads, 1, 1, 2)
            .repeat(1, self.num_levels, self.num_points, 1)
        )

        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))

        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

    def forward(
        self,
        query: Tensor,
        reference_points: Tensor,
        value: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass of multi-scale deformable attention.

        Args:
            query: Query embeddings of shape (batch, num_queries, embed_dim).
            reference_points: Reference points of shape (batch, num_queries, num_levels, 2)
                with values in [0, 1] range.
            value: Value features of shape (batch, num_keys, embed_dim).
            spatial_shapes: Spatial shapes of shape (num_levels, 2) containing (H, W).
            level_start_index: Starting indices of shape (num_levels,).
            attn_mask: Optional attention mask of shape (batch, num_heads, num_queries, num_keys).

        Returns:
            Output tensor of shape (batch, num_queries, embed_dim).
        """
        batch, num_queries, _ = query.shape
        batch, num_keys, _ = value.shape

        value = self.value_proj(value)
        if attn_mask is not None:
            value = value.masked_fill(attn_mask[..., None], float(0))
        value = value.view(batch, num_keys, self.num_heads, self.head_dim)

        sampling_offsets = self.sampling_offsets(query).view(
            batch, num_queries, self.num_heads, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            batch, num_queries, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = F.softmax(attention_weights, -1).view(
            batch, num_queries, self.num_heads, self.num_levels, self.num_points
        )

        if reference_points.shape[-1] == 2:
            offset_normalizer = spatial_shapes.flip([1]).view(
                1, 1, 1, self.num_levels, 1, 2
            )
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets
                / self.num_points
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )
        else:
            raise ValueError(
                f"Last dim of reference_points must be 2 or 4, but got {reference_points.shape[-1]}"
            )

        output = self._sample_and_aggregate(
            value,
            spatial_shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
        )

        output = self.output_proj(output)

        return output

    def _sample_and_aggregate(
        self,
        value: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        sampling_locations: Tensor,
        attention_weights: Tensor,
    ) -> Tensor:
        """Sample features at sampling locations and aggregate with attention weights.

        Args:
            value: Value features of shape (batch, num_keys, num_heads, head_dim).
            spatial_shapes: Spatial shapes of shape (num_levels, 2).
            level_start_index: Starting indices of shape (num_levels,).
            sampling_locations: Sampling locations of shape
                (batch, num_queries, num_heads, num_levels, num_points, 2).
            attention_weights: Attention weights of shape
                (batch, num_queries, num_heads, num_levels, num_points).

        Returns:
            Aggregated output of shape (batch, num_queries, embed_dim).
        """
        batch, num_queries, num_heads, num_levels, num_points, _ = (
            sampling_locations.shape
        )

        output_list = []
        for level_idx in range(num_levels):
            H, W = spatial_shapes[level_idx]
            start_idx = level_start_index[level_idx]
            end_idx = (
                level_start_index[level_idx + 1]
                if level_idx < num_levels - 1
                else value.shape[1]
            )

            value_level = value[:, start_idx:end_idx, :, :].view(
                batch, H, W, num_heads, self.head_dim
            )
            value_level = value_level.permute(0, 3, 4, 1, 2).contiguous()
            value_level = value_level.view(batch * num_heads, self.head_dim, H, W)

            sampling_loc_level = sampling_locations[:, :, :, level_idx, :, :]
            sampling_loc_level = sampling_loc_level.permute(0, 2, 1, 3, 4).contiguous()
            sampling_loc_level = sampling_loc_level.view(
                batch * num_heads, num_queries, num_points, 2
            )

            sampling_grid = 2.0 * sampling_loc_level - 1.0

            sampled_value = F.grid_sample(
                value_level,
                sampling_grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )

            sampled_value = sampled_value.view(
                batch, num_heads, self.head_dim, num_queries, num_points
            ).permute(0, 3, 1, 4, 2)

            output_list.append(sampled_value)

        output = torch.stack(output_list, dim=3)
        output = (output * attention_weights[..., None]).sum(dim=(3, 4))
        output = output.view(batch, num_queries, self.embed_dim)

        return output
