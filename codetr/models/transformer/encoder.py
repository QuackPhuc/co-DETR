"""Transformer Encoder for Co-Deformable DETR.

This module implements the multi-scale deformable transformer encoder that
processes multi-scale feature maps to generate enhanced representations for
object detection.

References:
    Deformable DETR: https://arxiv.org/abs/2010.04159
    Co-DETR: https://arxiv.org/abs/2211.12860
"""

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .attention import MultiScaleDeformableAttention


class DeformableTransformerEncoderLayer(nn.Module):
    """Single layer of the deformable transformer encoder.

    Each layer consists of:
    1. Multi-scale deformable self-attention
    2. Feed-forward network (FFN)
    3. Residual connections and layer normalization

    Args:
        embed_dim: Embedding dimension (default: 256).
        num_heads: Number of attention heads (default: 8).
        feedforward_dim: Dimension of FFN hidden layer (default: 1024).
        dropout: Dropout rate (default: 0.1).
        activation: Activation function name ('relu' or 'gelu', default: 'relu').
        num_levels: Number of feature pyramid levels (default: 4).
        num_points: Number of sampling points per attention head (default: 4).

    Shape:
        - Input: (batch, num_keys, embed_dim)
        - Output: (batch, num_keys, embed_dim)
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        feedforward_dim: int = 1024,
        dropout: float = 0.1,
        activation: str = "relu",
        num_levels: int = 4,
        num_points: int = 4,
    ) -> None:
        super().__init__()

        self.self_attn = MultiScaleDeformableAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
            dropout=dropout,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            nn.ReLU() if activation == "relu" else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        src: Tensor,
        reference_points: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass of encoder layer.

        Args:
            src: Source features of shape (batch, num_keys, embed_dim).
            reference_points: Reference points of shape (batch, num_keys, num_levels, 2).
            spatial_shapes: Spatial shapes of shape (num_levels, 2).
            level_start_index: Starting indices of shape (num_levels,).
            attn_mask: Optional attention mask.

        Returns:
            Output tensor of shape (batch, num_keys, embed_dim).
        """
        src2 = self.self_attn(
            query=src,
            reference_points=reference_points,
            value=src,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            attn_mask=attn_mask,
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.ffn(src)
        src = src + src2
        src = self.norm2(src)

        return src


class CoDeformableDetrTransformerEncoder(nn.Module):
    """Multi-layer deformable transformer encoder for Co-Deformable DETR.

    This encoder processes multi-scale feature maps using deformable attention
    to capture long-range dependencies while maintaining computational efficiency.

    Args:
        encoder_layer: Single encoder layer module.
        num_layers: Number of encoder layers (default: 6).

    Shape:
        - Input: (batch, num_keys, embed_dim)
        - Output: (batch, num_keys, embed_dim)

    Example:
        >>> layer = DeformableTransformerEncoderLayer(embed_dim=256, num_heads=8)
        >>> encoder = CoDeformableDetrTransformerEncoder(layer, num_layers=6)
        >>> src = torch.randn(2, 12000, 256)
        >>> reference_points = torch.rand(2, 12000, 4, 2)
        >>> spatial_shapes = torch.tensor([[100, 100], [50, 50], [25, 25], [13, 13]])
        >>> level_start_index = torch.tensor([0, 10000, 12500, 13125])
        >>> output = encoder(src, reference_points, spatial_shapes, level_start_index)
        >>> output.shape
        torch.Size([2, 12000, 256])
    """

    def __init__(
        self,
        encoder_layer: DeformableTransformerEncoderLayer,
        num_layers: int = 6,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(
        self,
        src: Tensor,
        reference_points: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass through all encoder layers.

        Args:
            src: Source features of shape (batch, num_keys, embed_dim).
            reference_points: Reference points of shape (batch, num_keys, num_levels, 2).
            spatial_shapes: Spatial shapes of shape (num_levels, 2).
            level_start_index: Starting indices of shape (num_levels,).
            valid_ratios: Valid ratios of shape (batch, num_levels, 2) for padding masks.
            attn_mask: Optional attention mask.

        Returns:
            Output tensor of shape (batch, num_keys, embed_dim).
        """
        output = src

        for layer in self.layers:
            output = layer(
                src=output,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                attn_mask=attn_mask,
            )

        return output

    @staticmethod
    def get_reference_points(
        spatial_shapes: Tensor,
        valid_ratios: Tensor,
        device: torch.device,
    ) -> Tensor:
        """Generate reference points for multi-scale features.

        Creates a grid of normalized reference points for each feature level,
        adjusted by valid ratios to account for padding.

        Args:
            spatial_shapes: Spatial shapes of shape (num_levels, 2) containing (H, W).
            valid_ratios: Valid ratios of shape (batch, num_levels, 2) in [0, 1].
            device: Target device for the reference points.

        Returns:
            Reference points of shape (batch, num_keys, num_levels, 2) in [0, 1].

        Example:
            >>> spatial_shapes = torch.tensor([[100, 100], [50, 50]])
            >>> valid_ratios = torch.ones(2, 2, 2)
            >>> ref_points = CoDeformableDetrTransformerEncoder.get_reference_points(
            ...     spatial_shapes, valid_ratios, torch.device('cpu')
            ... )
            >>> ref_points.shape
            torch.Size([2, 12500, 2, 2])
        """
        reference_points_list = []

        for level, (H, W) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device),
                indexing="ij",
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, level, 1] * H)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, level, 0] * W)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)

        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]

        return reference_points
