"""Transformer Decoder for Co-Deformable DETR.

This module implements the multi-scale deformable transformer decoder that
refines object queries through cross-attention with encoder features and
supports iterative bounding box refinement.

References:
    Deformable DETR: https://arxiv.org/abs/2010.04159
    Co-DETR: https://arxiv.org/abs/2211.12860
"""

from typing import List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from .attention import MultiScaleDeformableAttention


class DeformableTransformerDecoderLayer(nn.Module):
    """Single layer of the deformable transformer decoder.

    Each layer consists of:
    1. Self-attention on queries
    2. Multi-scale deformable cross-attention (query -> encoder features)
    3. Feed-forward network (FFN)
    4. Residual connections and layer normalization

    Args:
        embed_dim: Embedding dimension (default: 256).
        num_heads: Number of attention heads (default: 8).
        feedforward_dim: Dimension of FFN hidden layer (default: 1024).
        dropout: Dropout rate (default: 0.1).
        activation: Activation function name ('relu' or 'gelu', default: 'relu').
        num_levels: Number of feature pyramid levels (default: 4).
        num_points: Number of sampling points per attention head (default: 4).

    Shape:
        - Input query: (batch, num_queries, embed_dim)
        - Input memory: (batch, num_keys, embed_dim)
        - Output: (batch, num_queries, embed_dim)
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

        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.cross_attn = MultiScaleDeformableAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
            dropout=dropout,
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            nn.ReLU() if activation == "relu" else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        tgt: Tensor,
        reference_points: Tensor,
        memory: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        self_attn_mask: Optional[Tensor] = None,
        cross_attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass of decoder layer.

        Args:
            tgt: Target queries of shape (batch, num_queries, embed_dim).
            reference_points: Reference points of shape (batch, num_queries, num_levels, 2) or
                (batch, num_queries, num_levels, 4) for box format.
            memory: Encoder output of shape (batch, num_keys, embed_dim).
            spatial_shapes: Spatial shapes of shape (num_levels, 2).
            level_start_index: Starting indices of shape (num_levels,).
            self_attn_mask: Optional self-attention mask for query denoising.
            cross_attn_mask: Optional cross-attention mask.

        Returns:
            Output tensor of shape (batch, num_queries, embed_dim).
        """
        tgt2, _ = self.self_attn(
            query=tgt,
            key=tgt,
            value=tgt,
            attn_mask=self_attn_mask,
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.cross_attn(
            query=tgt,
            reference_points=reference_points,
            value=memory,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            attn_mask=cross_attn_mask,
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.ffn(tgt)
        tgt = tgt + tgt2
        tgt = self.norm3(tgt)

        return tgt


class CoDeformableDetrTransformerDecoder(nn.Module):
    """Multi-layer deformable transformer decoder for Co-Deformable DETR.

    This decoder refines object queries through iterative cross-attention with
    encoder features and supports bounding box refinement at each layer.

    Args:
        decoder_layer: Single decoder layer module.
        num_layers: Number of decoder layers (default: 6).
        return_intermediate: Whether to return outputs from all layers (default: True).

    Shape:
        - Input query: (batch, num_queries, embed_dim)
        - Input memory: (batch, num_keys, embed_dim)
        - Output: (num_layers, batch, num_queries, embed_dim) if return_intermediate
                  else (batch, num_queries, embed_dim)

    Example:
        >>> layer = DeformableTransformerDecoderLayer(embed_dim=256, num_heads=8)
        >>> decoder = CoDeformableDetrTransformerDecoder(layer, num_layers=6)
        >>> tgt = torch.randn(2, 300, 256)
        >>> memory = torch.randn(2, 12000, 256)
        >>> reference_points = torch.rand(2, 300, 4, 2)
        >>> spatial_shapes = torch.tensor([[100, 100], [50, 50], [25, 25], [13, 13]])
        >>> level_start_index = torch.tensor([0, 10000, 12500, 13125])
        >>> output = decoder(tgt, reference_points, memory, spatial_shapes, level_start_index)
        >>> output.shape
        torch.Size([6, 2, 300, 256])
    """

    def __init__(
        self,
        decoder_layer: DeformableTransformerDecoderLayer,
        num_layers: int = 6,
        return_intermediate: bool = True,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

        self.bbox_embed = None
        self.class_embed = None

    def forward(
        self,
        tgt: Tensor,
        reference_points: Tensor,
        memory: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        self_attn_mask: Optional[Tensor] = None,
        cross_attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass through all decoder layers.

        Args:
            tgt: Target queries of shape (batch, num_queries, embed_dim).
            reference_points: Initial reference points of shape (batch, num_queries, 2) or
                (batch, num_queries, 4) for box format.
            memory: Encoder output of shape (batch, num_keys, embed_dim).
            spatial_shapes: Spatial shapes of shape (num_levels, 2).
            level_start_index: Starting indices of shape (num_levels,).
            valid_ratios: Valid ratios of shape (batch, num_levels, 2).
            query_pos: Query positional embeddings of shape (batch, num_queries, embed_dim).
            self_attn_mask: Optional self-attention mask for query denoising.
            cross_attn_mask: Optional cross-attention mask.

        Returns:
            Output tensor of shape (num_layers, batch, num_queries, embed_dim) if
            return_intermediate is True, else (batch, num_queries, embed_dim).
        """
        output = tgt

        intermediate = []
        intermediate_reference_points = []

        for layer_idx, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = (
                    reference_points[:, :, None]
                    * torch.cat([valid_ratios, valid_ratios], -1)[:, None, :]
                )
            else:
                num_levels = valid_ratios.shape[1]
                reference_points_input = (
                    reference_points[:, :, None, :].repeat(1, 1, num_levels, 1)
                    * valid_ratios[:, None, :, :]
                )

            if query_pos is not None:
                output = output + query_pos

            output = layer(
                tgt=output,
                reference_points=reference_points_input,
                memory=memory,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                self_attn_mask=self_attn_mask,
                cross_attn_mask=cross_attn_mask,
            )

            if self.bbox_embed is not None:
                tmp = self.bbox_embed[layer_idx](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + self._inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[
                        ..., :2
                    ] + self._inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points

    @staticmethod
    def _inverse_sigmoid(x: Tensor, eps: float = 1e-5) -> Tensor:
        """Compute inverse sigmoid (logit) function.

        Args:
            x: Input tensor with values in (0, 1).
            eps: Small epsilon for numerical stability.

        Returns:
            Inverse sigmoid of x.
        """
        x = x.clamp(min=0, max=1)
        x1 = x.clamp(min=eps)
        x2 = (1 - x).clamp(min=eps)
        return torch.log(x1 / x2)
