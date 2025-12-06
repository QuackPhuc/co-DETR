"""Main Transformer Module for Co-Deformable DETR.

This module integrates the encoder and decoder to form the complete transformer
architecture with two-stage design and query generation from encoder features.

References:
    Deformable DETR: https://arxiv.org/abs/2010.04159
    Co-DETR: https://arxiv.org/abs/2211.12860
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .encoder import (
    CoDeformableDetrTransformerEncoder,
    DeformableTransformerEncoderLayer,
)
from .decoder import (
    CoDeformableDetrTransformerDecoder,
    DeformableTransformerDecoderLayer,
)


class CoDeformableDetrTransformer(nn.Module):
    """Complete transformer module for Co-Deformable DETR.

    This module implements the two-stage transformer architecture:
    1. Encoder: Processes multi-scale features with deformable attention
    2. Decoder: Refines object queries with iterative bounding box refinement

    The two-stage design generates object queries from encoder features rather
    than using learnable queries, improving convergence and performance.

    Args:
        embed_dim: Embedding dimension (default: 256).
        num_heads: Number of attention heads (default: 8).
        num_encoder_layers: Number of encoder layers (default: 6).
        num_decoder_layers: Number of decoder layers (default: 6).
        feedforward_dim: Dimension of FFN hidden layer (default: 1024).
        dropout: Dropout rate (default: 0.1).
        activation: Activation function name ('relu' or 'gelu', default: 'relu').
        num_feature_levels: Number of feature pyramid levels (default: 4).
        num_points: Number of sampling points per attention head (default: 4).
        num_queries: Number of object queries (default: 300).
        two_stage: Whether to use two-stage design (default: True).

    Shape:
        - Input features: List of (batch, C_i, H_i, W_i) for each level
        - Output: (num_decoder_layers, batch, num_queries, embed_dim)

    Example:
        >>> transformer = CoDeformableDetrTransformer(
        ...     embed_dim=256, num_heads=8, num_encoder_layers=6,
        ...     num_decoder_layers=6, num_queries=300
        ... )
        >>> srcs = [torch.randn(2, 256, 100, 100), torch.randn(2, 256, 50, 50)]
        >>> masks = [torch.zeros(2, 100, 100, dtype=torch.bool),
        ...          torch.zeros(2, 50, 50, dtype=torch.bool)]
        >>> pos_embeds = [torch.randn(2, 256, 100, 100), torch.randn(2, 256, 50, 50)]
        >>> output = transformer(srcs, masks, pos_embeds)
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        feedforward_dim: int = 1024,
        dropout: float = 0.1,
        activation: str = "relu",
        num_feature_levels: int = 4,
        num_points: int = 4,
        num_queries: int = 300,
        two_stage: bool = True,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_feature_levels = num_feature_levels
        self.num_queries = num_queries
        self.two_stage = two_stage

        encoder_layer = DeformableTransformerEncoderLayer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
            activation=activation,
            num_levels=num_feature_levels,
            num_points=num_points,
        )
        self.encoder = CoDeformableDetrTransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers,
        )

        decoder_layer = DeformableTransformerDecoderLayer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
            activation=activation,
            num_levels=num_feature_levels,
            num_points=num_points,
        )
        self.decoder = CoDeformableDetrTransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_decoder_layers,
            return_intermediate=True,
        )

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, embed_dim))

        if two_stage:
            self.enc_output = nn.Linear(embed_dim, embed_dim)
            self.enc_output_norm = nn.LayerNorm(embed_dim)
            self.pos_trans = nn.Linear(embed_dim * 2, embed_dim * 2)
            self.pos_trans_norm = nn.LayerNorm(embed_dim * 2)
        else:
            self.reference_points = nn.Linear(embed_dim, 2)

        self.tgt_embed = nn.Embedding(num_queries, embed_dim)
        self.query_pos_embed = nn.Embedding(num_queries, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize parameters."""
        nn.init.normal_(self.level_embed)
        nn.init.normal_(self.tgt_embed.weight)
        nn.init.normal_(self.query_pos_embed.weight)

        if self.two_stage:
            nn.init.xavier_uniform_(self.enc_output.weight)
            nn.init.constant_(self.enc_output.bias, 0.0)
            nn.init.xavier_uniform_(self.pos_trans.weight)
            nn.init.constant_(self.pos_trans.bias, 0.0)
        else:
            nn.init.xavier_uniform_(self.reference_points.weight)
            nn.init.constant_(self.reference_points.bias, 0.0)

    def forward(
        self,
        srcs: list,
        masks: list,
        pos_embeds: list,
        query_embed: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Forward pass of the transformer.

        Args:
            srcs: List of source features, each of shape (batch, embed_dim, H_i, W_i).
            masks: List of masks, each of shape (batch, H_i, W_i).
            pos_embeds: List of positional embeddings, each of shape (batch, embed_dim, H_i, W_i).
            query_embed: Optional custom query embeddings of shape (num_queries, embed_dim).
            attn_mask: Optional attention mask for query denoising.

        Returns:
            Tuple of:
                - hs: Decoder outputs of shape (num_decoder_layers, batch, num_queries, embed_dim)
                - init_reference: Initial reference points of shape (batch, num_queries, 2)
                - inter_references: Intermediate reference points of shape
                    (num_decoder_layers, batch, num_queries, 2)
                - enc_outputs_class: Encoder classification outputs (for two-stage)
                - enc_outputs_coord: Encoder coordinate outputs (for two-stage)
        """
        assert len(srcs) == len(masks) == len(pos_embeds) == self.num_feature_levels

        batch_size = srcs[0].shape[0]
        device = srcs[0].device

        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []

        for level, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shapes.append((h, w))

            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[level].view(1, 1, -1)

            src_flatten.append(src)
            mask_flatten.append(mask)
            lvl_pos_embed_flatten.append(lvl_pos_embed)

        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)

        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=device
        )
        level_start_index = torch.cat(
            [spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]]
        )
        valid_ratios = torch.stack([self._get_valid_ratio(m, device) for m in masks], 1)

        reference_points = self.encoder.get_reference_points(
            spatial_shapes, valid_ratios, device=device
        )

        memory = self.encoder(
            src=src_flatten + lvl_pos_embed_flatten,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            attn_mask=None,
        )

        if self.two_stage:
            output_memory = self.enc_output_norm(self.enc_output(memory))
            enc_outputs_class = None
            enc_outputs_coord_unact = None

            topk = self.num_queries
            topk_proposals = torch.topk(output_memory.max(-1)[0], topk, dim=1)[1]

            reference_points = torch.gather(
                reference_points,
                1,
                topk_proposals.unsqueeze(-1)
                .unsqueeze(-1)
                .repeat(1, 1, self.num_feature_levels, 2),
            )
            reference_points = reference_points.mean(dim=2)
            reference_points = reference_points.detach()

            tgt = torch.gather(
                output_memory,
                1,
                topk_proposals.unsqueeze(-1).repeat(1, 1, self.embed_dim),
            )
            tgt = tgt.detach()

            query_pos = self.query_pos_embed.weight.unsqueeze(0).repeat(
                batch_size, 1, 1
            )
        else:
            tgt = self.tgt_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)
            query_pos = self.query_pos_embed.weight.unsqueeze(0).repeat(
                batch_size, 1, 1
            )
            reference_points = self.reference_points(query_pos).sigmoid()
            enc_outputs_class = None
            enc_outputs_coord_unact = None

        hs, inter_references = self.decoder(
            tgt=tgt,
            reference_points=reference_points,
            memory=memory,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            query_pos=query_pos,
            self_attn_mask=attn_mask,
        )

        return (
            hs,
            reference_points,
            inter_references,
            enc_outputs_class,
            enc_outputs_coord_unact,
        )

    @staticmethod
    def _get_valid_ratio(mask: Tensor, device: torch.device) -> Tensor:
        """Compute valid ratio for a single mask.

        Args:
            mask: Binary mask of shape (batch, H, W) where True indicates padding.
            device: Target device.

        Returns:
            Valid ratio of shape (batch, 2) in [0, 1].
        """
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio
