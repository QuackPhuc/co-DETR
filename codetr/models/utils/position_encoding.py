"""Positional encoding for transformers.

This module implements sine/cosine positional encoding for 2D feature maps,
which is essential for transformers to understand spatial relationships.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class PositionEmbeddingSine(nn.Module):
    """2D sine-cosine positional embedding for feature maps.

    This module generates positional encodings using sine and cosine functions
    of different frequencies, similar to the original Transformer paper but
    extended to 2D spatial dimensions.

    Args:
        num_pos_feats: Number of positional features (half of embedding dim).
            The total embedding dimension will be 2 * num_pos_feats.
            Default: 128 (total embedding dim: 256).
        temperature: Temperature parameter for frequency scaling.
            Higher temperature leads to lower frequencies. Default: 10000.
        normalize: Whether to normalize coordinates to [0, 1] range.
            Default: True.
        scale: Scale factor applied after normalization. If None and normalize
            is True, uses 2*pi. Default: None.
        offset: Offset to add to normalized coordinates. Default: 0.0.
        eps: Small epsilon for numerical stability. Default: 1e-6.

    Examples:
        >>> pos_enc = PositionEmbeddingSine(num_pos_feats=128)
        >>> # Create dummy feature map (batch=2, channels=256, height=25, width=25)
        >>> feat = torch.randn(2, 256, 25, 25)
        >>> # Generate mask (True means padding, False means valid)
        >>> mask = torch.zeros(2, 25, 25, dtype=torch.bool)
        >>> pos = pos_enc(feat, mask)
        >>> pos.shape
        torch.Size([2, 256, 25, 25])
    """

    def __init__(
        self,
        num_pos_feats: int = 128,
        temperature: int = 10000,
        normalize: bool = True,
        scale: Optional[float] = None,
        offset: float = 0.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.offset = offset
        self.eps = eps

        if scale is not None and normalize is False:
            raise ValueError("scale should be None if normalize is False")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Generate 2D positional embeddings.

        Args:
            x: Input feature map. Shape: (batch, channels, height, width).
            mask: Binary mask indicating padding positions. Shape: (batch, height, width).
                True means padding/invalid, False means valid position.
                If None, all positions are considered valid.

        Returns:
            Positional embeddings. Shape: (batch, 2*num_pos_feats, height, width).
        """
        batch_size, _, height, width = x.shape

        if mask is None:
            mask = torch.zeros(
                (batch_size, height, width), dtype=torch.bool, device=x.device
            )

        # Ensure mask is boolean
        mask = mask.to(torch.bool)

        # mask convention: True = padding, False = valid
        # Invert to get valid positions for cumsum
        not_mask = ~mask

        # Compute cumulative sum along height and width (on valid positions)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            # Normalize to [0, 1] range
            y_embed = (
                (y_embed + self.offset) / (y_embed[:, -1:, :] + self.eps) * self.scale
            )
            x_embed = (
                (x_embed + self.offset) / (x_embed[:, :, -1:] + self.eps) * self.scale
            )

        # Generate frequency bands
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        # Compute sine and cosine embeddings for x and y coordinates
        # Shape: (batch, height, width, num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        # Apply sine to even indices, cosine to odd indices
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=4,
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            dim=4,
        ).flatten(3)

        # Concatenate x and y embeddings
        # Shape: (batch, height, width, 2*num_pos_feats)
        pos = torch.cat((pos_y, pos_x), dim=3)

        # Permute to (batch, 2*num_pos_feats, height, width)
        pos = pos.permute(0, 3, 1, 2)

        # Zero out masked (padding) positions
        pos = pos.masked_fill(mask.unsqueeze(1), 0.0)

        return pos

    def __repr__(self) -> str:
        """String representation of the module."""
        return (
            f"{self.__class__.__name__}("
            f"num_pos_feats={self.num_pos_feats}, "
            f"temperature={self.temperature}, "
            f"normalize={self.normalize}, "
            f"scale={self.scale}, "
            f"offset={self.offset})"
        )
