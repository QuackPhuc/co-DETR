"""Channel Mapper for creating feature pyramids with uniform channels.

This module implements a simple channel mapper that projects multi-scale
features to a uniform channel dimension and optionally adds extra levels.
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelMapper(nn.Module):
    """Channel mapper for feature pyramid construction.

    This module takes multi-scale features with different channel dimensions
    and projects them to a uniform channel dimension using 1x1 convolutions.
    It can also add extra pyramid levels through downsampling.

    Args:
        in_channels: List of input channel dimensions for each level.
            For ResNet-50: [512, 1024, 2048] for C3, C4, C5.
        out_channels: Output channel dimension for all levels. Default: 256.
        num_extra_levels: Number of extra levels to add via downsampling.
            Default: 1 (adds one level by applying stride-2 conv on last level).
        norm_type: Type of normalization. Options: 'GN' (GroupNorm), 'BN'
            (BatchNorm), None. Default: 'GN'.
        num_groups: Number of groups for GroupNorm. Default: 32.
        activation: Whether to apply activation after projection. Default: False.

    Attributes:
        lateral_convs: List of 1x1 convolutions for channel projection.
        extra_convs: List of 3x3 stride-2 convolutions for extra levels.

    Examples:
        >>> mapper = ChannelMapper(
        ...     in_channels=[512, 1024, 2048],
        ...     out_channels=256,
        ...     num_extra_levels=1
        ... )
        >>> feats = [
        ...     torch.randn(2, 512, 100, 100),   # C3
        ...     torch.randn(2, 1024, 50, 50),    # C4
        ...     torch.randn(2, 2048, 25, 25),    # C5
        ... ]
        >>> outputs = mapper(feats)
        >>> for i, out in enumerate(outputs):
        ...     print(f"P{i+3}: {out.shape}")
        P3: torch.Size([2, 256, 100, 100])
        P4: torch.Size([2, 256, 50, 50])
        P5: torch.Size([2, 256, 25, 25])
        P6: torch.Size([2, 256, 13, 13])
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int = 256,
        num_extra_levels: int = 1,
        norm_type: Optional[str] = "GN",
        num_groups: int = 32,
        activation: bool = False,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_extra_levels = num_extra_levels
        self.norm_type = norm_type
        self.num_groups = num_groups
        self.activation = activation

        # Build lateral (1x1) convolutions for channel projection
        self.lateral_convs = nn.ModuleList()
        for in_ch in in_channels:
            conv = nn.Conv2d(in_ch, out_channels, kernel_size=1)
            self.lateral_convs.append(conv)

        # Build normalization layers if needed
        if norm_type is not None:
            self.lateral_norms = nn.ModuleList()
            for _ in in_channels:
                if norm_type == "GN":
                    norm = nn.GroupNorm(num_groups, out_channels)
                elif norm_type == "BN":
                    norm = nn.BatchNorm2d(out_channels)
                else:
                    raise ValueError(f"Unsupported norm_type: {norm_type}")
                self.lateral_norms.append(norm)
        else:
            self.lateral_norms = None

        # Build extra levels (stride-2 convolutions)
        self.extra_convs = nn.ModuleList()
        if num_extra_levels > 0:
            for i in range(num_extra_levels):
                conv = nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                )
                self.extra_convs.append(conv)

            # Add normalization for extra levels if needed
            if norm_type is not None:
                self.extra_norms = nn.ModuleList()
                for _ in range(num_extra_levels):
                    if norm_type == "GN":
                        norm = nn.GroupNorm(num_groups, out_channels)
                    elif norm_type == "BN":
                        norm = nn.BatchNorm2d(out_channels)
                    else:
                        raise ValueError(f"Unsupported norm_type: {norm_type}")
                    self.extra_norms.append(norm)
            else:
                self.extra_norms = None
        else:
            self.extra_norms = None

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize convolutional weights using Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """Forward pass to project features to uniform channels.

        Args:
            features: List of input feature tensors with different channels.
                Expected to be in order from high to low resolution.
                For ResNet-50: [C3, C4, C5] with shapes:
                    [(B, 512, H/8, W/8), (B, 1024, H/16, W/16), (B, 2048, H/32, W/32)]

        Returns:
            List of output feature tensors with uniform channels.
            If num_extra_levels=1: [P3, P4, P5, P6]
        """
        assert len(features) == len(
            self.in_channels
        ), f"Expected {len(self.in_channels)} features, got {len(features)}"

        # Apply lateral convolutions
        outputs = []
        for i, feat in enumerate(features):
            out = self.lateral_convs[i](feat)

            # Apply normalization if exists
            if self.lateral_norms is not None:
                out = self.lateral_norms[i](out)

            # Apply activation if needed
            if self.activation:
                out = F.relu(out, inplace=True)

            outputs.append(out)

        # Add extra levels by downsampling the last level
        if self.num_extra_levels > 0:
            # Start from the last projected feature
            x = outputs[-1]
            for i in range(self.num_extra_levels):
                x = self.extra_convs[i](x)

                # Apply normalization if exists
                if self.extra_norms is not None:
                    x = self.extra_norms[i](x)

                # Apply activation if needed
                if self.activation:
                    x = F.relu(x, inplace=True)

                outputs.append(x)

        return outputs
