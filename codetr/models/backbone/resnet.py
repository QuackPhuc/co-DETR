"""ResNet backbone wrapper for multi-scale feature extraction.

This module provides a wrapper around torchvision's ResNet-50 to extract
multi-scale features from different stages (C3, C4, C5) for object detection.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class ResNetBackbone(nn.Module):
    """ResNet-50 backbone for multi-scale feature extraction.

    This class wraps torchvision's ResNet-50 and extracts features from
    stages C3, C4, and C5, which correspond to output strides of 8, 16, and 32.

    Args:
        pretrained: Whether to load pretrained ImageNet weights.
        frozen_stages: Number of stages to freeze (0-4). Freezing includes
            both convolutional layers and batch normalization layers.
        norm_eval: Whether to set normalization layers to eval mode, even
            during training. This prevents updating of running stats.
        out_indices: Indices of output stages. Default is (1, 2, 3) for
            stages C3, C4, C5.

    Attributes:
        out_channels: List of output channel dimensions for each stage.

    Examples:
        >>> backbone = ResNetBackbone(pretrained=True, frozen_stages=1)
        >>> x = torch.randn(2, 3, 800, 800)
        >>> features = backbone(x)
        >>> for i, feat in enumerate(features):
        ...     print(f"C{i+3}: {feat.shape}")
        C3: torch.Size([2, 512, 100, 100])
        C4: torch.Size([2, 1024, 50, 50])
        C5: torch.Size([2, 2048, 25, 25])
    """

    def __init__(
        self,
        pretrained: bool = True,
        frozen_stages: int = 1,
        norm_eval: bool = True,
        out_indices: Tuple[int, ...] = (1, 2, 3),
    ) -> None:
        super().__init__()

        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        self.out_indices = out_indices

        # Load pretrained ResNet-50
        if pretrained:
            weights = ResNet50_Weights.IMAGENET1K_V1
            resnet = resnet50(weights=weights)
        else:
            resnet = resnet50(weights=None)

        # Extract ResNet stages
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        # ResNet stages (C2, C3, C4, C5)
        self.layer1 = resnet.layer1  # C2: stride 4, channels 256
        self.layer2 = resnet.layer2  # C3: stride 8, channels 512
        self.layer3 = resnet.layer3  # C4: stride 16, channels 1024
        self.layer4 = resnet.layer4  # C5: stride 32, channels 2048

        # Output channels for each stage
        self.out_channels = [512, 1024, 2048]  # C3, C4, C5

        # Freeze stages if needed
        self._freeze_stages()

    def _freeze_stages(self) -> None:
        """Freeze specified number of stages.

        frozen_stages = 0: nothing frozen
        frozen_stages = 1: freeze stem (conv1, bn1)
        frozen_stages = 2: freeze stem + layer1 (C2)
        frozen_stages = 3: freeze stem + layer1 + layer2 (C3)
        frozen_stages = 4: freeze stem + layer1 + layer2 + layer3 (C4)
        """
        if self.frozen_stages >= 1:
            # Freeze stem
            self.bn1.eval()
            for param in [self.conv1.parameters(), self.bn1.parameters()]:
                for p in param:
                    p.requires_grad = False

        # Freeze subsequent stages
        for i in range(1, self.frozen_stages):
            layer = getattr(self, f"layer{i}")
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

    def train(self, mode: bool = True) -> "ResNetBackbone":
        """Set training mode while respecting frozen stages and norm_eval.

        Args:
            mode: Whether to set training mode (True) or evaluation mode (False).

        Returns:
            Self for method chaining.
        """
        super().train(mode)
        self._freeze_stages()

        if mode and self.norm_eval:
            # Set all BatchNorm layers to eval mode
            for module in self.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()

        return self

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass to extract multi-scale features.

        Args:
            x: Input tensor of shape (batch_size, 3, height, width).

        Returns:
            List of feature tensors from stages specified in out_indices.
            Default returns [C3, C4, C5] features.
        """
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # ResNet stages
        outputs = []

        x = self.layer1(x)  # C2
        if 0 in self.out_indices:
            outputs.append(x)

        x = self.layer2(x)  # C3
        if 1 in self.out_indices:
            outputs.append(x)

        x = self.layer3(x)  # C4
        if 2 in self.out_indices:
            outputs.append(x)

        x = self.layer4(x)  # C5
        if 3 in self.out_indices:
            outputs.append(x)

        return outputs
