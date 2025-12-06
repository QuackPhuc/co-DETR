"""
Tests for ResNet backbone.

This module tests the ResNet-50 backbone wrapper for multi-scale feature
extraction, verifying output shapes, channel dimensions, and frozen stages.
"""

import pytest
import torch
import torch.nn as nn

from codetr.models.backbone.resnet import ResNetBackbone


class TestResNetBackboneOutputShapes:
    """Tests for backbone output shapes and channel dimensions."""
    
    def test_default_output_shapes(self):
        """Test default output shapes for C3, C4, C5.
        
        ResNet-50 output strides:
        - C3 (layer2): stride 8, channels 512
        - C4 (layer3): stride 16, channels 1024
        - C5 (layer4): stride 32, channels 2048
        """
        backbone = ResNetBackbone(pretrained=False, frozen_stages=0)
        
        # Input: 800x800 image
        x = torch.randn(2, 3, 800, 800)
        features = backbone(x)
        
        assert len(features) == 3, f"Expected 3 feature maps, got {len(features)}"
        
        # C3: stride 8 -> 800/8 = 100, channels 512
        assert features[0].shape == (2, 512, 100, 100)
        
        # C4: stride 16 -> 800/16 = 50, channels 1024
        assert features[1].shape == (2, 1024, 50, 50)
        
        # C5: stride 32 -> 800/32 = 25, channels 2048
        assert features[2].shape == (2, 2048, 25, 25)
    
    def test_output_shapes_different_input_size(self):
        """Test output shapes scale correctly with input size."""
        backbone = ResNetBackbone(pretrained=False, frozen_stages=0)
        
        x = torch.randn(1, 3, 640, 480)  # 4:3 aspect ratio
        features = backbone(x)
        
        # C3: 640/8=80, 480/8=60
        assert features[0].shape == (1, 512, 80, 60)
        
        # C4: 640/16=40, 480/16=30
        assert features[1].shape == (1, 1024, 40, 30)
        
        # C5: 640/32=20, 480/32=15
        assert features[2].shape == (1, 2048, 20, 15)
    
    def test_out_channels_attribute(self):
        """Test out_channels attribute matches actual outputs."""
        backbone = ResNetBackbone(pretrained=False)
        
        assert backbone.out_channels == [512, 1024, 2048]
    
    def test_custom_out_indices(self):
        """Test custom out_indices to output different stages."""
        # Output only C4 and C5
        backbone = ResNetBackbone(pretrained=False, out_indices=(2, 3))
        
        x = torch.randn(1, 3, 256, 256)
        features = backbone(x)
        
        assert len(features) == 2
        # C4: 256/16=16
        assert features[0].shape == (1, 1024, 16, 16)
        # C5: 256/32=8
        assert features[1].shape == (1, 2048, 8, 8)


class TestResNetBackboneFrozenStages:
    """Tests for backbone stage freezing functionality."""
    
    def test_frozen_stages_0_all_trainable(self):
        """With frozen_stages=0, all parameters should be trainable."""
        backbone = ResNetBackbone(pretrained=False, frozen_stages=0)
        
        trainable_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in backbone.parameters())
        
        assert trainable_params == total_params
    
    def test_frozen_stages_1_stem_frozen(self):
        """With frozen_stages=1, stem (conv1, bn1) should be frozen."""
        backbone = ResNetBackbone(pretrained=False, frozen_stages=1)
        
        # conv1 should be frozen
        assert not backbone.conv1.weight.requires_grad
        
        # bn1 should be frozen
        for param in backbone.bn1.parameters():
            assert not param.requires_grad
        
        # layer1 should still be trainable
        for param in backbone.layer1.parameters():
            assert param.requires_grad
    
    def test_frozen_stages_2(self):
        """With frozen_stages=2, stem + layer1 should be frozen."""
        backbone = ResNetBackbone(pretrained=False, frozen_stages=2)
        
        # Stem frozen
        assert not backbone.conv1.weight.requires_grad
        
        # layer1 frozen
        for param in backbone.layer1.parameters():
            assert not param.requires_grad
        
        # layer2 trainable
        for param in backbone.layer2.parameters():
            assert param.requires_grad


class TestResNetBackboneNormEval:
    """Tests for BatchNorm eval mode behavior."""
    
    def test_norm_eval_true_bn_in_eval(self):
        """With norm_eval=True, BatchNorm layers stay in eval mode during training."""
        backbone = ResNetBackbone(pretrained=False, norm_eval=True)
        backbone.train()
        
        # All BatchNorm layers should be in eval mode
        for module in backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                assert not module.training, "BatchNorm should be in eval mode"
    
    def test_norm_eval_false_bn_in_train(self):
        """With norm_eval=False, BatchNorm layers train normally."""
        backbone = ResNetBackbone(pretrained=False, norm_eval=False, frozen_stages=0)
        backbone.train()
        
        # At least some BatchNorm layers should be in training mode
        bn_training = False
        for module in backbone.modules():
            if isinstance(module, nn.BatchNorm2d) and module.training:
                bn_training = True
                break
        
        assert bn_training, "Some BatchNorm should be in training mode"


class TestResNetBackboneGradientFlow:
    """Tests for gradient flow through backbone."""
    
    def test_gradient_flow_unfrozen(self):
        """Test gradients flow correctly with no frozen stages."""
        backbone = ResNetBackbone(pretrained=False, frozen_stages=0)
        
        x = torch.randn(1, 3, 224, 224, requires_grad=True)
        features = backbone(x)
        
        # Compute loss and backprop
        loss = sum(f.sum() for f in features)
        loss.backward()
        
        # Input should have gradients
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        
        # Trainable parameters should have gradients
        for name, param in backbone.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
    
    def test_no_gradient_frozen_params(self):
        """Test frozen parameters don't accumulate gradients."""
        backbone = ResNetBackbone(pretrained=False, frozen_stages=1)
        
        x = torch.randn(1, 3, 224, 224)
        features = backbone(x)
        
        loss = sum(f.sum() for f in features)
        loss.backward()
        
        # Frozen conv1 should have no gradient
        assert backbone.conv1.weight.grad is None


class TestResNetBackbonePretrainedLoading:
    """Tests for pretrained weight loading."""
    
    def test_pretrained_true_loads_weights(self):
        """Test pretrained=True loads ImageNet weights."""
        # This test may print a warning about downloading, which is expected
        backbone = ResNetBackbone(pretrained=True, frozen_stages=1)
        
        # Weights should be non-zero (initialized from pretrained)
        assert backbone.conv1.weight.abs().sum() > 0
        
        # Check that weights are not simple initialization patterns
        # Pretrained weights should have varied values
        std = backbone.conv1.weight.std()
        assert std > 0.01, "Weights appear to be uniform, not pretrained"
    
    def test_pretrained_false_random_init(self):
        """Test pretrained=False uses random initialization."""
        backbone = ResNetBackbone(pretrained=False)
        
        # Should still work, just with random weights
        x = torch.randn(1, 3, 224, 224)
        features = backbone(x)
        
        assert len(features) == 3
        assert not torch.isnan(features[0]).any()
