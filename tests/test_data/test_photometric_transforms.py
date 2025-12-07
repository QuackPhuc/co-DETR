"""
Tests for photometric augmentation transforms.

This module tests color and intensity transforms ensuring they
do not modify bounding box coordinates.
"""

import pytest
import torch
from PIL import Image

from codetr.data.transforms import (
    ColorJitter,
    RandomGrayscale,
    RandomChannelShuffle,
    GaussianBlur,
    RandomSharpness,
    RandomEqualize,
    RandomPosterize,
    RandomSolarize,
    RandomAutocontrast,
)


class TestColorJitter:
    """Tests for ColorJitter transform."""
    
    def test_boxes_unchanged(self):
        """Color jitter should not modify boxes."""
        transform = ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, p=1.0)
        
        original_boxes = torch.tensor([
            [0.3, 0.4, 0.2, 0.3],
            [0.7, 0.6, 0.15, 0.25],
        ])
        target = {'boxes': original_boxes.clone(), 'labels': torch.tensor([0, 1])}
        
        _, out = transform(torch.rand(3, 100, 100), target)
        
        assert torch.equal(out['boxes'], original_boxes)
    
    def test_image_values_changed(self):
        """Image pixel values should be modified."""
        transform = ColorJitter(brightness=0.5, contrast=0.5, p=1.0)
        
        image = torch.ones(3, 50, 50) * 0.5
        
        # Run multiple times to ensure at least one changes
        changed = False
        for _ in range(10):
            jittered, _ = transform(image.clone(), None)
            if not torch.allclose(jittered, image, atol=0.01):
                changed = True
                break
        
        assert changed, "ColorJitter should modify image values"
    
    def test_output_range_valid(self):
        """Output values should generally be in reasonable range."""
        transform = ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05, p=1.0)
        
        image = torch.rand(3, 50, 50)
        jittered, _ = transform(image, None)
        
        # Values should be finite
        assert torch.isfinite(jittered).all()
    
    def test_works_with_pil(self):
        """Should work with PIL images."""
        transform = ColorJitter(brightness=0.2, p=1.0)
        
        image = Image.new('RGB', (50, 50), color=(128, 128, 128))
        target = {'boxes': torch.tensor([[0.5, 0.5, 0.2, 0.2]])}
        
        result, out_target = transform(image, target)
        
        assert torch.equal(out_target['boxes'], target['boxes'])


class TestRandomGrayscale:
    """Tests for RandomGrayscale transform."""
    
    def test_boxes_unchanged(self):
        """Grayscale should not modify boxes."""
        transform = RandomGrayscale(p=1.0)
        
        original_boxes = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
        target = {'boxes': original_boxes.clone()}
        
        _, out = transform(torch.rand(3, 50, 50), target)
        
        assert torch.equal(out['boxes'], original_boxes)
    
    def test_output_channels_preserved(self):
        """Output should have same number of channels as input."""
        transform = RandomGrayscale(p=1.0)
        
        image = torch.rand(3, 50, 50)
        gray, _ = transform(image, None)
        
        assert gray.shape[0] == 3  # Still 3 channels
    
    def test_grayscale_has_equal_channels(self):
        """Grayscale image should have equal R, G, B values."""
        transform = RandomGrayscale(p=1.0)
        
        image = torch.rand(3, 50, 50)
        gray, _ = transform(image, None)
        
        # R, G, B should be very close
        assert torch.allclose(gray[0], gray[1], atol=1e-4)
        assert torch.allclose(gray[1], gray[2], atol=1e-4)
    
    def test_probability_respected(self):
        """With p=0, should never convert to grayscale."""
        transform = RandomGrayscale(p=0.0)
        
        image = torch.rand(3, 50, 50)
        # Make channels distinct
        image[0] = 0.2
        image[1] = 0.5
        image[2] = 0.8
        
        for _ in range(5):
            result, _ = transform(image.clone(), None)
            # Channels should still be distinct
            assert not torch.allclose(result[0], result[1], atol=0.1)


class TestRandomChannelShuffle:
    """Tests for RandomChannelShuffle transform."""
    
    def test_boxes_unchanged(self):
        """Channel shuffle should not modify boxes."""
        transform = RandomChannelShuffle(p=1.0)
        
        original_boxes = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
        target = {'boxes': original_boxes.clone()}
        
        _, out = transform(torch.rand(3, 50, 50), target)
        
        assert torch.equal(out['boxes'], original_boxes)
    
    def test_channel_values_shuffled(self):
        """Channel values should be permuted."""
        transform = RandomChannelShuffle(p=1.0)
        
        image = torch.zeros(3, 10, 10)
        image[0] = 0.1
        image[1] = 0.5
        image[2] = 0.9
        
        shuffled, _ = transform(image.clone(), None)
        
        # The set of channel mean values should be same
        original_means = {0.1, 0.5, 0.9}
        shuffled_means = {
            round(shuffled[0].mean().item(), 1),
            round(shuffled[1].mean().item(), 1),
            round(shuffled[2].mean().item(), 1),
        }
        assert original_means == shuffled_means
    
    def test_works_with_pil(self):
        """Should work with PIL images."""
        transform = RandomChannelShuffle(p=1.0)
        
        image = Image.new('RGB', (50, 50), color=(10, 128, 255))
        result, _ = transform(image, None)
        
        # Should return valid image
        assert result is not None


class TestGaussianBlur:
    """Tests for GaussianBlur transform."""
    
    def test_boxes_unchanged(self):
        """Blur should not modify boxes."""
        transform = GaussianBlur(kernel_size_range=(3, 5), p=1.0)
        
        original_boxes = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
        target = {'boxes': original_boxes.clone()}
        
        _, out = transform(torch.rand(3, 50, 50), target)
        
        assert torch.equal(out['boxes'], original_boxes)
    
    def test_blur_reduces_variance(self):
        """Blur should reduce high-frequency content (variance)."""
        transform = GaussianBlur(kernel_size_range=(5, 7), sigma_range=(1.5, 2.0), p=1.0)
        
        # Create noisy image
        torch.manual_seed(42)
        image = torch.rand(3, 50, 50)
        
        blurred, _ = transform(image.clone(), None)
        
        # Variance of difference between adjacent pixels should decrease
        original_var = (image[:, 1:, :] - image[:, :-1, :]).var()
        blurred_var = (blurred[:, 1:, :] - blurred[:, :-1, :]).var()
        
        assert blurred_var < original_var
    
    def test_output_shape_unchanged(self):
        """Blur should preserve image shape."""
        transform = GaussianBlur(p=1.0)
        
        image = torch.rand(3, 100, 80)
        blurred, _ = transform(image, None)
        
        assert blurred.shape == image.shape


class TestRandomSharpness:
    """Tests for RandomSharpness transform."""
    
    def test_boxes_unchanged(self):
        """Sharpness should not modify boxes."""
        transform = RandomSharpness(sharpness_range=(0.5, 2.0), p=1.0)
        
        original_boxes = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
        target = {'boxes': original_boxes.clone()}
        
        _, out = transform(torch.rand(3, 50, 50), target)
        
        assert torch.equal(out['boxes'], original_boxes)
    
    def test_output_shape_unchanged(self):
        """Sharpness should preserve image shape."""
        transform = RandomSharpness(p=1.0)
        
        image = torch.rand(3, 100, 80)
        result, _ = transform(image, None)
        
        assert result.shape == image.shape


class TestRandomEqualize:
    """Tests for RandomEqualize transform."""
    
    def test_boxes_unchanged(self):
        """Equalize should not modify boxes."""
        transform = RandomEqualize(p=1.0)
        
        original_boxes = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
        target = {'boxes': original_boxes.clone()}
        
        _, out = transform(torch.rand(3, 50, 50), target)
        
        assert torch.equal(out['boxes'], original_boxes)
    
    def test_output_range_normalized(self):
        """Output should be in [0, 1] for float tensors."""
        transform = RandomEqualize(p=1.0)
        
        image = torch.rand(3, 50, 50)
        result, _ = transform(image, None)
        
        assert result.min() >= 0.0
        assert result.max() <= 1.0


class TestRandomPosterize:
    """Tests for RandomPosterize transform."""
    
    def test_boxes_unchanged(self):
        """Posterize should not modify boxes."""
        transform = RandomPosterize(bits_range=(4, 6), p=1.0)
        
        original_boxes = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
        target = {'boxes': original_boxes.clone()}
        
        _, out = transform(torch.rand(3, 50, 50), target)
        
        assert torch.equal(out['boxes'], original_boxes)
    
    def test_reduces_unique_values(self):
        """Posterize should reduce number of unique intensity values."""
        transform = RandomPosterize(bits_range=(2, 2), p=1.0)
        
        image = torch.rand(3, 50, 50)
        original_unique = len(torch.unique(image))
        
        result, _ = transform(image, None)
        result_unique = len(torch.unique(result))
        
        # With 2 bits, should have at most 4 values per channel
        assert result_unique < original_unique


class TestRandomSolarize:
    """Tests for RandomSolarize transform."""
    
    def test_boxes_unchanged(self):
        """Solarize should not modify boxes."""
        transform = RandomSolarize(threshold_range=(0.3, 0.5), p=1.0)
        
        original_boxes = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
        target = {'boxes': original_boxes.clone()}
        
        _, out = transform(torch.rand(3, 50, 50), target)
        
        assert torch.equal(out['boxes'], original_boxes)
    
    def test_inverts_above_threshold(self):
        """Solarize should invert pixels above threshold."""
        # Fixed low threshold to ensure inversion
        transform = RandomSolarize(threshold_range=(0.3, 0.4), p=1.0)
        
        # Image with high values that should be inverted
        image = torch.ones(3, 10, 10) * 0.9  # Above threshold
        
        result, _ = transform(image.clone(), None)
        
        # High values should become low after solarization
        # (inverted values: ~1 - 0.9 = 0.1 in uint8 space)
        assert result.mean() < 0.5


class TestRandomAutocontrast:
    """Tests for RandomAutocontrast transform."""
    
    def test_boxes_unchanged(self):
        """Autocontrast should not modify boxes."""
        transform = RandomAutocontrast(p=1.0)
        
        original_boxes = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
        target = {'boxes': original_boxes.clone()}
        
        _, out = transform(torch.rand(3, 50, 50), target)
        
        assert torch.equal(out['boxes'], original_boxes)
    
    def test_stretches_contrast(self):
        """Autocontrast should stretch the dynamic range."""
        transform = RandomAutocontrast(p=1.0)
        
        # Low contrast image
        image = torch.rand(3, 50, 50) * 0.3 + 0.35  # Range [0.35, 0.65]
        
        result, _ = transform(image, None)
        
        # After autocontrast, range should be larger
        assert result.max() - result.min() > image.max() - image.min()


class TestPhotometricEdgeCases:
    """Edge case tests for photometric transforms."""
    
    def test_all_transforms_handle_empty_boxes(self):
        """All photometric transforms should handle empty boxes."""
        transforms = [
            ColorJitter(p=1.0),
            RandomGrayscale(p=1.0),
            RandomChannelShuffle(p=1.0),
            GaussianBlur(p=1.0),
            RandomSharpness(p=1.0),
            RandomEqualize(p=1.0),
            RandomPosterize(p=1.0),
            RandomSolarize(p=1.0),
            RandomAutocontrast(p=1.0),
        ]
        
        for t in transforms:
            image = torch.rand(3, 50, 50)
            target = {'boxes': torch.zeros(0, 4), 'labels': torch.zeros(0)}
            
            _, out = t(image, target)
            
            assert out['boxes'].shape == (0, 4), f"Failed for {type(t).__name__}"
    
    def test_all_transforms_preserve_labels(self):
        """All photometric transforms should preserve labels."""
        transforms = [
            ColorJitter(p=1.0),
            GaussianBlur(p=1.0),
            RandomEqualize(p=1.0),
        ]
        
        for t in transforms:
            labels = torch.tensor([1, 2, 3, 4, 5])
            target = {
                'boxes': torch.rand(5, 4),
                'labels': labels.clone(),
            }
            
            _, out = t(torch.rand(3, 50, 50), target)
            
            assert torch.equal(out['labels'], labels), f"Failed for {type(t).__name__}"
    
    def test_uniform_image_handling(self):
        """Transforms should handle uniform (single color) images."""
        transforms = [
            ColorJitter(brightness=0.2, p=1.0),
            GaussianBlur(p=1.0),
            RandomEqualize(p=1.0),
        ]
        
        for t in transforms:
            # Uniform image
            image = torch.ones(3, 50, 50) * 0.5
            
            result, _ = t(image, None)
            
            # Should not produce NaN or Inf
            assert torch.isfinite(result).all(), f"Failed for {type(t).__name__}"
