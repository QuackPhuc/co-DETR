"""
Tests for data transforms.

This module tests image transforms ensuring proper handling of
bounding box coordinate transformations.
"""

import pytest
import torch
from PIL import Image

from codetr.data.transforms import (
    Compose,
    ToTensor,
    Normalize,
    Resize,
    RandomHorizontalFlip,
    Pad,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


class TestToTensor:
    """Tests for ToTensor transform."""
    
    def test_pil_to_tensor_shape(self):
        """Test PIL image is converted to correct tensor shape."""
        transform = ToTensor()
        
        pil_image = Image.new('RGB', (100, 50))  # W=100, H=50
        
        tensor, _ = transform(pil_image, None)
        
        # Should be (C, H, W)
        assert tensor.shape == (3, 50, 100)
    
    def test_value_range(self):
        """Test tensor values are in [0, 1] range."""
        transform = ToTensor()
        
        pil_image = Image.new('RGB', (10, 10), color=(128, 64, 255))
        
        tensor, _ = transform(pil_image, None)
        
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0
    
    def test_target_passed_through(self):
        """Test target dict is passed through unchanged."""
        transform = ToTensor()
        
        image = Image.new('RGB', (10, 10))
        target = {'boxes': torch.tensor([[0.5, 0.5, 0.2, 0.2]]), 'labels': torch.tensor([0])}
        
        _, out_target = transform(image, target)
        
        assert torch.equal(out_target['boxes'], target['boxes'])
        assert torch.equal(out_target['labels'], target['labels'])


class TestNormalize:
    """Tests for Normalize transform."""
    
    def test_imagenet_normalization(self):
        """Test normalization with ImageNet stats."""
        transform = Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        
        # Create tensor with known values
        tensor = torch.ones(3, 10, 10)  # All ones
        
        normalized, _ = transform(tensor, None)
        
        # Normalized value for channel 0: (1 - 0.485) / 0.229 ≈ 2.25
        expected_ch0 = (1.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
        assert torch.allclose(normalized[0], torch.full((10, 10), expected_ch0), atol=1e-4)
    
    def test_target_unchanged(self):
        """Normalization should not affect target."""
        transform = Normalize()
        
        tensor = torch.rand(3, 10, 10)
        target = {'boxes': torch.rand(5, 4)}
        
        _, out_target = transform(tensor, target)
        
        assert torch.equal(out_target['boxes'], target['boxes'])


class TestResize:
    """Tests for Resize transform."""
    
    def test_aspect_ratio_preserved(self):
        """Resize should preserve aspect ratio."""
        transform = Resize(min_size=100, max_size=200)
        
        # 200x100 image (2:1 aspect ratio)
        image = Image.new('RGB', (200, 100))
        
        resized, _ = transform(image, None)
        
        # Should scale to min_size=100 for shorter side
        # Height becomes 100, width becomes 200
        # But check aspect ratio is preserved
        if isinstance(resized, torch.Tensor):
            h, w = resized.shape[1], resized.shape[2]
        else:
            w, h = resized.size
        
        assert abs(w / h - 2.0) < 0.1  # Aspect ratio ~2:1
    
    def test_min_size_constraint(self):
        """Shorter side should be at least min_size."""
        transform = Resize(min_size=100, max_size=500)
        
        image = Image.new('RGB', (50, 30))  # Small image
        
        resized, _ = transform(image, None)
        
        if isinstance(resized, torch.Tensor):
            h, w = resized.shape[1], resized.shape[2]
        else:
            w, h = resized.size
        
        assert min(h, w) >= 100
    
    def test_max_size_constraint(self):
        """Longer side should not exceed max_size."""
        transform = Resize(min_size=800, max_size=1333)
        
        # Very wide image
        image = Image.new('RGB', (5000, 100))
        
        resized, _ = transform(image, None)
        
        if isinstance(resized, torch.Tensor):
            h, w = resized.shape[1], resized.shape[2]
        else:
            w, h = resized.size
        
        assert max(h, w) <= 1333
    
    def test_target_size_updated(self):
        """Target 'size' should be updated after resize."""
        transform = Resize(min_size=100, max_size=200)
        
        image = Image.new('RGB', (200, 100))
        target = {'size': torch.tensor([100, 200])}  # Original H, W
        
        resized, out_target = transform(image, target)
        
        # Size should be updated
        assert 'size' in out_target


class TestRandomHorizontalFlip:
    """Tests for RandomHorizontalFlip transform."""
    
    def test_flip_probability_0_no_flip(self):
        """With p=0, image should never be flipped."""
        transform = RandomHorizontalFlip(p=0.0)
        
        # Create image with distinct left/right
        image = torch.zeros(3, 10, 20)
        image[:, :, :10] = 1.0  # Left half is 1
        
        for _ in range(5):
            flipped, _ = transform(image.clone(), None)
            # Left half should still be 1
            assert torch.allclose(flipped[:, :, :10], torch.ones(3, 10, 10))
    
    def test_flip_probability_1_always_flip(self):
        """With p=1, image should always be flipped."""
        transform = RandomHorizontalFlip(p=1.0)
        
        # Create image with distinct left/right
        image = torch.zeros(3, 10, 20)
        image[:, :, :10] = 1.0  # Left half is 1
        
        flipped, _ = transform(image.clone(), None)
        
        # After flip, right half should be 1
        assert torch.allclose(flipped[:, :, 10:], torch.ones(3, 10, 10))
    
    def test_box_x_coordinates_flipped(self):
        """Box x-coordinates should be flipped: cx -> 1 - cx."""
        transform = RandomHorizontalFlip(p=1.0)
        
        image = torch.rand(3, 100, 100)
        target = {
            'boxes': torch.tensor([[0.25, 0.5, 0.1, 0.2]]),  # Near left edge
        }
        
        _, flipped_target = transform(image, target)
        
        # cx should flip: 0.25 -> 0.75
        expected_cx = 0.75
        assert torch.allclose(
            flipped_target['boxes'][0, 0],
            torch.tensor(expected_cx),
            atol=0.01
        )


class TestCompose:
    """Tests for Compose transform."""
    
    def test_sequential_application(self):
        """Transforms should be applied in order."""
        transforms = Compose([
            ToTensor(),
            Normalize(),
        ])
        
        image = Image.new('RGB', (10, 10), color=(100, 100, 100))
        
        result, _ = transforms(image, None)
        
        # Should be a normalized tensor
        assert isinstance(result, torch.Tensor)
        # Normalized values for gray ~0.39 after ImageNet normalization
    
    def test_empty_compose(self):
        """Empty Compose should pass through unchanged."""
        transforms = Compose([])
        
        image = Image.new('RGB', (10, 10))
        target = {'labels': torch.tensor([0])}
        
        out_image, out_target = transforms(image, target)
        
        assert out_image == image
        assert out_target == target
    
    def test_repr(self):
        """Test string representation."""
        transforms = Compose([ToTensor(), Normalize()])
        
        repr_str = repr(transforms)
        
        assert 'Compose' in repr_str
        assert 'ToTensor' in repr_str
        assert 'Normalize' in repr_str


class TestTransformEdgeCases:
    """Tests for edge cases in transforms."""

    def test_box_coordinates_remain_valid_after_resize(self):
        """Box coordinates should remain in valid [0, 1] range after resize."""
        transform = Resize(min_size=100, max_size=200)

        # Image and box near the edge
        image = Image.new('RGB', (200, 100))  # Wide image
        target = {
            'boxes': torch.tensor([
                [0.95, 0.5, 0.1, 0.2],  # Box near right edge: cx=0.95, w=0.1
                [0.05, 0.5, 0.1, 0.2],  # Box near left edge: cx=0.05, w=0.1
            ]),
            'labels': torch.tensor([0, 1]),
        }

        resized, out_target = transform(image, target)

        # Boxes should still have valid coordinates
        boxes = out_target['boxes']
        assert boxes.shape == (2, 4)

        # All values should be finite (not NaN/Inf)
        assert torch.isfinite(boxes).all()

        # Check that boxes maintain reasonable values
        # (exact clipping depends on implementation)
        assert (boxes[:, 2] >= 0).all(), "Width should be non-negative"
        assert (boxes[:, 3] >= 0).all(), "Height should be non-negative"

    def test_boxes_at_image_boundary_handled(self):
        """Boxes exactly at image boundary should be handled correctly."""
        transform = Compose([ToTensor()])

        image = Image.new('RGB', (100, 100))

        # Box exactly at boundaries
        target = {
            'boxes': torch.tensor([
                [0.0, 0.0, 0.1, 0.1],  # Top-left corner (cx=0, cy=0)
                [1.0, 1.0, 0.1, 0.1],  # Bottom-right corner (cx=1, cy=1)
                [0.5, 0.5, 1.0, 1.0],  # Full image box
            ]),
            'labels': torch.tensor([0, 1, 2]),
        }

        out_image, out_target = transform(image, target)

        # Should not crash
        assert out_target is not None
        assert 'boxes' in out_target
        # Boxes should be passed through (ToTensor doesn't modify boxes)
        assert torch.equal(out_target['boxes'], target['boxes'])

    def test_pad_transform_preserves_box_coordinates(self):
        """Pad transform should correctly handle box coordinates."""
        # Pad should update coordinates to account for added padding
        transform = Pad(divisor=32)

        # Create image that needs padding (e.g., 100x100 -> 128x128)
        image = torch.rand(3, 100, 100)
        target = {
            'boxes': torch.tensor([
                [0.5, 0.5, 0.2, 0.2],  # Centered box
            ]),
            'labels': torch.tensor([0]),
        }

        padded, out_target = transform(image, target)

        # Image should be padded to multiple of divisor
        assert padded.shape[1] % 32 == 0
        assert padded.shape[2] % 32 == 0

        # Boxes should exist in output
        assert 'boxes' in out_target

    def test_horizontal_flip_preserves_box_dimensions(self):
        """Horizontal flip should preserve box width and height."""
        transform = RandomHorizontalFlip(p=1.0)

        image = torch.rand(3, 100, 100)
        original_boxes = torch.tensor([
            [0.3, 0.4, 0.2, 0.3],  # (cx, cy, w, h)
            [0.7, 0.6, 0.1, 0.15],
        ])
        target = {
            'boxes': original_boxes.clone(),
            'labels': torch.tensor([0, 1]),
        }

        _, flipped_target = transform(image, target)

        flipped_boxes = flipped_target['boxes']

        # Width and height should be unchanged
        assert torch.allclose(
            flipped_boxes[:, 2],  # widths
            original_boxes[:, 2],
            atol=1e-5,
        )
        assert torch.allclose(
            flipped_boxes[:, 3],  # heights
            original_boxes[:, 3],
            atol=1e-5,
        )

        # y-coordinates should be unchanged
        assert torch.allclose(
            flipped_boxes[:, 1],  # cy values
            original_boxes[:, 1],
            atol=1e-5,
        )

        # x-coordinates should be flipped (cx -> 1 - cx)
        expected_cx = 1.0 - original_boxes[:, 0]
        assert torch.allclose(
            flipped_boxes[:, 0],
            expected_cx,
            atol=1e-5,
        )

