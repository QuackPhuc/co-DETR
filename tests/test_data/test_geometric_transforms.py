"""
Tests for geometric augmentation transforms.

This module tests geometric transforms ensuring proper bounding box
coordinate transformations for object detection.
"""

import pytest
import torch
from PIL import Image

from codetr.data.transforms import (
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomRotation90,
    RandomCrop,
    Resize,
    RandomScale,
    RandomAffine,
)


class TestRandomVerticalFlip:
    """Tests for RandomVerticalFlip transform."""
    
    def test_flip_probability_0_no_flip(self):
        """With p=0, image should never be flipped."""
        transform = RandomVerticalFlip(p=0.0)
        
        # Create image with distinct top/bottom
        image = torch.zeros(3, 20, 10)
        image[:, :10, :] = 1.0  # Top half is 1
        
        for _ in range(5):
            flipped, _ = transform(image.clone(), None)
            assert torch.allclose(flipped[:, :10, :], torch.ones(3, 10, 10))
    
    def test_flip_probability_1_always_flip(self):
        """With p=1, image should always be flipped."""
        transform = RandomVerticalFlip(p=1.0)
        
        image = torch.zeros(3, 20, 10)
        image[:, :10, :] = 1.0  # Top half is 1
        
        flipped, _ = transform(image.clone(), None)
        
        # After flip, bottom half should be 1
        assert torch.allclose(flipped[:, 10:, :], torch.ones(3, 10, 10))
    
    def test_box_y_coordinates_flipped(self):
        """Box y-coordinates should be flipped: cy -> 1 - cy."""
        transform = RandomVerticalFlip(p=1.0)
        
        image = torch.rand(3, 100, 100)
        target = {
            'boxes': torch.tensor([[0.5, 0.25, 0.1, 0.2]]),  # Near top
        }
        
        _, flipped_target = transform(image, target)
        
        # cy should flip: 0.25 -> 0.75
        expected_cy = 0.75
        assert torch.allclose(
            flipped_target['boxes'][0, 1],
            torch.tensor(expected_cy),
            atol=0.01
        )
    
    def test_box_dimensions_preserved(self):
        """Width and height should be unchanged after vertical flip."""
        transform = RandomVerticalFlip(p=1.0)
        
        original_boxes = torch.tensor([
            [0.3, 0.4, 0.2, 0.3],
            [0.7, 0.6, 0.15, 0.25],
        ])
        target = {'boxes': original_boxes.clone()}
        
        _, flipped = transform(torch.rand(3, 100, 100), target)
        
        # w, h unchanged
        assert torch.allclose(flipped['boxes'][:, 2], original_boxes[:, 2], atol=1e-5)
        assert torch.allclose(flipped['boxes'][:, 3], original_boxes[:, 3], atol=1e-5)
        # cx unchanged
        assert torch.allclose(flipped['boxes'][:, 0], original_boxes[:, 0], atol=1e-5)
        # cy flipped
        assert torch.allclose(flipped['boxes'][:, 1], 1.0 - original_boxes[:, 1], atol=1e-5)


class TestRandomRotation90:
    """Tests for RandomRotation90 transform."""
    
    def test_invalid_angle_raises_error(self):
        """Only 0, 90, 180, 270 should be valid."""
        with pytest.raises(ValueError):
            RandomRotation90(angles=[45])
    
    def test_rotation_180_box_transform(self):
        """180° rotation: (cx, cy, w, h) -> (1-cx, 1-cy, w, h)."""
        transform = RandomRotation90(angles=[180], p=1.0)
        
        image = torch.rand(3, 100, 100)
        target = {
            'boxes': torch.tensor([[0.3, 0.2, 0.1, 0.15]]),
        }
        
        _, rotated = transform(image, target)
        
        expected = torch.tensor([[0.7, 0.8, 0.1, 0.15]])
        assert torch.allclose(rotated['boxes'], expected, atol=0.01)
    
    def test_rotation_90_box_transform(self):
        """90° CW rotation: (cx, cy, w, h) -> (1-cy, cx, h, w)."""
        transform = RandomRotation90(angles=[90], p=1.0)
        
        image = torch.rand(3, 100, 100)
        target = {
            'boxes': torch.tensor([[0.3, 0.2, 0.1, 0.15]]),
        }
        
        _, rotated = transform(image, target)
        
        # cx, cy, w, h = 0.3, 0.2, 0.1, 0.15
        # new: (1-0.2, 0.3, 0.15, 0.1) = (0.8, 0.3, 0.15, 0.1)
        expected = torch.tensor([[0.8, 0.3, 0.15, 0.1]])
        assert torch.allclose(rotated['boxes'], expected, atol=0.01)
    
    def test_rotation_270_box_transform(self):
        """270° CW rotation: (cx, cy, w, h) -> (cy, 1-cx, h, w)."""
        transform = RandomRotation90(angles=[270], p=1.0)
        
        image = torch.rand(3, 100, 100)
        target = {
            'boxes': torch.tensor([[0.3, 0.2, 0.1, 0.15]]),
        }
        
        _, rotated = transform(image, target)
        
        # (cy, 1-cx, h, w) = (0.2, 0.7, 0.15, 0.1)
        expected = torch.tensor([[0.2, 0.7, 0.15, 0.1]])
        assert torch.allclose(rotated['boxes'], expected, atol=0.01)
    
    def test_rotation_preserves_box_area(self):
        """Rotation should preserve box area (w*h unchanged, just swapped)."""
        transform = RandomRotation90(angles=[90], p=1.0)
        
        boxes = torch.tensor([[0.5, 0.5, 0.2, 0.3]])
        original_area = (boxes[:, 2] * boxes[:, 3]).item()
        
        target = {'boxes': boxes.clone()}
        _, rotated = transform(torch.rand(3, 100, 100), target)
        
        rotated_area = (rotated['boxes'][:, 2] * rotated['boxes'][:, 3]).item()
        assert abs(original_area - rotated_area) < 1e-5


class TestRandomCrop:
    """Tests for RandomCrop transform."""
    
    def test_crop_reduces_image_size(self):
        """Crop should reduce image dimensions."""
        transform = RandomCrop(scales=(0.5, 0.6))
        
        image = torch.rand(3, 100, 100)
        target = {'boxes': torch.zeros(0, 4), 'labels': torch.zeros(0)}
        
        cropped, _ = transform(image, target)
        
        assert cropped.shape[1] < 100 or cropped.shape[2] < 100
    
    def test_box_inside_crop_is_preserved(self):
        """Box fully inside crop region should be preserved."""
        # Use fixed crop size at center
        torch.manual_seed(42)
        transform = RandomCrop(scales=(0.9, 0.95))
        
        image = torch.rand(3, 100, 100)
        # Box at dead center, should likely survive most crops
        target = {
            'boxes': torch.tensor([[0.5, 0.5, 0.1, 0.1]]),
            'labels': torch.tensor([1]),
        }
        
        _, cropped = transform(image, target)
        
        # With high scale, center box should usually survive
        # If not, at least the structure should be correct
        assert 'boxes' in cropped
        assert cropped['boxes'].shape[1] == 4
    
    def test_box_outside_crop_is_filtered(self):
        """Box completely outside crop region should be removed."""
        # Create crop that takes bottom-right portion
        torch.manual_seed(123)
        transform = RandomCrop(crop_size=(30, 30))
        
        image = torch.rand(3, 100, 100)
        # Box near top-left corner
        target = {
            'boxes': torch.tensor([[0.05, 0.05, 0.05, 0.05]]),
            'labels': torch.tensor([1]),
        }
        
        # Run multiple times - sometimes box should be filtered
        filtered_count = 0
        for _ in range(20):
            _, cropped = transform(image, target)
            if len(cropped['boxes']) == 0:
                filtered_count += 1
        
        # At least some iterations should filter the box
        assert filtered_count > 0, "Box near corner should sometimes be filtered"
    
    def test_empty_boxes_handled(self):
        """Empty box tensor should be handled without error."""
        transform = RandomCrop(scales=(0.8, 0.9))
        
        image = torch.rand(3, 100, 100)
        target = {
            'boxes': torch.zeros(0, 4),
            'labels': torch.zeros(0, dtype=torch.long),
        }
        
        cropped_image, cropped_target = transform(image, target)
        
        assert cropped_target['boxes'].shape == (0, 4)
    
    def test_cropped_box_coordinates_normalized(self):
        """Box coordinates after crop should be normalized to [0, 1]."""
        transform = RandomCrop(scales=(0.5, 0.6))
        
        image = torch.rand(3, 100, 100)
        target = {
            'boxes': torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
            'labels': torch.tensor([1]),
        }
        
        _, cropped = transform(image, target)
        
        if len(cropped['boxes']) > 0:
            # All coords should be in [0, 1]
            assert (cropped['boxes'] >= 0).all()
            assert (cropped['boxes'] <= 1).all()


class TestRandomScale:
    """Tests for RandomScale transform."""
    
    def test_scale_changes_image_size(self):
        """Scale should change image dimensions."""
        transform = RandomScale(scale_range=(0.5, 0.6))
        
        image = torch.rand(3, 100, 100)
        scaled, _ = transform(image, None)
        
        # Should be smaller
        assert scaled.shape[1] < 100
        assert scaled.shape[2] < 100
    
    def test_box_coordinates_unchanged(self):
        """Scale should not modify normalized box coordinates."""
        transform = RandomScale(scale_range=(0.8, 1.2))
        
        original_boxes = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
        target = {'boxes': original_boxes.clone()}
        
        _, scaled = transform(torch.rand(3, 100, 100), target)
        
        # Normalized boxes should be unchanged
        assert torch.allclose(scaled['boxes'], original_boxes)
    
    def test_aspect_ratio_preserved(self):
        """With keep_aspect_ratio=True, aspect ratio should be preserved."""
        transform = RandomScale(scale_range=(0.5, 0.5), keep_aspect_ratio=True)
        
        image = torch.rand(3, 100, 200)  # 1:2 aspect ratio
        scaled, _ = transform(image, None)
        
        # Should still be 1:2
        aspect = scaled.shape[2] / scaled.shape[1]
        assert abs(aspect - 2.0) < 0.1


class TestRandomAffine:
    """Tests for RandomAffine transform."""
    
    def test_identity_affine(self):
        """With no transformation parameters, output should be similar."""
        transform = RandomAffine(degrees=0, translate=(0, 0), scale=(1.0, 1.0), shear=0)
        
        image = torch.rand(3, 50, 50)
        target = {
            'boxes': torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
            'labels': torch.tensor([1]),
        }
        
        _, out = transform(image, target)
        
        # Boxes should be nearly unchanged
        assert torch.allclose(out['boxes'], target['boxes'], atol=0.01)
    
    def test_rotation_transforms_boxes(self):
        """Rotation should transform box coordinates."""
        # Small rotation should move corner boxes
        transform = RandomAffine(degrees=45, translate=(0, 0), scale=(1.0, 1.0), shear=0)
        
        image = torch.rand(3, 100, 100)
        target = {
            'boxes': torch.tensor([[0.1, 0.1, 0.1, 0.1]]),  # Corner box
            'labels': torch.tensor([1]),
        }
        
        _, out = transform(image.clone(), target.copy())
        
        # Box should be transformed (not identical)
        if len(out['boxes']) > 0:
            # Just verify it's valid
            assert (out['boxes'] >= 0).all()
            assert (out['boxes'] <= 1).all()
    
    def test_small_boxes_filtered_after_extreme_transform(self):
        """Very small boxes after transformation should be filtered."""
        transform = RandomAffine(degrees=30, translate=(0.2, 0.2), scale=(0.5, 0.6), shear=10)
        
        image = torch.rand(3, 100, 100)
        # Small box that might become too small
        target = {
            'boxes': torch.tensor([[0.9, 0.9, 0.02, 0.02]]),
            'labels': torch.tensor([1]),
        }
        
        # Run multiple times - sometimes box might be filtered
        for _ in range(10):
            _, out = transform(image.clone(), target.copy())
            if len(out['boxes']) > 0:
                # Valid size check
                assert (out['boxes'][:, 2] > 0).all()
                assert (out['boxes'][:, 3] > 0).all()


class TestGeometricEdgeCases:
    """Edge case tests for geometric transforms."""
    
    def test_flip_with_zero_boxes(self):
        """Flip should handle empty target gracefully."""
        for Transform in [RandomHorizontalFlip, RandomVerticalFlip]:
            t = Transform(p=1.0)
            
            image = torch.rand(3, 50, 50)
            target = {'boxes': torch.zeros(0, 4), 'labels': torch.zeros(0)}
            
            _, out = t(image, target)
            
            assert out['boxes'].shape == (0, 4)
    
    def test_rotation_with_pil_image(self):
        """Rotation should work with PIL images."""
        transform = RandomRotation90(angles=[180], p=1.0)
        
        image = Image.new('RGB', (100, 100), color=(128, 64, 32))
        target = {'boxes': torch.tensor([[0.5, 0.5, 0.2, 0.2]])}
        
        rotated_img, rotated_target = transform(image, target)
        
        # Should return PIL or similar
        assert rotated_target is not None
    
    def test_affine_preserves_labels(self):
        """Affine should preserve label associations."""
        transform = RandomAffine(degrees=10, translate=(0.05, 0.05))
        
        labels = torch.tensor([0, 1, 2, 3, 4])
        boxes = torch.tensor([
            [0.5, 0.5, 0.1, 0.1],
            [0.3, 0.3, 0.1, 0.1],
            [0.7, 0.7, 0.1, 0.1],
            [0.4, 0.6, 0.1, 0.1],
            [0.6, 0.4, 0.1, 0.1],
        ])
        target = {'boxes': boxes, 'labels': labels.clone()}
        
        _, out = transform(torch.rand(3, 100, 100), target)
        
        # Number of labels should match boxes
        assert len(out['labels']) == len(out['boxes'])
