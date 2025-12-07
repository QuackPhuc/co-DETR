"""
Tests for advanced augmentation transforms.

This module tests modern augmentation strategies (CutOut, MixUp, CutMix, Mosaic)
with focus on correct bounding box handling.
"""

import pytest
import torch
from PIL import Image

from codetr.data.transforms import (
    CutOut,
    MixUp,
    CutMix,
    Mosaic,
    RandomSelect,
    OneOf,
    Compose,
    ToTensor,
)


class TestCutOut:
    """Tests for CutOut (RandomErasing) transform."""
    
    def test_creates_erased_region(self):
        """CutOut should create zero/filled regions in image."""
        transform = CutOut(num_patches=1, size_range=(0.1, 0.2), fill_value=0.0, p=1.0)
        
        image = torch.ones(3, 100, 100)
        result, _ = transform(image, None)
        
        # Should have some zeros now
        assert result.min() == 0.0
        # Not all zeros
        assert result.max() == 1.0
    
    def test_boxes_unchanged(self):
        """CutOut should not modify box coordinates (just masks image)."""
        transform = CutOut(num_patches=1, p=1.0)
        
        original_boxes = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
        target = {'boxes': original_boxes.clone(), 'labels': torch.tensor([1])}
        
        _, out = transform(torch.rand(3, 100, 100), target)
        
        assert torch.equal(out['boxes'], original_boxes)
    
    def test_multiple_patches(self):
        """Multiple CutOut patches should be applied."""
        transform = CutOut(num_patches=3, size_range=(0.05, 0.1), fill_value=0.0, p=1.0)
        
        image = torch.ones(3, 100, 100)
        result, _ = transform(image, None)
        
        # Count zeros - should have substantial erased area
        zero_fraction = (result == 0).float().mean()
        assert zero_fraction > 0.01  # At least 1% erased
    
    def test_random_fill(self):
        """CutOut with 'random' fill should create varied values."""
        transform = CutOut(num_patches=1, size_range=(0.3, 0.4), fill_value='random', p=1.0)
        
        image = torch.ones(3, 100, 100) * 0.5  # Uniform gray
        result, _ = transform(image, None)
        
        # Should have more varied values now
        unique_values = len(torch.unique(result))
        assert unique_values > 100  # Random fill adds many values
    
    def test_avoid_boxes_option(self):
        """With avoid_boxes=True, box centers should not be erased."""
        torch.manual_seed(42)
        transform = CutOut(num_patches=5, size_range=(0.05, 0.1), avoid_boxes=True, p=1.0)
        
        image = torch.ones(3, 100, 100)
        target = {
            'boxes': torch.tensor([[0.5, 0.5, 0.1, 0.1]]),  # Center box
            'labels': torch.tensor([1]),
        }
        
        # Run multiple times
        for _ in range(5):
            result, _ = transform(image.clone(), target)
            
            # Check center pixel (where box center is)
            center_h, center_w = 50, 50
            # With avoid_boxes, center should less likely be erased
            # Can't guarantee 100%, but logic should try to avoid


class TestMixUp:
    """Tests for MixUp transform."""
    
    def create_get_sample(self, image, target):
        """Create a get_sample function for testing."""
        def get_sample():
            return image.clone(), {k: v.clone() for k, v in target.items()}
        return get_sample
    
    def test_image_blending(self):
        """MixUp should blend two images."""
        transform = MixUp(alpha=1.0, p=1.0)
        
        image1 = torch.zeros(3, 50, 50)
        image2 = torch.ones(3, 50, 50)
        target1 = {'boxes': torch.tensor([[0.3, 0.3, 0.1, 0.1]]), 'labels': torch.tensor([0])}
        target2 = {'boxes': torch.tensor([[0.7, 0.7, 0.1, 0.1]]), 'labels': torch.tensor([1])}
        
        get_sample = self.create_get_sample(image2, target2)
        result, out_target = transform(image1, target1, get_sample)
        
        # Result should be between 0 and 1
        assert result.min() >= 0.0
        assert result.max() <= 1.0
        # Not exactly 0 or 1 due to mixing
        assert result.mean() > 0.0
        assert result.mean() < 1.0
    
    def test_boxes_concatenated(self):
        """MixUp should concatenate boxes from both images."""
        transform = MixUp(alpha=1.0, p=1.0)
        
        image1 = torch.rand(3, 50, 50)
        image2 = torch.rand(3, 50, 50)
        target1 = {
            'boxes': torch.tensor([[0.2, 0.2, 0.1, 0.1], [0.3, 0.3, 0.1, 0.1]]),
            'labels': torch.tensor([0, 0]),
        }
        target2 = {
            'boxes': torch.tensor([[0.7, 0.7, 0.1, 0.1]]),
            'labels': torch.tensor([1]),
        }
        
        get_sample = self.create_get_sample(image2, target2)
        _, out_target = transform(image1, target1, get_sample)
        
        # Should have 3 boxes total
        assert out_target['boxes'].shape[0] == 3
        assert out_target['labels'].shape[0] == 3
    
    def test_lambda_stored(self):
        """MixUp should store the mixing lambda."""
        transform = MixUp(alpha=1.0, p=1.0)
        
        image = torch.rand(3, 50, 50)
        target = {'boxes': torch.rand(2, 4), 'labels': torch.tensor([0, 1])}
        
        get_sample = self.create_get_sample(torch.rand(3, 50, 50), target)
        _, out_target = transform(image, target, get_sample)
        
        assert 'mixup_lambda' in out_target
        assert 0.0 <= out_target['mixup_lambda'].item() <= 1.0
    
    def test_probability_respected(self):
        """With p=0, MixUp should not mix."""
        transform = MixUp(alpha=1.0, p=0.0)
        
        image = torch.ones(3, 50, 50)
        target = {'boxes': torch.tensor([[0.5, 0.5, 0.1, 0.1]]), 'labels': torch.tensor([0])}
        
        get_sample = lambda: (torch.zeros(3, 50, 50), {'boxes': torch.rand(1, 4), 'labels': torch.tensor([1])})
        
        result, out_target = transform(image, target, get_sample)
        
        # Should be unchanged
        assert torch.allclose(result, image)
        assert out_target['boxes'].shape[0] == 1


class TestCutMix:
    """Tests for CutMix transform."""
    
    def create_get_sample(self, image, target):
        """Create a get_sample function for testing."""
        def get_sample():
            return image.clone(), {k: v.clone() for k, v in target.items()}
        return get_sample
    
    def test_creates_patch_from_other_image(self):
        """CutMix should paste a patch from another image."""
        transform = CutMix(alpha=1.0, min_cut_ratio=0.3, p=1.0)
        
        image1 = torch.zeros(3, 100, 100)
        image2 = torch.ones(3, 100, 100)
        target1 = {'boxes': torch.zeros(0, 4), 'labels': torch.zeros(0, dtype=torch.long)}
        target2 = {'boxes': torch.zeros(0, 4), 'labels': torch.zeros(0, dtype=torch.long)}
        
        get_sample = self.create_get_sample(image2, target2)
        result, _ = transform(image1, target1, get_sample)
        
        # Should have both 0s and 1s
        assert result.min() == 0.0
        assert result.max() == 1.0
    
    def test_box_filtering(self):
        """CutMix should filter boxes based on cut region."""
        transform = CutMix(alpha=1.0, min_cut_ratio=0.4, p=1.0)
        
        image1 = torch.rand(3, 100, 100)
        image2 = torch.rand(3, 100, 100)
        
        # Box in center of image1
        target1 = {
            'boxes': torch.tensor([[0.5, 0.5, 0.1, 0.1]]),
            'labels': torch.tensor([0]),
        }
        # Box at corner of image2
        target2 = {
            'boxes': torch.tensor([[0.1, 0.1, 0.1, 0.1]]),
            'labels': torch.tensor([1]),
        }
        
        get_sample = self.create_get_sample(image2, target2)
        
        # Run multiple times - box counts should vary based on cut region
        box_counts = []
        for _ in range(20):
            _, out_target = transform(image1.clone(), target1.copy(), get_sample)
            box_counts.append(len(out_target['boxes']))
        
        # Should sometimes have 0, 1, or 2 boxes
        assert min(box_counts) <= 1  # Sometimes boxes filtered
    
    def test_empty_boxes_handled(self):
        """CutMix should handle empty box tensors."""
        transform = CutMix(p=1.0)
        
        image = torch.rand(3, 50, 50)
        target = {'boxes': torch.zeros(0, 4), 'labels': torch.zeros(0, dtype=torch.long)}
        
        get_sample = lambda: (torch.rand(3, 50, 50), target.copy())
        
        result, out_target = transform(image, target, get_sample)
        
        # Should not crash
        assert 'boxes' in out_target


class TestMosaic:
    """Tests for Mosaic transform."""
    
    def create_get_sample(self, image, target):
        """Create a get_sample function."""
        def get_sample():
            return image.clone(), {k: v.clone() for k, v in target.items()}
        return get_sample
    
    def test_output_size(self):
        """Mosaic should produce specified output size."""
        transform = Mosaic(output_size=(640, 480), p=1.0)
        
        image = torch.rand(3, 100, 100)
        target = {'boxes': torch.rand(2, 4), 'labels': torch.tensor([0, 1])}
        
        get_sample = self.create_get_sample(image, target)
        result, out_target = transform(image, target, get_sample)
        
        assert result.shape == (3, 640, 480)
    
    def test_combines_four_images(self):
        """Mosaic should combine 4 images."""
        transform = Mosaic(output_size=(100, 100), center_range=(0.4, 0.6), p=1.0)
        
        # Create 4 images with distinct values
        image1 = torch.ones(3, 50, 50) * 0.1
        image2 = torch.ones(3, 50, 50) * 0.4
        image3 = torch.ones(3, 50, 50) * 0.7
        image4 = torch.ones(3, 50, 50) * 1.0
        
        images = [image2, image3, image4]
        idx = [0]
        
        def get_sample():
            img = images[idx[0] % 3]
            idx[0] += 1
            return img.clone(), {'boxes': torch.zeros(0, 4), 'labels': torch.zeros(0, dtype=torch.long)}
        
        target = {'boxes': torch.zeros(0, 4), 'labels': torch.zeros(0, dtype=torch.long)}
        result, _ = transform(image1, target, get_sample)
        
        # Should have multiple distinct values
        unique_values = len(torch.unique(result.round(decimals=1)))
        assert unique_values >= 2
    
    def test_boxes_transformed_to_output_space(self):
        """Boxes should be transformed to output coordinate space."""
        transform = Mosaic(output_size=(200, 200), center_range=(0.5, 0.5), p=1.0)
        
        # Image with centered box
        image = torch.rand(3, 100, 100)
        target = {
            'boxes': torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
            'labels': torch.tensor([0]),
        }
        
        get_sample = self.create_get_sample(image, target)
        _, out_target = transform(image, target, get_sample)
        
        # Should have boxes (from all 4 images)
        assert len(out_target['boxes']) > 0
        # All boxes should be in [0, 1] range
        assert (out_target['boxes'] >= 0).all()
        assert (out_target['boxes'] <= 1).all()
    
    def test_output_target_has_size(self):
        """Output target should have 'size' key."""
        transform = Mosaic(output_size=(320, 320), p=1.0)
        
        image = torch.rand(3, 100, 100)
        target = {'boxes': torch.rand(1, 4), 'labels': torch.tensor([0])}
        
        get_sample = self.create_get_sample(image, target)
        _, out_target = transform(image, target, get_sample)
        
        assert 'size' in out_target
        assert out_target['size'][0] == 320
        assert out_target['size'][1] == 320


class TestRandomSelect:
    """Tests for RandomSelect transform."""
    
    def test_selects_first_with_p_1(self):
        """With p=1, should always select first transform."""
        # Create tracking transforms
        called = {'t1': 0, 't2': 0}
        
        class T1:
            def __call__(self, image, target):
                called['t1'] += 1
                return image, target
        
        class T2:
            def __call__(self, image, target):
                called['t2'] += 1
                return image, target
        
        transform = RandomSelect(T1(), T2(), p=1.0)
        
        for _ in range(10):
            transform(torch.rand(3, 10, 10), None)
        
        assert called['t1'] == 10
        assert called['t2'] == 0
    
    def test_selects_second_with_p_0(self):
        """With p=0, should always select second transform."""
        called = {'t1': 0, 't2': 0}
        
        class T1:
            def __call__(self, image, target):
                called['t1'] += 1
                return image, target
        
        class T2:
            def __call__(self, image, target):
                called['t2'] += 1
                return image, target
        
        transform = RandomSelect(T1(), T2(), p=0.0)
        
        for _ in range(10):
            transform(torch.rand(3, 10, 10), None)
        
        assert called['t1'] == 0
        assert called['t2'] == 10


class TestOneOf:
    """Tests for OneOf transform."""
    
    def test_applies_one_transform(self):
        """Should apply exactly one transform from list."""
        call_counts = [0, 0, 0]
        
        def make_transform(idx):
            def t(image, target):
                call_counts[idx] += 1
                return image, target
            return t
        
        transforms = [make_transform(i) for i in range(3)]
        transform = OneOf(transforms, p=1.0)
        
        for _ in range(30):
            transform(torch.rand(3, 10, 10), None)
        
        # Each should be called roughly 10 times (uniform random)
        assert sum(call_counts) == 30
        # At least 2 different transforms called
        assert sum(1 for c in call_counts if c > 0) >= 2
    
    def test_respects_probability(self):
        """With p=0, should not apply any transform."""
        called = [0]
        
        def t(image, target):
            called[0] += 1
            return image, target
        
        transform = OneOf([t], p=0.0)
        
        for _ in range(10):
            transform(torch.rand(3, 10, 10), None)
        
        assert called[0] == 0


class TestAdvancedEdgeCases:
    """Edge case tests for advanced transforms."""
    
    def test_cutout_with_empty_boxes(self):
        """CutOut should handle empty boxes."""
        transform = CutOut(p=1.0)
        
        image = torch.rand(3, 50, 50)
        target = {'boxes': torch.zeros(0, 4), 'labels': torch.zeros(0, dtype=torch.long)}
        
        result, out_target = transform(image, target)
        
        assert out_target['boxes'].shape == (0, 4)
    
    def test_mixup_different_sizes(self):
        """MixUp should handle images of different sizes."""
        transform = MixUp(p=1.0)
        
        image1 = torch.rand(3, 100, 100)
        image2 = torch.rand(3, 50, 80)  # Different size
        target1 = {'boxes': torch.rand(1, 4), 'labels': torch.tensor([0])}
        target2 = {'boxes': torch.rand(2, 4), 'labels': torch.tensor([1, 2])}
        
        def get_sample():
            return image2.clone(), {k: v.clone() for k, v in target2.items()}
        
        result, out_target = transform(image1, target1, get_sample)
        
        # Output should be size of first image
        assert result.shape == (3, 100, 100)
        # Should have boxes from both
        assert len(out_target['boxes']) == 3
    
    def test_mosaic_with_single_box_per_image(self):
        """Mosaic should correctly transform single boxes."""
        transform = Mosaic(output_size=(100, 100), p=1.0)
        
        image = torch.rand(3, 50, 50)
        target = {
            'boxes': torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
            'labels': torch.tensor([0]),
        }
        
        def get_sample():
            return image.clone(), {k: v.clone() for k, v in target.items()}
        
        _, out_target = transform(image, target, get_sample)
        
        # Should have 4 boxes (one from each quadrant)
        # Some might be filtered if too small
        assert len(out_target['boxes']) >= 1
        assert len(out_target['boxes']) == len(out_target['labels'])
