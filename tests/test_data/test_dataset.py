"""
Tests for YOLOv5 format dataset loader.

This module tests the dataset loading, annotation parsing, and
coordinate handling for YOLOv5 format datasets.
"""

import pytest
import torch
import tempfile
from pathlib import Path

from codetr.data.datasets.yolo_dataset import YOLODataset


class TestYOLODatasetAnnotationParsing:
    """Tests for annotation parsing from .txt files."""
    
    @pytest.fixture
    def sample_dataset(self, tmp_path):
        """Create a minimal sample dataset for testing."""
        # Create directory structure
        images_dir = tmp_path / "images" / "train"
        labels_dir = tmp_path / "labels" / "train"
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)
        
        # Create dummy images (1x1 pixel PNGs)
        from PIL import Image
        for i in range(3):
            img = Image.new('RGB', (100, 100), color=(255, 0, 0))
            img.save(images_dir / f"image_{i}.jpg")
            
            # Create corresponding label file
            with open(labels_dir / f"image_{i}.txt", 'w') as f:
                # YOLOv5 format: class_id cx cy w h (normalized)
                f.write(f"0 0.5 0.5 0.2 0.2\n")
                f.write(f"1 0.25 0.75 0.1 0.1\n")
        
        return tmp_path
    
    def test_dataset_length(self, sample_dataset):
        """Test dataset returns correct number of samples."""
        dataset = YOLODataset(
            data_root=str(sample_dataset),
            split='train',
        )
        
        assert len(dataset) == 3
    
    def test_getitem_returns_correct_format(self, sample_dataset):
        """Test __getitem__ returns (image, target) tuple."""
        from codetr.data.transforms import Compose, ToTensor
        
        # Use transforms to get Tensor output
        transforms = Compose([ToTensor()])
        dataset = YOLODataset(
            data_root=str(sample_dataset),
            split='train',
            transforms=transforms,
        )
        
        image, target = dataset[0]
        
        # Image should be tensor after ToTensor transform
        assert isinstance(image, torch.Tensor)
        assert image.dim() == 3  # (C, H, W)
        
        # Target should be dict
        assert isinstance(target, dict)
        assert 'labels' in target
        assert 'boxes' in target
    
    def test_boxes_normalized_cxcywh(self, sample_dataset):
        """Test boxes are in normalized cxcywh format."""
        dataset = YOLODataset(
            data_root=str(sample_dataset),
            split='train',
        )
        
        _, target = dataset[0]
        
        boxes = target['boxes']
        
        # Boxes should be (N, 4)
        assert boxes.shape[1] == 4
        
        # Normalized coordinates should be in [0, 1]
        assert (boxes >= 0).all()
        assert (boxes <= 1).all()
    
    def test_labels_are_integers(self, sample_dataset):
        """Test labels are integer class indices."""
        dataset = YOLODataset(
            data_root=str(sample_dataset),
            split='train',
        )
        
        _, target = dataset[0]
        
        labels = target['labels']
        
        # Should be 1D tensor of integers
        assert labels.dim() == 1
        assert labels.dtype in [torch.int32, torch.int64, torch.long]


class TestYOLODatasetEdgeCases:
    """Tests for edge cases in dataset loading."""
    
    @pytest.fixture
    def empty_annotation_dataset(self, tmp_path):
        """Create dataset with empty annotation file."""
        images_dir = tmp_path / "images" / "train"
        labels_dir = tmp_path / "labels" / "train"
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)
        
        from PIL import Image
        img = Image.new('RGB', (100, 100))
        img.save(images_dir / "empty.jpg")
        
        # Empty label file
        (labels_dir / "empty.txt").touch()
        
        return tmp_path
    
    def test_empty_annotation_returns_empty_tensors(self, empty_annotation_dataset):
        """Empty annotation file should return empty boxes/labels."""
        dataset = YOLODataset(
            data_root=str(empty_annotation_dataset),
            split='train',
        )
        
        _, target = dataset[0]
        
        assert target['boxes'].shape[0] == 0
        assert target['labels'].shape[0] == 0


class TestYOLODatasetWithTransforms:
    """Tests for dataset with data transforms."""
    
    @pytest.fixture
    def sample_dataset(self, tmp_path):
        """Create sample dataset."""
        images_dir = tmp_path / "images" / "train"
        labels_dir = tmp_path / "labels" / "train"
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)
        
        from PIL import Image
        img = Image.new('RGB', (200, 100))  # Non-square
        img.save(images_dir / "rect.jpg")
        
        with open(labels_dir / "rect.txt", 'w') as f:
            f.write("0 0.5 0.5 0.5 0.5\n")
        
        return tmp_path
    
    def test_transforms_applied(self, sample_dataset):
        """Test that transforms are applied to image."""
        from codetr.data.transforms import Compose, ToTensor, Normalize
        
        transforms = Compose([
            ToTensor(),
            Normalize(),
        ])
        
        dataset = YOLODataset(
            data_root=str(sample_dataset),
            split='train',
            transforms=transforms,
        )
        
        image, _ = dataset[0]
        
        # After ToTensor, should be float in [0, 1] range
        # After Normalize, values can be outside [0, 1]
        assert image.dtype == torch.float32


class TestYOLODatasetRobustEdgeCases:
    """Tests for robust handling of edge cases that occur in real datasets."""
    
    @pytest.fixture
    def missing_label_dataset(self, tmp_path):
        """Create dataset where label file is missing."""
        images_dir = tmp_path / "images" / "train"
        labels_dir = tmp_path / "labels" / "train"
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)
        
        from PIL import Image
        # Create image but NO corresponding label file
        img = Image.new('RGB', (100, 100))
        img.save(images_dir / "no_label.jpg")
        
        return tmp_path
    
    def test_missing_label_file_handled(self, missing_label_dataset):
        """Missing label file should return empty annotations (not crash)."""
        try:
            dataset = YOLODataset(
                data_root=str(missing_label_dataset),
                split='train',
            )
            
            _, target = dataset[0]
            
            # Should return empty annotations
            assert target['boxes'].shape[0] == 0
            assert target['labels'].shape[0] == 0
        except FileNotFoundError:
            # Alternative acceptable behavior: raise error
            pass
    
    @pytest.fixture  
    def invalid_class_id_dataset(self, tmp_path):
        """Create dataset with class_id outside valid range."""
        images_dir = tmp_path / "images" / "train"
        labels_dir = tmp_path / "labels" / "train"
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)
        
        from PIL import Image
        img = Image.new('RGB', (100, 100))
        img.save(images_dir / "invalid_class.jpg")
        
        # Create label with very high class ID
        with open(labels_dir / "invalid_class.txt", 'w') as f:
            f.write("999 0.5 0.5 0.2 0.2\n")  # class_id=999 likely > num_classes
        
        return tmp_path
    
    def test_high_class_id_loaded(self, invalid_class_id_dataset):
        """Dataset should load annotations even with high class IDs."""
        dataset = YOLODataset(
            data_root=str(invalid_class_id_dataset),
            split='train',
        )
        
        _, target = dataset[0]
        
        # Label should be loaded as-is (validation happens later)
        assert target['labels'].shape[0] == 1
        assert target['labels'][0] == 999
    
    @pytest.fixture
    def out_of_range_boxes_dataset(self, tmp_path):
        """Create dataset with box coordinates outside [0, 1]."""
        images_dir = tmp_path / "images" / "train"
        labels_dir = tmp_path / "labels" / "train"
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)
        
        from PIL import Image
        img = Image.new('RGB', (100, 100))
        img.save(images_dir / "out_of_range.jpg")
        
        with open(labels_dir / "out_of_range.txt", 'w') as f:
            # Intentionally invalid: center at 1.5 (outside image)
            f.write("0 1.5 0.5 0.2 0.2\n")
        
        return tmp_path
    
    def test_out_of_range_boxes_loaded(self, out_of_range_boxes_dataset):
        """Dataset should handle boxes with coordinates > 1 (clipping is acceptable)."""
        dataset = YOLODataset(
            data_root=str(out_of_range_boxes_dataset),
            split='train',
        )
        
        _, target = dataset[0]
        
        # Box should be loaded
        assert target['boxes'].shape[0] == 1
        # Dataset may clip coordinates to [0, 1] - this is acceptable behavior
        # The first coordinate (cx) should be <= 1.0 after clipping
        assert target['boxes'][0, 0] <= 1.0
    
    @pytest.fixture
    def zero_size_box_dataset(self, tmp_path):
        """Create dataset with zero-area boxes."""
        images_dir = tmp_path / "images" / "train"
        labels_dir = tmp_path / "labels" / "train"
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)
        
        from PIL import Image
        img = Image.new('RGB', (100, 100))
        img.save(images_dir / "zero_box.jpg")
        
        with open(labels_dir / "zero_box.txt", 'w') as f:
            # Zero width and height
            f.write("0 0.5 0.5 0.0 0.0\n")
        
        return tmp_path
    
    def test_zero_size_box_loaded(self, zero_size_box_dataset):
        """Dataset should handle zero-area boxes gracefully (filtering is acceptable)."""
        dataset = YOLODataset(
            data_root=str(zero_size_box_dataset),
            split='train',
        )
        
        _, target = dataset[0]
        
        # Zero-size boxes may be filtered out by the dataset (acceptable behavior)
        # Either the box is loaded or it's filtered - both are valid
        assert target['boxes'].shape[0] >= 0
        assert target['labels'].shape[0] == target['boxes'].shape[0]
    
    @pytest.fixture
    def malformed_label_dataset(self, tmp_path):
        """Create dataset with malformed label format."""
        images_dir = tmp_path / "images" / "train"
        labels_dir = tmp_path / "labels" / "train"
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)
        
        from PIL import Image
        img = Image.new('RGB', (100, 100))
        img.save(images_dir / "malformed.jpg")
        
        with open(labels_dir / "malformed.txt", 'w') as f:
            # Wrong number of columns (only 3 values instead of 5)
            f.write("0 0.5 0.5\n")
        
        return tmp_path
    
    def test_malformed_label_handling(self, malformed_label_dataset):
        """Malformed labels should be handled gracefully."""
        try:
            dataset = YOLODataset(
                data_root=str(malformed_label_dataset),
                split='train',
            )
            
            _, target = dataset[0]
            
            # Either skip malformed line or handle error
            # Empty is acceptable
            assert target['boxes'].shape[1] == 4 or target['boxes'].shape[0] == 0
        except (ValueError, IndexError):
            # Raising a clear error is also acceptable
            pass


class TestYOLODatasetGrayscaleImage:
    """Tests for handling grayscale (1-channel) images.
    
    Real datasets may contain grayscale images, which need to be converted
    to RGB (3-channel) before being processed by the model.
    
    Mathematical correctness:
    - Grayscale to RGB: R = G = B = Gray value
    - Shape transformation: (H, W) -> (3, H, W) for tensors
    """
    
    @pytest.fixture
    def grayscale_dataset(self, tmp_path):
        """Create dataset with a grayscale image."""
        images_dir = tmp_path / "images" / "train"
        labels_dir = tmp_path / "labels" / "train"
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)
        
        from PIL import Image
        # Create grayscale image (mode 'L' = 8-bit luminance)
        img = Image.new('L', (100, 100), color=128)
        img.save(images_dir / "grayscale.jpg")
        
        with open(labels_dir / "grayscale.txt", 'w') as f:
            f.write("0 0.5 0.5 0.2 0.2\n")
        
        return tmp_path
    
    def test_grayscale_image_converted_to_rgb(self, grayscale_dataset):
        """Grayscale images should be converted to 3-channel RGB.
        
        The dataset uses PIL's .convert('RGB') which replicates the
        grayscale channel across R, G, and B.
        """
        from codetr.data.transforms import Compose, ToTensor
        
        transforms = Compose([ToTensor()])
        dataset = YOLODataset(
            data_root=str(grayscale_dataset),
            split='train',
            transforms=transforms,
        )
        
        image, target = dataset[0]
        
        # Image should have 3 channels after conversion
        assert image.shape[0] == 3, f"Expected 3 channels, got {image.shape[0]}"
        # All channels should have same values (R=G=B for grayscale)
        assert torch.allclose(image[0], image[1], atol=1e-5)
        assert torch.allclose(image[1], image[2], atol=1e-5)
    
    def test_grayscale_rgba_image_handled(self, tmp_path):
        """Test RGBA images (4-channel with alpha) are handled correctly.
        
        Some image formats support transparency, which should be dropped
        when converting to RGB.
        """
        images_dir = tmp_path / "images" / "train"
        labels_dir = tmp_path / "labels" / "train"
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)
        
        from PIL import Image
        # Create RGBA image (4-channel with alpha)
        img = Image.new('RGBA', (100, 100), color=(255, 0, 0, 128))
        img.save(images_dir / "rgba.png")
        
        with open(labels_dir / "rgba.txt", 'w') as f:
            f.write("0 0.5 0.5 0.2 0.2\n")
        
        from codetr.data.transforms import Compose, ToTensor
        
        transforms = Compose([ToTensor()])
        dataset = YOLODataset(
            data_root=str(tmp_path),
            split='train',
            transforms=transforms,
        )
        
        image, _ = dataset[0]
        
        # Should have exactly 3 channels (alpha dropped)
        assert image.shape[0] == 3, f"Expected 3 channels, got {image.shape[0]}"


class TestYOLODatasetCorruptedImage:
    """Tests for handling corrupted or truncated image files.
    
    Real-world datasets often contain corrupted images due to:
    - Incomplete downloads
    - Storage corruption
    - Network transfer errors
    
    The dataset should handle these gracefully by either:
    1. Raising an informative error
    2. Skipping the corrupted file
    """
    
    @pytest.fixture
    def corrupted_image_dataset(self, tmp_path):
        """Create dataset with a corrupted (truncated) image file."""
        images_dir = tmp_path / "images" / "train"
        labels_dir = tmp_path / "labels" / "train"
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)
        
        # Create a file with random bytes that is NOT a valid image
        corrupted_file = images_dir / "corrupted.jpg"
        with open(corrupted_file, 'wb') as f:
            # Write partial JPEG header followed by garbage
            # This simulates a truncated/corrupted download
            f.write(b'\xff\xd8\xff\xe0\x00\x10JFIF\x00')  # Partial JPEG header
            f.write(b'\x00' * 50)  # Random garbage bytes
        
        # Create corresponding label file
        with open(labels_dir / "corrupted.txt", 'w') as f:
            f.write("0 0.5 0.5 0.2 0.2\n")
        
        return tmp_path
    
    def test_corrupted_image_raises_error(self, corrupted_image_dataset):
        """Corrupted images should raise an informative error.
        
        When PIL.Image.open() encounters a corrupted file, it will raise
        one of several exceptions (OSError, IOError, or PIL-specific errors).
        
        The dataset should either:
        1. Let this error propagate with a clear message
        2. Catch and re-raise with more context
        """
        from PIL import Image, UnidentifiedImageError
        
        dataset = YOLODataset(
            data_root=str(corrupted_image_dataset),
            split='train',
        )
        
        # Attempting to load corrupted image should raise an error
        with pytest.raises((OSError, IOError, UnidentifiedImageError, Exception)):
            _ = dataset[0]
    
    @pytest.fixture
    def empty_file_dataset(self, tmp_path):
        """Create dataset with an empty file (0 bytes)."""
        images_dir = tmp_path / "images" / "train"
        labels_dir = tmp_path / "labels" / "train"
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)
        
        # Create empty file with image extension
        empty_file = images_dir / "empty.jpg"
        empty_file.touch()  # Creates 0-byte file
        
        with open(labels_dir / "empty.txt", 'w') as f:
            f.write("0 0.5 0.5 0.2 0.2\n")
        
        return tmp_path
    
    def test_empty_image_file_raises_error(self, empty_file_dataset):
        """Empty image files should raise an informative error.
        
        A 0-byte file is clearly invalid and should be rejected.
        """
        from PIL import Image, UnidentifiedImageError
        
        dataset = YOLODataset(
            data_root=str(empty_file_dataset),
            split='train',
        )
        
        # Loading empty file should raise error
        with pytest.raises((OSError, IOError, UnidentifiedImageError, Exception)):
            _ = dataset[0]
