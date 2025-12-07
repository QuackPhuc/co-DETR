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
        """Dataset should load boxes even if coordinates are > 1."""
        dataset = YOLODataset(
            data_root=str(out_of_range_boxes_dataset),
            split='train',
        )
        
        _, target = dataset[0]
        
        # Box should be loaded (validation/clipping happens at transform stage)
        assert target['boxes'].shape[0] == 1
        # The first coordinate (cx) should be 1.5
        assert target['boxes'][0, 0] == pytest.approx(1.5, rel=0.01)
    
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
        """Dataset should load zero-area boxes without crashing."""
        dataset = YOLODataset(
            data_root=str(zero_size_box_dataset),
            split='train',
        )
        
        _, target = dataset[0]
        
        # Zero-size boxes should be loaded
        assert target['boxes'].shape[0] == 1
        # Width and height should be 0
        assert target['boxes'][0, 2] == pytest.approx(0.0)
        assert target['boxes'][0, 3] == pytest.approx(0.0)
    
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
