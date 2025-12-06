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
        dataset = YOLODataset(
            data_root=str(sample_dataset),
            split='train',
        )
        
        image, target = dataset[0]
        
        # Image should be tensor
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
