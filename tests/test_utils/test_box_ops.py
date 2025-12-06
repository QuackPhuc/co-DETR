"""
Tests for bounding box operations.

This module tests the bounding box utility functions including coordinate
transformations, IoU computation, and GIoU calculation with mathematical
correctness verification.
"""

import pytest
import torch
from torch import Tensor

from codetr.models.utils.box_ops import (
    bbox_xyxy_to_cxcywh,
    bbox_cxcywh_to_xyxy,
    box_area,
    box_iou,
    generalized_box_iou,
    inverse_sigmoid,
)


class TestBboxCoordinateConversion:
    """Tests for bounding box coordinate format conversions."""
    
    def test_bbox_xyxy_to_cxcywh_basic(self):
        """Test basic xyxy to cxcywh conversion.
        
        Formula: cx = (x1+x2)/2, cy = (y1+y2)/2, w = x2-x1, h = y2-y1
        """
        boxes_xyxy = torch.tensor([
            [0.0, 0.0, 10.0, 10.0],  # Expected: cx=5, cy=5, w=10, h=10
            [5.0, 5.0, 15.0, 15.0],  # Expected: cx=10, cy=10, w=10, h=10
        ])
        
        result = bbox_xyxy_to_cxcywh(boxes_xyxy)
        
        expected = torch.tensor([
            [5.0, 5.0, 10.0, 10.0],
            [10.0, 10.0, 10.0, 10.0],
        ])
        
        assert result.shape == (2, 4), f"Shape mismatch: {result.shape}"
        assert torch.allclose(result, expected), f"Values mismatch: {result} vs {expected}"
    
    def test_bbox_cxcywh_to_xyxy_basic(self):
        """Test basic cxcywh to xyxy conversion.
        
        Formula: x1 = cx - w/2, y1 = cy - h/2, x2 = cx + w/2, y2 = cy + h/2
        """
        boxes_cxcywh = torch.tensor([
            [5.0, 5.0, 10.0, 10.0],   # Expected: x1=0, y1=0, x2=10, y2=10
            [10.0, 10.0, 10.0, 10.0], # Expected: x1=5, y1=5, x2=15, y2=15
        ])
        
        result = bbox_cxcywh_to_xyxy(boxes_cxcywh)
        
        expected = torch.tensor([
            [0.0, 0.0, 10.0, 10.0],
            [5.0, 5.0, 15.0, 15.0],
        ])
        
        assert result.shape == (2, 4)
        assert torch.allclose(result, expected)
    
    def test_bbox_conversion_roundtrip(self):
        """Test that xyxy -> cxcywh -> xyxy returns original boxes."""
        original = torch.tensor([
            [10.0, 20.0, 50.0, 80.0],
            [0.0, 0.0, 100.0, 100.0],
            [25.5, 30.5, 75.5, 90.5],
        ])
        
        cxcywh = bbox_xyxy_to_cxcywh(original)
        recovered = bbox_cxcywh_to_xyxy(cxcywh)
        
        assert torch.allclose(original, recovered, atol=1e-5)
    
    def test_bbox_conversion_batch_dimension(self):
        """Test conversion with batch dimension."""
        # Shape: (batch, num_boxes, 4)
        boxes = torch.rand(2, 10, 4) * 100
        boxes[..., 2:] = boxes[..., :2] + boxes[..., 2:].abs()  # Ensure x2>x1, y2>y1
        
        cxcywh = bbox_xyxy_to_cxcywh(boxes)
        recovered = bbox_cxcywh_to_xyxy(cxcywh)
        
        assert cxcywh.shape == (2, 10, 4)
        assert torch.allclose(boxes, recovered, atol=1e-5)


class TestBoxArea:
    """Tests for box area computation."""
    
    def test_box_area_basic(self):
        """Test area computation.
        
        Formula: area = (x2 - x1) * (y2 - y1)
        """
        boxes = torch.tensor([
            [0.0, 0.0, 10.0, 10.0],  # Area = 100
            [5.0, 5.0, 15.0, 25.0],  # Area = 10 * 20 = 200
            [0.0, 0.0, 1.0, 1.0],    # Area = 1
        ])
        
        areas = box_area(boxes)
        expected = torch.tensor([100.0, 200.0, 1.0])
        
        assert areas.shape == (3,)
        assert torch.allclose(areas, expected)
    
    def test_box_area_zero_size(self):
        """Test degenerate boxes with zero size."""
        boxes = torch.tensor([
            [5.0, 5.0, 5.0, 5.0],  # Zero area (point)
            [0.0, 0.0, 10.0, 0.0], # Zero area (line)
        ])
        
        areas = box_area(boxes)
        
        assert torch.allclose(areas, torch.tensor([0.0, 0.0]))


class TestBoxIoU:
    """Tests for Intersection over Union computation."""
    
    def test_box_iou_identical_boxes(self):
        """Identical boxes should have IoU = 1.0."""
        boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        
        iou, union = box_iou(boxes, boxes)
        
        assert iou.shape == (1, 1)
        assert torch.allclose(iou, torch.tensor([[1.0]]))
    
    def test_box_iou_no_overlap(self):
        """Non-overlapping boxes should have IoU = 0.0."""
        boxes1 = torch.tensor([[0.0, 0.0, 10.0, 10.0]])  # Left box
        boxes2 = torch.tensor([[20.0, 20.0, 30.0, 30.0]]) # Right box, no overlap
        
        iou, union = box_iou(boxes1, boxes2)
        
        assert torch.allclose(iou, torch.tensor([[0.0]]))
    
    def test_box_iou_partial_overlap(self):
        """Test IoU for partially overlapping boxes.
        
        Box1: [0,0,10,10] area=100
        Box2: [5,5,15,15] area=100
        Intersection: [5,5,10,10] area=25
        Union: 100 + 100 - 25 = 175
        IoU = 25/175 = 0.142857...
        """
        boxes1 = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        boxes2 = torch.tensor([[5.0, 5.0, 15.0, 15.0]])
        
        iou, union = box_iou(boxes1, boxes2)
        
        expected_iou = 25.0 / 175.0  # ≈ 0.1429
        assert iou.shape == (1, 1)
        assert torch.allclose(iou, torch.tensor([[expected_iou]]), atol=1e-4)
    
    def test_box_iou_enclosed_box(self):
        """Test IoU when one box encloses another.
        
        Box1: [0,0,10,10] area=100
        Box2: [0,0,20,20] area=400
        Intersection = 100 (entire box1)
        Union = 400 (entire box2)
        IoU = 100/400 = 0.25
        """
        boxes1 = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        boxes2 = torch.tensor([[0.0, 0.0, 20.0, 20.0]])
        
        iou, union = box_iou(boxes1, boxes2)
        
        assert torch.allclose(iou, torch.tensor([[0.25]]), atol=1e-4)
    
    def test_box_iou_matrix_shape(self):
        """Test IoU returns correct matrix shape for N×M computation."""
        boxes1 = torch.rand(5, 4)
        boxes1[..., 2:] = boxes1[..., :2] + boxes1[..., 2:].abs()
        
        boxes2 = torch.rand(3, 4)
        boxes2[..., 2:] = boxes2[..., :2] + boxes2[..., 2:].abs()
        
        iou, union = box_iou(boxes1, boxes2)
        
        assert iou.shape == (5, 3)
        assert union.shape == (5, 3)
    
    def test_box_iou_value_range(self):
        """IoU values should be in [0, 1]."""
        boxes1 = torch.rand(10, 4) * 100
        boxes1[..., 2:] = boxes1[..., :2] + (boxes1[..., 2:].abs() + 1)
        
        boxes2 = torch.rand(8, 4) * 100
        boxes2[..., 2:] = boxes2[..., :2] + (boxes2[..., 2:].abs() + 1)
        
        iou, _ = box_iou(boxes1, boxes2)
        
        assert (iou >= 0).all(), "IoU has negative values"
        assert (iou <= 1).all(), "IoU exceeds 1"


class TestGeneralizedBoxIoU:
    """Tests for Generalized IoU computation."""
    
    def test_giou_identical_boxes(self):
        """Identical boxes should have GIoU = 1.0."""
        boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        
        giou = generalized_box_iou(boxes, boxes)
        
        assert torch.allclose(giou, torch.tensor([[1.0]]))
    
    def test_giou_no_overlap_penalty(self):
        """GIoU should penalize non-overlapping boxes.
        
        Unlike IoU which is 0 for non-overlapping boxes, GIoU can be negative
        due to the enclosing box penalty term.
        
        GIoU = IoU - (|C| - |A ∪ B|) / |C|
        where C is the smallest enclosing box.
        """
        boxes1 = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        boxes2 = torch.tensor([[20.0, 20.0, 30.0, 30.0]])
        
        giou = generalized_box_iou(boxes1, boxes2)
        
        # GIoU should be negative for non-overlapping boxes
        assert giou[0, 0] < 0, "GIoU should be negative for non-overlapping boxes"
    
    def test_giou_range(self):
        """GIoU values should be in [-1, 1]."""
        boxes1 = torch.rand(10, 4) * 100
        boxes1[..., 2:] = boxes1[..., :2] + (boxes1[..., 2:].abs() + 1)
        
        boxes2 = torch.rand(8, 4) * 100
        boxes2[..., 2:] = boxes2[..., :2] + (boxes2[..., 2:].abs() + 1)
        
        giou = generalized_box_iou(boxes1, boxes2)
        
        assert (giou >= -1).all(), "GIoU below -1"
        assert (giou <= 1).all(), "GIoU exceeds 1"
    
    def test_giou_manual_calculation(self):
        """Manually verify GIoU calculation.
        
        Box1: [0,0,10,10] area=100
        Box2: [5,5,15,15] area=100
        Intersection: 25
        Union: 175
        IoU: 25/175 ≈ 0.1429
        Enclosing box C: [0,0,15,15] area=225
        GIoU = IoU - (225-175)/225 = 0.1429 - 50/225 ≈ 0.1429 - 0.222 ≈ -0.079
        """
        boxes1 = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        boxes2 = torch.tensor([[5.0, 5.0, 15.0, 15.0]])
        
        giou = generalized_box_iou(boxes1, boxes2)
        
        expected_iou = 25.0 / 175.0
        enclosing_area = 15.0 * 15.0
        expected_giou = expected_iou - (enclosing_area - 175.0) / enclosing_area
        
        assert torch.allclose(giou, torch.tensor([[expected_giou]]), atol=1e-4)


class TestInverseSigmoid:
    """Tests for inverse sigmoid (logit) function."""
    
    def test_inverse_sigmoid_roundtrip(self):
        """sigmoid(inverse_sigmoid(x)) should equal x for x in (0, 1)."""
        x = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
        
        logit = inverse_sigmoid(x)
        recovered = torch.sigmoid(logit)
        
        assert torch.allclose(recovered, x, atol=1e-5)
    
    def test_inverse_sigmoid_midpoint(self):
        """inverse_sigmoid(0.5) should be 0."""
        x = torch.tensor([0.5])
        
        result = inverse_sigmoid(x)
        
        assert torch.allclose(result, torch.tensor([0.0]), atol=1e-5)
    
    def test_inverse_sigmoid_symmetry(self):
        """inverse_sigmoid(p) = -inverse_sigmoid(1-p)."""
        p = torch.tensor([0.2, 0.3, 0.4])
        
        logit_p = inverse_sigmoid(p)
        logit_1_minus_p = inverse_sigmoid(1 - p)
        
        assert torch.allclose(logit_p, -logit_1_minus_p, atol=1e-5)
    
    def test_inverse_sigmoid_clamping(self):
        """Values at boundaries should be clamped to avoid inf."""
        x = torch.tensor([0.0, 1.0])  # Boundary values
        
        result = inverse_sigmoid(x, eps=1e-5)
        
        # Should not be inf
        assert torch.isfinite(result).all()
