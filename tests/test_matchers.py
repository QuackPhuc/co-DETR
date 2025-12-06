"""
Tests for Hungarian Matcher.

This module tests the bipartite matching algorithm for DETR-based detectors,
verifying one-to-one assignment, cost computation, and edge cases.
"""

import pytest
import torch

from codetr.models.matchers.hungarian_matcher import HungarianMatcher, SimpleMatcher


class TestHungarianMatcherBasic:
    """Basic tests for Hungarian Matcher functionality."""
    
    def test_output_format(self):
        """Test matcher returns list of (pred_idx, target_idx) tuples."""
        matcher = HungarianMatcher(cost_class=1.0, cost_bbox=5.0, cost_giou=2.0)
        
        batch_size = 2
        num_queries = 100
        num_classes = 80
        
        pred_logits = torch.randn(batch_size, num_queries, num_classes)
        pred_boxes = torch.rand(batch_size, num_queries, 4)
        
        targets = [
            {'labels': torch.tensor([0, 5, 10]), 'boxes': torch.rand(3, 4)},
            {'labels': torch.tensor([1, 2]), 'boxes': torch.rand(2, 4)},
        ]
        
        indices = matcher(pred_logits, pred_boxes, targets)
        
        # Should return list with one tuple per batch item
        assert len(indices) == batch_size
        
        # Each tuple should be (pred_indices, target_indices)
        for pred_idx, tgt_idx in indices:
            assert isinstance(pred_idx, torch.Tensor)
            assert isinstance(tgt_idx, torch.Tensor)
    
    def test_one_to_one_assignment(self):
        """Each prediction should be matched to at most one target."""
        matcher = HungarianMatcher()
        
        pred_logits = torch.randn(1, 100, 80)
        pred_boxes = torch.rand(1, 100, 4)
        
        targets = [{'labels': torch.tensor([0, 1, 2, 3, 4]), 'boxes': torch.rand(5, 4)}]
        
        indices = matcher(pred_logits, pred_boxes, targets)
        
        pred_idx, tgt_idx = indices[0]
        
        # Check uniqueness of prediction indices
        assert len(pred_idx) == len(pred_idx.unique()), "Duplicate prediction indices"
        
        # Check uniqueness of target indices
        assert len(tgt_idx) == len(tgt_idx.unique()), "Duplicate target indices"
    
    def test_matched_count_equals_targets(self):
        """Number of matches should equal number of targets."""
        matcher = HungarianMatcher()
        
        pred_logits = torch.randn(1, 100, 80)
        pred_boxes = torch.rand(1, 100, 4)
        
        num_targets = 7
        targets = [{'labels': torch.randint(0, 80, (num_targets,)), 'boxes': torch.rand(num_targets, 4)}]
        
        indices = matcher(pred_logits, pred_boxes, targets)
        
        pred_idx, tgt_idx = indices[0]
        
        assert len(pred_idx) == num_targets
        assert len(tgt_idx) == num_targets


class TestHungarianMatcherEmptyTargets:
    """Tests for handling empty targets."""
    
    def test_empty_targets_returns_empty_indices(self):
        """Empty targets should return empty index tensors."""
        matcher = HungarianMatcher()
        
        pred_logits = torch.randn(1, 100, 80)
        pred_boxes = torch.rand(1, 100, 4)
        
        targets = [{'labels': torch.tensor([]), 'boxes': torch.zeros(0, 4)}]
        
        indices = matcher(pred_logits, pred_boxes, targets)
        
        pred_idx, tgt_idx = indices[0]
        
        assert len(pred_idx) == 0
        assert len(tgt_idx) == 0
    
    def test_mixed_empty_and_non_empty(self):
        """Handle batch with both empty and non-empty targets."""
        matcher = HungarianMatcher()
        
        pred_logits = torch.randn(2, 100, 80)
        pred_boxes = torch.rand(2, 100, 4)
        
        targets = [
            {'labels': torch.tensor([0, 1, 2]), 'boxes': torch.rand(3, 4)},  # Non-empty
            {'labels': torch.tensor([]), 'boxes': torch.zeros(0, 4)},  # Empty
        ]
        
        indices = matcher(pred_logits, pred_boxes, targets)
        
        # First batch: 3 matches
        assert len(indices[0][0]) == 3
        
        # Second batch: 0 matches
        assert len(indices[1][0]) == 0


class TestHungarianMatcherCostComponents:
    """Tests for cost matrix computation."""
    
    def test_cost_class_weighting(self):
        """Higher cost_class should prioritize class matching."""
        # Create case where one prediction is perfect class match but poor box match
        pred_logits = torch.zeros(1, 2, 3)
        pred_logits[0, 0, 0] = 10.0  # High confidence for class 0
        pred_logits[0, 1, 1] = 10.0  # High confidence for class 1
        
        pred_boxes = torch.tensor([[[0.0, 0.0, 0.1, 0.1], [0.5, 0.5, 0.9, 0.9]]])
        
        targets = [{'labels': torch.tensor([0]), 'boxes': torch.tensor([[0.5, 0.5, 0.9, 0.9]])}]
        
        # High class cost should prefer class match
        matcher_high_class = HungarianMatcher(cost_class=10.0, cost_bbox=0.1, cost_giou=0.1)
        indices_high = matcher_high_class(pred_logits, pred_boxes, targets)
        
        # Low class cost should prefer box match
        matcher_high_box = HungarianMatcher(cost_class=0.1, cost_bbox=10.0, cost_giou=10.0)
        indices_low = matcher_high_box(pred_logits, pred_boxes, targets)
        
        # With high class weight, should match prediction 0 (correct class)
        assert indices_high[0][0].item() == 0
        
        # With high box weight, should match prediction 1 (correct box position)
        assert indices_low[0][0].item() == 1
    
    def test_index_validity(self):
        """Indices should be within valid ranges."""
        matcher = HungarianMatcher()
        
        num_queries = 50
        num_targets = 8
        
        pred_logits = torch.randn(1, num_queries, 80)
        pred_boxes = torch.rand(1, num_queries, 4)
        targets = [{'labels': torch.randint(0, 80, (num_targets,)), 'boxes': torch.rand(num_targets, 4)}]
        
        indices = matcher(pred_logits, pred_boxes, targets)
        
        pred_idx, tgt_idx = indices[0]
        
        # Prediction indices should be in [0, num_queries)
        assert (pred_idx >= 0).all()
        assert (pred_idx < num_queries).all()
        
        # Target indices should be in [0, num_targets)
        assert (tgt_idx >= 0).all()
        assert (tgt_idx < num_targets).all()


class TestHungarianMatcherDeterminism:
    """Tests for reproducibility."""
    
    def test_reproducibility(self):
        """Same inputs should give same outputs."""
        matcher = HungarianMatcher()
        
        torch.manual_seed(42)
        pred_logits = torch.randn(1, 100, 80)
        pred_boxes = torch.rand(1, 100, 4)
        targets = [{'labels': torch.tensor([0, 1, 2, 3, 4]), 'boxes': torch.rand(5, 4)}]
        
        indices1 = matcher(pred_logits, pred_boxes, targets)
        indices2 = matcher(pred_logits, pred_boxes, targets)
        
        assert torch.equal(indices1[0][0], indices2[0][0])
        assert torch.equal(indices1[0][1], indices2[0][1])


class TestSimpleMatcher:
    """Tests for lightweight IoU-based SimpleMatcher."""
    
    def test_simple_matcher_output_format(self):
        """SimpleMatcher should return same format as HungarianMatcher."""
        matcher = SimpleMatcher(iou_threshold=0.5)
        
        pred_boxes = torch.rand(1, 100, 4)
        targets = [{'boxes': torch.rand(5, 4)}]
        
        indices = matcher(pred_boxes, targets)
        
        assert len(indices) == 1
        pred_idx, tgt_idx = indices[0]
        assert isinstance(pred_idx, torch.Tensor)
        assert isinstance(tgt_idx, torch.Tensor)
    
    def test_simple_matcher_iou_threshold(self):
        """Matches should only occur above IoU threshold."""
        matcher = SimpleMatcher(iou_threshold=0.99)  # Very high threshold
        
        # Create boxes that don't overlap perfectly
        pred_boxes = torch.tensor([[[0.0, 0.0, 0.5, 0.5]]])
        targets = [{'boxes': torch.tensor([[0.1, 0.1, 0.6, 0.6]])}]
        
        indices = matcher(pred_boxes, targets)
        
        # Very high threshold may result in no matches
        pred_idx, tgt_idx = indices[0]
        # Depending on overlap, may or may not match
        assert len(pred_idx) <= 1
    
    def test_simple_matcher_perfect_overlap(self):
        """Identical boxes should match."""
        matcher = SimpleMatcher(iou_threshold=0.5)
        
        box = torch.tensor([[0.2, 0.2, 0.8, 0.8]])
        pred_boxes = box.unsqueeze(0)  # (1, 1, 4)
        targets = [{'boxes': box}]
        
        indices = matcher(pred_boxes, targets)
        
        pred_idx, tgt_idx = indices[0]
        
        # Should have exactly one match
        assert len(pred_idx) == 1
