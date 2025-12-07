"""
Tests for detection evaluator.

This module tests the custom mAP computation including IoU matrix calculation,
precision-recall curves, and AP aggregation.
"""

import pytest
import torch

from codetr.engine.evaluator import (
    compute_iou_matrix,
    compute_ap,
    match_predictions_to_gt,
    DetectionEvaluator,
    build_evaluator,
)


class TestComputeIoUMatrix:
    """Tests for IoU matrix computation."""
    
    def test_output_shape(self):
        """Test IoU matrix has shape (N, M)."""
        pred_boxes = torch.rand(5, 4) * 100
        pred_boxes[:, 2:] = pred_boxes[:, :2] + pred_boxes[:, 2:].abs() + 1
        
        gt_boxes = torch.rand(3, 4) * 100
        gt_boxes[:, 2:] = gt_boxes[:, :2] + gt_boxes[:, 2:].abs() + 1
        
        iou_matrix = compute_iou_matrix(pred_boxes, gt_boxes)
        
        assert iou_matrix.shape == (5, 3)
    
    def test_identical_boxes_iou_one(self):
        """Identical boxes should have IoU = 1."""
        boxes = torch.tensor([
            [0.0, 0.0, 10.0, 10.0],
            [20.0, 20.0, 30.0, 30.0],
        ])
        
        iou_matrix = compute_iou_matrix(boxes, boxes)
        
        # Diagonal should be 1
        assert torch.allclose(torch.diag(iou_matrix), torch.ones(2))
    
    def test_non_overlapping_iou_zero(self):
        """Non-overlapping boxes should have IoU = 0."""
        pred = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        gt = torch.tensor([[100.0, 100.0, 110.0, 110.0]])
        
        iou_matrix = compute_iou_matrix(pred, gt)
        
        assert torch.allclose(iou_matrix, torch.zeros(1, 1))
    
    def test_value_range(self):
        """IoU values should be in [0, 1]."""
        pred = torch.rand(10, 4) * 100
        pred[:, 2:] = pred[:, :2] + pred[:, 2:].abs() + 1
        
        gt = torch.rand(5, 4) * 100
        gt[:, 2:] = gt[:, :2] + gt[:, 2:].abs() + 1
        
        iou_matrix = compute_iou_matrix(pred, gt)
        
        assert (iou_matrix >= 0).all()
        assert (iou_matrix <= 1).all()


class TestComputeAP:
    """Tests for Average Precision computation."""
    
    def test_perfect_detector_ap_one(self):
        """Perfect detector should have AP = 1."""
        # Perfect detector: all predictions correct, recall goes from 0 to 1
        recalls = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        precisions = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
        
        ap = compute_ap(recalls, precisions)
        
        assert abs(ap - 1.0) < 1e-3
    
    def test_no_detections_ap_zero(self):
        """No true positives should give AP = 0."""
        recalls = torch.tensor([0.0])
        precisions = torch.tensor([0.0])
        
        ap = compute_ap(recalls, precisions)
        
        assert ap <= 1e-3


class TestMatchPredictionsToGT:
    """Tests for matching predictions to ground truth."""
    
    def test_match_perfect_overlap(self):
        """Perfect overlap should match."""
        pred_boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        pred_scores = torch.tensor([0.9])
        pred_labels = torch.tensor([0])
        
        gt_boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        gt_labels = torch.tensor([0])
        
        tp, matched_gt = match_predictions_to_gt(
            pred_boxes, pred_scores, pred_labels,
            gt_boxes, gt_labels,
            iou_threshold=0.5
        )
        
        # Should match prediction 0 to gt 0
        assert tp[0] == True  # Is a true positive
        assert matched_gt[0] == 0  # Matched to GT index 0
    
    def test_no_match_wrong_class(self):
        """Different class labels should not match."""
        pred_boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        pred_scores = torch.tensor([0.9])
        pred_labels = torch.tensor([0])
        
        gt_boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        gt_labels = torch.tensor([1])  # Different class
        
        tp, matched_gt = match_predictions_to_gt(
            pred_boxes, pred_scores, pred_labels,
            gt_boxes, gt_labels,
            iou_threshold=0.5
        )
        
        # Should not match (different class)
        assert tp[0] == False
        assert matched_gt[0] == -1
    
    def test_no_match_low_iou(self):
        """Low IoU should not match."""
        pred_boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        pred_scores = torch.tensor([0.9])
        pred_labels = torch.tensor([0])
        
        gt_boxes = torch.tensor([[50.0, 50.0, 60.0, 60.0]])  # Far away
        gt_labels = torch.tensor([0])
        
        tp, matched_gt = match_predictions_to_gt(
            pred_boxes, pred_scores, pred_labels,
            gt_boxes, gt_labels,
            iou_threshold=0.5
        )
        
        # Should not match (low IoU)
        assert tp[0] == False
        assert matched_gt[0] == -1


class TestDetectionEvaluator:
    """Tests for DetectionEvaluator class."""
    
    def test_add_predictions_format(self):
        """Test add_predictions accepts correct format."""
        evaluator = DetectionEvaluator(num_classes=20)
        
        predictions = [{
            'boxes': torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
            'scores': torch.tensor([0.9]),
            'labels': torch.tensor([0]),
        }]
        
        targets = [{
            'boxes': torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
            'labels': torch.tensor([0]),
        }]
        
        # Should not raise
        evaluator.add_predictions(predictions, targets)
    
    def test_compute_metrics_returns_dict(self):
        """compute_metrics should return dict with standard keys."""
        evaluator = DetectionEvaluator(num_classes=5)
        
        predictions = [{
            'boxes': torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
            'scores': torch.tensor([0.9]),
            'labels': torch.tensor([0]),
        }]
        
        targets = [{
            'boxes': torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
            'labels': torch.tensor([0]),
        }]
        
        evaluator.add_predictions(predictions, targets)
        metrics = evaluator.compute_metrics()
        
        assert 'mAP' in metrics
        assert 'AP50' in metrics
        assert 'AP75' in metrics
    
    def test_reset_clears_state(self):
        """reset() should clear accumulated predictions."""
        evaluator = DetectionEvaluator(num_classes=5)
        
        predictions = [{'boxes': torch.rand(3, 4), 'scores': torch.rand(3), 'labels': torch.zeros(3, dtype=torch.long)}]
        targets = [{'boxes': torch.rand(2, 4), 'labels': torch.zeros(2, dtype=torch.long)}]
        
        evaluator.add_predictions(predictions, targets)
        evaluator.reset()
        
        # After reset, internal state should be empty
        assert len(evaluator.predictions) == 0
        assert len(evaluator.ground_truths) == 0
    
    def test_ap_value_range(self):
        """AP values should be in [0, 1]."""
        evaluator = DetectionEvaluator(num_classes=5)
        
        for _ in range(5):
            predictions = [{
                'boxes': torch.rand(10, 4) * 100,
                'scores': torch.rand(10),
                'labels': torch.randint(0, 5, (10,)),
            }]
            targets = [{
                'boxes': torch.rand(5, 4) * 100,
                'labels': torch.randint(0, 5, (5,)),
            }]
            # Fix box coordinates
            predictions[0]['boxes'][:, 2:] = predictions[0]['boxes'][:, :2] + predictions[0]['boxes'][:, 2:].abs() + 1
            targets[0]['boxes'][:, 2:] = targets[0]['boxes'][:, :2] + targets[0]['boxes'][:, 2:].abs() + 1
            
            evaluator.add_predictions(predictions, targets)
        
        metrics = evaluator.compute_metrics()
        
        assert 0 <= metrics['mAP'] <= 1
        assert 0 <= metrics['AP50'] <= 1
        assert 0 <= metrics['AP75'] <= 1


class TestBuildEvaluator:
    """Tests for build_evaluator factory function."""
    
    def test_build_evaluator_default(self):
        """Test building evaluator with defaults."""
        evaluator = build_evaluator(num_classes=80)
        
        assert isinstance(evaluator, DetectionEvaluator)
    
    def test_build_evaluator_custom_thresholds(self):
        """Test building evaluator with custom IoU thresholds."""
        evaluator = build_evaluator(
            num_classes=20,
            iou_thresholds=[0.5, 0.75]
        )
        
        assert len(evaluator.iou_thresholds) == 2


class TestEvaluatorMultiImage:
    """Tests for evaluator with multiple images.
    
    These tests verify the evaluator correctly accumulates predictions
    across many images and computes per-class AP correctly.
    """

    def test_evaluator_with_many_images(self):
        """Test evaluator handles 20+ images correctly.
        
        This verifies the accumulation logic works for realistic batch counts.
        """
        evaluator = DetectionEvaluator(num_classes=5)

        # Add 25 images worth of predictions
        for img_idx in range(25):
            # Create 2-5 predictions per image
            num_preds = (img_idx % 4) + 2
            num_gts = (img_idx % 3) + 1

            pred_boxes = torch.rand(num_preds, 4) * 100
            pred_boxes[:, 2:] = pred_boxes[:, :2] + 20  # Ensure valid xyxy

            gt_boxes = torch.rand(num_gts, 4) * 100
            gt_boxes[:, 2:] = gt_boxes[:, :2] + 20

            predictions = [{
                'boxes': pred_boxes,
                'scores': torch.rand(num_preds),
                'labels': torch.randint(0, 5, (num_preds,)),
            }]

            targets = [{
                'boxes': gt_boxes,
                'labels': torch.randint(0, 5, (num_gts,)),
            }]

            evaluator.add_predictions(predictions=predictions, targets=targets)

        # Should have accumulated all images
        assert len(evaluator.predictions) == 25
        assert len(evaluator.ground_truths) == 25

        # Should compute metrics without error
        metrics = evaluator.compute_metrics()
        assert 'mAP' in metrics
        assert 0 <= metrics['mAP'] <= 1

    def test_per_class_ap_computation(self):
        """Verify per-class AP is computed correctly.
        
        Class 0 should have higher AP than class 1 when class 0 predictions
        are more accurate.
        """
        evaluator = DetectionEvaluator(num_classes=3)

        # Class 0: Perfect predictions (exact match)
        predictions_class0 = [{
            'boxes': torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
            'scores': torch.tensor([0.99]),
            'labels': torch.tensor([0]),
        }]
        targets_class0 = [{
            'boxes': torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
            'labels': torch.tensor([0]),
        }]

        # Class 1: Wrong predictions (no overlap)
        predictions_class1 = [{
            'boxes': torch.tensor([[100.0, 100.0, 150.0, 150.0]]),
            'scores': torch.tensor([0.95]),
            'labels': torch.tensor([1]),
        }]
        targets_class1 = [{
            'boxes': torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
            'labels': torch.tensor([1]),
        }]

        evaluator.add_predictions(predictions=predictions_class0, targets=targets_class0)
        evaluator.add_predictions(predictions=predictions_class1, targets=targets_class1)

        metrics = evaluator.compute_metrics()

        # Should have per-class AP in metrics
        assert 'AP_per_class' in metrics
        # Class 0 should have high AP (perfect match)
        # Class 1 should have low AP (no match)
        per_class_ap = metrics['AP_per_class']
        assert per_class_ap[0] > per_class_ap[1], (
            f"Class 0 AP ({per_class_ap[0]}) should > Class 1 AP ({per_class_ap[1]})"
        )

    def test_all_false_positives_ap_zero(self):
        """All predictions wrong class → AP should be ~0.
        
        Mathematical guarantee: If no true positives, precision = 0 → AP = 0.
        """
        evaluator = DetectionEvaluator(num_classes=3)

        # Predictions all class 0
        predictions = [{
            'boxes': torch.tensor([
                [10.0, 10.0, 50.0, 50.0],
                [60.0, 60.0, 100.0, 100.0],
            ]),
            'scores': torch.tensor([0.9, 0.8]),
            'labels': torch.tensor([0, 0]),  # All class 0
        }]

        # But GT is all class 1
        targets = [{
            'boxes': torch.tensor([
                [10.0, 10.0, 50.0, 50.0],
                [60.0, 60.0, 100.0, 100.0],
            ]),
            'labels': torch.tensor([1, 1]),  # All class 1
        }]

        evaluator.add_predictions(predictions=predictions, targets=targets)
        metrics = evaluator.compute_metrics()

        # mAP should be very low (all predictions are false positives)
        assert metrics['mAP'] < 0.1, f"mAP should be ~0, got {metrics['mAP']}"
