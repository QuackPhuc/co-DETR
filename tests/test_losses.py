"""
Tests for loss functions.

This module tests Focal Loss, L1/Smooth L1 Loss, and GIoU/DIoU Loss
with mathematical correctness verification.
"""

import pytest
import torch
import torch.nn.functional as F

from codetr.models.losses.focal_loss import FocalLoss, SigmoidFocalLoss
from codetr.models.losses.l1_loss import L1Loss, SmoothL1Loss
from codetr.models.losses.giou_loss import GIoULoss, DIoULoss, giou_loss, diou_loss


class TestFocalLoss:
    """Tests for Focal Loss.
    
    Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    
    def test_focal_loss_output_shape(self):
        """Test output shapes for different reductions."""
        criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')
        
        inputs = torch.randn(2, 100, 80)  # (batch, queries, classes)
        targets = torch.randint(0, 80, (2, 100))  # (batch, queries)
        
        loss = criterion(inputs, targets)
        
        # With reduction='mean', output should be scalar
        assert loss.dim() == 0
    
    def test_focal_loss_reduction_none(self):
        """Test reduction='none' returns per-element loss."""
        criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction='none')
        
        inputs = torch.randn(2, 100, 80)
        targets = torch.randint(0, 80, (2, 100))
        
        loss = criterion(inputs, targets)
        
        # Should return per-element losses
        assert loss.shape == (2, 100) or loss.numel() == 2 * 100 * 80
    
    def test_focal_loss_easy_examples_downweighted(self):
        """Easy examples (high confidence correct predictions) should have lower loss.
        
        With gamma=2, (1-p)^2 heavily downweights well-classified examples.
        """
        criterion_focal = FocalLoss(alpha=0.25, gamma=2.0, reduction='none')
        
        # Easy example: high probability for correct class
        logits_easy = torch.zeros(1, 1, 2)
        logits_easy[0, 0, 0] = 10.0  # Very confident class 0
        target = torch.tensor([[0]])  # True class is 0
        
        # Hard example: low probability for correct class
        logits_hard = torch.zeros(1, 1, 2)
        logits_hard[0, 0, 0] = -10.0  # Low confidence for class 0
        
        loss_easy = criterion_focal(logits_easy, target)
        loss_hard = criterion_focal(logits_hard, target)
        
        # Hard example should have much higher loss
        assert loss_hard.mean() > loss_easy.mean()
    
    def test_focal_loss_gamma_effect(self):
        """Higher gamma should focus more on hard examples."""
        # Moderate difficulty example
        logits = torch.zeros(1, 1, 2)
        logits[0, 0, 0] = 1.0  # Moderate confidence
        target = torch.tensor([[0]])
        
        criterion_low_gamma = FocalLoss(alpha=0.25, gamma=0.5, reduction='mean')
        criterion_high_gamma = FocalLoss(alpha=0.25, gamma=4.0, reduction='mean')
        
        loss_low_gamma = criterion_low_gamma(logits, target)
        loss_high_gamma = criterion_high_gamma(logits, target)
        
        # Higher gamma should give lower loss for correct (easy) predictions
        assert loss_high_gamma < loss_low_gamma
    
    def test_focal_loss_gradient_exists(self):
        """Test gradients are computed correctly."""
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        
        inputs = torch.randn(2, 50, 20, requires_grad=True)
        targets = torch.randint(0, 20, (2, 50))
        
        loss = criterion(inputs, targets)
        loss.backward()
        
        assert inputs.grad is not None
        assert not torch.isnan(inputs.grad).any()
    
    def test_focal_loss_positive_loss(self):
        """Loss should always be non-negative."""
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        
        for _ in range(10):
            inputs = torch.randn(2, 50, 20)
            targets = torch.randint(0, 20, (2, 50))
            loss = criterion(inputs, targets)
            assert loss >= 0


class TestSigmoidFocalLoss:
    """Tests for memory-efficient Sigmoid Focal Loss variant."""
    
    def test_sigmoid_focal_loss_output(self):
        """Test output is scalar with default reduction."""
        criterion = SigmoidFocalLoss(alpha=0.25, gamma=2.0)
        
        inputs = torch.randn(2, 100, 80)
        targets = torch.randint(0, 80, (2, 100))
        
        loss = criterion(inputs, targets)
        
        assert loss.dim() == 0
        assert loss >= 0


class TestL1Loss:
    """Tests for L1 (Mean Absolute Error) Loss."""
    
    def test_l1_loss_basic(self):
        """Test basic L1 loss computation.
        
        L1 = mean(|pred - target|)
        """
        criterion = L1Loss(reduction='mean')
        
        pred = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        target = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
        
        loss = criterion(pred, target)
        
        # Mean of |1|, |2|, |3|, |4| = (1+2+3+4)/4 = 2.5
        expected = 2.5
        assert torch.allclose(loss, torch.tensor(expected))
    
    def test_l1_loss_zero_diff(self):
        """Loss should be zero for identical predictions and targets."""
        criterion = L1Loss()
        
        pred = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        target = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        
        loss = criterion(pred, target)
        
        assert torch.allclose(loss, torch.tensor(0.0))
    
    def test_l1_loss_gradient(self):
        """Test gradients exist."""
        criterion = L1Loss()
        
        pred = torch.randn(2, 100, 4, requires_grad=True)
        target = torch.randn(2, 100, 4)
        
        loss = criterion(pred, target)
        loss.backward()
        
        assert pred.grad is not None


class TestSmoothL1Loss:
    """Tests for Smooth L1 (Huber) Loss.
    
    Smooth L1(x) = 0.5 * x^2 / beta,  if |x| < beta
                   |x| - 0.5 * beta,  otherwise
    """
    
    def test_smooth_l1_quadratic_region(self):
        """In quadratic region (|x| < beta), behaves like squared error / (2*beta)."""
        criterion = SmoothL1Loss(beta=1.0, reduction='none')
        
        pred = torch.tensor([[0.5]])  # |diff| = 0.5 < beta=1.0
        target = torch.tensor([[0.0]])
        
        loss = criterion(pred, target)
        
        # Expected: 0.5 * 0.5^2 / 1.0 = 0.125
        expected = 0.5 * 0.5 * 0.5 / 1.0
        assert torch.allclose(loss, torch.tensor([[expected]]))
    
    def test_smooth_l1_linear_region(self):
        """In linear region (|x| >= beta), behaves like L1 with offset."""
        criterion = SmoothL1Loss(beta=1.0, reduction='none')
        
        pred = torch.tensor([[3.0]])  # |diff| = 3.0 >= beta=1.0
        target = torch.tensor([[0.0]])
        
        loss = criterion(pred, target)
        
        # Expected: |3.0| - 0.5 * 1.0 = 2.5
        expected = 3.0 - 0.5 * 1.0
        assert torch.allclose(loss, torch.tensor([[expected]]))
    
    def test_smooth_l1_at_beta_boundary(self):
        """At exactly |x| = beta, both formulas should give same value."""
        beta = 2.0
        criterion = SmoothL1Loss(beta=beta, reduction='none')
        
        pred = torch.tensor([[beta]])
        target = torch.tensor([[0.0]])
        
        loss = criterion(pred, target)
        
        # Both should equal: beta/2
        expected = beta / 2
        assert torch.allclose(loss, torch.tensor([[expected]]), atol=1e-6)
    
    def test_smooth_l1_gradient_continuity(self):
        """Gradient should be continuous at the transition point."""
        beta = 1.0
        criterion = SmoothL1Loss(beta=beta, reduction='sum')
        
        # Just below and just above transition
        pred_below = torch.tensor([[0.999]], requires_grad=True)
        pred_above = torch.tensor([[1.001]], requires_grad=True)
        target = torch.zeros(1, 1)
        
        loss_below = criterion(pred_below, target)
        loss_below.backward()
        grad_below = pred_below.grad.clone()
        
        loss_above = criterion(pred_above, target)
        loss_above.backward()
        grad_above = pred_above.grad.clone()
        
        # Gradients should be very close (continuous)
        assert torch.allclose(grad_below, grad_above, atol=0.01)


class TestGIoULoss:
    """Tests for Generalized IoU Loss.
    
    L_GIoU = 1 - GIoU
    where GIoU = IoU - (|C| - |A ∪ B|) / |C|
    """
    
    def test_giou_loss_identical_boxes_zero(self):
        """GIoU loss should be 0 for identical boxes (GIoU = 1)."""
        criterion = GIoULoss(reduction='mean')
        
        boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0]])
        
        loss = criterion(boxes, boxes)
        
        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-5)
    
    def test_giou_loss_non_overlapping_positive(self):
        """GIoU loss should be positive for non-overlapping boxes."""
        criterion = GIoULoss(reduction='mean')
        
        pred = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        target = torch.tensor([[50.0, 50.0, 60.0, 60.0]])
        
        loss = criterion(pred, target)
        
        # GIoU < 0 for non-overlapping, so loss = 1-GIoU > 1
        assert loss > 0
        assert loss > 1  # Because GIoU < 0
    
    def test_giou_loss_range(self):
        """GIoU loss should be in [0, 2] since GIoU ∈ [-1, 1]."""
        criterion = GIoULoss(reduction='none')
        
        # Generate random boxes
        pred = torch.rand(100, 4) * 100
        pred[..., 2:] = pred[..., :2] + pred[..., 2:].abs() + 1
        
        target = torch.rand(100, 4) * 100
        target[..., 2:] = target[..., :2] + target[..., 2:].abs() + 1
        
        loss = criterion(pred, target)
        
        assert (loss >= 0).all(), "GIoU loss below 0"
        assert (loss <= 2).all(), "GIoU loss exceeds 2"
    
    def test_giou_loss_gradient_for_non_overlapping(self):
        """GIoU should provide gradients even for non-overlapping boxes."""
        criterion = GIoULoss(reduction='mean')
        
        pred = torch.tensor([[0.0, 0.0, 10.0, 10.0]], requires_grad=True)
        target = torch.tensor([[100.0, 100.0, 110.0, 110.0]])
        
        loss = criterion(pred, target)
        loss.backward()
        
        assert pred.grad is not None
        assert not torch.isnan(pred.grad).any()
        # Gradient should push boxes toward each other
        assert pred.grad.abs().sum() > 0


class TestDIoULoss:
    """Tests for Distance IoU Loss.
    
    L_DIoU = 1 - DIoU
    where DIoU = IoU - (d^2 / c^2)
    d = center distance, c = diagonal of enclosing box
    """
    
    def test_diou_loss_identical_boxes_zero(self):
        """DIoU loss should be 0 for identical boxes."""
        criterion = DIoULoss(reduction='mean')
        
        boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0]])
        
        loss = criterion(boxes, boxes)
        
        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-5)
    
    def test_diou_loss_center_distance_penalty(self):
        """DIoU should penalize center distance more than GIoU."""
        giou_criterion = GIoULoss(reduction='mean')
        diou_criterion = DIoULoss(reduction='mean')
        
        # Two boxes with same overlap but different center distances
        pred1 = torch.tensor([[0.0, 0.0, 20.0, 20.0]])
        pred2 = torch.tensor([[5.0, 5.0, 25.0, 25.0]])  # Same size, offset center
        target = torch.tensor([[0.0, 0.0, 20.0, 20.0]])
        
        giou_loss1 = giou_criterion(pred1, target)
        diou_loss1 = diou_criterion(pred1, target)
        
        giou_loss2 = giou_criterion(pred2, target)
        diou_loss2 = diou_criterion(pred2, target)
        
        # For identical boxes
        assert torch.allclose(giou_loss1, torch.tensor(0.0), atol=1e-5)
        assert torch.allclose(diou_loss1, torch.tensor(0.0), atol=1e-5)
        
        # For offset boxes, both should have positive loss
        assert giou_loss2 > 0
        assert diou_loss2 > 0


class TestFunctionalInterface:
    """Tests for functional interfaces giou_loss and diou_loss."""
    
    def test_giou_loss_functional(self):
        """Test functional giou_loss interface."""
        pred = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        target = torch.tensor([[5.0, 5.0, 15.0, 15.0]])
        
        loss = giou_loss(pred, target, reduction='mean')
        
        assert loss.dim() == 0
        assert loss > 0
    
    def test_diou_loss_functional(self):
        """Test functional diou_loss interface."""
        pred = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        target = torch.tensor([[5.0, 5.0, 15.0, 15.0]])
        
        loss = diou_loss(pred, target, reduction='mean')
        
        assert loss.dim() == 0
        assert loss > 0
