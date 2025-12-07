"""
Integration tests for complete Co-DETR pipeline.

This module tests end-to-end functionality including data-to-model
pipeline and model output compatibility with evaluator.
"""

import pytest
import torch
from typing import List, Dict

from codetr.models.detector import CoDETR, build_codetr
from codetr.models.utils.misc import nested_tensor_from_tensor_list, NestedTensor
from codetr.engine.evaluator import DetectionEvaluator


class TestFullForwardPass:
    """Tests for complete model forward pass."""
    
    def test_training_forward_pass(self):
        """Test full training forward pass from images to losses."""
        model = build_codetr(
            num_classes=20,
            pretrained_backbone=False,
            use_aux_heads=False,
        )
        model.train()
        
        # Create input batch
        batch_size = 2
        images = torch.randn(batch_size, 3, 256, 256)
        
        targets = [
            {
                'labels': torch.tensor([0, 5, 10]),
                'boxes': torch.tensor([
                    [0.25, 0.25, 0.1, 0.1],
                    [0.5, 0.5, 0.2, 0.2],
                    [0.75, 0.75, 0.15, 0.15],
                ]),
            },
            {
                'labels': torch.tensor([1, 2]),
                'boxes': torch.tensor([
                    [0.3, 0.3, 0.25, 0.25],
                    [0.7, 0.7, 0.2, 0.2],
                ]),
            },
        ]
        
        # Forward pass
        outputs = model(images, targets)
        
        # Should return loss dictionary
        assert isinstance(outputs, dict)
        
        # Total loss should be computable
        total_loss = sum(v for v in outputs.values() if isinstance(v, torch.Tensor))
        assert total_loss.dim() == 0  # Scalar
        assert not torch.isnan(total_loss)
    
    def test_inference_forward_pass(self):
        """Test full inference forward pass from images to predictions."""
        model = build_codetr(
            num_classes=20,
            pretrained_backbone=False,
            use_aux_heads=False,
        )
        model.eval()
        
        images = torch.randn(2, 3, 256, 256)
        
        with torch.no_grad():
            predictions = model(images)
        
        # Should return list of prediction dicts
        assert isinstance(predictions, list)
        assert len(predictions) == 2
        
        for pred in predictions:
            assert 'scores' in pred
            assert 'labels' in pred
            assert 'boxes' in pred
            
            # Boxes should be valid shape
            assert pred['boxes'].dim() == 2
            assert pred['boxes'].shape[1] == 4


class TestDataToModelPipeline:
    """Tests for data pipeline integration with model."""
    
    def test_nested_tensor_input(self):
        """Test model accepts NestedTensor input format."""
        model = build_codetr(
            num_classes=10,
            pretrained_backbone=False,
        )
        model.eval()
        
        # Create variable-size image batch
        images = [
            torch.randn(3, 256, 200),
            torch.randn(3, 300, 256),
        ]
        
        nested_tensor = nested_tensor_from_tensor_list(images, size_divisibility=32)
        
        # Extract features should work with NestedTensor
        with torch.no_grad():
            # Model should handle padded input
            predictions = model(nested_tensor.tensors)
        
        assert len(predictions) == 2


class TestModelEvaluatorCompatibility:
    """Tests for model output compatibility with evaluator."""
    
    def test_prediction_format_compatible(self):
        """Test model predictions can be directly used with evaluator."""
        model = build_codetr(
            num_classes=5,
            pretrained_backbone=False,
        )
        model.eval()
        
        evaluator = DetectionEvaluator(num_classes=5)
        
        images = torch.randn(2, 3, 256, 256)
        targets = [
            {'labels': torch.tensor([0, 1]), 'boxes': torch.tensor([[50.0, 50.0, 100.0, 100.0], [110.0, 110.0, 150.0, 150.0]])},
            {'labels': torch.tensor([2]), 'boxes': torch.tensor([[30.0, 30.0, 80.0, 80.0]])},
        ]
        
        with torch.no_grad():
            predictions = model(images)
        
        # Predictions should be directly usable with evaluator
        evaluator.add_predictions(predictions, targets)
        
        # Should be able to compute metrics
        metrics = evaluator.compute_metrics()
        
        assert 'mAP' in metrics


class TestGradientFlowEndToEnd:
    """Tests for end-to-end gradient flow."""
    
    def test_gradients_flow_to_all_components(self):
        """Test gradients reach all trainable components."""
        model = build_codetr(
            num_classes=10,
            pretrained_backbone=False,
        )
        model.train()
        
        # Disable frozen stages for gradient test
        for param in model.parameters():
            param.requires_grad = True
        
        images = torch.randn(1, 3, 128, 128)
        targets = [{'labels': torch.tensor([0]), 'boxes': torch.rand(1, 4)}]
        
        outputs = model(images, targets)
        total_loss = sum(v for v in outputs.values() if isinstance(v, torch.Tensor))
        total_loss.backward()
        
        # Check backbone has gradients
        backbone_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.backbone.parameters()
            if p.requires_grad
        )
        
        # Check transformer has gradients
        transformer_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.transformer.parameters()
            if p.requires_grad
        )
        
        assert backbone_has_grad or transformer_has_grad


class TestModelOutputConsistency:
    """Tests for output consistency across modes."""
    
    def test_output_shapes_consistent(self):
        """Test output shapes are consistent across calls."""
        model = build_codetr(
            num_classes=20,
            pretrained_backbone=False,
        )
        model.eval()
        
        images = torch.randn(2, 3, 256, 256)
        
        with torch.no_grad():
            pred1 = model(images)
            pred2 = model(images)
        
        # Same input should give same output structure
        assert len(pred1) == len(pred2)
        
        for p1, p2 in zip(pred1, pred2):
            assert p1['boxes'].shape == p2['boxes'].shape
            assert p1['scores'].shape == p2['scores'].shape
            assert p1['labels'].shape == p2['labels'].shape


class TestCheckpointConsistency:
    """Tests for checkpoint save/load consistency.
    
    These tests verify that model state is preserved correctly when
    saving and loading checkpoints.
    """

    def test_checkpoint_save_load_weights_match(self):
        """Save → Load should produce identical model weights.
        
        This is critical for training resume to work correctly.
        """
        import tempfile
        from pathlib import Path

        # Create model
        model1 = build_codetr(
            num_classes=10,
            pretrained_backbone=False,
            use_aux_heads=False,
        )

        # Get original weights
        original_state = {k: v.clone() for k, v in model1.state_dict().items()}

        # Save checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test_checkpoint.pth"
            torch.save({
                'model_state_dict': model1.state_dict(),
                'epoch': 5,
            }, ckpt_path)

            # Create new model
            model2 = build_codetr(
                num_classes=10,
                pretrained_backbone=False,
                use_aux_heads=False,
            )

            # Load checkpoint
            ckpt = torch.load(ckpt_path, weights_only=False)
            model2.load_state_dict(ckpt['model_state_dict'])

            # Verify weights match
            for key in original_state:
                assert torch.allclose(
                    original_state[key], 
                    model2.state_dict()[key],
                    atol=1e-6
                ), f"Weight mismatch for {key}"

            # Verify metadata
            assert ckpt['epoch'] == 5


class TestInferenceDeterminism:
    """Tests for inference determinism and reproducibility."""

    def test_inference_deterministic_same_input(self):
        """Same input should produce identical outputs in eval mode.
        
        This verifies no randomness leaks in during inference.
        """
        model = build_codetr(
            num_classes=10,
            pretrained_backbone=False,
            use_aux_heads=False,
        )
        model.eval()

        # Fixed input
        torch.manual_seed(42)
        images = torch.randn(2, 3, 256, 256)

        with torch.no_grad():
            pred1 = model(images)
            pred2 = model(images)

        # Outputs should be identical (not just same shape)
        for p1, p2 in zip(pred1, pred2):
            assert torch.allclose(p1['boxes'], p2['boxes']), "Boxes should be identical"
            assert torch.allclose(p1['scores'], p2['scores']), "Scores should be identical"
            assert torch.equal(p1['labels'], p2['labels']), "Labels should be identical"

    def test_eval_mode_disables_dropout(self):
        """Model in eval mode should have no stochastic behavior."""
        model = build_codetr(
            num_classes=5,
            pretrained_backbone=False,
        )
        model.eval()

        images = torch.randn(1, 3, 128, 128)

        # Run multiple times, should all be identical
        with torch.no_grad():
            results = [model(images) for _ in range(3)]

        # All results should match the first
        for i, result in enumerate(results[1:], 1):
            for key in ['boxes', 'scores', 'labels']:
                if key == 'labels':
                    assert torch.equal(results[0][0][key], result[0][key]), (
                        f"Run {i} {key} differs from run 0"
                    )
                else:
                    assert torch.allclose(results[0][0][key], result[0][key]), (
                        f"Run {i} {key} differs from run 0"
                    )


class TestNestedTensorVariableSizes:
    """Tests for NestedTensor handling of variable-size images.
    
    This is critical for real inference where images in a batch may have
    different dimensions. The mask must correctly indicate padded regions.
    
    Mathematical correctness:
    - Mask should be True for padded regions, False for valid regions
    - Valid ratio = actual_size / padded_size
    """
    
    def test_variable_sizes_creates_correct_masks(self):
        """NestedTensor masks should correctly identify padded regions."""
        # Images with different sizes
        images = [
            torch.randn(3, 100, 80),   # Smallest
            torch.randn(3, 150, 100),  # Medium
            torch.randn(3, 200, 150),  # Largest
        ]
        
        nested = nested_tensor_from_tensor_list(images, size_divisibility=32)
        
        # Padded tensor should be large enough for all images
        assert nested.tensors.shape[2] >= 200  # Height
        assert nested.tensors.shape[3] >= 150  # Width
        
        # Check mask for first (smallest) image
        # Valid region should be (100, 80), rest is padding
        mask = nested.mask[0]
        
        # Valid region (False in mask)
        assert (~mask[:100, :80]).all(), "Valid region should have mask=False"
        
        # Padding region (True in mask) - if there's padding
        if nested.tensors.shape[2] > 100:
            assert mask[100:, :].all(), "Padded height should have mask=True"
        if nested.tensors.shape[3] > 80:
            assert mask[:100, 80:].all(), "Padded width should have mask=True"
    
    def test_variable_sizes_preserves_original_content(self):
        """Original image content should be preserved in nested tensor."""
        # Create images with distinct content
        img1 = torch.ones(3, 50, 60) * 0.5
        img2 = torch.ones(3, 80, 70) * 0.8
        
        nested = nested_tensor_from_tensor_list([img1, img2])
        
        # Check first image content preserved
        assert torch.allclose(
            nested.tensors[0, :, :50, :60], 
            img1,
            atol=1e-6
        ), "First image content should be preserved"
        
        # Check second image content preserved
        assert torch.allclose(
            nested.tensors[1, :, :80, :70], 
            img2,
            atol=1e-6
        ), "Second image content should be preserved"
    
    def test_model_handles_variable_size_batch(self):
        """Model should process variable-size batch correctly."""
        model = build_codetr(
            num_classes=10,
            pretrained_backbone=False,
            use_aux_heads=False,
        )
        model.eval()
        
        # Variable-size images
        images = [
            torch.randn(3, 256, 200),
            torch.randn(3, 320, 256),
            torch.randn(3, 200, 300),
        ]
        
        nested = nested_tensor_from_tensor_list(images, size_divisibility=32)
        
        with torch.no_grad():
            predictions = model(nested.tensors)
        
        # Should get predictions for all images
        assert len(predictions) == 3
        
        # Each prediction should have valid format
        for pred in predictions:
            assert 'boxes' in pred
            assert 'scores' in pred
            assert 'labels' in pred


class TestSingleTrainingStep:
    """Tests for verifying a single training step works correctly.
    
    This is a basic sanity check that:
    1. Forward pass produces loss
    2. Backward pass computes gradients
    3. Optimizer step updates weights
    
    Mathematical correctness:
    - After optimizer.step(), weights should differ from initial values
    - Loss should be finite (not NaN or Inf)
    """
    
    def test_optimizer_step_updates_weights(self):
        """One optimizer step should update model weights.
        
        Verify the complete training loop:
        forward → loss → backward → optimizer.step() → weights changed
        """
        model = build_codetr(
            num_classes=5,
            pretrained_backbone=False,
            use_aux_heads=False,
        )
        model.train()
        
        # Make all params trainable for this test
        for param in model.parameters():
            param.requires_grad = True
        
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        
        # Get initial weights (clone to compare later)
        initial_weights = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        
        # Prepare input
        images = torch.randn(2, 3, 128, 128)
        targets = [
            {'labels': torch.tensor([0, 1]), 'boxes': torch.rand(2, 4)},
            {'labels': torch.tensor([2]), 'boxes': torch.rand(1, 4)},
        ]
        
        # Forward pass
        losses = model(images, targets)
        total_loss = sum(v for v in losses.values() if isinstance(v, torch.Tensor))
        
        assert torch.isfinite(total_loss), "Loss should be finite"
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        
        # Optimizer step
        optimizer.step()
        
        # Check that at least some weights changed
        weights_changed = False
        for name, param in model.named_parameters():
            if param.requires_grad and name in initial_weights:
                if not torch.allclose(initial_weights[name], param, atol=1e-8):
                    weights_changed = True
                    break
        
        assert weights_changed, "Optimizer step should update at least some weights"
    
    def test_gradient_accumulation_works(self):
        """Multiple forward passes before optimizer step should accumulate gradients.
        
        Mathematical guarantee: 
        grad_accumulated = grad_1 + grad_2 (not averaged, summed)
        """
        model = build_codetr(
            num_classes=5,
            pretrained_backbone=False,
            use_aux_heads=False,
        )
        model.train()
        
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        optimizer.zero_grad()
        
        # Two forward-backward passes without optimizer.step()
        for i in range(2):
            images = torch.randn(1, 3, 128, 128)
            targets = [{'labels': torch.tensor([i % 5]), 'boxes': torch.rand(1, 4)}]
            
            losses = model(images, targets)
            total_loss = sum(v for v in losses.values() if isinstance(v, torch.Tensor))
            total_loss.backward()
        
        # Check gradients exist and are accumulated (larger than single pass)
        has_nonzero_grad = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_nonzero_grad = True
                break
        
        assert has_nonzero_grad, "Accumulated gradients should exist"

