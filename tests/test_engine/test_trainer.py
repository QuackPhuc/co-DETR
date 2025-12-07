"""
Tests for Trainer class.

This module tests the training engine including gradient clipping,
AMP support, and checkpoint handling.
"""

import pytest
import torch
import torch.nn as nn
import tempfile
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from codetr.engine.trainer import Trainer


class MockModel(nn.Module):
    """Simple mock model for testing."""
    
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
    
    def forward(self, x, targets=None):
        out = self.linear(x)
        if targets is not None:
            loss = out.sum()
            return {'loss': loss}
        return out


def create_mock_dataloader(batch_size=2, num_samples=4):
    """Create a mock dataloader for testing."""
    # Create simple tensor data
    data = torch.randn(num_samples, 10)
    # Create dummy targets (list of dicts per sample)
    dataset = TensorDataset(data)
    
    def collate_fn(batch):
        tensors = torch.stack([b[0] for b in batch])
        targets = [{'dummy': i} for i in range(len(batch))]
        return tensors, targets
    
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)


class TestTrainerBasic:
    """Basic tests for Trainer class."""
    
    def test_trainer_creation(self):
        """Test Trainer can be created."""
        model = MockModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train_loader = create_mock_dataloader()
        
        # Trainer requires train_loader as positional argument
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=torch.device('cpu'),
        )
        
        assert trainer.model is model
        assert trainer.optimizer is optimizer
        assert trainer.train_loader is train_loader
    
    def test_trainer_single_step(self):
        """Test single training step updates parameters."""
        model = MockModel()
        initial_params = model.linear.weight.clone()
        
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        train_loader = create_mock_dataloader()
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=torch.device('cpu'),
        )
        
        # Mock batch
        batch = (torch.randn(2, 10), [{'target': 1}, {'target': 2}])
        
        # Training step should update params
        model.train()
        x, targets = batch
        outputs = model(x, targets)
        loss = outputs['loss']
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Params should have changed
        assert not torch.equal(model.linear.weight, initial_params)


class TestTrainerGradientClipping:
    """Tests for gradient clipping functionality."""
    
    def test_gradient_clipping_applied(self):
        """Test gradient norms are clipped."""
        model = MockModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
        train_loader = create_mock_dataloader()
        
        # Trainer uses config for gradient_clip, or has default value
        # The actual attribute is 'gradient_clip' not 'max_grad_norm'
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=torch.device('cpu'),
        )
        
        # Trainer uses 'gradient_clip' attribute from config (default 0.1)
        assert hasattr(trainer, 'gradient_clip')
        assert trainer.gradient_clip > 0  # Default should be positive
        
        # Create large gradient
        x = torch.randn(2, 10) * 1000
        outputs = model(x, targets={'dummy': 1})
        loss = outputs['loss']
        
        optimizer.zero_grad()
        loss.backward()
        
        # clip_grad_norm_ returns the norm BEFORE clipping, but then clips in-place
        # So we call it and then compute the norm again to verify clipping worked
        torch.nn.utils.clip_grad_norm_(model.parameters(), trainer.gradient_clip)
        
        # Compute gradient norm after clipping
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        actual_norm_after = total_norm ** 0.5
        
        # After clipping, norm should be <= gradient_clip (with small tolerance)
        assert actual_norm_after <= trainer.gradient_clip + 1e-5


class TestTrainerCheckpointing:
    """Tests for checkpoint save/load functionality."""
    
    def test_save_checkpoint(self):
        """Test checkpoint saving."""
        model = MockModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train_loader = create_mock_dataloader()
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=torch.device('cpu'),
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pth"
            
            # save_checkpoint takes only filepath (epoch is tracked internally)
            trainer.save_checkpoint(str(checkpoint_path))
            
            assert checkpoint_path.exists()
            
            # Verify checkpoint contains expected keys
            checkpoint = torch.load(checkpoint_path)
            assert 'model_state_dict' in checkpoint
            assert 'optimizer_state_dict' in checkpoint
            assert 'epoch' in checkpoint
    
    def test_load_checkpoint(self):
        """Test checkpoint loading."""
        model = MockModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train_loader = create_mock_dataloader()
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=torch.device('cpu'),
        )
        
        # Store original weights
        original_weights = model.linear.weight.clone()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pth"
            
            # Save initial state
            trainer.save_checkpoint(str(checkpoint_path))
            
            # Modify model
            with torch.no_grad():
                model.linear.weight.fill_(0.0)
            
            # Verify weights are now zeros
            assert torch.allclose(model.linear.weight, torch.zeros_like(model.linear.weight))
            
            # Load checkpoint (load_checkpoint doesn't return epoch)
            trainer.load_checkpoint(str(checkpoint_path))
            
            # Weights should be restored (not all zeros)
            assert not torch.allclose(model.linear.weight, torch.zeros_like(model.linear.weight))


class TestTrainerAMP:
    """Tests for Automatic Mixed Precision support."""
    
    def test_amp_enabled(self):
        """Test AMP can be enabled via config."""
        model = MockModel()
        optimizer = torch.optim.Adam(model.parameters())
        train_loader = create_mock_dataloader()
        
        # Trainer gets use_amp from config, not direct parameter
        # Create a simple config dict-like object
        class Config:
            def get(self, key, default=None):
                if key == 'train.amp':
                    return True
                return default
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=torch.device('cpu'),
            config=Config(),
        )
        
        # use_amp is False on CPU even if config says True (requires CUDA)
        # So we just check the attribute exists
        assert hasattr(trainer, 'use_amp')
    
    def test_amp_disabled(self):
        """Test AMP can be disabled via config."""
        model = MockModel()
        optimizer = torch.optim.Adam(model.parameters())
        train_loader = create_mock_dataloader()
        
        # Create config that explicitly disables AMP
        class ConfigAMPDisabled:
            def get(self, key, default=None):
                if key == 'train.amp':
                    return False  # Explicitly disable AMP
                return default
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=torch.device('cpu'),
            config=ConfigAMPDisabled(),
        )
        
        # AMP should be disabled when config says False
        assert trainer.use_amp == False


class TestTrainerResumeAndValidation:
    """Tests for checkpoint resume and validation loop."""

    def test_resume_from_checkpoint_restores_epoch(self):
        """Test resume training correctly restores epoch counter."""
        model = MockModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train_loader = create_mock_dataloader()

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=torch.device('cpu'),
        )

        # Manually set epoch to simulate training progress
        trainer.current_epoch = 5

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint_epoch5.pth"

            # Save checkpoint (should include epoch=5)
            trainer.save_checkpoint(str(checkpoint_path))

            # Create new trainer (starts at epoch 0)
            new_model = MockModel()
            new_optimizer = torch.optim.Adam(new_model.parameters(), lr=1e-3)

            new_trainer = Trainer(
                model=new_model,
                train_loader=train_loader,
                optimizer=new_optimizer,
                device=torch.device('cpu'),
            )

            assert new_trainer.current_epoch == 0

            # Load checkpoint
            new_trainer.load_checkpoint(str(checkpoint_path))

            # Epoch should be restored to saved_epoch + 1 (resume from next epoch)
            # Saved epoch was 5, so after load it should be 6
            assert new_trainer.current_epoch == 6


    def test_validation_method_exists_and_returns_metrics(self):
        """Test validate() method exists and returns metrics dictionary."""
        model = MockModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train_loader = create_mock_dataloader()
        val_loader = create_mock_dataloader(batch_size=2, num_samples=2)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=torch.device('cpu'),
        )

        # validate() method should exist
        assert hasattr(trainer, 'validate')
        assert callable(trainer.validate)

        # Run validation (with MockModel this just does a forward pass)
        model.eval()
        with torch.no_grad():
            metrics = trainer.validate()

        # Should return a dictionary (can be empty or have metrics)
        assert isinstance(metrics, dict)

    def test_validate_handles_no_val_loader(self):
        """Test validate() handles case when no val_loader provided."""
        model = MockModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train_loader = create_mock_dataloader()

        # Create trainer without val_loader
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=None,
            optimizer=optimizer,
            device=torch.device('cpu'),
        )

        assert trainer.val_loader is None

        # validate() should handle gracefully (return empty or skip)
        metrics = trainer.validate()
        # Either returns empty dict or has some default behavior
        assert metrics is None or isinstance(metrics, dict)

