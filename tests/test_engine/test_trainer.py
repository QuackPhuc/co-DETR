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


class TestTrainerBasic:
    """Basic tests for Trainer class."""
    
    def test_trainer_creation(self):
        """Test Trainer can be created."""
        model = MockModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            device=torch.device('cpu'),
        )
        
        assert trainer.model is model
        assert trainer.optimizer is optimizer
    
    def test_trainer_single_step(self):
        """Test single training step updates parameters."""
        model = MockModel()
        initial_params = model.linear.weight.clone()
        
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        trainer = Trainer(
            model=model,
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
        
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            device=torch.device('cpu'),
            max_grad_norm=0.1,
        )
        
        # Create large gradient
        x = torch.randn(2, 10) * 1000
        outputs = model(x, targets={'dummy': 1})
        loss = outputs['loss']
        
        optimizer.zero_grad()
        loss.backward()
        
        # Before clipping, gradient norm may be large
        grad_norm_before = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
        
        # After clipping
        actual_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), trainer.max_grad_norm)
        
        # Clipped norm should be <= max_grad_norm
        assert actual_norm <= trainer.max_grad_norm + 1e-6


class TestTrainerCheckpointing:
    """Tests for checkpoint save/load functionality."""
    
    def test_save_checkpoint(self):
        """Test checkpoint saving."""
        model = MockModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            device=torch.device('cpu'),
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pth"
            
            trainer.save_checkpoint(checkpoint_path, epoch=5)
            
            assert checkpoint_path.exists()
            
            # Verify checkpoint contains expected keys
            checkpoint = torch.load(checkpoint_path)
            assert 'model_state_dict' in checkpoint
            assert 'optimizer_state_dict' in checkpoint
            assert 'epoch' in checkpoint
            assert checkpoint['epoch'] == 5
    
    def test_load_checkpoint(self):
        """Test checkpoint loading."""
        model = MockModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            device=torch.device('cpu'),
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pth"
            
            # Save initial state
            trainer.save_checkpoint(checkpoint_path, epoch=10)
            
            # Modify model
            with torch.no_grad():
                model.linear.weight.fill_(0.0)
            
            # Load checkpoint
            epoch = trainer.load_checkpoint(checkpoint_path)
            
            assert epoch == 10
            # Weights should be restored (not all zeros)
            assert not torch.allclose(model.linear.weight, torch.zeros_like(model.linear.weight))


class TestTrainerAMP:
    """Tests for Automatic Mixed Precision support."""
    
    def test_amp_enabled(self):
        """Test AMP can be enabled."""
        model = MockModel()
        optimizer = torch.optim.Adam(model.parameters())
        
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            device=torch.device('cpu'),
            use_amp=True,
        )
        
        assert trainer.use_amp == True
        assert trainer.scaler is not None
    
    def test_amp_disabled(self):
        """Test AMP can be disabled."""
        model = MockModel()
        optimizer = torch.optim.Adam(model.parameters())
        
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            device=torch.device('cpu'),
            use_amp=False,
        )
        
        assert trainer.use_amp == False
