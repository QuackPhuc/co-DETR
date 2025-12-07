"""
Tests for Learning Rate Schedulers.

This module tests the warmup learning rate schedulers used in Co-DETR training:
- WarmupStepLR
- WarmupMultiStepLR  
- WarmupCosineLR
- build_lr_scheduler factory function
"""

import pytest
import torch
import torch.nn as nn
import math

from codetr.engine.lr_scheduler import (
    WarmupStepLR,
    WarmupMultiStepLR,
    WarmupCosineLR,
    build_lr_scheduler,
)


class TestWarmupStepLR:
    """Tests for WarmupStepLR scheduler."""
    
    @pytest.fixture
    def optimizer(self):
        """Create a simple optimizer for testing."""
        model = nn.Linear(10, 10)
        return torch.optim.SGD(model.parameters(), lr=0.1)
    
    def test_warmup_phase_increases_lr(self, optimizer):
        """LR should linearly increase during warmup phase."""
        scheduler = WarmupStepLR(
            optimizer,
            step_size=10,
            gamma=0.1,
            warmup_epochs=5,
            warmup_lr_ratio=0.001,
        )
        
        base_lr = 0.1
        warmup_start_lr = base_lr * 0.001  # warmup_lr_ratio
        
        lrs = []
        for epoch in range(5):  # Warmup epochs only
            lrs.append(scheduler.get_lr()[0])
            scheduler.step()
        
        # LR should monotonically increase during warmup
        for i in range(1, len(lrs)):
            assert lrs[i] > lrs[i-1], \
                f"LR should increase during warmup: {lrs[i-1]} -> {lrs[i]}"
        
        # First LR should be close to warmup_start_lr
        assert lrs[0] == pytest.approx(warmup_start_lr, rel=0.1)
    
    def test_step_decay_after_warmup(self, optimizer):
        """LR should decay at step milestones after warmup."""
        scheduler = WarmupStepLR(
            optimizer,
            step_size=5,
            gamma=0.1,
            warmup_epochs=2,
            warmup_lr_ratio=0.001,
        )
        
        # Run through warmup
        for _ in range(2):
            scheduler.step()
        
        # Get LR at epoch 2 (end of warmup)
        lr_post_warmup = scheduler.get_lr()[0]
        
        # Step until first decay (epoch 2 + 5 = 7)
        for _ in range(5):
            scheduler.step()
        
        lr_after_decay = scheduler.get_lr()[0]
        
        # LR should have decayed by gamma
        expected_lr = lr_post_warmup * 0.1
        assert lr_after_decay == pytest.approx(expected_lr, rel=0.01)
    
    def test_no_warmup(self, optimizer):
        """Scheduler should work without warmup."""
        scheduler = WarmupStepLR(
            optimizer,
            step_size=5,
            gamma=0.1,
            warmup_epochs=0,  # No warmup
        )
        
        base_lr = 0.1
        
        # First epoch should be base LR
        assert scheduler.get_lr()[0] == pytest.approx(base_lr)
        
        # Step to decay point
        for _ in range(5):
            scheduler.step()
        
        # After 5 epochs, should decay
        assert scheduler.get_lr()[0] == pytest.approx(base_lr * 0.1)


class TestWarmupMultiStepLR:
    """Tests for WarmupMultiStepLR scheduler."""
    
    @pytest.fixture
    def optimizer(self):
        """Create optimizer for testing."""
        model = nn.Linear(10, 10)
        return torch.optim.SGD(model.parameters(), lr=0.1)
    
    def test_warmup_phase(self, optimizer):
        """LR should increase during warmup."""
        scheduler = WarmupMultiStepLR(
            optimizer,
            milestones=[8, 11],
            gamma=0.1,
            warmup_epochs=3,
            warmup_lr_ratio=0.001,
        )
        
        lrs = []
        for _ in range(3):
            lrs.append(scheduler.get_lr()[0])
            scheduler.step()
        
        # Should increase during warmup
        for i in range(1, len(lrs)):
            assert lrs[i] > lrs[i-1]
    
    def test_multi_milestone_decay(self, optimizer):
        """LR should decay at each milestone."""
        scheduler = WarmupMultiStepLR(
            optimizer,
            milestones=[3, 6],
            gamma=0.1,
            warmup_epochs=0,  # No warmup for simpler test
        )
        
        base_lr = 0.1
        lr_history = []
        
        for epoch in range(8):
            lr_history.append(scheduler.get_lr()[0])
            scheduler.step()
        
        # Before first milestone (epochs 0-2): base_lr
        assert lr_history[0] == pytest.approx(base_lr)
        assert lr_history[2] == pytest.approx(base_lr)
        
        # After first milestone (epoch 3+): base_lr * 0.1
        assert lr_history[3] == pytest.approx(base_lr * 0.1)
        
        # After second milestone (epoch 6+): base_lr * 0.1 * 0.1
        assert lr_history[6] == pytest.approx(base_lr * 0.01)
    
    def test_milestones_sorted(self, optimizer):
        """Milestones should be automatically sorted."""
        scheduler = WarmupMultiStepLR(
            optimizer,
            milestones=[10, 5, 8],  # Unsorted
            gamma=0.1,
        )
        
        assert scheduler.milestones == [5, 8, 10]


class TestWarmupCosineLR:
    """Tests for WarmupCosineLR scheduler."""
    
    @pytest.fixture
    def optimizer(self):
        """Create optimizer for testing."""
        model = nn.Linear(10, 10)
        return torch.optim.SGD(model.parameters(), lr=0.1)
    
    def test_warmup_phase(self, optimizer):
        """LR should increase during warmup."""
        scheduler = WarmupCosineLR(
            optimizer,
            max_epochs=20,
            warmup_epochs=5,
            warmup_lr_ratio=0.001,
        )
        
        lrs = []
        for _ in range(5):
            lrs.append(scheduler.get_lr()[0])
            scheduler.step()
        
        # Should increase during warmup
        for i in range(1, len(lrs)):
            assert lrs[i] > lrs[i-1]
    
    def test_cosine_decay_after_warmup(self, optimizer):
        """LR should follow cosine curve after warmup."""
        scheduler = WarmupCosineLR(
            optimizer,
            max_epochs=12,
            warmup_epochs=2,
            warmup_lr_ratio=0.001,
            min_lr_ratio=0.0,
        )
        
        base_lr = 0.1
        
        # Skip warmup
        for _ in range(2):
            scheduler.step()
        
        # After warmup, should be at max LR
        lr_at_warmup_end = scheduler.get_lr()[0]
        assert lr_at_warmup_end == pytest.approx(base_lr, rel=0.05)
        
        # At end of training, should approach min_lr
        for _ in range(10):  # max_epochs - warmup_epochs
            scheduler.step()
        
        lr_at_end = scheduler.get_lr()[0]
        # Should be close to min_lr (0)
        assert lr_at_end < base_lr * 0.1
    
    def test_cosine_midpoint(self, optimizer):
        """At midpoint of cosine phase, LR should be ~50% of max."""
        scheduler = WarmupCosineLR(
            optimizer,
            max_epochs=10,
            warmup_epochs=0,  # No warmup for simpler test
            min_lr_ratio=0.0,
        )
        
        base_lr = 0.1
        
        # At epoch 0 (start), LR = base_lr
        assert scheduler.get_lr()[0] == pytest.approx(base_lr)
        
        # Step to midpoint
        for _ in range(5):
            scheduler.step()
        
        lr_midpoint = scheduler.get_lr()[0]
        # Cosine at pi/2 should give factor of 0.5
        expected = base_lr * 0.5
        assert lr_midpoint == pytest.approx(expected, rel=0.1)
    
    def test_min_lr_ratio(self, optimizer):
        """LR should not go below min_lr_ratio * base_lr."""
        scheduler = WarmupCosineLR(
            optimizer,
            max_epochs=10,
            warmup_epochs=0,
            min_lr_ratio=0.1,  # Minimum 10% of base
        )
        
        base_lr = 0.1
        
        # Run to end
        for _ in range(20):  # Well past max_epochs
            scheduler.step()
        
        lr_final = scheduler.get_lr()[0]
        # Should be at or above min_lr_ratio * base_lr
        assert lr_final >= base_lr * 0.1 - 1e-6


class TestBuildLRScheduler:
    """Tests for build_lr_scheduler factory function."""
    
    @pytest.fixture
    def optimizer(self):
        """Create optimizer for testing."""
        model = nn.Linear(10, 10)
        return torch.optim.SGD(model.parameters(), lr=0.1)
    
    def test_build_step_scheduler(self, optimizer):
        """Factory should build WarmupStepLR for 'step' type."""
        scheduler = build_lr_scheduler(
            optimizer,
            scheduler_type='step',
            max_epochs=12,
            step_size=8,
            gamma=0.1,
            warmup_epochs=1,
        )
        
        assert isinstance(scheduler, WarmupStepLR)
        assert scheduler.step_size == 8
        assert scheduler.gamma == 0.1
    
    def test_build_multistep_scheduler(self, optimizer):
        """Factory should build WarmupMultiStepLR for 'multistep' type."""
        scheduler = build_lr_scheduler(
            optimizer,
            scheduler_type='multistep',
            max_epochs=12,
            milestones=[8, 11],
            gamma=0.1,
        )
        
        assert isinstance(scheduler, WarmupMultiStepLR)
        assert scheduler.milestones == [8, 11]
    
    def test_build_cosine_scheduler(self, optimizer):
        """Factory should build WarmupCosineLR for 'cosine' type."""
        scheduler = build_lr_scheduler(
            optimizer,
            scheduler_type='cosine',
            max_epochs=12,
            warmup_epochs=1,
        )
        
        assert isinstance(scheduler, WarmupCosineLR)
        assert scheduler.max_epochs == 12
    
    def test_case_insensitive_type(self, optimizer):
        """Scheduler type should be case-insensitive."""
        scheduler1 = build_lr_scheduler(optimizer, 'STEP', max_epochs=12, step_size=8)
        scheduler2 = build_lr_scheduler(optimizer, 'Step', max_epochs=12, step_size=8)
        scheduler3 = build_lr_scheduler(optimizer, 'step', max_epochs=12, step_size=8)
        
        assert all(isinstance(s, WarmupStepLR) for s in [scheduler1, scheduler2, scheduler3])
    
    def test_unknown_type_raises_error(self, optimizer):
        """Factory should raise ValueError for unknown scheduler types."""
        with pytest.raises(ValueError) as excinfo:
            build_lr_scheduler(
                optimizer,
                scheduler_type='unknown',
                max_epochs=12,
            )
        
        assert 'Unknown scheduler type' in str(excinfo.value)
    
    def test_default_step_size(self, optimizer):
        """Step scheduler should use default step_size if not provided."""
        scheduler = build_lr_scheduler(
            optimizer,
            scheduler_type='step',
            max_epochs=12,
            # step_size not provided
        )
        
        # Default is max_epochs - 1
        assert scheduler.step_size == 11
    
    def test_default_milestones(self, optimizer):
        """MultiStep scheduler should use default milestones if not provided."""
        scheduler = build_lr_scheduler(
            optimizer,
            scheduler_type='multistep',
            max_epochs=20,
            # milestones not provided
        )
        
        # Default is [int(0.7 * max_epochs), int(0.9 * max_epochs)]
        assert scheduler.milestones == [14, 18]


class TestLRSchedulerMultiParamGroups:
    """Tests for schedulers with multiple parameter groups."""
    
    @pytest.fixture
    def multi_param_optimizer(self):
        """Create optimizer with multiple param groups."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.Linear(20, 10),
        )
        return torch.optim.SGD([
            {'params': model[0].parameters(), 'lr': 0.1},
            {'params': model[1].parameters(), 'lr': 0.01},
        ])
    
    def test_warmup_step_multiple_groups(self, multi_param_optimizer):
        """WarmupStepLR should handle multiple param groups."""
        scheduler = WarmupStepLR(
            multi_param_optimizer,
            step_size=5,
            gamma=0.1,
            warmup_epochs=2,
            warmup_lr_ratio=0.001,
        )
        
        lrs = scheduler.get_lr()
        
        # Should return list with one LR per param group
        assert len(lrs) == 2
        # LRs should be different (based on different base_lrs)
        assert lrs[0] != lrs[1]
    
    def test_cosine_multiple_groups(self, multi_param_optimizer):
        """WarmupCosineLR should handle multiple param groups."""
        scheduler = WarmupCosineLR(
            multi_param_optimizer,
            max_epochs=10,
            warmup_epochs=2,
        )
        
        # Step through some epochs
        for _ in range(5):
            scheduler.step()
        
        lrs = scheduler.get_lr()
        
        # Should maintain relative ratio between groups
        assert len(lrs) == 2
        # Ratio should be preserved (0.1 / 0.01 = 10)
        ratio = lrs[0] / lrs[1]
        assert ratio == pytest.approx(10.0, rel=0.01)
