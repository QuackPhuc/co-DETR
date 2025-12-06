"""Learning rate schedulers for Co-DETR training.

This module provides custom learning rate schedulers with warmup support
that are commonly used in object detection training.

Example:
    >>> optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002)
    >>> scheduler = WarmupStepLR(
    ...     optimizer,
    ...     step_size=11,
    ...     gamma=0.1,
    ...     warmup_epochs=1,
    ...     warmup_lr_ratio=0.001,
    ... )
"""

import math
from typing import List, Optional

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class WarmupStepLR(_LRScheduler):
    """Step learning rate scheduler with linear warmup.
    
    Implements linear warmup for the first few epochs, then step decay
    at specified milestones.
    
    Args:
        optimizer: Wrapped optimizer.
        step_size: Epoch at which to decay learning rate.
        gamma: Multiplicative factor of learning rate decay.
        warmup_epochs: Number of warmup epochs.
        warmup_lr_ratio: Initial warmup LR as ratio of base LR.
        last_epoch: Index of last epoch.
        
    Example:
        >>> scheduler = WarmupStepLR(optimizer, step_size=11, gamma=0.1)
        >>> for epoch in range(12):
        ...     train(...)
        ...     scheduler.step()
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        step_size: int,
        gamma: float = 0.1,
        warmup_epochs: int = 0,
        warmup_lr_ratio: float = 0.001,
        last_epoch: int = -1,
    ):
        self.step_size = step_size
        self.gamma = gamma
        self.warmup_epochs = warmup_epochs
        self.warmup_lr_ratio = warmup_lr_ratio
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """Compute learning rate for current epoch."""
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / max(self.warmup_epochs, 1)
            warmup_factor = self.warmup_lr_ratio + (1 - self.warmup_lr_ratio) * alpha
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Step decay after warmup
            effective_epoch = self.last_epoch - self.warmup_epochs
            num_decays = effective_epoch // self.step_size
            return [base_lr * (self.gamma ** num_decays) for base_lr in self.base_lrs]


class WarmupMultiStepLR(_LRScheduler):
    """Multi-step learning rate scheduler with linear warmup.
    
    Decays learning rate at multiple milestones, with optional warmup.
    
    Args:
        optimizer: Wrapped optimizer.
        milestones: List of epoch indices at which to decay LR.
        gamma: Multiplicative factor of learning rate decay.
        warmup_epochs: Number of warmup epochs.
        warmup_lr_ratio: Initial warmup LR as ratio of base LR.
        last_epoch: Index of last epoch.
        
    Example:
        >>> scheduler = WarmupMultiStepLR(
        ...     optimizer,
        ...     milestones=[8, 11],
        ...     gamma=0.1,
        ...     warmup_epochs=1,
        ... )
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        milestones: List[int],
        gamma: float = 0.1,
        warmup_epochs: int = 0,
        warmup_lr_ratio: float = 0.001,
        last_epoch: int = -1,
    ):
        self.milestones = sorted(milestones)
        self.gamma = gamma
        self.warmup_epochs = warmup_epochs
        self.warmup_lr_ratio = warmup_lr_ratio
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """Compute learning rate for current epoch."""
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / max(self.warmup_epochs, 1)
            warmup_factor = self.warmup_lr_ratio + (1 - self.warmup_lr_ratio) * alpha
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Multi-step decay after warmup
            effective_epoch = self.last_epoch - self.warmup_epochs
            num_decays = sum(1 for m in self.milestones if effective_epoch >= m)
            return [base_lr * (self.gamma ** num_decays) for base_lr in self.base_lrs]


class WarmupCosineLR(_LRScheduler):
    """Cosine annealing learning rate scheduler with linear warmup.
    
    Implements linear warmup followed by cosine annealing decay.
    
    Args:
        optimizer: Wrapped optimizer.
        max_epochs: Total number of training epochs.
        warmup_epochs: Number of warmup epochs.
        warmup_lr_ratio: Initial warmup LR as ratio of base LR.
        min_lr_ratio: Minimum LR as ratio of base LR.
        last_epoch: Index of last epoch.
        
    Example:
        >>> scheduler = WarmupCosineLR(
        ...     optimizer,
        ...     max_epochs=12,
        ...     warmup_epochs=1,
        ... )
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        max_epochs: int,
        warmup_epochs: int = 0,
        warmup_lr_ratio: float = 0.001,
        min_lr_ratio: float = 0.0,
        last_epoch: int = -1,
    ):
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self.warmup_lr_ratio = warmup_lr_ratio
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """Compute learning rate for current epoch."""
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / max(self.warmup_epochs, 1)
            warmup_factor = self.warmup_lr_ratio + (1 - self.warmup_lr_ratio) * alpha
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine annealing after warmup
            effective_epoch = self.last_epoch - self.warmup_epochs
            total_epochs = self.max_epochs - self.warmup_epochs
            progress = effective_epoch / max(total_epochs, 1)
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            factor = self.min_lr_ratio + (1 - self.min_lr_ratio) * cosine_factor
            return [base_lr * factor for base_lr in self.base_lrs]


def build_lr_scheduler(
    optimizer: Optimizer,
    scheduler_type: str,
    max_epochs: int,
    step_size: Optional[int] = None,
    milestones: Optional[List[int]] = None,
    gamma: float = 0.1,
    warmup_epochs: int = 0,
    warmup_lr_ratio: float = 0.001,
) -> _LRScheduler:
    """Build learning rate scheduler from configuration.
    
    Args:
        optimizer: Wrapped optimizer.
        scheduler_type: Type of scheduler ("step", "multistep", "cosine").
        max_epochs: Total number of training epochs.
        step_size: Step size for step scheduler.
        milestones: Milestones for multi-step scheduler.
        gamma: LR decay factor.
        warmup_epochs: Number of warmup epochs.
        warmup_lr_ratio: Initial warmup LR ratio.
        
    Returns:
        Configured LR scheduler.
        
    Raises:
        ValueError: If scheduler type is unknown.
    """
    scheduler_type = scheduler_type.lower()
    
    if scheduler_type == "step":
        if step_size is None:
            step_size = max_epochs - 1  # Decay at last epoch by default
        return WarmupStepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
            warmup_epochs=warmup_epochs,
            warmup_lr_ratio=warmup_lr_ratio,
        )
    elif scheduler_type == "multistep":
        if milestones is None:
            milestones = [int(0.7 * max_epochs), int(0.9 * max_epochs)]
        return WarmupMultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=gamma,
            warmup_epochs=warmup_epochs,
            warmup_lr_ratio=warmup_lr_ratio,
        )
    elif scheduler_type == "cosine":
        return WarmupCosineLR(
            optimizer,
            max_epochs=max_epochs,
            warmup_epochs=warmup_epochs,
            warmup_lr_ratio=warmup_lr_ratio,
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
