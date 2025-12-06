"""Training hooks for Co-DETR.

This module provides modular callbacks for training, including:
    - Checkpoint saving
    - Logging to console and TensorBoard
    - Periodic evaluation

Example:
    >>> hooks = [
    ...     CheckpointHook(save_dir="checkpoints", interval=1),
    ...     LoggingHook(log_interval=50),
    ... ]
    >>> trainer = Trainer(model, dataloader, hooks=hooks)
"""

import logging
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import nn, Tensor

# Optional TensorBoard import
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False


class Hook(ABC):
    """Base class for training hooks.
    
    Hooks provide a modular way to extend training functionality
    without modifying the core training loop.
    """
    
    def before_train(self, trainer: Any) -> None:
        """Called before training starts."""
        pass
    
    def after_train(self, trainer: Any) -> None:
        """Called after training ends."""
        pass
    
    def before_epoch(self, trainer: Any, epoch: int) -> None:
        """Called before each epoch."""
        pass
    
    def after_epoch(self, trainer: Any, epoch: int) -> None:
        """Called after each epoch."""
        pass
    
    def before_iter(self, trainer: Any, iteration: int) -> None:
        """Called before each iteration."""
        pass
    
    def after_iter(self, trainer: Any, iteration: int, outputs: Dict) -> None:
        """Called after each iteration with outputs."""
        pass


class CheckpointHook(Hook):
    """Hook for saving model checkpoints.
    
    Saves checkpoints at regular intervals and tracks best model.
    
    Args:
        save_dir: Directory to save checkpoints.
        interval: Save checkpoint every N epochs.
        save_best: Whether to save best model based on metric.
        metric_key: Metric key to track for best model.
        mode: 'min' or 'max' for metric comparison.
        
    Example:
        >>> hook = CheckpointHook("checkpoints", interval=1, save_best=True)
    """
    
    def __init__(
        self,
        save_dir: str,
        interval: int = 1,
        save_best: bool = True,
        metric_key: str = "val_loss",
        mode: str = "min",
    ):
        self.save_dir = Path(save_dir)
        self.interval = interval
        self.save_best = save_best
        self.metric_key = metric_key
        self.mode = mode
        
        self.best_metric = float("inf") if mode == "min" else float("-inf")
        
        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def after_epoch(self, trainer: Any, epoch: int) -> None:
        """Save checkpoint after epoch if conditions met."""
        if (epoch + 1) % self.interval == 0:
            checkpoint_path = self.save_dir / f"epoch_{epoch + 1}.pth"
            trainer.save_checkpoint(str(checkpoint_path))
        
        # Save best model
        if self.save_best and hasattr(trainer, "metrics"):
            current_metric = trainer.metrics.get(self.metric_key, None)
            if current_metric is not None:
                is_best = (
                    (self.mode == "min" and current_metric < self.best_metric) or
                    (self.mode == "max" and current_metric > self.best_metric)
                )
                if is_best:
                    self.best_metric = current_metric
                    best_path = self.save_dir / "best.pth"
                    trainer.save_checkpoint(str(best_path))
    
    def after_train(self, trainer: Any) -> None:
        """Save final checkpoint after training."""
        final_path = self.save_dir / "final.pth"
        trainer.save_checkpoint(str(final_path))


class LoggingHook(Hook):
    """Hook for logging training progress.
    
    Logs to console and optionally TensorBoard.
    
    Args:
        log_interval: Log every N iterations.
        log_dir: Directory for TensorBoard logs (optional).
        
    Example:
        >>> hook = LoggingHook(log_interval=50, log_dir="logs")
    """
    
    def __init__(
        self,
        log_interval: int = 50,
        log_dir: Optional[str] = None,
    ):
        self.log_interval = log_interval
        self.log_dir = log_dir
        
        self.logger = logging.getLogger("codetr.trainer")
        self.writer = None
        
        self.epoch_start_time = 0.0
        self.iter_start_time = 0.0
        self.running_loss = 0.0
        self.num_iters = 0
    
    def before_train(self, trainer: Any) -> None:
        """Initialize TensorBoard writer if log_dir specified."""
        if self.log_dir and HAS_TENSORBOARD:
            log_path = Path(self.log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(str(log_path))
    
    def after_train(self, trainer: Any) -> None:
        """Close TensorBoard writer."""
        if self.writer:
            self.writer.close()
    
    def before_epoch(self, trainer: Any, epoch: int) -> None:
        """Reset epoch tracking."""
        self.epoch_start_time = time.time()
        self.running_loss = 0.0
        self.num_iters = 0
    
    def after_epoch(self, trainer: Any, epoch: int) -> None:
        """Log epoch summary."""
        epoch_time = time.time() - self.epoch_start_time
        avg_loss = self.running_loss / max(self.num_iters, 1)
        
        lr = trainer.get_current_lr()
        
        self.logger.info(
            f"Epoch [{epoch + 1}/{trainer.num_epochs}] "
            f"Loss: {avg_loss:.4f} "
            f"LR: {lr:.6f} "
            f"Time: {epoch_time:.1f}s"
        )
        
        if self.writer:
            self.writer.add_scalar("train/epoch_loss", avg_loss, epoch + 1)
            self.writer.add_scalar("train/lr", lr, epoch + 1)
    
    def before_iter(self, trainer: Any, iteration: int) -> None:
        """Record iteration start time."""
        self.iter_start_time = time.time()
    
    def after_iter(self, trainer: Any, iteration: int, outputs: Dict) -> None:
        """Log iteration progress."""
        loss = outputs.get("loss", 0.0)
        if isinstance(loss, Tensor):
            loss = loss.item()
        
        self.running_loss += loss
        self.num_iters += 1
        
        # Log at intervals
        if (iteration + 1) % self.log_interval == 0:
            iter_time = time.time() - self.iter_start_time
            avg_loss = self.running_loss / self.num_iters
            
            self.logger.info(
                f"Iter [{iteration + 1}/{trainer.iters_per_epoch}] "
                f"Loss: {loss:.4f} (avg: {avg_loss:.4f}) "
                f"Time: {iter_time:.3f}s"
            )
        
        # TensorBoard logging
        if self.writer:
            global_step = trainer.current_epoch * trainer.iters_per_epoch + iteration
            self.writer.add_scalar("train/iter_loss", loss, global_step)


class EvalHook(Hook):
    """Hook for periodic evaluation.
    
    Runs validation at specified intervals.
    
    Args:
        interval: Evaluate every N epochs.
        
    Example:
        >>> hook = EvalHook(interval=1)
    """
    
    def __init__(self, interval: int = 1):
        self.interval = interval
        self.logger = logging.getLogger("codetr.trainer")
    
    def after_epoch(self, trainer: Any, epoch: int) -> None:
        """Run evaluation after epoch if interval met."""
        if (epoch + 1) % self.interval == 0 and hasattr(trainer, "validate"):
            self.logger.info(f"Running validation at epoch {epoch + 1}")
            metrics = trainer.validate()
            
            if metrics:
                metric_str = " ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
                self.logger.info(f"Validation metrics: {metric_str}")
