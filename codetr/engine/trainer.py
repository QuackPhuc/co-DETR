"""Training engine for Co-DETR.

This module provides the main Trainer class for end-to-end training with:
    - Mixed precision (AMP) support
    - Gradient clipping
    - Learning rate scheduling with warmup
    - Checkpoint save/resume
    - Distributed training support
    - Modular hooks for extensibility

Example:
    >>> from codetr.engine import Trainer
    >>> from codetr.configs import Config
    >>> 
    >>> config = Config.from_file("config.yaml")
    >>> trainer = Trainer(model, train_loader, val_loader, config)
    >>> trainer.train(num_epochs=12)
"""

import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import nn, Tensor
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .hooks import Hook, CheckpointHook, LoggingHook
from .lr_scheduler import build_lr_scheduler, WarmupStepLR

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("codetr.trainer")


class Trainer:
    """Main training engine for Co-DETR.
    
    Handles the complete training loop including forward/backward passes,
    optimizer updates, learning rate scheduling, checkpointing, and logging.
    
    Args:
        model: The CoDETR model to train.
        train_loader: DataLoader for training data.
        val_loader: Optional DataLoader for validation.
        optimizer: PyTorch optimizer. If None, builds from config.
        lr_scheduler: LR scheduler. If None, builds from config.
        config: Configuration object with training parameters.
        device: Device to train on.
        hooks: List of training hooks.
        resume_from: Path to checkpoint to resume from.
        
    Attributes:
        model: The model being trained.
        optimizer: The optimizer.
        lr_scheduler: The learning rate scheduler.
        current_epoch: Current training epoch.
        global_step: Global iteration counter.
        metrics: Dictionary of tracked metrics.
        
    Example:
        >>> trainer = Trainer(
        ...     model=build_codetr(num_classes=80),
        ...     train_loader=train_loader,
        ...     config=config,
        ... )
        >>> trainer.train(num_epochs=12)
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[Optimizer] = None,
        lr_scheduler: Optional[Any] = None,
        config: Optional[Any] = None,
        device: Optional[torch.device] = None,
        hooks: Optional[List[Hook]] = None,
        resume_from: Optional[str] = None,
    ):
        # Set device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        
        # Move model to device
        self.model = model.to(self.device)
        
        # Store config
        self.config = config
        
        # DataLoaders
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.iters_per_epoch = len(train_loader)
        
        # Training state - must be initialized before building scheduler
        self.current_epoch = 0
        self.global_step = 0
        self.num_epochs = self._get_config("train.epochs", 12)
        self.metrics: Dict[str, float] = {}
        
        # Build optimizer if not provided
        if optimizer is None:
            self.optimizer = self._build_optimizer()
        else:
            self.optimizer = optimizer
        
        # Build LR scheduler if not provided (after num_epochs is set)
        if lr_scheduler is None:
            self.lr_scheduler = self._build_lr_scheduler()
        else:
            self.lr_scheduler = lr_scheduler
        
        # Mixed precision
        self.use_amp = self._get_config("train.amp", True) and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None
        
        # Gradient clipping
        self.gradient_clip = self._get_config("train.gradient_clip", 0.1)
        
        # Hooks
        self.hooks = hooks if hooks is not None else self._default_hooks()
        
        # Resume from checkpoint
        if resume_from:
            self.load_checkpoint(resume_from)
    
    def _get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value with fallback to default."""
        if self.config is None:
            return default
        return self.config.get(key, default)
    
    def _build_optimizer(self) -> Optimizer:
        """Build optimizer from configuration."""
        lr = self._get_config("train.optimizer.lr", 0.0002)
        weight_decay = self._get_config("train.optimizer.weight_decay", 0.0001)
        backbone_lr_mult = self._get_config("train.optimizer.backbone_lr_mult", 0.1)
        
        # Separate backbone and other parameters
        backbone_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "backbone" in name:
                backbone_params.append(param)
            else:
                other_params.append(param)
        
        param_groups = [
            {"params": other_params, "lr": lr},
            {"params": backbone_params, "lr": lr * backbone_lr_mult},
        ]
        
        optimizer_type = self._get_config("train.optimizer.type", "AdamW")
        
        if optimizer_type.lower() == "adamw":
            return torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)
        elif optimizer_type.lower() == "sgd":
            return torch.optim.SGD(
                param_groups, lr=lr, weight_decay=weight_decay, momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    def _build_lr_scheduler(self) -> Any:
        """Build LR scheduler from configuration."""
        scheduler_type = self._get_config("train.lr_scheduler.type", "step")
        step_size = self._get_config("train.lr_scheduler.step_size", 11)
        gamma = self._get_config("train.lr_scheduler.gamma", 0.1)
        warmup_epochs = self._get_config("train.lr_scheduler.warmup_epochs", 0)
        warmup_lr_ratio = self._get_config("train.lr_scheduler.warmup_lr_ratio", 0.001)
        
        return build_lr_scheduler(
            self.optimizer,
            scheduler_type=scheduler_type,
            max_epochs=self.num_epochs,
            step_size=step_size,
            gamma=gamma,
            warmup_epochs=warmup_epochs,
            warmup_lr_ratio=warmup_lr_ratio,
        )
    
    def _default_hooks(self) -> List[Hook]:
        """Create default training hooks."""
        log_interval = self._get_config("train.log_interval", 50)
        return [
            LoggingHook(log_interval=log_interval),
        ]
    
    def train(self, num_epochs: Optional[int] = None) -> Dict[str, float]:
        """Run full training loop.
        
        Args:
            num_epochs: Number of epochs to train. Uses config if None.
            
        Returns:
            Dictionary of final metrics.
        """
        if num_epochs is not None:
            self.num_epochs = num_epochs
        
        logger.info(f"Starting training for {self.num_epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"AMP: {self.use_amp}")
        logger.info(f"Gradient clip: {self.gradient_clip}")
        
        # Pre-training hooks
        for hook in self.hooks:
            hook.before_train(self)
        
        try:
            for epoch in range(self.current_epoch, self.num_epochs):
                self.current_epoch = epoch
                
                # Pre-epoch hooks
                for hook in self.hooks:
                    hook.before_epoch(self, epoch)
                
                # Train one epoch
                epoch_metrics = self.train_one_epoch()
                self.metrics.update(epoch_metrics)
                
                # Step LR scheduler
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                
                # Validation
                if self.val_loader is not None:
                    val_metrics = self.validate()
                    self.metrics.update(val_metrics)
                
                # Post-epoch hooks
                for hook in self.hooks:
                    hook.after_epoch(self, epoch)
        
        finally:
            # Post-training hooks
            for hook in self.hooks:
                hook.after_train(self)
        
        logger.info("Training complete!")
        return self.metrics
    
    def train_one_epoch(self) -> Dict[str, float]:
        """Train for one epoch.
        
        Returns:
            Dictionary of epoch metrics.
        """
        self.model.train()
        
        total_loss = 0.0
        loss_components: Dict[str, float] = {}
        num_batches = 0
        
        for iteration, (images, targets) in enumerate(self.train_loader):
            # Pre-iteration hooks
            for hook in self.hooks:
                hook.before_iter(self, iteration)
            
            # Forward and backward
            outputs = self.train_step(images, targets)
            
            # Accumulate losses
            loss = outputs.get("loss", 0.0)
            if isinstance(loss, Tensor):
                loss = loss.item()
            total_loss += loss
            
            # Accumulate component losses
            for key, value in outputs.items():
                if key != "loss":
                    if isinstance(value, Tensor):
                        value = value.item()
                    loss_components[key] = loss_components.get(key, 0.0) + value
            
            num_batches += 1
            self.global_step += 1
            
            # Post-iteration hooks
            for hook in self.hooks:
                hook.after_iter(self, iteration, outputs)
        
        # Compute averages
        avg_loss = total_loss / max(num_batches, 1)
        avg_components = {k: v / max(num_batches, 1) for k, v in loss_components.items()}
        
        return {"train_loss": avg_loss, **avg_components}
    
    def train_step(
        self,
        images: Any,
        targets: List[Dict[str, Tensor]],
    ) -> Dict[str, Tensor]:
        """Execute single training step.
        
        Args:
            images: Input images (NestedTensor or Tensor).
            targets: List of target dictionaries.
            
        Returns:
            Dictionary with loss values.
        """
        # Move data to device
        if hasattr(images, "tensors"):
            # NestedTensor
            images_tensor = images.tensors.to(self.device)
        else:
            images_tensor = images.to(self.device)
        
        targets = [
            {k: v.to(self.device) if isinstance(v, Tensor) else v 
             for k, v in t.items()}
            for t in targets
        ]
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass with optional AMP
        if self.use_amp:
            with autocast():
                losses = self.model(images_tensor, targets)
                if isinstance(losses, dict):
                    loss = sum(losses.values())
                else:
                    loss = losses
                    losses = {"loss": loss}
        else:
            losses = self.model(images_tensor, targets)
            if isinstance(losses, dict):
                loss = sum(losses.values())
            else:
                loss = losses
                losses = {"loss": loss}
        
        # Backward pass
        if self.use_amp:
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            if self.gradient_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip
                )
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip
                )
            
            self.optimizer.step()
        
        # Add total loss to outputs
        outputs = {k: v.detach() for k, v in losses.items()}
        outputs["loss"] = loss.detach()
        
        return outputs
    
    def validate(self) -> Dict[str, float]:
        """Run validation loop.
        
        Returns:
            Dictionary of validation metrics.
        """
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for images, targets in self.val_loader:
                # Move data to device
                if hasattr(images, "tensors"):
                    images_tensor = images.tensors.to(self.device)
                else:
                    images_tensor = images.to(self.device)
                
                targets = [
                    {k: v.to(self.device) if isinstance(v, Tensor) else v 
                     for k, v in t.items()}
                    for t in targets
                ]
                
                # Forward pass
                if self.use_amp:
                    with autocast():
                        losses = self.model(images_tensor, targets)
                else:
                    losses = self.model(images_tensor, targets)
                
                if isinstance(losses, dict):
                    loss = sum(losses.values())
                else:
                    loss = losses
                
                total_loss += loss.item()
                num_batches += 1
        
        self.model.train()
        
        avg_loss = total_loss / max(num_batches, 1)
        return {"val_loss": avg_loss}
    
    def get_current_lr(self) -> float:
        """Get current learning rate.
        
        Returns:
            Current learning rate from first param group.
        """
        return self.optimizer.param_groups[0]["lr"]
    
    def save_checkpoint(self, filepath: str) -> None:
        """Save training checkpoint.
        
        Args:
            filepath: Path to save checkpoint.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Handle DDP model
        model_state = self.model.state_dict()
        if isinstance(self.model, DistributedDataParallel):
            model_state = self.model.module.state_dict()
        
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": model_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": self.metrics,
        }
        
        if self.lr_scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.lr_scheduler.state_dict()
        
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        torch.save(checkpoint, filepath)
        logger.info(f"Saved checkpoint to {filepath}")
    
    def load_checkpoint(self, filepath: str) -> None:
        """Load training checkpoint.
        
        Args:
            filepath: Path to checkpoint file.
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Load model state
        if isinstance(self.model, DistributedDataParallel):
            self.model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Load scheduler state
        if self.lr_scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        # Load scaler state
        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        # Restore training state
        self.current_epoch = checkpoint.get("epoch", 0) + 1  # Resume from next epoch
        self.global_step = checkpoint.get("global_step", 0)
        self.metrics = checkpoint.get("metrics", {})
        
        logger.info(f"Loaded checkpoint from {filepath} (epoch {self.current_epoch})")


def build_trainer(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    config: Optional[Any] = None,
    **kwargs,
) -> Trainer:
    """Build Trainer from model and dataloaders.
    
    Convenience function for creating a Trainer with standard settings.
    
    Args:
        model: The model to train.
        train_loader: Training data loader.
        val_loader: Optional validation loader.
        config: Configuration object.
        **kwargs: Additional arguments for Trainer.
        
    Returns:
        Configured Trainer instance.
    """
    return Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        **kwargs,
    )
