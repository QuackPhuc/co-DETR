"""Training and evaluation engine for Co-DETR.

This module provides:
    - Trainer: Main training engine with AMP, gradient clipping, checkpointing
    - LR Schedulers: WarmupStepLR, WarmupMultiStepLR, WarmupCosineLR
    - Hooks: CheckpointHook, LoggingHook, EvalHook
    - Evaluator: Detection evaluation with mAP metrics

Example:
    >>> from codetr.engine import Trainer, build_trainer
    >>> trainer = Trainer(model, train_loader, config=config)
    >>> trainer.train(num_epochs=12)
"""

from .trainer import Trainer, build_trainer
from .lr_scheduler import (
    WarmupStepLR,
    WarmupMultiStepLR,
    WarmupCosineLR,
    build_lr_scheduler,
)
from .hooks import (
    Hook,
    CheckpointHook,
    LoggingHook,
    EvalHook,
)
from .evaluator import (
    DetectionEvaluator,
    build_evaluator,
    compute_ap,
    compute_iou_matrix,
)

__all__ = [
    # Trainer
    "Trainer",
    "build_trainer",
    # LR Schedulers
    "WarmupStepLR",
    "WarmupMultiStepLR",
    "WarmupCosineLR",
    "build_lr_scheduler",
    # Hooks
    "Hook",
    "CheckpointHook",
    "LoggingHook",
    "EvalHook",
    # Evaluator
    "DetectionEvaluator",
    "build_evaluator",
    "compute_ap",
    "compute_iou_matrix",
]

