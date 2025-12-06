#!/usr/bin/env python3
"""Training script for Co-DETR object detection.

This script provides a command-line interface for training Co-DETR models
on YOLOv5 format datasets with support for:
    - Single and multi-GPU training (DDP)
    - Mixed precision (AMP)
    - Resume from checkpoint
    - Configuration overrides via CLI
    - TensorBoard logging

Usage:
    Basic training:
        python tools/train.py --config configs/co_deformable_detr_r50_yolo.yaml
    
    With custom data path:
        python tools/train.py --config configs/config.yaml \\
            --opts data.train_root=/path/to/train data.val_root=/path/to/val
    
    Resume training:
        python tools/train.py --config configs/config.yaml --resume checkpoints/latest.pth
    
    Multi-GPU (using torchrun):
        torchrun --nproc_per_node=2 tools/train.py --config configs/config.yaml

Author: Co-DETR PyTorch Team
"""

import argparse
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from codetr.configs import Config
from codetr.data.datasets.yolo_dataset import YOLODataset
from codetr.data.transforms import Compose, ToTensor, Normalize, Resize, RandomHorizontalFlip
from codetr.data.dataloader import build_dataloader, build_val_dataloader
from codetr.engine.trainer import Trainer
from codetr.engine.hooks import CheckpointHook, LoggingHook
from codetr.models.detector import CoDETR
from codetr.utils.distributed import init_distributed_mode, is_main_process, get_rank, get_world_size

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("train")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.
    
    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train Co-DETR object detector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic training
    python tools/train.py --config configs/co_deformable_detr_r50_yolo.yaml
    
    # Override batch size
    python tools/train.py --config configs/config.yaml --opts train.batch_size=4
    
    # Resume training
    python tools/train.py --config configs/config.yaml --resume checkpoints/epoch_5.pth
    
    # Multi-GPU training
    torchrun --nproc_per_node=2 tools/train.py --config configs/config.yaml
        """,
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save checkpoints and logs (default: outputs)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--opts",
        nargs="*",
        default=[],
        help="Override config options (format: key=value, e.g., train.batch_size=4)",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Run evaluation only (no training)",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validation during training",
    )
    
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def parse_opts(opts: List[str]) -> Dict[str, Any]:
    """Parse CLI option overrides.
    
    Args:
        opts: List of "key=value" strings.
        
    Returns:
        Dictionary of parsed overrides.
    """
    overrides = {}
    for opt in opts:
        if "=" not in opt:
            raise ValueError(f"Invalid option format: {opt}. Use key=value format.")
        key, value = opt.split("=", 1)
        
        # Try to parse value as int, float, or bool
        if value.lower() == "true":
            value = True
        elif value.lower() == "false":
            value = False
        elif value.lower() == "none":
            value = None
        else:
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass  # Keep as string
        
        overrides[key] = value
    
    return overrides


def apply_overrides(config: Config, overrides: Dict[str, Any]) -> None:
    """Apply CLI overrides to configuration.
    
    Args:
        config: Configuration object.
        overrides: Dictionary of overrides.
    """
    for key, value in overrides.items():
        config.set(key, value)
        logger.info(f"Override: {key} = {value}")


def build_transforms(config: Config, is_train: bool = True) -> Compose:
    """Build data transformation pipeline.
    
    Args:
        config: Configuration object.
        is_train: Whether to build training transforms.
        
    Returns:
        Composed transform pipeline.
    """
    img_size = config.get("data.img_size", 800)
    max_size = config.get("data.max_size", 1333)
    
    transforms_list = []
    
    if is_train:
        transforms_list.append(RandomHorizontalFlip(p=0.5))
    
    transforms_list.extend([
        Resize(min_size=img_size, max_size=max_size),
        ToTensor(),
        Normalize(),
    ])
    
    return Compose(transforms_list)


def build_datasets(
    config: Config,
    distributed: bool = False,
) -> Tuple[Optional[YOLODataset], Optional[YOLODataset]]:
    """Build training and validation datasets.
    
    Args:
        config: Configuration object.
        distributed: Whether using distributed training.
        
    Returns:
        Tuple of (train_dataset, val_dataset).
    """
    train_root = config.get("data.train_root", "data/train")
    val_root = config.get("data.val_root", "data/val")
    
    train_transforms = build_transforms(config, is_train=True)
    val_transforms = build_transforms(config, is_train=False)
    
    train_dataset = None
    val_dataset = None
    
    # Build training dataset
    if train_root and Path(train_root).exists():
        train_dataset = YOLODataset(
            data_root=train_root,
            split="train",
            transforms=train_transforms,
        )
        if is_main_process():
            logger.info(f"Training dataset: {len(train_dataset)} images")
    else:
        if is_main_process():
            logger.warning(f"Training data not found at: {train_root}")
    
    # Build validation dataset
    if val_root and Path(val_root).exists():
        val_dataset = YOLODataset(
            data_root=val_root,
            split="val",
            transforms=val_transforms,
        )
        if is_main_process():
            logger.info(f"Validation dataset: {len(val_dataset)} images")
    else:
        if is_main_process():
            logger.warning(f"Validation data not found at: {val_root}")
    
    return train_dataset, val_dataset


def build_model(config: Config, device: torch.device) -> torch.nn.Module:
    """Build Co-DETR model from configuration.
    
    Args:
        config: Configuration object.
        device: Device to place model on.
        
    Returns:
        CoDETR model.
    """
    num_classes = config.get("model.num_classes", 80)
    embed_dim = config.get("model.embed_dim", 256)
    num_queries = config.get("model.num_queries", 300)
    num_feature_levels = config.get("model.num_feature_levels", 4)
    num_encoder_layers = config.get("model.num_encoder_layers", 6)
    num_decoder_layers = config.get("model.num_decoder_layers", 6)
    use_rpn = config.get("model.use_rpn", True)
    use_roi = config.get("model.use_roi", True)
    use_atss = config.get("model.use_atss", True)
    use_dn = config.get("model.use_dn", True)
    pretrained = config.get("model.pretrained_backbone", True)
    frozen_stages = config.get("model.frozen_backbone_stages", 1)
    
    model = CoDETR(
        num_classes=num_classes,
        embed_dim=embed_dim,
        num_queries=num_queries,
        num_feature_levels=num_feature_levels,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        use_rpn=use_rpn,
        use_roi=use_roi,
        use_atss=use_atss,
        use_dn=use_dn,
        pretrained_backbone=pretrained,
        frozen_backbone_stages=frozen_stages,
    )
    
    model = model.to(device)
    
    if is_main_process():
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    return model


def setup_distributed() -> Tuple[bool, int, torch.device]:
    """Setup distributed training environment.
    
    Returns:
        Tuple of (distributed, rank, device).
    """
    distributed = False
    rank = 0
    
    # Check if running with torchrun
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        init_distributed_mode()
        distributed = True
        rank = get_rank()
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    return distributed, rank, device


def main() -> None:
    """Main training entry point."""
    args = parse_args()
    
    # Setup distributed training
    distributed, rank, device = setup_distributed()
    
    # Only log on main process
    if not is_main_process():
        logging.getLogger().setLevel(logging.WARNING)
    
    logger.info("=" * 60)
    logger.info("Co-DETR Training Script")
    logger.info("=" * 60)
    
    # Set random seed
    set_seed(args.seed + rank)
    logger.info(f"Random seed: {args.seed + rank}")
    
    # Load configuration
    logger.info(f"Loading config from: {args.config}")
    config = Config.from_file(args.config)
    
    # Apply CLI overrides
    if args.opts:
        overrides = parse_opts(args.opts)
        apply_overrides(config, overrides)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        # Save config to output dir
        config.save(output_dir / "config.yaml")
        logger.info(f"Output directory: {output_dir}")
    
    # Build model
    logger.info("Building model...")
    model = build_model(config, device)
    
    # Wrap model for distributed training
    if distributed:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        logger.info(f"Using DistributedDataParallel on rank {rank}")
    
    # Build datasets
    logger.info("Building datasets...")
    train_dataset, val_dataset = build_datasets(config, distributed)
    
    if train_dataset is None:
        logger.error("No training data found. Please check data.train_root in config.")
        return
    
    # Build dataloaders
    batch_size = config.get("train.batch_size", 2)
    num_workers = config.get("train.num_workers", 4)
    
    train_loader = build_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        distributed=distributed,
    )
    
    val_loader = None
    if val_dataset is not None and not args.no_validate:
        val_loader = build_val_dataloader(
            val_dataset,
            batch_size=1,
            num_workers=num_workers,
            distributed=distributed,
        )
    
    logger.info(f"Training batches per epoch: {len(train_loader)}")
    if val_loader:
        logger.info(f"Validation batches: {len(val_loader)}")
    
    # Setup hooks
    hooks = []
    
    # Checkpoint hook
    checkpoint_interval = config.get("train.checkpoint_interval", 1)
    if is_main_process():
        hooks.append(CheckpointHook(
            save_dir=str(output_dir / "checkpoints"),
            save_interval=checkpoint_interval,
        ))
    
    # Logging hook
    log_interval = config.get("train.log_interval", 50)
    hooks.append(LoggingHook(log_interval=log_interval))
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        hooks=hooks,
        resume_from=args.resume,
    )
    
    # Run training
    if args.eval_only:
        logger.info("Running evaluation only...")
        if val_loader is None:
            logger.error("No validation data found for evaluation.")
            return
        metrics = trainer.validate()
        logger.info(f"Evaluation results: {metrics}")
    else:
        logger.info("Starting training...")
        start_time = time.time()
        
        num_epochs = config.get("train.epochs", 12)
        trainer.train(num_epochs=num_epochs)
        
        elapsed = time.time() - start_time
        logger.info(f"Training complete! Total time: {elapsed/3600:.2f} hours")
        
        # Save final checkpoint
        if is_main_process():
            final_path = output_dir / "checkpoints" / "final.pth"
            trainer.save_checkpoint(str(final_path))
            logger.info(f"Saved final checkpoint to: {final_path}")
    
    # Cleanup distributed
    if distributed:
        dist.destroy_process_group()
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
