"""DataLoader utilities for detection datasets.

This module provides utilities for building dataloaders with proper
collation for object detection tasks, handling variable-size images
and creating padding masks as required by DETR-style detectors.

Key features:
    - Custom collate function for detection
    - Padding to max batch size with mask generation
    - Support for distributed training
    - NestedTensor output for CoDETR compatibility

Example:
    >>> dataset = YOLODataset(data_root, split="train", transforms=transforms)
    >>> dataloader = build_dataloader(dataset, batch_size=2, shuffle=True)
    >>> for images, targets in dataloader:
    ...     # images is NestedTensor, targets is list of dicts
    ...     pass
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, DistributedSampler, RandomSampler, SequentialSampler

# Import NestedTensor from models.utils.misc
from codetr.models.utils.misc import NestedTensor


def collate_fn(
    batch: List[Tuple[Tensor, Dict[str, Tensor]]]
) -> Tuple[NestedTensor, List[Dict[str, Tensor]]]:
    """Collate function for detection batches.
    
    Handles variable-size images by padding to the maximum size in the batch
    and creating corresponding masks.
    
    Args:
        batch: List of (image, target) tuples from dataset.
            - image: Tensor of shape (3, H_i, W_i)
            - target: Dict with 'boxes', 'labels', etc.
            
    Returns:
        Tuple of:
            - images: NestedTensor with:
                - tensors: (B, 3, H_max, W_max) padded images
                - mask: (B, H_max, W_max) True where padding
            - targets: List of target dicts
            
    Example:
        >>> batch = [(img1, target1), (img2, target2)]
        >>> images, targets = collate_fn(batch)
        >>> print(images.tensors.shape)  # (2, 3, H, W)
    """
    images = []
    targets = []
    
    for image, target in batch:
        images.append(image)
        targets.append(target)
    
    # Find max dimensions
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)
    
    # Pad to divisible size (32)
    divisor = 32
    max_h = ((max_h + divisor - 1) // divisor) * divisor
    max_w = ((max_w + divisor - 1) // divisor) * divisor
    
    batch_size = len(images)
    num_channels = images[0].shape[0]
    
    # Create padded tensor and mask
    padded = torch.zeros(batch_size, num_channels, max_h, max_w, dtype=images[0].dtype)
    mask = torch.ones(batch_size, max_h, max_w, dtype=torch.bool)
    
    for i, img in enumerate(images):
        c, h, w = img.shape
        padded[i, :, :h, :w] = img
        mask[i, :h, :w] = False  # False = valid, True = padding
    
    nested_tensor = NestedTensor(padded, mask)
    
    return nested_tensor, targets


def build_dataloader(
    dataset: Dataset,
    batch_size: int = 2,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
    distributed: bool = False,
) -> DataLoader:
    """Build a DataLoader for detection.
    
    Creates a DataLoader with the custom collate function for detection,
    with support for distributed training.
    
    Args:
        dataset: Dataset to load from.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle data (ignored if distributed).
        num_workers: Number of data loading workers.
        pin_memory: Whether to pin memory for GPU transfer.
        drop_last: Whether to drop the last incomplete batch.
        distributed: Whether to use distributed sampler.
        
    Returns:
        Configured DataLoader.
        
    Example:
        >>> dataloader = build_dataloader(
        ...     dataset,
        ...     batch_size=2,
        ...     shuffle=True,
        ...     num_workers=4,
        ... )
    """
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        # shuffle is handled by sampler when distributed
        shuffle = False
    else:
        if shuffle:
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    
    return dataloader


def build_val_dataloader(
    dataset: Dataset,
    batch_size: int = 1,
    num_workers: int = 4,
    distributed: bool = False,
) -> DataLoader:
    """Build a validation DataLoader.
    
    Convenience function for validation with no shuffling.
    
    Args:
        dataset: Validation dataset.
        batch_size: Batch size (usually 1 for accurate metrics).
        num_workers: Number of data loading workers.
        distributed: Whether to use distributed sampler.
        
    Returns:
        Configured DataLoader for validation.
    """
    return build_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        distributed=distributed,
    )
