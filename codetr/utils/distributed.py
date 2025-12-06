"""Distributed training utilities for Co-DETR.

This module provides utilities for multi-GPU distributed training using
PyTorch's DistributedDataParallel (DDP).

Example:
    >>> from codetr.utils.distributed import init_distributed_mode, is_main_process
    >>> init_distributed_mode()
    >>> if is_main_process():
    ...     print("Logging from main process only")
"""

import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import Tensor


def is_dist_available_and_initialized() -> bool:
    """Check if distributed training is available and initialized.
    
    Returns:
        True if distributed is available and initialized.
    """
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size() -> int:
    """Get the number of processes in the distributed group.
    
    Returns:
        World size (1 if not distributed).
    """
    if not is_dist_available_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    """Get the rank of the current process.
    
    Returns:
        Process rank (0 if not distributed).
    """
    if not is_dist_available_and_initialized():
        return 0
    return dist.get_rank()


def get_local_rank() -> int:
    """Get the local rank of the current process.
    
    The local rank is the rank within a single machine.
    
    Returns:
        Local rank (0 if not distributed or not set).
    """
    if not is_dist_available_and_initialized():
        return 0
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main_process() -> bool:
    """Check if this is the main process (rank 0).
    
    Use this to guard logging and saving operations.
    
    Returns:
        True if this is the main process.
    """
    return get_rank() == 0


def synchronize() -> None:
    """Synchronize all processes (barrier).
    
    All processes wait until every process reaches this point.
    """
    if not is_dist_available_and_initialized():
        return
    if get_world_size() == 1:
        return
    dist.barrier()


def init_distributed_mode(
    backend: str = "nccl",
    init_method: Optional[str] = None,
) -> bool:
    """Initialize distributed training mode.
    
    Automatically detects environment variables set by launchers
    (torchrun, torch.distributed.launch, SLURM, etc.).
    
    Args:
        backend: Distributed backend ("nccl" for GPU, "gloo" for CPU).
        init_method: URL specifying how to initialize process group.
        
    Returns:
        True if distributed mode was initialized, False otherwise.
        
    Example:
        >>> # Launch with: torchrun --nproc_per_node=2 train.py
        >>> init_distributed_mode()
    """
    # Check if already initialized
    if is_dist_available_and_initialized():
        return True
    
    # Check for environment variables from various launchers
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    elif "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ.get("SLURM_NTASKS", 1))
        local_rank = int(os.environ.get("SLURM_LOCALID", 0))
    else:
        # Not in distributed mode
        print("Not running in distributed mode")
        return False
    
    # Set CUDA device
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    
    # Initialize process group
    if init_method is None:
        init_method = "env://"
    
    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        world_size=world_size,
        rank=rank,
    )
    
    # Synchronize after initialization
    synchronize()
    
    if is_main_process():
        print(f"Initialized distributed mode: world_size={world_size}")
    
    return True


def cleanup_distributed() -> None:
    """Clean up distributed process group."""
    if is_dist_available_and_initialized():
        dist.destroy_process_group()


def reduce_dict(
    input_dict: Dict[str, Tensor],
    average: bool = True,
) -> Dict[str, Tensor]:
    """Reduce dictionary of tensors across all processes.
    
    Args:
        input_dict: Dictionary with tensor values to reduce.
        average: If True, average the values; otherwise, sum them.
        
    Returns:
        Dictionary with reduced values.
        
    Example:
        >>> losses = {"cls": tensor(1.0), "bbox": tensor(2.0)}
        >>> reduced = reduce_dict(losses)
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    
    with torch.no_grad():
        names = []
        values = []
        
        # Sort keys for consistent ordering
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        
        if average:
            values /= world_size
        
        reduced_dict = {k: v for k, v in zip(names, values)}
    
    return reduced_dict


def all_gather(data: Any) -> List[Any]:
    """Gather data from all processes.
    
    Args:
        data: Any picklable data to gather.
        
    Returns:
        List of data from all processes.
        
    Example:
        >>> local_data = {"rank": get_rank(), "value": 42}
        >>> all_data = all_gather(local_data)
        >>> print(len(all_data))  # world_size
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    
    # Serialize to bytes
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Gather sizes
    local_size = torch.tensor([tensor.numel()], device=tensor.device)
    size_list = [torch.tensor([0], device=tensor.device) for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)
    
    # Pad tensors to same size
    tensor_list = []
    for _ in range(world_size):
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device=tensor.device))
    
    if local_size != max_size:
        padding = torch.empty((max_size - local_size,), dtype=torch.uint8, device=tensor.device)
        tensor = torch.cat([tensor, padding], dim=0)
    
    dist.all_gather(tensor_list, tensor)
    
    # Deserialize
    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor[:size].cpu().numpy().tobytes()
        data_list.append(pickle.loads(buffer))
    
    return data_list


def all_gather_tensor(tensor: Tensor) -> Tensor:
    """Gather tensors from all processes and concatenate.
    
    Args:
        tensor: Tensor to gather (must have same shape on all ranks).
        
    Returns:
        Concatenated tensor from all processes.
    """
    world_size = get_world_size()
    if world_size == 1:
        return tensor
    
    # Ensure tensor is contiguous
    tensor = tensor.contiguous()
    
    # Create output list
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    
    # Gather
    dist.all_gather(tensor_list, tensor)
    
    return torch.cat(tensor_list, dim=0)


def broadcast(data: Any, src: int = 0) -> Any:
    """Broadcast data from source process to all others.
    
    Args:
        data: Data to broadcast (only meaningful on source).
        src: Source rank.
        
    Returns:
        Broadcast data on all processes.
    """
    world_size = get_world_size()
    if world_size == 1:
        return data
    
    if get_rank() == src:
        buffer = pickle.dumps(data)
        storage = torch.ByteStorage.from_buffer(buffer)
        tensor = torch.ByteTensor(storage)
    else:
        tensor = torch.ByteTensor()
    
    # Broadcast size
    size = torch.tensor([tensor.numel()], dtype=torch.long)
    dist.broadcast(size, src)
    
    # Resize on non-source
    if get_rank() != src:
        tensor = torch.ByteTensor(size.item())
    
    # Move to GPU if available
    if torch.cuda.is_available():
        tensor = tensor.cuda()
        size = size.cuda()
    
    # Broadcast data
    dist.broadcast(tensor, src)
    
    # Deserialize
    buffer = tensor.cpu().numpy().tobytes()
    return pickle.loads(buffer)
