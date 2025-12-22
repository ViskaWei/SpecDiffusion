"""Hardware detection and configuration utilities.

Provides:
- GPU/CPU/MPS accelerator detection
- Optimal worker count allocation for data loading
- Training strategy selection
"""

from __future__ import annotations

import os
import subprocess
from typing import Optional, Tuple

import torch


def detect_system_gpus() -> int:
    """Detect total physical GPUs in the system.
    
    Returns:
        int: Total number of physical GPUs
    """
    try:
        nvidia_smi = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'],
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()
        
        if nvidia_smi:
            return len(nvidia_smi.split('\n'))
    except Exception:
        pass
    
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def select_accelerator_and_devices(num_gpus: Optional[int] = None) -> Tuple[str, int]:
    """Select optimal accelerator and device count.
    
    Priority: CUDA > MPS > CPU
    
    Args:
        num_gpus: Desired number of GPUs (None = auto-detect)
        
    Returns:
        Tuple[str, int]: (accelerator_type, device_count)
    """
    if num_gpus and num_gpus > 0:
        if torch.cuda.is_available():
            return 'gpu', min(num_gpus, torch.cuda.device_count())
        if torch.backends.mps.is_available():
            return 'mps', 1
    
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return 'gpu', torch.cuda.device_count()
    if torch.backends.mps.is_available():
        return 'mps', 1
    
    return 'cpu', 1


def get_training_strategy(device_count: int) -> str:
    """Determine optimal distributed training strategy.
    
    Args:
        device_count: Number of devices being used
        
    Returns:
        str: PyTorch Lightning strategy
    """
    return 'ddp' if device_count and device_count > 1 else 'auto'


def is_server_environment(cpu_count: int, total_gpus: int) -> bool:
    """Determine if running on a server vs local machine."""
    return cpu_count >= 32 and total_gpus >= 4


def calculate_optimal_workers(
    cpu_count: int,
    gpu_count: int,
    total_system_gpus: int,
    batch_size: int,
    is_server: bool
) -> int:
    """Calculate optimal number of dataloader workers.
    
    Strategy:
    - Server + Single GPU: Divide CPUs fairly for parallel jobs
    - Server + Multi GPU: Moderate workers per GPU
    - Local: Use 0 workers to avoid multiprocessing issues
    """
    if not is_server:
        return 0
    
    if gpu_count == 1:
        workers_per_job = max(1, (cpu_count - total_system_gpus) // total_system_gpus)
        
        if batch_size < 128:
            base_workers = min(workers_per_job, 12)
        elif batch_size < 512:
            base_workers = min(workers_per_job, 10)
        else:
            base_workers = min(workers_per_job, 8)
        
        return max(4, min(base_workers, 12))
    
    elif gpu_count <= 4:
        base_workers = min(6 * gpu_count, 32)
        return min(base_workers, cpu_count - 1, 48)
    
    else:
        if batch_size >= 512:
            base_workers = min(4 * gpu_count, 32)
        elif batch_size >= 128:
            base_workers = min(6 * gpu_count, 48)
        else:
            base_workers = min(8 * gpu_count, 63)
        return min(base_workers, cpu_count - 1, 63)


def auto_detect_num_workers(
    gpu_count: Optional[int] = None,
    batch_size: int = 256,
    verbose: bool = True
) -> int:
    """Automatically detect optimal number of dataloader workers.
    
    Priority:
    1. Environment variable NUM_WORKERS
    2. Auto-detection based on system resources
    """
    env_num_workers = os.environ.get('NUM_WORKERS')
    if env_num_workers is not None:
        num_workers = int(env_num_workers)
        if verbose:
            print(f"[Hardware] Using NUM_WORKERS from environment: {num_workers}")
        return num_workers
    
    cpu_count = os.cpu_count() or 1
    if gpu_count is None:
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    total_system_gpus = detect_system_gpus()
    is_server = is_server_environment(cpu_count, total_system_gpus)
    
    num_workers = calculate_optimal_workers(
        cpu_count=cpu_count,
        gpu_count=gpu_count,
        total_system_gpus=total_system_gpus,
        batch_size=batch_size,
        is_server=is_server
    )
    
    if verbose:
        env_type = "SERVER" if is_server else "LOCAL"
        print(
            f"[Hardware] Auto-detected {env_type}: "
            f"{cpu_count} CPUs, {gpu_count} GPUs, batch_size={batch_size} "
            f"-> using {num_workers} workers"
        )
    
    return num_workers


def get_num_workers_from_config(config: dict, verbose: bool = True) -> Tuple[int, int]:
    """Extract num_workers and batch_size from config with smart defaults.
    
    Returns:
        Tuple[int, int]: (num_workers, batch_size)
    """
    train_config = config.get('train', {})
    
    num_workers_config = train_config.get('num_workers', train_config.get('workers', None))
    batch_size = train_config.get('batch_size', 256)
    
    if num_workers_config is not None:
        num_workers = int(num_workers_config)
        if verbose:
            print(f"[Hardware] Using num_workers from config: {num_workers}")
        return num_workers, batch_size
    
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    num_workers = auto_detect_num_workers(
        gpu_count=gpu_count,
        batch_size=batch_size,
        verbose=verbose
    )
    
    return num_workers, batch_size


__all__ = [
    'detect_system_gpus',
    'select_accelerator_and_devices',
    'get_training_strategy',
    'is_server_environment',
    'calculate_optimal_workers',
    'auto_detect_num_workers',
    'get_num_workers_from_config',
]

