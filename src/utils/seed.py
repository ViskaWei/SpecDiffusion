"""Seed utilities for reproducible experiments.

Provides comprehensive seed management for:
- PyTorch (CPU and GPU)
- NumPy
- Python random
- CUDA deterministic algorithms
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np

# Global seed for reproducibility
GLOBAL_SEED = 42

# Seeds for different data splits
TRAIN_SEED = 42
VAL_SEED = 43
TEST_SEED = 44


def set_all_seeds(seed: int = GLOBAL_SEED, deterministic: bool = True) -> None:
    """Set all random seeds for complete reproducibility.
    
    Args:
        seed: Random seed value
        deterministic: If True, force CUDA deterministic algorithms
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            
            if deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_split_seed(split: str) -> int:
    """Get the appropriate seed for a data split."""
    seeds = {
        'train': TRAIN_SEED,
        'val': VAL_SEED,
        'test': TEST_SEED,
    }
    return seeds.get(split, GLOBAL_SEED)


class SeedContext:
    """Context manager for temporary seed changes.
    
    Example:
        >>> with SeedContext(123):
        ...     noise = torch.randn(100)  # Always the same
        >>> # Original random state restored
    """
    
    def __init__(self, seed: int):
        self.seed = seed
        self.saved_torch_state = None
        self.saved_numpy_state = None
        self.saved_random_state = None
    
    def __enter__(self):
        import torch
        
        self.saved_torch_state = torch.get_rng_state()
        self.saved_numpy_state = np.random.get_state()
        self.saved_random_state = random.getstate()
        
        set_all_seeds(self.seed, deterministic=False)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import torch
        
        torch.set_rng_state(self.saved_torch_state)
        np.random.set_state(self.saved_numpy_state)
        random.setstate(self.saved_random_state)
        return False


__all__ = [
    "GLOBAL_SEED",
    "TRAIN_SEED",
    "VAL_SEED",
    "TEST_SEED",
    "set_all_seeds",
    "get_split_seed",
    "SeedContext",
]

