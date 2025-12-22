"""Lightning DataModule for diffusion model training.

Provides:
- Automatic worker detection for optimal data loading
- Train/val/test split management
- Memory-efficient data loading with pinned memory
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Type

import lightning as L
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.dataloader.base import BaseDataset, BaseDiffusionDataset
from src.utils.hardware import get_num_workers_from_config


class BaseDataModule(L.LightningDataModule):
    """Lightning DataModule with smart configuration.
    
    Features:
    - Automatic worker count detection based on system resources
    - Configurable batch size and data loading parameters
    - Support for custom dataset classes
    """
    
    def __init__(
        self,
        batch_size: int = 256,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
        debug: bool = False,
        dataset_cls: Optional[Type[BaseDataset]] = None,
        config: Dict[str, Any] = None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.prefetch_factor = prefetch_factor if num_workers > 0 else None
        self.debug = debug
        self.dataset_cls = dataset_cls or BaseDiffusionDataset
        self.config = config or {}
        
        # Dataset instances
        self.train_dataset: Optional[BaseDataset] = None
        self.val_dataset: Optional[BaseDataset] = None
        self.test_dataset: Optional[BaseDataset] = None

    @classmethod
    def from_config(cls, config: Dict[str, Any], dataset_cls: Type[BaseDataset] = None):
        """Create DataModule from configuration dictionary.
        
        Worker detection priority:
        1. Environment variable NUM_WORKERS
        2. Config train.num_workers
        3. Auto-detection based on system resources
        """
        train_config = config.get('train', {})
        
        # Get optimal worker count
        num_workers, batch_size = get_num_workers_from_config(config, verbose=True)
        
        return cls(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=train_config.get('pin_memory', True),
            persistent_workers=train_config.get('persistent_workers', True),
            prefetch_factor=train_config.get('prefetch_factor', 2),
            debug=train_config.get('debug', False),
            dataset_cls=dataset_cls or BaseDiffusionDataset,
            config=config,
        )

    def prepare_data(self) -> None:
        """Called only on rank 0 for downloading/preparing data."""
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up datasets for the specified stage.
        
        Args:
            stage: 'fit', 'validate', 'test', or None (all)
        """
        if stage == 'fit' or stage is None:
            # Create and load training dataset
            self.train_dataset = self.dataset_cls.from_config(self.config)
            self.train_dataset.load_data(stage='train')
            
            # Create and load validation dataset
            self.val_dataset = self.dataset_cls.from_config(self.config)
            self.val_dataset.load_data(stage='val')
            
            # Share normalization stats from training set
            if hasattr(self.train_dataset, 'data_mean'):
                self.val_dataset.data_mean = self.train_dataset.data_mean
                self.val_dataset.data_std = self.train_dataset.data_std
                self.val_dataset.data_min = getattr(self.train_dataset, 'data_min', None)
                self.val_dataset.data_max = getattr(self.train_dataset, 'data_max', None)
        
        if stage == 'test' or stage is None:
            self.test_dataset = self.dataset_cls.from_config(self.config)
            self.test_dataset.load_data(stage='test')
            
            # Share normalization stats from training set if available
            if self.train_dataset is not None and hasattr(self.train_dataset, 'data_mean'):
                self.test_dataset.data_mean = self.train_dataset.data_mean
                self.test_dataset.data_std = self.train_dataset.data_std
                self.test_dataset.data_min = getattr(self.train_dataset, 'data_min', None)
                self.test_dataset.data_max = getattr(self.train_dataset, 'data_max', None)

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            shuffle=not self.debug,
            drop_last=True,  # Important for batch norm and consistent batch sizes
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        """Create validation dataloader."""
        if self.val_dataset is None or len(self.val_dataset) == 0:
            print("[WARNING] Validation dataset is empty - validation will be skipped")
            return None
        
        val_batch_size = min(self.batch_size, len(self.val_dataset))
        
        return DataLoader(
            self.val_dataset,
            batch_size=val_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            shuffle=False,
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        """Create test dataloader."""
        if self.test_dataset is None or len(self.test_dataset) == 0:
            print("[WARNING] Test dataset is empty - testing will be skipped")
            return None
        
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            shuffle=False,
        )

    def get_sample_shape(self) -> tuple:
        """Get shape of a single sample from the training dataset."""
        if self.train_dataset is not None:
            return self.train_dataset.get_sample_shape()
        raise RuntimeError("Training dataset not initialized. Call setup() first.")

    def get_normalization_stats(self) -> Dict[str, Any]:
        """Get normalization statistics from training dataset."""
        if self.train_dataset is None:
            return {}
        
        stats = {}
        for attr in ['data_mean', 'data_std', 'data_min', 'data_max']:
            if hasattr(self.train_dataset, attr):
                val = getattr(self.train_dataset, attr)
                if val is not None:
                    stats[attr] = val
        return stats


class InMemoryDataModule(BaseDataModule):
    """DataModule that loads all data into GPU memory for maximum speed.
    
    Use when:
    - Dataset fits entirely in GPU memory
    - Training speed is critical
    - Multi-epoch training where data doesn't change
    """
    
    def __init__(self, device: str = 'cuda', **kwargs):
        super().__init__(**kwargs)
        self.device = device

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data and move to GPU."""
        super().setup(stage)
        
        # Move training data to GPU
        if self.train_dataset is not None and self.train_dataset.data is not None:
            self.train_dataset.data = self.train_dataset.data.to(self.device)
            if self.train_dataset.labels is not None:
                self.train_dataset.labels = self.train_dataset.labels.to(self.device)
        
        # Move validation data to GPU
        if self.val_dataset is not None and self.val_dataset.data is not None:
            self.val_dataset.data = self.val_dataset.data.to(self.device)
            if self.val_dataset.labels is not None:
                self.val_dataset.labels = self.val_dataset.labels.to(self.device)

    def train_dataloader(self) -> DataLoader:
        """Create dataloader with num_workers=0 for GPU-resident data."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=not self.debug,
            drop_last=True,
            num_workers=0,  # Data already on GPU
            pin_memory=False,
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        """Create validation dataloader with num_workers=0."""
        if self.val_dataset is None or len(self.val_dataset) == 0:
            return None
        
        return DataLoader(
            self.val_dataset,
            batch_size=min(self.batch_size, len(self.val_dataset)),
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )


__all__ = [
    "BaseDataModule",
    "InMemoryDataModule",
]

