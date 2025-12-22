"""Dataset primitives for diffusion model training.

This module provides base classes for data loading with:
- Configurable instantiation from config dictionaries
- HDF5/NumPy data loading support
- Normalization and preprocessing utilities
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class Configurable:
    """Mixin that instantiates objects from nested config dictionaries.
    
    Usage:
        class MyDataset(Configurable, Dataset):
            init_params = ['file_path', 'num_samples']
            config_section = 'data'
            
            def __init__(self, file_path, num_samples):
                ...
        
        dataset = MyDataset.from_config(config)
    """

    init_params: List[str] = []
    config_section: Optional[str] = None

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        """Create instance from a config dictionary.
        
        Walks through the class hierarchy (MRO) to collect parameters
        from the appropriate config sections.
        """
        params: Dict[str, Any] = {}
        for base in cls.__mro__[::-1]:
            if issubclass(base, Configurable) and base is not Configurable:
                if base.config_section:
                    section = config.get(base.config_section, {})
                    for param in base.init_params:
                        if param in section:
                            params[param] = section[param]
        return cls(**params)


class BaseDataset(Configurable, Dataset, ABC):
    """Abstract base dataset class for diffusion model training.
    
    Provides common functionality:
    - Config-based instantiation
    - Train/val/test split management
    - Data path resolution
    """
    
    init_params = [
        "file_path",
        "val_path",
        "test_path",
        "num_samples",
        "num_test_samples",
        "root_dir",
        "normalize",
    ]
    config_section = "data"

    def __init__(
        self,
        file_path: Optional[str] = None,
        val_path: Optional[str] = None,
        test_path: Optional[str] = None,
        num_samples: Optional[int] = None,
        num_test_samples: Optional[int] = None,
        root_dir: str = "./data",
        normalize: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        
        self.file_path = file_path
        self.val_path = val_path if val_path is not None else file_path
        self.test_path = test_path if test_path is not None else file_path
        self.num_samples = num_samples if num_samples is not None else 1
        self.num_test_samples = (
            num_test_samples if num_test_samples is not None 
            else min(10000, self.num_samples)
        )
        self.root_dir = root_dir
        self.normalize = normalize
        
        # Data containers (populated in load_data)
        self.data: Optional[torch.Tensor] = None
        self.labels: Optional[torch.Tensor] = None
        
        # Normalization stats
        self.data_mean: Optional[torch.Tensor] = None
        self.data_std: Optional[torch.Tensor] = None

    def get_path_and_samples(self, stage: Optional[str]) -> Tuple[str, int]:
        """Get appropriate file path and sample count for a stage."""
        if stage in {"fit", "train", None}:
            return self.file_path, self.num_samples
        load_path = self.test_path if stage == "test" else self.val_path
        return load_path, self.num_test_samples

    @abstractmethod
    def load_data(self, stage: Optional[str] = None) -> None:
        """Load the data from disk. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def __getitem__(self, idx: int):
        """Return a single data sample."""
        pass

    def __len__(self) -> int:
        return self.num_samples if self.data is None else len(self.data)

    def compute_normalization_stats(self) -> None:
        """Compute mean and std for normalization."""
        if self.data is not None:
            self.data_mean = self.data.mean(dim=0, keepdim=True)
            self.data_std = self.data.std(dim=0, keepdim=True)
            # Avoid division by zero
            self.data_std = torch.clamp(self.data_std, min=1e-6)
    
    def normalize_data(self, data: torch.Tensor) -> torch.Tensor:
        """Normalize data using stored stats."""
        if self.data_mean is not None and self.data_std is not None:
            return (data - self.data_mean) / self.data_std
        return data
    
    def denormalize_data(self, data: torch.Tensor) -> torch.Tensor:
        """Denormalize data using stored stats."""
        if self.data_mean is not None and self.data_std is not None:
            return data * self.data_std + self.data_mean
        return data


class BaseDiffusionDataset(BaseDataset):
    """Dataset class optimized for diffusion model training.
    
    Features:
    - Automatic normalization to [-1, 1] or [0, 1]
    - Support for conditional generation (labels)
    - HDF5 and NumPy file loading
    """
    
    init_params = BaseDataset.init_params + [
        "data_key",
        "label_key",
        "normalize_range",
        "cache_in_memory",
    ]
    
    def __init__(
        self,
        data_key: str = "data",
        label_key: Optional[str] = None,
        normalize_range: str = "[-1,1]",  # or "[0,1]" or "none"
        cache_in_memory: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        
        self.data_key = data_key
        self.label_key = label_key
        self.normalize_range = normalize_range
        self.cache_in_memory = cache_in_memory
        
        # HDF5 file handle (for memory-mapped loading)
        self._h5_file: Optional[h5py.File] = None

    def load_data(self, stage: Optional[str] = None) -> None:
        """Load data from HDF5 or NumPy file."""
        load_path, num_samples = self.get_path_and_samples(stage)
        
        if load_path is None:
            raise ValueError(f"No data path specified for stage '{stage}'")
        
        print(f"[{stage or 'train'}] Loading data from {load_path}, num_samples={num_samples}")
        
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Data file not found: {load_path}")
        
        # Determine file type and load
        if load_path.endswith('.h5') or load_path.endswith('.hdf5'):
            self._load_hdf5(load_path, num_samples)
        elif load_path.endswith('.npy') or load_path.endswith('.npz'):
            self._load_numpy(load_path, num_samples)
        else:
            raise ValueError(f"Unsupported file format: {load_path}")
        
        # Apply normalization
        if self.normalize:
            self._apply_normalization(stage)
        
        self.num_samples = len(self.data)
        print(f"[{stage or 'train'}] Loaded {self.num_samples} samples, shape: {self.data.shape}")

    def _load_hdf5(self, path: str, num_samples: int) -> None:
        """Load data from HDF5 file."""
        with h5py.File(path, 'r') as f:
            # Try common data paths
            data_paths = [self.data_key, 'data', 'images', 'flux', 'dataset/arrays/flux/value']
            
            data = None
            for dpath in data_paths:
                if dpath in f:
                    data = f[dpath][:num_samples]
                    break
                # Try nested paths
                try:
                    keys = dpath.split('/')
                    obj = f
                    for key in keys:
                        obj = obj[key]
                    data = obj[:num_samples]
                    break
                except (KeyError, TypeError):
                    continue
            
            if data is None:
                available_keys = list(f.keys())
                raise KeyError(f"Data key '{self.data_key}' not found. Available: {available_keys}")
            
            self.data = torch.tensor(data, dtype=torch.float32)
            
            # Load labels if specified
            if self.label_key:
                label_paths = [self.label_key, 'labels', 'label', 'y']
                for lpath in label_paths:
                    if lpath in f:
                        self.labels = torch.tensor(f[lpath][:num_samples])
                        break

    def _load_numpy(self, path: str, num_samples: int) -> None:
        """Load data from NumPy file."""
        if path.endswith('.npz'):
            data = np.load(path)
            self.data = torch.tensor(data[self.data_key][:num_samples], dtype=torch.float32)
            if self.label_key and self.label_key in data:
                self.labels = torch.tensor(data[self.label_key][:num_samples])
        else:
            data = np.load(path)[:num_samples]
            self.data = torch.tensor(data, dtype=torch.float32)

    def _apply_normalization(self, stage: Optional[str] = None) -> None:
        """Apply data normalization."""
        if self.normalize_range == "none":
            return
        
        is_train = stage in (None, "fit", "train")
        
        if is_train:
            # Compute stats from training data
            self.data_min = self.data.min()
            self.data_max = self.data.max()
            self.data_mean = self.data.mean()
            self.data_std = self.data.std()
        
        if self.normalize_range == "[-1,1]":
            # Scale to [-1, 1]
            if self.data_min is not None and self.data_max is not None:
                self.data = 2.0 * (self.data - self.data_min) / (self.data_max - self.data_min + 1e-8) - 1.0
        elif self.normalize_range == "[0,1]":
            # Scale to [0, 1]
            if self.data_min is not None and self.data_max is not None:
                self.data = (self.data - self.data_min) / (self.data_max - self.data_min + 1e-8)
        elif self.normalize_range == "standard":
            # Z-score normalization
            if self.data_mean is not None and self.data_std is not None:
                self.data = (self.data - self.data_mean) / (self.data_std + 1e-8)

    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Return a single sample, optionally with label."""
        if self.labels is not None:
            return self.data[idx], self.labels[idx]
        return self.data[idx]

    def get_sample_shape(self) -> Tuple[int, ...]:
        """Get shape of a single sample."""
        if self.data is not None:
            return tuple(self.data.shape[1:])
        raise RuntimeError("Data not loaded. Call load_data() first.")


class ImageDiffusionDataset(BaseDiffusionDataset):
    """Dataset for image diffusion models.
    
    Handles image-specific preprocessing:
    - Channel dimension handling (CHW vs HWC)
    - Image augmentations
    """
    
    init_params = BaseDiffusionDataset.init_params + [
        "image_size",
        "channels",
        "channel_first",
    ]
    
    def __init__(
        self,
        image_size: int = 32,
        channels: int = 3,
        channel_first: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.image_size = image_size
        self.channels = channels
        self.channel_first = channel_first

    def load_data(self, stage: Optional[str] = None) -> None:
        """Load and preprocess image data."""
        super().load_data(stage)
        
        # Ensure correct shape (N, C, H, W)
        if len(self.data.shape) == 3:
            # Add channel dimension
            self.data = self.data.unsqueeze(1)
        
        if not self.channel_first and len(self.data.shape) == 4:
            # Convert from (N, H, W, C) to (N, C, H, W)
            self.data = self.data.permute(0, 3, 1, 2)


class Spectrum1DDiffusionDataset(BaseDiffusionDataset):
    """Dataset for 1D spectral data diffusion models.
    
    Specialized for spectral data with:
    - Wavelength grid support
    - Error/uncertainty handling
    - SNR-based filtering
    """
    
    init_params = BaseDiffusionDataset.init_params + [
        "wave_key",
        "error_key",
        "min_snr",
    ]
    
    def __init__(
        self,
        wave_key: str = "wave",
        error_key: Optional[str] = "error",
        min_snr: Optional[float] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.wave_key = wave_key
        self.error_key = error_key
        self.min_snr = min_snr
        
        self.wave: Optional[torch.Tensor] = None
        self.error: Optional[torch.Tensor] = None

    def _load_hdf5(self, path: str, num_samples: int) -> None:
        """Load spectral data from HDF5."""
        with h5py.File(path, 'r') as f:
            # Load flux data
            flux_paths = [self.data_key, 'flux', 'dataset/arrays/flux/value', 'spectrumdataset/flux']
            for fpath in flux_paths:
                try:
                    keys = fpath.split('/')
                    obj = f
                    for key in keys:
                        obj = obj[key]
                    self.data = torch.tensor(obj[:num_samples], dtype=torch.float32)
                    break
                except (KeyError, TypeError):
                    continue
            
            if self.data is None:
                raise KeyError(f"Could not find flux data in {path}")
            
            # Load wavelength grid
            wave_paths = [self.wave_key, 'wave', 'wavelength', 'spectrumdataset/wave']
            for wpath in wave_paths:
                try:
                    keys = wpath.split('/')
                    obj = f
                    for key in keys:
                        obj = obj[key]
                    self.wave = torch.tensor(obj[()], dtype=torch.float32)
                    break
                except (KeyError, TypeError):
                    continue
            
            # Load error if specified
            if self.error_key:
                error_paths = [self.error_key, 'error', 'dataset/arrays/error/value']
                for epath in error_paths:
                    try:
                        keys = epath.split('/')
                        obj = f
                        for key in keys:
                            obj = obj[key]
                        self.error = torch.tensor(obj[:num_samples], dtype=torch.float32)
                        break
                    except (KeyError, TypeError):
                        continue

    def load_data(self, stage: Optional[str] = None) -> None:
        """Load spectral data with optional SNR filtering."""
        super().load_data(stage)
        
        # Ensure 2D shape (N, L)
        if len(self.data.shape) == 1:
            self.data = self.data.unsqueeze(0)
        
        # Apply SNR filtering if specified
        if self.min_snr is not None and self.error is not None:
            snr = self.data.norm(dim=-1) / self.error.norm(dim=-1)
            mask = snr >= self.min_snr
            self.data = self.data[mask]
            self.error = self.error[mask] if self.error is not None else None
            if self.labels is not None:
                self.labels = self.labels[mask]
            self.num_samples = len(self.data)
            print(f"After SNR filtering: {self.num_samples} samples")

    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Return spectrum with optional error and labels."""
        flux = self.data[idx]
        
        if self.error is not None and self.labels is not None:
            return flux, self.error[idx], self.labels[idx]
        elif self.error is not None:
            return flux, self.error[idx]
        elif self.labels is not None:
            return flux, self.labels[idx]
        return flux


__all__ = [
    "Configurable",
    "BaseDataset",
    "BaseDiffusionDataset",
    "ImageDiffusionDataset",
    "Spectrum1DDiffusionDataset",
]

