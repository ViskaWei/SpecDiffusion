"""Configuration utilities.

Provides:
- YAML config loading with OmegaConf
- Config merging and validation
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config or {}


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple configurations (later configs override earlier).
    
    Args:
        *configs: Configuration dictionaries to merge
        
    Returns:
        Merged configuration
    """
    result = {}
    for config in configs:
        result = _deep_merge(result, config)
    return result


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dictionaries."""
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration for required fields.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    required_sections = ['data', 'model', 'train']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
    
    # Validate data section
    data_config = config['data']
    if 'file_path' not in data_config:
        raise ValueError("Missing 'data.file_path' in config")
    
    # Validate model section
    model_config = config['model']
    if 'type' not in model_config:
        config['model']['type'] = '1d'  # Default to 1D
    
    return True


def get_default_config() -> Dict[str, Any]:
    """Get default configuration."""
    return {
        'data': {
            'file_path': None,
            'num_samples': 10000,
            'num_test_samples': 1000,
            'normalize': True,
            'normalize_range': '[-1,1]',
        },
        'model': {
            'type': '1d',
            'in_channels': 1,
            'out_channels': 1,
            'base_channels': 64,
            'channel_mults': [1, 2, 4, 8],
            'num_res_blocks': 2,
            'attention_resolutions': [4],
            'num_heads': 4,
            'dropout': 0.0,
        },
        'diffusion': {
            'num_timesteps': 1000,
            'beta_start': 1e-4,
            'beta_end': 0.02,
            'beta_schedule': 'linear',
            'prediction_type': 'epsilon',
            'loss_type': 'mse',
        },
        'train': {
            'epochs': 100,
            'batch_size': 64,
            'precision': '16-mixed',
            'grad_clip': 1.0,
            'save': True,
            'use_ema': True,
            'ema_decay': 0.9999,
            'sample_interval': 10,
        },
        'opt': {
            'lr': 1e-4,
            'type': 'adamw',
            'weight_decay': 1e-4,
            'lr_sch': 'cosine',
            'warmup_epochs': 5,
        },
    }


__all__ = [
    "load_config",
    "merge_configs",
    "validate_config",
    "get_default_config",
]

