"""Trainer utilities for diffusion models.

Provides:
- DiffusionTrainer: Lightning Trainer wrapper with smart defaults
- Quick training functions for experiments
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

import lightning as L
import torch

from src.utils.hardware import select_accelerator_and_devices, get_training_strategy
from src.utils.seed import set_all_seeds

# Enable PyTorch 2.x optimizations
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")


class DiffusionTrainer(L.Trainer):
    """Lightning Trainer with optimized defaults for diffusion models.
    
    Features:
    - Automatic hardware detection
    - Mixed precision training
    - Gradient clipping
    - Smart checkpointing
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        logger: Any = None,
        num_gpus: Optional[int] = None,
        callbacks: list = None,
    ):
        """Initialize trainer.
        
        Args:
            config: Training configuration
            logger: Lightning logger instance
            num_gpus: Number of GPUs (None = auto-detect)
            callbacks: Additional callbacks
        """
        train_config = config.get('train', {})
        
        # Hardware detection
        acc, devices = select_accelerator_and_devices(num_gpus or config.get('gpus'))
        strategy = get_training_strategy(devices)
        
        # Training parameters
        epochs = train_config.get('epochs', 100)
        precision = str(train_config.get('precision', '16-mixed'))
        grad_clip = train_config.get('grad_clip', 1.0)
        
        # Checkpointing
        enable_checkpointing = train_config.get('save', True)
        
        # Validation
        val_check_interval = train_config.get('val_check_interval', 1.0)
        check_val_every_n_epoch = train_config.get('check_val_every_n_epoch', 1)
        
        # Debug mode
        fast_dev_run = train_config.get('debug', False)
        
        # Initialize callbacks
        all_callbacks = callbacks or []
        
        # Add checkpoint callback if enabled
        if enable_checkpointing:
            from lightning.pytorch.callbacks import ModelCheckpoint
            
            checkpoint_callback = ModelCheckpoint(
                dirpath=train_config.get('checkpoint_dir', './checkpoints'),
                filename='diffusion-{epoch:02d}-{val_loss:.4f}',
                monitor='val_loss',
                mode='min',
                save_top_k=3,
                save_last=True,
            )
            all_callbacks.append(checkpoint_callback)
        
        # Add progress bar customization
        if not train_config.get('enable_progress_bar', True):
            from lightning.pytorch.callbacks import TQDMProgressBar
            # Disable or customize progress bar
            pass
        
        super().__init__(
            max_epochs=epochs,
            devices=devices,
            accelerator=acc,
            strategy=strategy,
            logger=logger,
            precision=precision,
            gradient_clip_val=grad_clip,
            callbacks=all_callbacks,
            enable_checkpointing=enable_checkpointing,
            enable_progress_bar=train_config.get('enable_progress_bar', True),
            enable_model_summary=train_config.get('enable_model_summary', True),
            val_check_interval=val_check_interval,
            check_val_every_n_epoch=check_val_every_n_epoch,
            fast_dev_run=fast_dev_run,
            deterministic=True,
        )
        
        self.config = config


def train_experiment(
    config: Dict[str, Any],
    gpu_id: int = 0,
    seed: int = 42,
    use_wandb: bool = False,
    project_name: str = "SpecDiffusion",
) -> Dict[str, Any]:
    """Train a single diffusion experiment.
    
    Args:
        config: Experiment configuration
        gpu_id: GPU device ID
        seed: Random seed
        use_wandb: Whether to use Weights & Biases logging
        project_name: W&B project name
        
    Returns:
        Dictionary with training results
    """
    from src.nn.lightning_module import DiffusionLightningModule
    from src.dataloader.datamodule import BaseDataModule
    from src.dataloader.base import BaseDiffusionDataset
    
    start_time = time.perf_counter()
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Set seeds
    set_all_seeds(seed)
    L.seed_everything(seed, workers=True)
    
    result = {
        "status": "ok",
        "metrics": {},
        "training_time_sec": 0.0,
        "error_message": None,
        "config": config,
        "gpu_id": gpu_id,
        "seed": seed,
    }
    
    try:
        # Create logger
        logger = None
        if use_wandb:
            from lightning.pytorch.loggers import WandbLogger
            logger = WandbLogger(
                project=project_name,
                config=config,
            )
        
        # Create data module
        data_module = BaseDataModule.from_config(config, dataset_cls=BaseDiffusionDataset)
        
        # Create model
        lightning_module = DiffusionLightningModule.from_config(config)
        
        # Enable torch.compile if available
        if hasattr(torch, "compile") and config.get('train', {}).get('compile', True):
            try:
                lightning_module.model.model = torch.compile(
                    lightning_module.model.model,
                    mode="reduce-overhead",
                )
                print(f"[GPU {gpu_id}] torch.compile enabled")
            except Exception as e:
                print(f"[GPU {gpu_id}] torch.compile not available: {e}")
        
        # Create trainer
        trainer = DiffusionTrainer(config, logger=logger)
        
        # Train
        trainer.fit(lightning_module, datamodule=data_module)
        
        # Test
        trainer.test(lightning_module, datamodule=data_module)
        
        # Collect metrics
        result["metrics"] = dict(trainer.callback_metrics)
        result["training_time_sec"] = time.perf_counter() - start_time
        
    except Exception as e:
        result["status"] = "error"
        result["error_message"] = f"{type(e).__name__}: {e}"
        result["training_time_sec"] = time.perf_counter() - start_time
        
        import traceback
        traceback.print_exc()
    
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return result


def quick_train(
    config_path: str,
    gpu_id: int = 0,
    seed: int = 42,
    **overrides,
) -> Dict[str, Any]:
    """Quick training with config file and overrides.
    
    Args:
        config_path: Path to YAML config file
        gpu_id: GPU to use
        seed: Random seed
        **overrides: Config overrides
        
    Example:
        result = quick_train(
            "configs/spectrum_ddpm.yaml",
            gpu_id=0,
            epochs=100,
            lr=1e-4,
        )
    """
    from src.utils.config import load_config
    
    config = load_config(config_path)
    
    # Apply overrides
    for key, value in overrides.items():
        if '.' in key:
            parts = key.split('.')
            current = config
            for part in parts[:-1]:
                current = current.setdefault(part, {})
            current[parts[-1]] = value
        elif key in ('lr', 'learning_rate'):
            config.setdefault('opt', {})['lr'] = value
        elif key in ('epochs', 'ep'):
            config.setdefault('train', {})['epochs'] = value
        elif key == 'batch_size':
            config.setdefault('train', {})['batch_size'] = value
        else:
            # Try common locations
            for section in ['model', 'train', 'diffusion']:
                if section in config and key in config[section]:
                    config[section][key] = value
                    break
            else:
                config[key] = value
    
    return train_experiment(config, gpu_id=gpu_id, seed=seed)


__all__ = [
    "DiffusionTrainer",
    "train_experiment",
    "quick_train",
]

