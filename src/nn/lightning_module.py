"""Lightning module for diffusion model training.

Provides:
- Training/validation/test step implementations
- EMA (Exponential Moving Average) support
- Logging and visualization
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import lightning as L
import torch
import torch.nn as nn

from src.models.diffusion import BaseDiffusionModel, DDPM
from src.nn.optimizer import OptModule


class EMA:
    """Exponential Moving Average for model parameters."""
    
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update shadow parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1 - self.decay) * param.data
                )

    def apply_shadow(self):
        """Apply shadow parameters to model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """Restore original parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


class DiffusionLightningModule(L.LightningModule):
    """PyTorch Lightning module for diffusion model training.
    
    Features:
    - Automatic optimization with configurable schedulers
    - EMA support for stable training
    - Validation sampling and visualization
    """
    
    def __init__(
        self,
        model: BaseDiffusionModel,
        config: Dict[str, Any] = None,
        use_ema: bool = True,
        ema_decay: float = 0.9999,
    ):
        super().__init__()
        self.model = model
        self.config = config or {}
        self.use_ema = use_ema
        
        # EMA
        if use_ema:
            self.ema = EMA(model.model, decay=ema_decay)
        else:
            self.ema = None
        
        # Logging
        self.save_hyperparameters(ignore=['model'])

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        """Create from configuration dictionary."""
        model = DDPM.from_config(config)
        
        train_config = config.get('train', {})
        use_ema = train_config.get('use_ema', True)
        ema_decay = train_config.get('ema_decay', 0.9999)
        
        return cls(
            model=model,
            config=config,
            use_ema=use_ema,
            ema_decay=ema_decay,
        )

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        opt_config = self.config.get('opt', {})
        
        # Add training info for OneCycleLR
        train_config = self.config.get('train', {})
        data_config = self.config.get('data', {})
        
        batch_size = train_config.get('batch_size', 64)
        num_samples = data_config.get('num_samples', 10000)
        epochs = train_config.get('epochs', 100)
        
        opt_config['steps_per_epoch'] = (num_samples + batch_size - 1) // batch_size
        opt_config['epochs'] = epochs
        
        return OptModule.from_config(opt_config)(self.model)

    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        return self.model(x, **kwargs)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Training step."""
        # Handle different batch formats
        if isinstance(batch, (list, tuple)):
            x = batch[0]
            cond = batch[1] if len(batch) > 1 else None
        else:
            x = batch
            cond = None
        
        # Forward pass
        outputs = self.model(x, cond=cond)
        loss = outputs['loss']
        
        # Logging
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Update EMA after each training batch."""
        if self.ema is not None:
            self.ema.update()

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Validation step."""
        if isinstance(batch, (list, tuple)):
            x = batch[0]
            cond = batch[1] if len(batch) > 1 else None
        else:
            x = batch
            cond = None
        
        # Use EMA model for validation if available
        if self.ema is not None:
            self.ema.apply_shadow()
        
        outputs = self.model(x, cond=cond)
        loss = outputs['loss']
        
        if self.ema is not None:
            self.ema.restore()
        
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Test step."""
        if isinstance(batch, (list, tuple)):
            x = batch[0]
            cond = batch[1] if len(batch) > 1 else None
        else:
            x = batch
            cond = None
        
        if self.ema is not None:
            self.ema.apply_shadow()
        
        outputs = self.model(x, cond=cond)
        loss = outputs['loss']
        
        if self.ema is not None:
            self.ema.restore()
        
        self.log('test_loss', loss, on_epoch=True)
        
        return loss

    def on_validation_epoch_end(self):
        """Generate and log samples at end of validation."""
        if self.trainer.sanity_checking:
            return
        
        # Only generate samples every N epochs
        sample_interval = self.config.get('train', {}).get('sample_interval', 10)
        if self.current_epoch % sample_interval != 0:
            return
        
        # Generate samples
        try:
            samples = self.generate_samples(num_samples=4)
            
            # Log samples if logger supports it
            if hasattr(self.logger, 'experiment'):
                if hasattr(self.logger.experiment, 'log_image'):
                    # For WandB
                    import wandb
                    self.logger.experiment.log({
                        "samples": [wandb.Image(s) for s in samples.cpu()],
                        "epoch": self.current_epoch,
                    })
        except Exception as e:
            print(f"[Warning] Failed to generate samples: {e}")

    @torch.no_grad()
    def generate_samples(
        self,
        num_samples: int = 16,
        num_inference_steps: int = 50,
        use_ddim: bool = True,
    ) -> torch.Tensor:
        """Generate samples from the model.
        
        Args:
            num_samples: Number of samples to generate
            num_inference_steps: Steps for DDIM sampling
            use_ddim: Whether to use DDIM (faster) or DDPM sampling
            
        Returns:
            Generated samples
        """
        # Use EMA model
        if self.ema is not None:
            self.ema.apply_shadow()
        
        # Get sample shape from model
        sample_shape = self._get_sample_shape(num_samples)
        
        # Generate
        if use_ddim:
            samples = self.model.sample_ddim(
                shape=sample_shape,
                num_inference_steps=num_inference_steps,
                device=self.device,
            )
        else:
            samples = self.model.sample(
                shape=sample_shape,
                device=self.device,
            )
        
        if self.ema is not None:
            self.ema.restore()
        
        return samples

    def _get_sample_shape(self, batch_size: int) -> Tuple[int, ...]:
        """Get shape for sample generation."""
        model_config = self.config.get('model', {})
        
        if model_config.get('type', '1d') == '1d':
            in_channels = model_config.get('in_channels', 1)
            seq_len = self.config.get('data', {}).get('seq_len', 1024)
            return (batch_size, in_channels, seq_len)
        else:
            in_channels = model_config.get('in_channels', 3)
            image_size = self.config.get('data', {}).get('image_size', 32)
            return (batch_size, in_channels, image_size, image_size)

    def on_fit_start(self):
        """Log model info at start of training."""
        if self.logger is not None:
            num_params = sum(p.numel() for p in self.model.parameters())
            self.log('num_params_M', num_params / 1e6)

    def on_train_epoch_start(self):
        """Log learning rate at start of each epoch."""
        optimizer = self.trainer.optimizers[0]
        lr = optimizer.param_groups[0]['lr']
        self.log('lr', lr)


__all__ = ["DiffusionLightningModule", "EMA"]

