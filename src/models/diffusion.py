"""Diffusion model implementations.

Provides:
- BaseDiffusionModel: Abstract base class
- DDPM: Denoising Diffusion Probabilistic Model
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.noise_scheduler import DDPMScheduler, DDIMScheduler, NoiseScheduler
from src.models.unet import UNet1D, UNet2D


class BaseDiffusionModel(nn.Module, ABC):
    """Abstract base class for diffusion models."""
    
    def __init__(
        self,
        noise_scheduler: NoiseScheduler,
        model: nn.Module,
        model_name: str = "diffusion",
    ):
        super().__init__()
        self.noise_scheduler = noise_scheduler
        self.model = model
        self._model_name = model_name

    @property
    def name(self) -> str:
        return self._model_name

    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass for training."""
        pass

    @abstractmethod
    def sample(self, shape: Tuple[int, ...], **kwargs) -> torch.Tensor:
        """Generate samples from the model."""
        pass

    def compute_loss(
        self,
        model_output: torch.Tensor,
        target: torch.Tensor,
        loss_type: str = "mse",
    ) -> torch.Tensor:
        """Compute training loss.
        
        Args:
            model_output: Output from the denoising model
            target: Target (noise or sample, depending on prediction_type)
            loss_type: "mse", "l1", or "huber"
        """
        if loss_type == "mse":
            return F.mse_loss(model_output, target)
        elif loss_type == "l1":
            return F.l1_loss(model_output, target)
        elif loss_type == "huber":
            return F.smooth_l1_loss(model_output, target)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")


class DDPM(BaseDiffusionModel):
    """Denoising Diffusion Probabilistic Model.
    
    Implements the training and sampling procedures from:
    "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        prediction_type: str = "epsilon",
        loss_type: str = "mse",
        **kwargs,
    ):
        scheduler = DDPMScheduler(
            num_timesteps=num_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            prediction_type=prediction_type,
        )
        super().__init__(scheduler, model, **kwargs)
        
        self.num_timesteps = num_timesteps
        self.prediction_type = prediction_type
        self.loss_type = loss_type

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        """Create DDPM from configuration dictionary."""
        model_config = config.get('model', {})
        diffusion_config = config.get('diffusion', {})
        
        # Determine model type
        model_type = model_config.get('type', '1d')
        
        if model_type == '1d':
            model = UNet1D(
                in_channels=model_config.get('in_channels', 1),
                out_channels=model_config.get('out_channels', 1),
                base_channels=model_config.get('base_channels', 64),
                channel_mults=tuple(model_config.get('channel_mults', [1, 2, 4, 8])),
                num_res_blocks=model_config.get('num_res_blocks', 2),
                attention_resolutions=tuple(model_config.get('attention_resolutions', [4])),
                num_heads=model_config.get('num_heads', 4),
                dropout=model_config.get('dropout', 0.0),
            )
        else:
            model = UNet2D(
                in_channels=model_config.get('in_channels', 3),
                out_channels=model_config.get('out_channels', 3),
                base_channels=model_config.get('base_channels', 64),
                channel_mults=tuple(model_config.get('channel_mults', [1, 2, 4, 8])),
                num_res_blocks=model_config.get('num_res_blocks', 2),
                attention_resolutions=tuple(model_config.get('attention_resolutions', [2, 3])),
                num_heads=model_config.get('num_heads', 4),
                dropout=model_config.get('dropout', 0.0),
            )
        
        return cls(
            model=model,
            num_timesteps=diffusion_config.get('num_timesteps', 1000),
            beta_start=diffusion_config.get('beta_start', 1e-4),
            beta_end=diffusion_config.get('beta_end', 0.02),
            beta_schedule=diffusion_config.get('beta_schedule', 'linear'),
            prediction_type=diffusion_config.get('prediction_type', 'epsilon'),
            loss_type=diffusion_config.get('loss_type', 'mse'),
            model_name=model_config.get('name', 'ddpm'),
        )

    def forward(
        self,
        x: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Training forward pass.
        
        Args:
            x: Clean samples [B, ...]
            cond: Optional conditioning
            return_dict: Whether to return a dictionary
            
        Returns:
            Dictionary with loss and optionally other values
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Move scheduler to device if needed
        if self.noise_scheduler.betas.device != device:
            self.noise_scheduler.to(device)
        
        # Sample random timesteps
        timesteps = torch.randint(0, self.num_timesteps, (batch_size,), device=device)
        
        # Sample noise
        noise = torch.randn_like(x)
        
        # Add noise to get x_t
        x_t = self.noise_scheduler.add_noise(x, noise, timesteps)
        
        # Predict noise (or sample, depending on prediction_type)
        model_output = self.model(x_t, timesteps, cond)
        
        # Compute target based on prediction type
        if self.prediction_type == "epsilon":
            target = noise
        elif self.prediction_type == "sample":
            target = x
        elif self.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(x, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")
        
        # Compute loss
        loss = self.compute_loss(model_output, target, self.loss_type)
        
        if return_dict:
            return {
                "loss": loss,
                "model_output": model_output,
                "target": target,
                "x_t": x_t,
                "timesteps": timesteps,
            }
        return loss

    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, ...],
        cond: Optional[torch.Tensor] = None,
        device: torch.device = None,
        generator: Optional[torch.Generator] = None,
        return_intermediates: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, list]]:
        """Generate samples using DDPM sampling.
        
        Args:
            shape: Shape of samples to generate [B, ...]
            cond: Optional conditioning
            device: Device to use
            generator: Random generator for reproducibility
            return_intermediates: Whether to return intermediate samples
            
        Returns:
            Generated samples [B, ...]
        """
        device = device or next(self.model.parameters()).device
        
        # Move scheduler to device
        if self.noise_scheduler.betas.device != device:
            self.noise_scheduler.to(device)
        
        # Start from pure noise
        x = torch.randn(shape, device=device, generator=generator)
        
        intermediates = [x] if return_intermediates else None
        
        # Denoise step by step
        for t in reversed(range(self.num_timesteps)):
            timesteps = torch.full((shape[0],), t, device=device, dtype=torch.long)
            
            # Predict noise
            model_output = self.model(x, timesteps, cond)
            
            # Denoise
            x, pred_x0 = self.noise_scheduler.step(model_output, t, x, generator=generator)
            
            if return_intermediates and t % 100 == 0:
                intermediates.append(x.clone())
        
        if return_intermediates:
            return x, intermediates
        return x

    @torch.no_grad()
    def sample_ddim(
        self,
        shape: Tuple[int, ...],
        num_inference_steps: int = 50,
        eta: float = 0.0,
        cond: Optional[torch.Tensor] = None,
        device: torch.device = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Generate samples using DDIM sampling (faster).
        
        Args:
            shape: Shape of samples to generate
            num_inference_steps: Number of denoising steps (can be much less than num_timesteps)
            eta: DDIM eta parameter (0 = deterministic)
            cond: Optional conditioning
            device: Device to use
            generator: Random generator
            
        Returns:
            Generated samples
        """
        device = device or next(self.model.parameters()).device
        
        # Create DDIM scheduler
        ddim_scheduler = DDIMScheduler(
            num_timesteps=self.num_timesteps,
            beta_start=self.noise_scheduler.beta_start,
            beta_end=self.noise_scheduler.beta_end,
            beta_schedule=self.noise_scheduler.beta_schedule,
            prediction_type=self.prediction_type,
            eta=eta,
        )
        ddim_scheduler.to(device)
        
        # Get timesteps for inference
        timesteps = ddim_scheduler.set_timesteps(num_inference_steps)
        
        # Start from pure noise
        x = torch.randn(shape, device=device, generator=generator)
        
        # Denoise
        for i, t in enumerate(timesteps):
            timestep_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            
            model_output = self.model(x, timestep_batch, cond)
            
            prev_t = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            x, _ = ddim_scheduler.step(model_output, t.item(), x, prev_timestep=prev_t.item() if prev_t >= 0 else None, generator=generator)
        
        return x


__all__ = [
    "BaseDiffusionModel",
    "DDPM",
]

