"""Noise schedulers for diffusion models.

Implements:
- DDPM (Denoising Diffusion Probabilistic Models) scheduler
- DDIM (Denoising Diffusion Implicit Models) scheduler
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn


class NoiseScheduler(ABC):
    """Abstract base class for noise schedulers."""
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        clip_sample: bool = True,
        prediction_type: str = "epsilon",  # "epsilon" or "v_prediction" or "sample"
    ):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.clip_sample = clip_sample
        self.prediction_type = prediction_type
        
        # Compute beta schedule
        self.betas = self._get_beta_schedule()
        
        # Compute alpha schedule
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        
        # Precompute values for q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Precompute values for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )

    def _get_beta_schedule(self) -> torch.Tensor:
        """Generate beta schedule."""
        if self.beta_schedule == "linear":
            return torch.linspace(self.beta_start, self.beta_end, self.num_timesteps)
        elif self.beta_schedule == "cosine":
            return self._cosine_beta_schedule()
        elif self.beta_schedule == "quadratic":
            return torch.linspace(self.beta_start ** 0.5, self.beta_end ** 0.5, self.num_timesteps) ** 2
        elif self.beta_schedule == "sigmoid":
            betas = torch.linspace(-6, 6, self.num_timesteps)
            return torch.sigmoid(betas) * (self.beta_end - self.beta_start) + self.beta_start
        else:
            raise ValueError(f"Unknown beta schedule: {self.beta_schedule}")

    def _cosine_beta_schedule(self, s: float = 0.008) -> torch.Tensor:
        """Cosine schedule as proposed in https://arxiv.org/abs/2102.09672."""
        steps = self.num_timesteps + 1
        x = torch.linspace(0, self.num_timesteps, steps)
        alphas_cumprod = torch.cos(((x / self.num_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0.0001, 0.9999)

    def to(self, device: torch.device) -> "NoiseScheduler":
        """Move scheduler tensors to device."""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        self.posterior_log_variance_clipped = self.posterior_log_variance_clipped.to(device)
        self.posterior_mean_coef1 = self.posterior_mean_coef1.to(device)
        self.posterior_mean_coef2 = self.posterior_mean_coef2.to(device)
        return self

    def add_noise(
        self,
        x_0: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Add noise to samples according to the forward diffusion process.
        
        q(x_t | x_0) = N(x_t; sqrt(alpha_cumprod_t) * x_0, (1 - alpha_cumprod_t) * I)
        
        Args:
            x_0: Clean samples [B, ...]
            noise: Gaussian noise [B, ...]
            timesteps: Timesteps [B]
            
        Returns:
            Noisy samples x_t
        """
        sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[timesteps]
        
        # Reshape for broadcasting
        while len(sqrt_alpha_cumprod.shape) < len(x_0.shape):
            sqrt_alpha_cumprod = sqrt_alpha_cumprod.unsqueeze(-1)
            sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.unsqueeze(-1)
        
        return sqrt_alpha_cumprod * x_0 + sqrt_one_minus_alpha_cumprod * noise

    @abstractmethod
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform one denoising step.
        
        Args:
            model_output: Output from the denoising model
            timestep: Current timestep
            sample: Current noisy sample x_t
            
        Returns:
            Tuple of (predicted x_{t-1}, predicted x_0)
        """
        pass

    def get_velocity(
        self,
        x_0: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Get velocity for v-prediction parameterization.
        
        v = sqrt(alpha_cumprod) * noise - sqrt(1 - alpha_cumprod) * x_0
        """
        sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[timesteps]
        
        while len(sqrt_alpha_cumprod.shape) < len(x_0.shape):
            sqrt_alpha_cumprod = sqrt_alpha_cumprod.unsqueeze(-1)
            sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.unsqueeze(-1)
        
        return sqrt_alpha_cumprod * noise - sqrt_one_minus_alpha_cumprod * x_0


class DDPMScheduler(NoiseScheduler):
    """DDPM scheduler with stochastic sampling."""
    
    def __init__(self, variance_type: str = "fixed_small", **kwargs):
        super().__init__(**kwargs)
        self.variance_type = variance_type

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform one DDPM denoising step."""
        t = timestep
        
        # Get coefficients
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod_prev[t]
        beta_t = self.betas[t]
        
        # Predict x_0
        if self.prediction_type == "epsilon":
            # model_output is epsilon
            pred_x_0 = (sample - torch.sqrt(1 - alpha_prod_t) * model_output) / torch.sqrt(alpha_prod_t)
        elif self.prediction_type == "sample":
            # model_output is x_0
            pred_x_0 = model_output
        elif self.prediction_type == "v_prediction":
            # model_output is v
            pred_x_0 = torch.sqrt(alpha_prod_t) * sample - torch.sqrt(1 - alpha_prod_t) * model_output
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")
        
        # Clip sample if needed
        if self.clip_sample:
            pred_x_0 = torch.clamp(pred_x_0, -1, 1)
        
        # Compute posterior mean
        pred_mean = (
            self.posterior_mean_coef1[t] * pred_x_0 +
            self.posterior_mean_coef2[t] * sample
        )
        
        # Get variance
        if t == 0:
            pred_prev_sample = pred_mean
        else:
            if self.variance_type == "fixed_small":
                variance = self.posterior_variance[t]
            elif self.variance_type == "fixed_large":
                variance = self.betas[t]
            else:
                variance = self.posterior_variance[t]
            
            noise = torch.randn(
                model_output.shape,
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            pred_prev_sample = pred_mean + torch.sqrt(variance) * noise
        
        return pred_prev_sample, pred_x_0


class DDIMScheduler(NoiseScheduler):
    """DDIM scheduler with deterministic sampling."""
    
    def __init__(self, eta: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.eta = eta  # 0 = deterministic, 1 = DDPM-like

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        prev_timestep: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform one DDIM denoising step."""
        t = timestep
        t_prev = prev_timestep if prev_timestep is not None else max(t - 1, 0)
        
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0)
        
        # Predict x_0
        if self.prediction_type == "epsilon":
            pred_x_0 = (sample - torch.sqrt(1 - alpha_prod_t) * model_output) / torch.sqrt(alpha_prod_t)
        elif self.prediction_type == "sample":
            pred_x_0 = model_output
        elif self.prediction_type == "v_prediction":
            pred_x_0 = torch.sqrt(alpha_prod_t) * sample - torch.sqrt(1 - alpha_prod_t) * model_output
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")
        
        # Clip sample if needed
        if self.clip_sample:
            pred_x_0 = torch.clamp(pred_x_0, -1, 1)
        
        # Compute variance
        variance = self._get_variance(t, t_prev)
        std_dev_t = self.eta * torch.sqrt(variance)
        
        # Direction pointing to x_t
        pred_epsilon = (sample - torch.sqrt(alpha_prod_t) * pred_x_0) / torch.sqrt(1 - alpha_prod_t)
        
        # Compute x_{t-1}
        pred_prev_sample = (
            torch.sqrt(alpha_prod_t_prev) * pred_x_0 +
            torch.sqrt(1 - alpha_prod_t_prev - std_dev_t ** 2) * pred_epsilon
        )
        
        # Add noise if eta > 0
        if self.eta > 0 and t > 0:
            noise = torch.randn(
                model_output.shape,
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            pred_prev_sample = pred_prev_sample + std_dev_t * noise
        
        return pred_prev_sample, pred_x_0

    def _get_variance(self, t: int, t_prev: int) -> torch.Tensor:
        """Compute variance for DDIM."""
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0)
        
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        return variance

    def set_timesteps(self, num_inference_steps: int) -> torch.Tensor:
        """Set timesteps for inference (potentially fewer than training steps).
        
        Args:
            num_inference_steps: Number of steps to use for inference
            
        Returns:
            Tensor of timesteps to use
        """
        step_ratio = self.num_timesteps // num_inference_steps
        timesteps = torch.arange(0, num_inference_steps) * step_ratio
        timesteps = timesteps.flip(0)  # Reverse for denoising
        return timesteps.long()


__all__ = [
    "NoiseScheduler",
    "DDPMScheduler",
    "DDIMScheduler",
]

