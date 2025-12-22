"""
Utility functions for 1D Diffusion Models

Includes:
- Sinusoidal position embeddings for timestep encoding
- Beta schedules (linear, cosine)
- Extraction helper for alpha/beta values
- EMA for model parameters
- Spectrum normalization utilities
"""

import math
import torch
import torch.nn as nn
import numpy as np


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal positional embeddings for diffusion timesteps.
    
    Based on the positional encoding from "Attention Is All You Need",
    adapted for scalar timestep inputs.
    
    Args:
        dim: Embedding dimension (will be split between sin and cos)
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: (B,) tensor of integer timesteps
            
        Returns:
            (B, dim) tensor of embeddings
        """
        device = timesteps.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = timesteps[:, None].float() * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        
        # Handle odd dimensions
        if self.dim % 2 == 1:
            embeddings = torch.cat([embeddings, torch.zeros_like(embeddings[:, :1])], dim=-1)
        
        return embeddings


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """
    Linear beta schedule from DDPM paper.
    
    Args:
        timesteps: Number of diffusion timesteps
        beta_start: Starting beta value
        beta_end: Ending beta value
        
    Returns:
        (timesteps,) tensor of beta values
    """
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine beta schedule from "Improved DDPM" paper.
    
    Args:
        timesteps: Number of diffusion timesteps
        s: Small offset to prevent beta from being too small at t=0
        
    Returns:
        (timesteps,) tensor of beta values
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
    """
    Extract values from a tensor at specified timesteps.
    
    Args:
        a: (T,) tensor of values to extract from
        t: (B,) tensor of timestep indices
        x_shape: Shape of x tensor for broadcasting
        
    Returns:
        (B, 1, 1, ...) tensor of extracted values
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    # Reshape for broadcasting: (B,) -> (B, 1, 1, ...)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


class EMA:
    """
    Exponential Moving Average for model parameters.
    
    Args:
        model: The model to track
        decay: EMA decay rate (default: 0.9999)
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update EMA parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """Apply EMA parameters to model."""
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


def normalize_spectrum(flux: torch.Tensor, method: str = "minmax") -> tuple:
    """
    Normalize stellar spectrum for diffusion training.
    
    Args:
        flux: (N, L) tensor of flux values
        method: Normalization method ('minmax', 'standard', 'continuum')
        
    Returns:
        Tuple of (normalized_flux, normalization_params)
    """
    if method == "minmax":
        flux_min = flux.min(dim=-1, keepdim=True).values
        flux_max = flux.max(dim=-1, keepdim=True).values
        flux_range = flux_max - flux_min
        flux_range = torch.where(flux_range < 1e-8, torch.ones_like(flux_range), flux_range)
        normalized = (flux - flux_min) / flux_range
        params = {"min": flux_min, "max": flux_max}
    elif method == "standard":
        flux_mean = flux.mean(dim=-1, keepdim=True)
        flux_std = flux.std(dim=-1, keepdim=True)
        flux_std = torch.where(flux_std < 1e-8, torch.ones_like(flux_std), flux_std)
        normalized = (flux - flux_mean) / flux_std
        params = {"mean": flux_mean, "std": flux_std}
    elif method == "continuum":
        # Simple percentile-based continuum normalization
        # Use 95th percentile as pseudo-continuum
        continuum = torch.quantile(flux, 0.95, dim=-1, keepdim=True)
        continuum = torch.where(continuum < 1e-8, torch.ones_like(continuum), continuum)
        normalized = flux / continuum
        params = {"continuum": continuum}
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized, params


def denormalize_spectrum(normalized: torch.Tensor, params: dict, method: str = "minmax") -> torch.Tensor:
    """
    Denormalize stellar spectrum back to original scale.
    
    Args:
        normalized: (N, L) tensor of normalized flux values
        params: Normalization parameters from normalize_spectrum
        method: Normalization method used
        
    Returns:
        Denormalized flux tensor
    """
    if method == "minmax":
        flux_range = params["max"] - params["min"]
        return normalized * flux_range + params["min"]
    elif method == "standard":
        return normalized * params["std"] + params["mean"]
    elif method == "continuum":
        return normalized * params["continuum"]
    else:
        raise ValueError(f"Unknown normalization method: {method}")


__all__ = [
    "SinusoidalPositionEmbeddings",
    "linear_beta_schedule",
    "cosine_beta_schedule",
    "extract",
    "EMA",
    "normalize_spectrum",
    "denormalize_spectrum",
]

