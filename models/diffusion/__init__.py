"""
1D Diffusion Models for Stellar Spectra

This module provides implementations of:
- 1D U-Net architecture for noise prediction
- DDPM (Denoising Diffusion Probabilistic Models) training and sampling
- Conditional U-Net for supervised denoising
- Conditional DDPM for supervised spectrum denoising
"""

from .unet_1d import UNet1D, count_parameters
from .ddpm import DDPM, GaussianDiffusion
from .conditional_unet_1d import ConditionalUNet1D
from .conditional_ddpm import ConditionalDDPM, ConditionalGaussianDiffusion
from .utils import (
    SinusoidalPositionEmbeddings,
    linear_beta_schedule,
    cosine_beta_schedule,
    extract,
    EMA,
    normalize_spectrum,
    denormalize_spectrum,
)

__all__ = [
    "UNet1D",
    "ConditionalUNet1D",
    "count_parameters",
    "DDPM",
    "GaussianDiffusion",
    "ConditionalDDPM",
    "ConditionalGaussianDiffusion",
    "SinusoidalPositionEmbeddings",
    "linear_beta_schedule",
    "cosine_beta_schedule",
    "extract",
    "EMA",
    "normalize_spectrum",
    "denormalize_spectrum",
]

