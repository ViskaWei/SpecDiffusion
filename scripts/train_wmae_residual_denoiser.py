#!/usr/bin/env python3
"""
Training Script for wMAE Residual Denoiser (MVP-0.6)

Experiment ID: SD-20251204-diff-wmae-01
MVP Source: MVP-0.6
Description: Residual denoiser with weighted MAE loss for low-noise stellar spectra

Key design principles:
1. Noise model: y = x₀ + s·σ⊙ε, where s ∈ {0.0, 0.05, 0.1, 0.2}
2. Residual structure: x̂₀ = y + s·g_θ(y, s, σ)
   - s=0 → strict identity (x̂₀ = y)
   - s small → small corrections
   - s=0.2 → actual denoising
3. Loss function: wMAE = (1/N) Σ |x̂₀ᵢ - x₀ᵢ| / σᵢ
   - High SNR pixels (small σ) have larger weights → preserve good regions
   - Low SNR pixels have smaller weights → don't overfit noisy regions

Usage:
    python scripts/train_wmae_residual_denoiser.py --epochs 50
    python scripts/train_wmae_residual_denoiser.py --epochs 50 --s-levels 0.0 0.05 0.1 0.2

Author: Viska Wei
Date: 2025-12-04
"""

import os
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.diffusion.utils import EMA

# Set matplotlib style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


# ============================================
# Configuration
# ============================================

DEFAULT_DATA_ROOT = "/srv/local/tmp/swei20/data/bosz50000/z0"
KNOWLEDGE_CENTER = "/home/swei20/Physics_Informed_AI"

EXP_ID = "SD-20251204-diff-wmae-01"
EXP_NAME = "wMAE Residual Denoiser for Low-Noise Stellar Spectra"


# ============================================
# Sinusoidal Embedding for Noise Level s
# ============================================

class SinusoidalEmbedding(nn.Module):
    """Sinusoidal embedding for continuous noise level s."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s: (B,) tensor of noise levels in [0, 1]
        Returns:
            (B, dim) embedding
        """
        device = s.device
        half_dim = self.dim // 2
        
        # Scale s to range suitable for sinusoidal encoding
        # s ∈ [0, 0.2] -> scale to [0, 1000] for better frequency spread
        s_scaled = s * 5000
        
        embeddings = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = s_scaled[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        
        return embeddings


# ============================================
# Residual Denoiser Network
# ============================================

class GroupNorm32(nn.GroupNorm):
    """GroupNorm with 32 groups."""
    
    def __init__(self, num_channels: int):
        num_groups = min(32, num_channels)
        while num_channels % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        super().__init__(num_groups, num_channels)


class Swish(nn.Module):
    """Swish activation (SiLU)."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class ResBlock1D(nn.Module):
    """1D Residual block with noise level embedding injection."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.norm1 = GroupNorm32(in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.emb_proj = nn.Sequential(
            Swish(),
            nn.Linear(emb_dim, out_channels),
        )
        
        self.norm2 = GroupNorm32(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        
        if in_channels != out_channels:
            self.skip_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_conv = nn.Identity()
        
        self.act = Swish()
    
    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        
        # Add noise level embedding
        h = h + self.emb_proj(emb)[:, :, None]
        
        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.skip_conv(x)


class ConditionalResidualNet1D(nn.Module):
    """
    Conditional Residual Network for spectrum denoising.
    
    Core design: x̂₀ = y + s·g_θ(y, s, σ)
    
    - y: noisy observation
    - s: noise level factor
    - σ: per-pixel error vector
    - g_θ: learned residual function
    
    When s=0, x̂₀ = y (strict identity)
    When s>0, applies learned correction scaled by s
    """
    
    def __init__(
        self,
        in_channels: int = 2,  # [y, σ] concatenated
        out_channels: int = 1,  # residual prediction
        base_channels: int = 64,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        dropout: float = 0.1,
        emb_dim: int = 128,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        
        num_levels = len(channel_mults)
        
        # Noise level s embedding
        self.s_embedding = nn.Sequential(
            SinusoidalEmbedding(emb_dim),
            nn.Linear(emb_dim, emb_dim * 4),
            Swish(),
            nn.Linear(emb_dim * 4, emb_dim),
        )
        
        # Initial convolution
        self.conv_in = nn.Conv1d(in_channels, base_channels, kernel_size=7, padding=3)
        
        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        
        ch = base_channels
        self.skip_channels = []
        
        for level in range(num_levels):
            out_ch = base_channels * channel_mults[level]
            
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResBlock1D(ch, out_ch, emb_dim, dropout))
                ch = out_ch
            
            self.encoder_blocks.append(blocks)
            self.skip_channels.append(ch)
            
            # Downsample except at last level
            if level < num_levels - 1:
                self.downsamplers.append(
                    nn.Conv1d(ch, ch, kernel_size=3, stride=2, padding=1)
                )
            else:
                self.downsamplers.append(nn.Identity())
        
        # Middle block
        self.middle_block1 = ResBlock1D(ch, ch, emb_dim, dropout)
        self.middle_block2 = ResBlock1D(ch, ch, emb_dim, dropout)
        
        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        
        for level in reversed(range(num_levels)):
            out_ch = base_channels * channel_mults[level]
            skip_ch = self.skip_channels[level]
            
            blocks = nn.ModuleList()
            for i in range(num_res_blocks + 1):
                in_ch = ch + skip_ch if i == 0 else out_ch
                blocks.append(ResBlock1D(in_ch, out_ch, emb_dim, dropout))
                if i == 0:
                    ch = out_ch
            
            self.decoder_blocks.append(blocks)
            ch = out_ch
            
            # Upsample except at first level
            if level > 0:
                self.upsamplers.append(nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv1d(ch, ch, kernel_size=3, padding=1),
                ))
            else:
                self.upsamplers.append(nn.Identity())
        
        # Output
        self.norm_out = GroupNorm32(ch)
        self.act_out = Swish()
        self.conv_out = nn.Conv1d(ch, out_channels, kernel_size=3, padding=1)
        
        # Zero initialize output for residual learning
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)
    
    def forward(
        self,
        y: torch.Tensor,
        s: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass computing residual prediction.
        
        Args:
            y: (B, 1, L) noisy observation
            s: (B,) noise level factors
            sigma: (B, 1, L) per-pixel error vector
        
        Returns:
            (B, 1, L) predicted residual g_θ(y, s, σ)
        """
        # Get noise level embedding
        s_emb = self.s_embedding(s)
        
        # Concatenate input: [y, σ]
        x = torch.cat([y, sigma], dim=1)  # (B, 2, L)
        
        # Initial conv
        h = self.conv_in(x)
        
        # Encoder
        skips = []
        for blocks, downsample in zip(self.encoder_blocks, self.downsamplers):
            for block in blocks:
                h = block(h, s_emb)
            skips.append(h)
            h = downsample(h)
        
        # Middle
        h = self.middle_block1(h, s_emb)
        h = self.middle_block2(h, s_emb)
        
        # Decoder
        for blocks, upsample in zip(self.decoder_blocks, self.upsamplers):
            skip = skips.pop()
            
            # Handle size mismatch
            if h.shape[-1] != skip.shape[-1]:
                h = F.interpolate(h, size=skip.shape[-1], mode='nearest')
            
            h = torch.cat([h, skip], dim=1)
            
            for block in blocks:
                h = block(h, s_emb)
            
            h = upsample(h)
        
        # Final size adjustment
        if h.shape[-1] != y.shape[-1]:
            h = F.interpolate(h, size=y.shape[-1], mode='nearest')
        
        # Output
        h = self.norm_out(h)
        h = self.act_out(h)
        residual = self.conv_out(h)
        
        return residual


class ResidualDenoiser(nn.Module):
    """
    Residual Denoiser wrapper that applies the key formula:
    
    x̂₀ = y + s·g_θ(y, s, σ)
    
    This ensures:
    - s=0 → x̂₀ = y (strict identity)
    - s>0 → applies scaled correction
    """
    
    def __init__(self, backbone: ConditionalResidualNet1D):
        super().__init__()
        self.backbone = backbone
    
    def forward(
        self,
        y: torch.Tensor,
        s: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            y: (B, 1, L) noisy observation
            s: (B,) noise level factors
            sigma: (B, 1, L) per-pixel error
        
        Returns:
            (B, 1, L) denoised prediction x̂₀
        """
        # Get residual from backbone
        residual = self.backbone(y, s, sigma)
        
        # Apply residual formula: x̂₀ = y + s·g_θ(y, s, σ)
        # s needs to be reshaped for broadcasting: (B,) -> (B, 1, 1)
        s_reshaped = s.view(-1, 1, 1)
        
        x_hat = y + s_reshaped * residual
        
        return x_hat


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================
# Dataset
# ============================================

class WMAESpectraDataset(Dataset):
    """
    Dataset for wMAE residual denoising.
    
    Implements noise model: y = x₀ + s·σ⊙ε
    where:
        - x₀: clean spectrum (physically normalized)
        - σ: per-pixel error vector
        - s: noise level ∈ {0.0, 0.05, 0.1, 0.2}
        - ε: standard Gaussian noise
    """
    
    def __init__(
        self,
        file_path: str,
        num_samples: int = 10000,
        snr_threshold: float = 50.0,
        s_levels: List[float] = [0.0, 0.05, 0.1, 0.2],
        s_sampling_weights: Optional[List[float]] = None,
        sigma_percentile_floor: float = 5.0,  # 5th percentile floor for σ
    ):
        self.file_path = file_path
        self.num_samples = num_samples
        self.snr_threshold = snr_threshold
        self.s_levels = s_levels
        self.sigma_percentile_floor = sigma_percentile_floor
        
        # Sampling weights: give s=0 and s=0.05 higher probability (50% total)
        if s_sampling_weights is None:
            # Default: s=0 (25%), s=0.05 (25%), s=0.1 (25%), s=0.2 (25%)
            # Actually let's give more weight to low noise: 30%, 25%, 25%, 20%
            self.s_sampling_weights = [0.30, 0.25, 0.25, 0.20]
        else:
            self.s_sampling_weights = s_sampling_weights
        
        # Normalize weights
        total_w = sum(self.s_sampling_weights)
        self.s_sampling_weights = [w/total_w for w in self.s_sampling_weights]
        
        self.clean_flux = None
        self.sigma = None
        self.wave = None
        self.sigma_floor = None
        
        self._load_data()
    
    def _load_data(self):
        """Load and preprocess data with PHYSICAL normalization."""
        print(f"Loading data from: {self.file_path}")
        
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Data file not found: {self.file_path}")
        
        with h5py.File(self.file_path, "r") as f:
            # Load wavelength grid
            self.wave = torch.tensor(f["spectrumdataset/wave"][()], dtype=torch.float32)
            
            # Load flux and error
            flux = torch.tensor(
                f["dataset/arrays/flux/value"][:self.num_samples],
                dtype=torch.float32
            )
            error = torch.tensor(
                f["dataset/arrays/error/value"][:self.num_samples],
                dtype=torch.float32
            )
        
        # Basic cleaning
        flux = flux.clip(min=0.0)
        flux = torch.nan_to_num(flux, nan=0.0)
        error = torch.nan_to_num(error, nan=1.0)
        error = error.clip(min=1e-10)  # Prevent division by zero
        
        # Calculate SNR per spectrum
        snr = flux.norm(dim=-1) / (error.norm(dim=-1) + 1e-8)
        print(f"SNR range: {snr.min():.1f} - {snr.max():.1f}, median: {snr.median():.1f}")
        
        # Filter high SNR spectra (use lower threshold since SNR values seem low)
        # If SNR values are generally low, just use all data
        if self.snr_threshold > 0:
            high_snr_mask = snr >= self.snr_threshold
            n_high_snr = high_snr_mask.sum().item()
            print(f"High SNR (>={self.snr_threshold}) samples: {n_high_snr}/{len(flux)}")
            
            if n_high_snr >= 1000:
                flux = flux[high_snr_mask]
                error = error[high_snr_mask]
            else:
                # Use top 50% by SNR if threshold yields too few samples
                snr_threshold_actual = snr.median().item()
                high_snr_mask = snr >= snr_threshold_actual
                flux = flux[high_snr_mask]
                error = error[high_snr_mask]
                print(f"Using median SNR threshold: {snr_threshold_actual:.2f}, got {len(flux)} samples")
        
        # PHYSICAL NORMALIZATION (global, not per-spectrum)
        # Use numpy for large tensor quantile (torch.quantile has size limits)
        flux_np = flux.numpy()
        flux_p95 = float(np.percentile(flux_np, 95))
        flux_p05 = float(np.percentile(flux_np, 5))
        flux_range = flux_p95 - flux_p05
        
        if flux_range < 1e-8:
            # Fallback to per-spectrum normalization
            print("Warning: Using per-spectrum normalization due to small flux range")
            flux_min = flux.min(dim=-1, keepdim=True).values
            flux_max = flux.max(dim=-1, keepdim=True).values
            flux_range_per = flux_max - flux_min
            flux_range_per = torch.where(flux_range_per < 1e-8, torch.ones_like(flux_range_per), flux_range_per)
            self.clean_flux = (flux - flux_min) / flux_range_per
            self.sigma = error / flux_range_per
        else:
            # Global normalization
            self.clean_flux = (flux - flux_p05) / flux_range
            self.sigma = error / flux_range
        
        # Scale to [-1, 1] for network
        self.clean_flux = self.clean_flux * 2 - 1
        self.sigma = self.sigma * 2
        
        # Apply sigma floor (5th percentile) to prevent division issues
        sigma_np = self.sigma.numpy()
        sigma_floor_value = float(np.percentile(sigma_np, self.sigma_percentile_floor))
        self.sigma_floor = max(sigma_floor_value, 0.001)  # Ensure minimum floor
        print(f"Sigma floor (p{self.sigma_percentile_floor}): {self.sigma_floor:.6f}")
        
        # Clip sigma to floor
        self.sigma = self.sigma.clip(min=self.sigma_floor)
        
        self.num_samples = len(self.clean_flux)
        self.spectrum_length = self.clean_flux.shape[1]
        
        print(f"Loaded {self.num_samples} spectra with {self.spectrum_length} wavelength points")
        print(f"Flux range: [{self.clean_flux.min():.3f}, {self.clean_flux.max():.3f}]")
        print(f"Sigma range: [{self.sigma.min():.4f}, {self.sigma.max():.4f}]")
        print(f"s levels: {self.s_levels}")
        print(f"s sampling weights: {self.s_sampling_weights}")
    
    def add_noise(
        self,
        x0: torch.Tensor,
        sigma: torch.Tensor,
        s: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise: y = x₀ + s·σ⊙ε
        
        Args:
            x0: clean spectrum
            sigma: per-pixel error
            s: noise level
        
        Returns:
            (noisy_spectrum, noise)
        """
        epsilon = torch.randn_like(x0)
        noise = s * sigma * epsilon
        y = x0 + noise
        return y, epsilon
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Returns:
            (x0, y, sigma, s) - clean, noisy, error, noise_level
        """
        x0 = self.clean_flux[idx].unsqueeze(0)  # (1, L)
        sigma = self.sigma[idx].unsqueeze(0)  # (1, L)
        
        # Sample s with weights
        s = np.random.choice(self.s_levels, p=self.s_sampling_weights)
        
        # Add noise
        y, _ = self.add_noise(x0, sigma, s)
        
        return x0, y, sigma, s


# ============================================
# Loss Functions
# ============================================

def weighted_mae_loss(
    x_hat: torch.Tensor,
    x0: torch.Tensor,
    sigma: torch.Tensor,
    sigma_floor: float = 0.01,
) -> torch.Tensor:
    """
    Weighted MAE loss (Laplace negative log-likelihood).
    
    L_wMAE = (1/N) Σ |x̂₀ᵢ - x₀ᵢ| / σᵢ
    
    Args:
        x_hat: (B, 1, L) predicted clean spectrum
        x0: (B, 1, L) ground truth clean spectrum
        sigma: (B, 1, L) per-pixel error
        sigma_floor: minimum sigma value (prevent division by zero)
    
    Returns:
        Scalar loss
    """
    # Ensure sigma has floor
    sigma_safe = sigma.clamp(min=sigma_floor)
    
    # Compute weighted absolute error
    abs_error = torch.abs(x_hat - x0)
    weighted_error = abs_error / sigma_safe
    
    return weighted_error.mean()


def weighted_mse_loss(
    x_hat: torch.Tensor,
    x0: torch.Tensor,
    sigma: torch.Tensor,
    sigma_floor: float = 0.01,
) -> torch.Tensor:
    """
    Weighted MSE loss for comparison.
    """
    sigma_safe = sigma.clamp(min=sigma_floor)
    sq_error = (x_hat - x0) ** 2
    weighted_error = sq_error / (sigma_safe ** 2)
    return weighted_error.mean()


# ============================================
# Trainer
# ============================================

class WMAEResidualTrainer:
    """Trainer for wMAE Residual Denoiser."""
    
    def __init__(
        self,
        model: ResidualDenoiser,
        train_dataset: WMAESpectraDataset,
        device: str = "cuda",
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_epochs: int = 5,
        gradient_clip: float = 1.0,
        ema_decay: float = 0.9999,
        save_dir: str = "lightning_logs/diffusion/wmae_residual",
    ):
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.device = device
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.warmup_epochs = warmup_epochs
        self.gradient_clip = gradient_clip
        
        # Output directories
        self.save_dir = Path(save_dir)
        self.checkpoint_dir = self.save_dir / "checkpoints"
        self.figures_dir = Path(KNOWLEDGE_CENTER) / "logg/diffusion/img"
        
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Data loader
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        
        # Scheduler
        warmup_steps = warmup_epochs * len(self.train_loader)
        total_steps = epochs * len(self.train_loader)
        
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=1e-6
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )
        
        # EMA
        self.ema = EMA(model, decay=ema_decay)
        
        # History
        self.history = {
            "train_loss": [],
            "epoch_loss": [],
            "lr": [],
        }
        
        # Dataset info
        self.wave = train_dataset.wave
        self.spectrum_length = train_dataset.spectrum_length
        self.s_levels = train_dataset.s_levels
        self.sigma_floor = train_dataset.sigma_floor
    
    def train_epoch(self, epoch: int) -> float:
        """Train one epoch."""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
        for batch_idx, (x0, y, sigma, s) in enumerate(pbar):
            x0 = x0.to(self.device)
            y = y.to(self.device)
            sigma = sigma.to(self.device)
            s = torch.tensor(s, dtype=torch.float32).to(self.device)
            
            # Forward pass
            x_hat = self.model(y, s, sigma)
            
            # Compute wMAE loss
            loss = weighted_mae_loss(x_hat, x0, sigma, self.sigma_floor)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
                )
            
            self.optimizer.step()
            self.scheduler.step()
            self.ema.update()
            
            # Logging
            total_loss += loss.item()
            self.history["train_loss"].append(loss.item())
            self.history["lr"].append(self.scheduler.get_last_lr()[0])
            
            pbar.set_postfix({
                "loss": f"{loss.item():.6f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
        
        avg_loss = total_loss / len(self.train_loader)
        self.history["epoch_loss"].append(avg_loss)
        return avg_loss
    
    @torch.no_grad()
    def evaluate(self) -> Dict[float, Dict[str, float]]:
        """Evaluate at each noise level."""
        self.ema.apply_shadow()
        self.model.eval()
        
        results = {}
        
        for s in self.s_levels:
            wmae_noisy_list = []
            wmae_denoised_list = []
            mse_noisy_list = []
            mse_denoised_list = []
            
            n_eval = min(200, len(self.train_dataset))
            indices = np.random.choice(len(self.train_dataset), n_eval, replace=False)
            
            for idx in indices:
                x0 = self.train_dataset.clean_flux[idx].unsqueeze(0).unsqueeze(0)
                sigma = self.train_dataset.sigma[idx].unsqueeze(0).unsqueeze(0)
                
                y, _ = self.train_dataset.add_noise(x0.squeeze(0), sigma.squeeze(0), s)
                y = y.unsqueeze(0)
                
                x0 = x0.to(self.device)
                y = y.to(self.device)
                sigma = sigma.to(self.device)
                s_tensor = torch.tensor([s], dtype=torch.float32).to(self.device)
                
                # Denoise
                x_hat = self.model(y, s_tensor, sigma)
                
                # Compute metrics
                wmae_noisy = weighted_mae_loss(y, x0, sigma, self.sigma_floor).item()
                wmae_denoised = weighted_mae_loss(x_hat, x0, sigma, self.sigma_floor).item()
                mse_noisy = F.mse_loss(y, x0).item()
                mse_denoised = F.mse_loss(x_hat, x0).item()
                
                wmae_noisy_list.append(wmae_noisy)
                wmae_denoised_list.append(wmae_denoised)
                mse_noisy_list.append(mse_noisy)
                mse_denoised_list.append(mse_denoised)
            
            avg_wmae_noisy = np.mean(wmae_noisy_list)
            avg_wmae_denoised = np.mean(wmae_denoised_list)
            avg_mse_noisy = np.mean(mse_noisy_list)
            avg_mse_denoised = np.mean(mse_denoised_list)
            
            # Compute improvement
            if avg_wmae_noisy > 0:
                wmae_improvement = (avg_wmae_noisy - avg_wmae_denoised) / avg_wmae_noisy
            else:
                wmae_improvement = 0.0
            
            results[s] = {
                "wmae_noisy": avg_wmae_noisy,
                "wmae_denoised": avg_wmae_denoised,
                "wmae_improvement": wmae_improvement,
                "mse_noisy": avg_mse_noisy,
                "mse_denoised": avg_mse_denoised,
            }
        
        self.ema.restore()
        return results
    
    def save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """Save checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "ema_shadow": self.ema.shadow,
            "loss": loss,
            "history": self.history,
            "config": {
                "s_levels": self.s_levels,
                "sigma_floor": self.sigma_floor,
            },
        }
        
        latest_path = self.checkpoint_dir / "checkpoint_latest.pt"
        torch.save(checkpoint, latest_path)
        
        if is_best:
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(checkpoint, best_path)
            print(f"💾 Saved best checkpoint: {best_path}")
    
    def train(self):
        """Full training loop."""
        print(f"\n{'='*60}")
        print(f"🚀 Starting training: {EXP_ID}")
        print(f"{'='*60}")
        print(f"Model parameters: {count_parameters(self.model):,}")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.epochs}")
        print(f"Batch size: {self.batch_size}")
        print(f"Learning rate: {self.lr}")
        print(f"s levels: {self.s_levels}")
        print(f"Output dir: {self.save_dir}")
        print(f"Figures dir: {self.figures_dir}")
        print(f"{'='*60}\n")
        
        best_loss = float("inf")
        
        for epoch in range(self.epochs):
            avg_loss = self.train_epoch(epoch)
            print(f"\n📊 Epoch {epoch+1}/{self.epochs} - Loss: {avg_loss:.6f}")
            
            is_best = avg_loss < best_loss
            if is_best:
                best_loss = avg_loss
            self.save_checkpoint(epoch, avg_loss, is_best=is_best)
            
            # Evaluate periodically
            if (epoch + 1) % 10 == 0 or epoch == self.epochs - 1:
                print("🔍 Evaluating...")
                results = self.evaluate()
                for s, metrics in sorted(results.items()):
                    print(f"  s={s:.2f}: wMAE(noisy)={metrics['wmae_noisy']:.4f}, "
                          f"wMAE(denoised)={metrics['wmae_denoised']:.4f}, "
                          f"Improvement={metrics['wmae_improvement']*100:.1f}%")
                
                print("📈 Generating visualizations...")
                self.plot_denoising_samples(epoch)
        
        # Final visualizations
        print("\n📊 Generating final visualizations...")
        self.plot_loss_curve()
        final_results = self.evaluate()
        self.plot_wmae_comparison(final_results)
        self.plot_residual_distribution(final_results)
        self.plot_spectra_per_s_level(final_results)
        
        self.save_summary(final_results)
        
        print(f"\n{'='*60}")
        print("✅ Training completed!")
        print(f"Best loss: {best_loss:.6f}")
        print(f"{'='*60}\n")
        
        return self.history, final_results
    
    # ============================================
    # Visualization
    # ============================================
    
    def plot_loss_curve(self):
        """Plot training loss curve (fig_1)."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Per-step loss
        ax1 = axes[0]
        steps = np.arange(len(self.history["train_loss"]))
        ax1.plot(steps, self.history["train_loss"], alpha=0.5, linewidth=0.5, color='steelblue')
        
        window = min(100, len(self.history["train_loss"]) // 10)
        if window > 1:
            smoothed = np.convolve(
                self.history["train_loss"],
                np.ones(window)/window,
                mode='valid'
            )
            ax1.plot(steps[window-1:], smoothed, color='crimson', linewidth=2, label='Smoothed')
        
        ax1.set_xlabel("Training Step", fontsize=12)
        ax1.set_ylabel("wMAE Loss", fontsize=12)
        ax1.set_title("Training Loss per Step", fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Per-epoch loss
        ax2 = axes[1]
        epochs = np.arange(1, len(self.history["epoch_loss"]) + 1)
        ax2.plot(epochs, self.history["epoch_loss"], marker='o', linewidth=2, color='teal')
        ax2.set_xlabel("Epoch", fontsize=12)
        ax2.set_ylabel("wMAE Loss", fontsize=12)
        ax2.set_title("Training Loss per Epoch", fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        fig.suptitle(f"{EXP_ID} - wMAE Loss Curve", fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        path = self.figures_dir / "diff_wmae_loss_curve.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Saved: {path}")
    
    def plot_wmae_comparison(self, results: Dict):
        """Plot wMAE comparison bar chart (fig_2)."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        s_values = sorted(results.keys())
        x = np.arange(len(s_values))
        width = 0.35
        
        wmae_noisy = [results[s]["wmae_noisy"] for s in s_values]
        wmae_denoised = [results[s]["wmae_denoised"] for s in s_values]
        
        bars1 = ax.bar(x - width/2, wmae_noisy, width, label='Noisy', color='gray', alpha=0.7)
        bars2 = ax.bar(x + width/2, wmae_denoised, width, label='Denoised', color='forestgreen', alpha=0.9)
        
        # Add improvement annotations
        for i, s in enumerate(s_values):
            improvement = results[s]["wmae_improvement"] * 100
            y_pos = max(wmae_noisy[i], wmae_denoised[i])
            color = 'green' if improvement > 0 else 'red'
            sign = '↓' if improvement > 0 else '↑'
            ax.annotate(f'{abs(improvement):.1f}%{sign}',
                       (i, y_pos + 0.02),
                       ha='center',
                       fontsize=10,
                       fontweight='bold',
                       color=color)
        
        ax.set_xlabel("Noise Level (s)", fontsize=12)
        ax.set_ylabel("wMAE", fontsize=12)
        ax.set_title(f"{EXP_ID}\nwMAE: Noisy vs Denoised", fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f"{s:.2f}" for s in s_values])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        path = self.figures_dir / "diff_wmae_comparison.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Saved: {path}")
    
    def plot_denoising_samples(self, epoch: int):
        """Plot denoising samples grid."""
        self.ema.apply_shadow()
        self.model.eval()
        
        n_samples = 3
        fig, axes = plt.subplots(n_samples, 3, figsize=(18, 4*n_samples))
        
        indices = np.random.choice(len(self.train_dataset), n_samples, replace=False)
        wave = self.wave.numpy()
        s_show = 0.2  # Show highest noise level
        
        for i, idx in enumerate(indices):
            x0 = self.train_dataset.clean_flux[idx].unsqueeze(0).unsqueeze(0)
            sigma = self.train_dataset.sigma[idx].unsqueeze(0).unsqueeze(0)
            
            y, _ = self.train_dataset.add_noise(x0.squeeze(0), sigma.squeeze(0), s_show)
            y = y.unsqueeze(0)
            
            x0_gpu = x0.to(self.device)
            y_gpu = y.to(self.device)
            sigma_gpu = sigma.to(self.device)
            s_tensor = torch.tensor([s_show], dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                x_hat = self.model(y_gpu, s_tensor, sigma_gpu)
            
            x0_np = x0.squeeze().numpy()
            y_np = y.squeeze().numpy()
            x_hat_np = x_hat.cpu().squeeze().numpy()
            
            # Clean
            axes[i, 0].plot(wave, x0_np, color='royalblue', linewidth=0.8)
            axes[i, 0].set_ylabel(f"Sample {i+1}", fontsize=11, fontweight='bold')
            if i == 0:
                axes[i, 0].set_title("Clean (x₀)", fontsize=12)
            axes[i, 0].grid(True, alpha=0.3)
            
            # Noisy
            axes[i, 1].plot(wave, y_np, color='gray', linewidth=0.8, alpha=0.8)
            axes[i, 1].plot(wave, x0_np, color='royalblue', linewidth=0.5, alpha=0.3)
            if i == 0:
                axes[i, 1].set_title(f"Noisy (y, s={s_show})", fontsize=12)
            axes[i, 1].grid(True, alpha=0.3)
            
            # Denoised
            axes[i, 2].plot(wave, x_hat_np, color='forestgreen', linewidth=0.8)
            axes[i, 2].plot(wave, x0_np, color='royalblue', linewidth=0.5, alpha=0.3)
            wmae = weighted_mae_loss(
                torch.tensor(x_hat_np).unsqueeze(0).unsqueeze(0),
                torch.tensor(x0_np).unsqueeze(0).unsqueeze(0),
                sigma,
                self.sigma_floor
            ).item()
            if i == 0:
                axes[i, 2].set_title(f"Denoised (x̂₀), wMAE={wmae:.4f}", fontsize=12)
            else:
                axes[i, 2].set_title(f"wMAE={wmae:.4f}", fontsize=11)
            axes[i, 2].grid(True, alpha=0.3)
        
        for j in range(3):
            axes[-1, j].set_xlabel("Wavelength (Å)", fontsize=11)
        
        fig.suptitle(f"{EXP_ID} - Denoising Results (Epoch {epoch+1})", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        path = self.figures_dir / "diff_wmae_denoising_samples.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Saved: {path}")
        
        self.ema.restore()
    
    def plot_residual_distribution(self, results: Dict):
        """Plot (denoised - clean) / σ distribution (fig_4)."""
        self.ema.apply_shadow()
        self.model.eval()
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        n_samples = 100
        indices = np.random.choice(len(self.train_dataset), n_samples, replace=False)
        
        for ax_idx, s in enumerate(self.s_levels):
            residuals = []
            
            for idx in indices:
                x0 = self.train_dataset.clean_flux[idx].unsqueeze(0).unsqueeze(0)
                sigma = self.train_dataset.sigma[idx].unsqueeze(0).unsqueeze(0)
                
                y, _ = self.train_dataset.add_noise(x0.squeeze(0), sigma.squeeze(0), s)
                y = y.unsqueeze(0)
                
                x0 = x0.to(self.device)
                y = y.to(self.device)
                sigma_dev = sigma.to(self.device)
                s_tensor = torch.tensor([s], dtype=torch.float32).to(self.device)
                
                with torch.no_grad():
                    x_hat = self.model(y, s_tensor, sigma_dev)
                
                # Compute normalized residual
                res = ((x_hat - x0) / sigma_dev.clamp(min=self.sigma_floor)).cpu().flatten().numpy()
                residuals.extend(res)
            
            # Subsample for plotting
            residuals = np.random.choice(residuals, min(50000, len(residuals)), replace=False)
            
            ax = axes[ax_idx]
            ax.hist(residuals, bins=100, density=True, alpha=0.7, color='steelblue', edgecolor='black')
            
            # Fit Gaussian
            mu, std = np.mean(residuals), np.std(residuals)
            x_range = np.linspace(-5, 5, 200)
            gaussian = np.exp(-0.5 * ((x_range - mu) / std) ** 2) / (std * np.sqrt(2 * np.pi))
            ax.plot(x_range, gaussian, 'r-', linewidth=2, label=f'N({mu:.3f}, {std:.3f}²)')
            
            ax.axvline(0, color='black', linestyle='--', alpha=0.5)
            ax.set_xlabel("(x̂₀ - x₀) / σ", fontsize=11)
            ax.set_ylabel("Density", fontsize=11)
            ax.set_title(f"s = {s:.2f}", fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.set_xlim(-5, 5)
            ax.grid(True, alpha=0.3)
        
        fig.suptitle(f"{EXP_ID}\nNormalized Residual Distribution by Noise Level", 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        path = self.figures_dir / "diff_wmae_residual_dist.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Saved: {path}")
        
        self.ema.restore()
    
    def plot_spectra_per_s_level(self, results: Dict):
        """Plot spectra visualization for each s level (fig_3 a-d)."""
        self.ema.apply_shadow()
        self.model.eval()
        
        n_samples = 3
        indices = np.random.choice(len(self.train_dataset), n_samples, replace=False)
        wave = self.wave.numpy()
        
        for s in self.s_levels:
            fig, axes = plt.subplots(n_samples, 3, figsize=(18, 4*n_samples))
            
            for i, idx in enumerate(indices):
                x0 = self.train_dataset.clean_flux[idx].unsqueeze(0).unsqueeze(0)
                sigma = self.train_dataset.sigma[idx].unsqueeze(0).unsqueeze(0)
                
                y, _ = self.train_dataset.add_noise(x0.squeeze(0), sigma.squeeze(0), s)
                y = y.unsqueeze(0)
                
                x0_gpu = x0.to(self.device)
                y_gpu = y.to(self.device)
                sigma_gpu = sigma.to(self.device)
                s_tensor = torch.tensor([s], dtype=torch.float32).to(self.device)
                
                with torch.no_grad():
                    x_hat = self.model(y_gpu, s_tensor, sigma_gpu)
                
                x0_np = x0.squeeze().numpy()
                y_np = y.squeeze().numpy()
                x_hat_np = x_hat.cpu().squeeze().numpy()
                
                axes[i, 0].plot(wave, x0_np, color='royalblue', linewidth=0.8)
                axes[i, 0].set_ylabel(f"Sample {i+1}", fontsize=11)
                if i == 0:
                    axes[i, 0].set_title("Clean (x₀)", fontsize=12)
                
                axes[i, 1].plot(wave, y_np, color='gray', linewidth=0.8)
                if i == 0:
                    axes[i, 1].set_title("Noisy (y)", fontsize=12)
                
                axes[i, 2].plot(wave, x_hat_np, color='forestgreen', linewidth=0.8)
                if i == 0:
                    axes[i, 2].set_title("Denoised (x̂₀)", fontsize=12)
                
                for j in range(3):
                    axes[i, j].grid(True, alpha=0.3)
            
            for j in range(3):
                axes[-1, j].set_xlabel("Wavelength (Å)", fontsize=11)
            
            s_str = f"{s:.2f}".replace('.', 'p')
            fig.suptitle(f"{EXP_ID} - Spectra at s={s:.2f}", fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            path = self.figures_dir / f"diff_wmae_spectra_s{s_str}.png"
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"📊 Saved: {path}")
        
        self.ema.restore()
    
    def save_summary(self, results: Dict):
        """Save training summary."""
        summary = {
            "experiment_id": EXP_ID,
            "experiment_name": EXP_NAME,
            "date": datetime.now().isoformat(),
            "model": {
                "type": "ConditionalResidualNet1D + ResidualDenoiser",
                "formula": "x̂₀ = y + s·g_θ(y, s, σ)",
                "parameters": count_parameters(self.model),
            },
            "training": {
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "learning_rate": self.lr,
                "loss_function": "wMAE",
                "final_loss": self.history["epoch_loss"][-1] if self.history["epoch_loss"] else None,
                "best_loss": min(self.history["epoch_loss"]) if self.history["epoch_loss"] else None,
            },
            "data": {
                "num_samples": len(self.train_dataset),
                "spectrum_length": self.spectrum_length,
                "s_levels": self.s_levels,
                "sigma_floor": self.sigma_floor,
            },
            "results": {
                str(s): {
                    "wmae_noisy": m["wmae_noisy"],
                    "wmae_denoised": m["wmae_denoised"],
                    "wmae_improvement_pct": m["wmae_improvement"] * 100,
                    "mse_noisy": m["mse_noisy"],
                    "mse_denoised": m["mse_denoised"],
                } for s, m in results.items()
            },
            "success_criteria": {
                "s_0.0_identity": bool(results[0.0]["wmae_denoised"] < 0.1),  # Should be near 0
                "s_0.2_improvement_pct": float(results[0.2]["wmae_improvement"] * 100),
                "s_0.05_no_degradation": bool(results[0.05]["wmae_improvement"] >= -0.2),  # Not worse than 20%
                "s_0.1_no_degradation": bool(results[0.1]["wmae_improvement"] >= -0.2),
            },
        }
        
        path = self.save_dir / "training_summary.json"
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"📝 Saved: {path}")


# ============================================
# Main
# ============================================

def main():
    parser = argparse.ArgumentParser(description="Train wMAE Residual Denoiser (MVP-0.6)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--s-levels", type=float, nargs='+', default=[0.0, 0.05, 0.1, 0.2],
                       help="Noise levels s")
    parser.add_argument("--num-samples", type=int, default=10000, help="Training samples")
    parser.add_argument("--device", type=str, default=None, help="Device")
    parser.add_argument("--data-root", type=str, default=None, help="Data root")
    args = parser.parse_args()
    
    # Data file
    data_root = args.data_root or os.environ.get("DATA_ROOT", DEFAULT_DATA_ROOT)
    data_file_options = [
        os.path.join(data_root, "train_100k/dataset.h5"),
        os.path.join(data_root, "bosz50000_optical_R5000_synthetic_spectra.h5"),
        os.path.join(data_root, "dataset.h5"),
    ]
    data_file = None
    for opt in data_file_options:
        if os.path.exists(opt):
            data_file = opt
            break
    if data_file is None:
        data_file = data_file_options[0]
    
    print(f"Data file: {data_file}")
    
    # Device
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Dataset
    dataset = WMAESpectraDataset(
        file_path=data_file,
        num_samples=args.num_samples,
        snr_threshold=50.0,
        s_levels=args.s_levels,
    )
    
    # Model
    backbone = ConditionalResidualNet1D(
        in_channels=2,  # [y, σ]
        out_channels=1,  # residual
        base_channels=64,
        channel_mults=(1, 2, 4, 4),
        num_res_blocks=2,
        dropout=0.1,
        emb_dim=128,
    )
    model = ResidualDenoiser(backbone)
    
    # Trainer
    trainer = WMAEResidualTrainer(
        model=model,
        train_dataset=dataset,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    
    # Train
    history, results = trainer.train()
    
    # Print final results
    print("\n" + "="*60)
    print("📊 FINAL RESULTS")
    print("="*60)
    for s, metrics in sorted(results.items()):
        print(f"s={s:.2f}:")
        print(f"  wMAE(noisy):    {metrics['wmae_noisy']:.6f}")
        print(f"  wMAE(denoised): {metrics['wmae_denoised']:.6f}")
        print(f"  Improvement:    {metrics['wmae_improvement']*100:.1f}%")
    print("="*60)
    
    # Check success criteria
    print(f"\n🎯 MVP-0.6 SUCCESS CRITERIA CHECK:")
    
    # s=0 identity
    s0_wmae = results[0.0]["wmae_denoised"]
    s0_pass = s0_wmae < 0.1
    print(f"  s=0 identity (wMAE≈0): {s0_wmae:.6f} {'✅ PASS' if s0_pass else '❌ FAIL'}")
    
    # s=0.2 improvement
    s02_imp = results[0.2]["wmae_improvement"] * 100
    s02_pass = s02_imp >= 10
    print(f"  s=0.2 improvement (≥10%): {s02_imp:.1f}% {'✅ PASS' if s02_pass else '❌ FAIL'}")
    
    # s=0.05 no degradation
    s005_imp = results[0.05]["wmae_improvement"] * 100
    s005_pass = s005_imp >= -20
    print(f"  s=0.05 no degradation (>-20%): {s005_imp:.1f}% {'✅ PASS' if s005_pass else '❌ FAIL'}")
    
    # s=0.1 no degradation
    s01_imp = results[0.1]["wmae_improvement"] * 100
    s01_pass = s01_imp >= -20
    print(f"  s=0.1 no degradation (>-20%): {s01_imp:.1f}% {'✅ PASS' if s01_pass else '❌ FAIL'}")
    
    overall = s0_pass and s02_pass and s005_pass and s01_pass
    print(f"\n{'✅ MVP-0.6 PASSED!' if overall else '❌ MVP-0.6 FAILED'}")
    
    return history, results


if __name__ == "__main__":
    main()
