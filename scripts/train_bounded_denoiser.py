#!/usr/bin/env python3
"""
Training Script for Bounded Noise Denoiser (MVP-0.5)

Experiment ID: SD-20251204-diff-bounded-01
MVP Source: MVP-0.5
Description: Bounded noise denoiser that doesn't go to pure noise, 
             starts from observation and uses known per-pixel σ.

Key differences from standard DDPM:
- Noise model: y = x0 + λ * σ ⊙ ε, where λ ∈ [0, 0.5]
- Input: [x_t, σ] (2 channels)
- Inference: Start from observation (not pure noise)
- Steps: 10-20 (not 1000)

Usage:
    python scripts/train_bounded_denoiser.py --epochs 50
    python scripts/train_bounded_denoiser.py --epochs 50 --target eps

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

from models.diffusion.conditional_unet_1d import ConditionalUNet1D, count_parameters
from models.diffusion.utils import EMA, normalize_spectrum, denormalize_spectrum

# Set matplotlib style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


# ============================================
# Configuration
# ============================================

DEFAULT_DATA_ROOT = "/srv/local/tmp/swei20/data/bosz50000/z0"
KNOWLEDGE_CENTER = "/home/swei20/Physics_Informed_AI"

EXP_ID = "SD-20251204-diff-bounded-01"
EXP_NAME = "Bounded Noise Multi-Level Denoiser"


# ============================================
# Dataset
# ============================================

class BoundedNoiseSpectraDataset(Dataset):
    """
    Dataset for bounded noise spectrum denoising.
    
    Implements the noise model: y = x0 + λ * σ ⊙ ε
    where:
        - x0: clean spectrum
        - σ: per-pixel error vector (known)
        - λ: noise level factor ∈ [0.1, 0.2, 0.3, 0.4, 0.5]
        - ε: standard Gaussian noise
    """
    
    def __init__(
        self,
        file_path: str,
        num_samples: int = 10000,
        normalization: str = "minmax",
        snr_threshold: float = 50.0,
        lambda_values: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5],
        sigma_scale: float = 0.1,  # Scale factor for sigma
    ):
        self.file_path = file_path
        self.num_samples = num_samples
        self.normalization = normalization
        self.snr_threshold = snr_threshold
        self.lambda_values = lambda_values
        self.sigma_scale = sigma_scale
        
        self.clean_flux = None
        self.sigma = None  # Per-pixel error vector
        self.wave = None
        self.norm_params = None
        
        self._load_data()
    
    def _load_data(self):
        """Load and preprocess the data."""
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
        
        # Clip negative flux values
        flux = flux.clip(min=0.0)
        
        # Handle NaN values
        flux = torch.nan_to_num(flux, nan=0.0)
        error = torch.nan_to_num(error, nan=1.0)
        
        # Calculate SNR and filter for high SNR (clean) spectra
        snr = flux.norm(dim=-1) / error.norm(dim=-1)
        print(f"SNR range: {snr.min():.1f} - {snr.max():.1f}, median: {snr.median():.1f}")
        
        # Filter by SNR to get clean spectra
        if self.snr_threshold > 0:
            high_snr_mask = snr >= self.snr_threshold
            n_high_snr = high_snr_mask.sum().item()
            print(f"High SNR (>={self.snr_threshold}) samples: {n_high_snr}/{len(flux)}")
            
            if n_high_snr > 1000:
                flux = flux[high_snr_mask]
                error = error[high_snr_mask]
        
        # Normalize flux to [0, 1] range
        self.clean_flux, self.norm_params = normalize_spectrum(flux, method=self.normalization)
        
        # Scale from [0, 1] to [-1, 1] for training
        self.clean_flux = self.clean_flux * 2 - 1
        
        # Compute normalized sigma (per-pixel error vector)
        # Normalize error relative to flux range
        if self.normalization == "minmax":
            flux_range = self.norm_params["max"] - self.norm_params["min"]
            flux_range = torch.where(flux_range < 1e-8, torch.ones_like(flux_range), flux_range)
            self.sigma = (error / flux_range) * 2  # Scale to match [-1, 1] range
        else:
            self.sigma = error / (error.mean(dim=-1, keepdim=True) + 1e-8)
        
        # Apply additional scale factor
        self.sigma = self.sigma * self.sigma_scale
        
        # Clip sigma to reasonable range
        self.sigma = self.sigma.clip(min=0.01, max=1.0)
        
        self.num_samples = len(self.clean_flux)
        self.spectrum_length = self.clean_flux.shape[1]
        
        print(f"Loaded {self.num_samples} clean spectra with {self.spectrum_length} wavelength points")
        print(f"Flux range: [{self.clean_flux.min():.3f}, {self.clean_flux.max():.3f}]")
        print(f"Sigma range: [{self.sigma.min():.4f}, {self.sigma.max():.4f}]")
        print(f"Sigma mean: {self.sigma.mean():.4f}")
    
    def add_bounded_noise(
        self, 
        clean_spectrum: torch.Tensor, 
        sigma: torch.Tensor,
        lam: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add bounded noise to create noisy observation.
        
        Noise model: y = x0 + λ * σ ⊙ ε, where ε ~ N(0, I)
        
        Args:
            clean_spectrum: Clean spectrum x0
            sigma: Per-pixel error vector σ
            lam: Noise level factor λ ∈ [0, λ_max]
            
        Returns:
            Tuple of (noisy_spectrum, noise)
        """
        # Sample standard Gaussian noise
        epsilon = torch.randn_like(clean_spectrum)
        
        # Apply bounded noise model
        noise = lam * sigma * epsilon
        noisy_spectrum = clean_spectrum + noise
        
        return noisy_spectrum, epsilon
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Return clean spectrum, noisy observation, sigma, and lambda.
        
        Returns:
            Tuple of (clean, noisy, sigma, lambda)
        """
        clean = self.clean_flux[idx].unsqueeze(0)  # (1, L)
        sigma = self.sigma[idx].unsqueeze(0)  # (1, L)
        
        # Randomly select lambda
        lam = np.random.choice(self.lambda_values)
        
        # Create noisy observation
        noisy, epsilon = self.add_bounded_noise(clean, sigma, lam)
        
        return clean, noisy, sigma, lam, epsilon
    
    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Convert from [-1, 1] back to original flux scale."""
        x = (x + 1) / 2
        return denormalize_spectrum(x, self.norm_params, self.normalization)


# ============================================
# Bounded Noise Diffusion
# ============================================

class BoundedNoiseDiffusion(nn.Module):
    """
    Bounded Noise Diffusion for spectrum denoising.
    
    Key differences from standard DDPM:
    - Noise model: y = x0 + λ * σ ⊙ ε (not standard DDPM schedule)
    - λ ∈ [0, λ_max] (bounded, not going to pure noise)
    - Per-pixel σ is known and passed to network
    - Inference starts from observation y (not pure noise)
    """
    
    def __init__(
        self,
        model: nn.Module,
        lambda_values: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5],
        prediction_target: str = "x0",  # "x0" or "eps"
        loss_type: str = "l2",
    ):
        super().__init__()
        
        self.model = model
        self.lambda_values = sorted(lambda_values)
        self.lambda_max = max(lambda_values)
        self.prediction_target = prediction_target
        self.loss_type = loss_type
        
        # Register lambda schedule as buffer
        self.register_buffer('lambdas', torch.tensor(lambda_values, dtype=torch.float32))
        
        print(f"BoundedNoiseDiffusion initialized:")
        print(f"  Lambda values: {self.lambda_values}")
        print(f"  Prediction target: {self.prediction_target}")
        print(f"  Loss type: {self.loss_type}")
    
    def training_loss(
        self,
        x0: torch.Tensor,
        noisy: torch.Tensor,
        sigma: torch.Tensor,
        lam: torch.Tensor,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute training loss.
        
        Args:
            x0: Clean spectrum (B, 1, L)
            noisy: Noisy observation (B, 1, L)
            sigma: Per-pixel error vector (B, 1, L)
            lam: Noise level factor (B,)
            epsilon: Ground truth noise (B, 1, L)
        
        Returns:
            Loss value
        """
        device = x0.device
        batch_size = x0.shape[0]
        
        # Convert lambda to discrete timestep-like index
        # Map lambda to integer for time embedding
        # lambda 0.1 -> 0, 0.2 -> 1, ..., 0.5 -> 4
        t = ((lam - 0.1) / 0.1).long().clamp(0, len(self.lambda_values) - 1)
        
        # Forward pass: model predicts x0 or eps from noisy input
        # Input: concatenate [noisy, sigma] as 2-channel input
        # Note: we use noisy (not x_t) directly since we're doing single-shot prediction
        prediction = self.model(noisy, t, sigma)
        
        # Compute loss based on prediction target
        if self.prediction_target == "x0":
            target = x0
        elif self.prediction_target == "eps":
            target = epsilon
        else:
            raise ValueError(f"Unknown prediction target: {self.prediction_target}")
        
        # Compute loss
        if self.loss_type == "l1":
            loss = F.l1_loss(prediction, target)
        elif self.loss_type == "l2":
            loss = F.mse_loss(prediction, target)
        elif self.loss_type == "huber":
            loss = F.smooth_l1_loss(prediction, target)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return loss
    
    @torch.no_grad()
    def denoise(
        self,
        noisy: torch.Tensor,
        sigma: torch.Tensor,
        num_steps: int = 10,
        return_intermediates: bool = False,
    ) -> torch.Tensor:
        """
        Denoise from observation using iterative refinement.
        
        Args:
            noisy: Noisy observation y (B, 1, L)
            sigma: Per-pixel error vector σ (B, 1, L)
            num_steps: Number of denoising steps
            return_intermediates: Whether to return intermediate results
        
        Returns:
            Denoised spectrum x̂_0
        """
        device = noisy.device
        batch_size = noisy.shape[0]
        
        # Start from observation at lambda_max
        x_t = noisy.clone()
        
        # Lambda schedule: from lambda_max down to 0
        lambda_schedule = np.linspace(self.lambda_max, 0, num_steps + 1)
        
        intermediates = []
        
        for i in range(num_steps):
            lam_t = lambda_schedule[i]
            lam_next = lambda_schedule[i + 1]
            
            # Convert lambda to timestep
            t = torch.full((batch_size,), int((lam_t - 0.1) / 0.1), device=device, dtype=torch.long)
            t = t.clamp(0, len(self.lambda_values) - 1)
            
            # Model prediction
            if self.prediction_target == "x0":
                # Direct x0 prediction
                x0_pred = self.model(x_t, t, sigma)
            else:  # eps prediction
                # Predict noise, then compute x0
                eps_pred = self.model(x_t, t, sigma)
                # x_t = x0 + lam * sigma * eps => x0 = x_t - lam * sigma * eps
                x0_pred = x_t - lam_t * sigma * eps_pred
            
            # Clamp to valid range
            x0_pred = x0_pred.clamp(-1.5, 1.5)
            
            # Move to next noise level (unless at final step)
            if i < num_steps - 1 and lam_next > 0:
                # Re-add noise at lower level: x_{t-1} = x0_pred + lam_next * sigma * eps
                # Use same predicted noise direction for smoother trajectory
                if self.prediction_target == "eps":
                    x_t = x0_pred + lam_next * sigma * eps_pred
                else:
                    # For x0 prediction, estimate noise from current state
                    eps_est = (x_t - x0_pred) / (lam_t * sigma + 1e-8)
                    x_t = x0_pred + lam_next * sigma * eps_est
            else:
                x_t = x0_pred
            
            if return_intermediates:
                intermediates.append((lam_t, x_t.clone()))
        
        if return_intermediates:
            return x_t, intermediates
        return x_t
    
    @torch.no_grad()
    def single_step_denoise(
        self,
        noisy: torch.Tensor,
        sigma: torch.Tensor,
        lam: float,
    ) -> torch.Tensor:
        """
        Single-step denoising (for evaluation).
        
        Args:
            noisy: Noisy observation y (B, 1, L)
            sigma: Per-pixel error vector σ (B, 1, L)
            lam: Known noise level λ
        
        Returns:
            Denoised spectrum x̂_0
        """
        device = noisy.device
        batch_size = noisy.shape[0]
        
        # Convert lambda to timestep
        t = torch.full((batch_size,), int((lam - 0.1) / 0.1), device=device, dtype=torch.long)
        t = t.clamp(0, len(self.lambda_values) - 1)
        
        # Model prediction
        if self.prediction_target == "x0":
            x0_pred = self.model(noisy, t, sigma)
        else:
            eps_pred = self.model(noisy, t, sigma)
            x0_pred = noisy - lam * sigma * eps_pred
        
        return x0_pred.clamp(-1.5, 1.5)


# ============================================
# Trainer
# ============================================

class BoundedDenoiserTrainer:
    """Trainer class for bounded noise denoiser."""
    
    def __init__(
        self,
        model: nn.Module,
        diffusion: BoundedNoiseDiffusion,
        train_dataset: BoundedNoiseSpectraDataset,
        device: str = "cuda",
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_epochs: int = 5,
        gradient_clip: float = 1.0,
        ema_decay: float = 0.9999,
        save_dir: str = "lightning_logs/diffusion/bounded_noise",
    ):
        self.model = model.to(device)
        self.diffusion = diffusion.to(device)
        self.train_dataset = train_dataset
        self.device = device
        
        # Training config
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.gradient_clip = gradient_clip
        self.ema_decay = ema_decay
        
        # Output directories
        self.save_dir = Path(save_dir)
        self.checkpoint_dir = self.save_dir / "checkpoints"
        self.figures_dir = Path(KNOWLEDGE_CENTER) / "logg/diffusion/img"
        
        # Create directories
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Data loader
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        
        # Scheduler with warmup
        warmup_steps = self.warmup_epochs * len(self.train_loader)
        total_steps = self.epochs * len(self.train_loader)
        
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
        self.ema = EMA(self.model, decay=self.ema_decay)
        
        # History
        self.history = {
            "train_loss": [],
            "epoch_loss": [],
            "lr": [],
        }
        
        # Dataset info
        self.wave = train_dataset.wave
        self.spectrum_length = train_dataset.spectrum_length
        self.lambda_values = train_dataset.lambda_values
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
        for batch_idx, (clean, noisy, sigma, lam, epsilon) in enumerate(pbar):
            clean = clean.to(self.device)
            noisy = noisy.to(self.device)
            sigma = sigma.to(self.device)
            lam = torch.tensor(lam, dtype=torch.float32).to(self.device)
            epsilon = epsilon.to(self.device)
            
            # Forward pass
            loss = self.diffusion.training_loss(clean, noisy, sigma, lam, epsilon)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
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
    def evaluate(self) -> Dict[str, Dict[str, float]]:
        """Evaluate denoising performance at different lambda values."""
        self.ema.apply_shadow()
        self.model.eval()
        
        results = {}
        
        for lam in self.lambda_values:
            mse_noisy_list = []
            mse_denoised_list = []
            wmse_list = []
            
            # Evaluate on subset of data
            n_eval = min(100, len(self.train_dataset))
            indices = np.random.choice(len(self.train_dataset), n_eval, replace=False)
            
            for idx in indices:
                clean = self.train_dataset.clean_flux[idx].unsqueeze(0).unsqueeze(0)  # (1, 1, L)
                sigma = self.train_dataset.sigma[idx].unsqueeze(0).unsqueeze(0)  # (1, 1, L)
                
                # Create noisy observation
                noisy, _ = self.train_dataset.add_bounded_noise(clean.squeeze(0), sigma.squeeze(0), lam)
                noisy = noisy.unsqueeze(0)
                
                clean = clean.to(self.device)
                noisy = noisy.to(self.device)
                sigma = sigma.to(self.device)
                
                # Denoise
                denoised = self.diffusion.single_step_denoise(noisy, sigma, lam)
                
                # Compute metrics
                mse_noisy = F.mse_loss(noisy, clean).item()
                mse_denoised = F.mse_loss(denoised, clean).item()
                
                # Weighted MSE
                residual = (denoised - clean) / (sigma + 1e-8)
                wmse = (residual ** 2).mean().item()
                
                mse_noisy_list.append(mse_noisy)
                mse_denoised_list.append(mse_denoised)
                wmse_list.append(wmse)
            
            results[lam] = {
                "mse_noisy": np.mean(mse_noisy_list),
                "mse_denoised": np.mean(mse_denoised_list),
                "wmse": np.mean(wmse_list),
                "improvement": 1 - np.mean(mse_denoised_list) / np.mean(mse_noisy_list),
            }
        
        self.ema.restore()
        return results
    
    def save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "ema_shadow": self.ema.shadow,
            "loss": loss,
            "history": self.history,
            "diffusion_config": {
                "lambda_values": self.diffusion.lambda_values,
                "prediction_target": self.diffusion.prediction_target,
                "loss_type": self.diffusion.loss_type,
            },
        }
        
        path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"
        torch.save(checkpoint, path)
        
        # Also save as latest
        latest_path = self.checkpoint_dir / "checkpoint_latest.pt"
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best.ckpt"
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
        print(f"Lambda values: {self.lambda_values}")
        print(f"Prediction target: {self.diffusion.prediction_target}")
        print(f"Output dir: {self.save_dir}")
        print(f"Figures dir: {self.figures_dir}")
        print(f"{'='*60}\n")
        
        best_loss = float("inf")
        
        for epoch in range(self.epochs):
            # Train
            avg_loss = self.train_epoch(epoch)
            
            print(f"\n📊 Epoch {epoch+1}/{self.epochs} - Loss: {avg_loss:.6f}")
            
            # Save checkpoint
            is_best = avg_loss < best_loss
            if is_best:
                best_loss = avg_loss
            self.save_checkpoint(epoch, avg_loss, is_best=is_best)
            
            # Evaluate and visualize periodically
            if (epoch + 1) % 10 == 0 or epoch == self.epochs - 1:
                print("🔍 Evaluating...")
                results = self.evaluate()
                for lam, metrics in results.items():
                    print(f"  λ={lam:.1f}: MSE(noisy)={metrics['mse_noisy']:.6f}, "
                          f"MSE(denoised)={metrics['mse_denoised']:.6f}, "
                          f"Improvement={metrics['improvement']*100:.1f}%")
                
                print("📈 Generating visualizations...")
                self.plot_denoising_samples(epoch)
        
        # Final visualizations
        print("\n📊 Generating final visualizations...")
        self.plot_loss_curve()
        final_results = self.evaluate()
        self.plot_mse_comparison(final_results)
        self.plot_improvement_bars(final_results)
        self.plot_flux_distribution()
        
        # Save summary
        self.save_summary(final_results)
        
        print(f"\n{'='*60}")
        print("✅ Training completed!")
        print(f"Best loss: {best_loss:.6f}")
        print(f"Figures saved to: {self.figures_dir}")
        print(f"Checkpoints saved to: {self.checkpoint_dir}")
        print(f"{'='*60}\n")
        
        return self.history, final_results
    
    # ============================================
    # Visualization Methods
    # ============================================
    
    def plot_loss_curve(self):
        """Plot training loss curve (fig_1)."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Per-step loss
        ax1 = axes[0]
        steps = np.arange(len(self.history["train_loss"]))
        ax1.plot(steps, self.history["train_loss"], alpha=0.5, linewidth=0.5, color='steelblue')
        # Smoothed
        window = min(100, len(self.history["train_loss"]) // 10)
        if window > 1:
            smoothed = np.convolve(
                self.history["train_loss"], 
                np.ones(window)/window, 
                mode='valid'
            )
            ax1.plot(
                steps[window-1:], smoothed, 
                color='crimson', linewidth=2, label='Smoothed'
            )
        ax1.set_xlabel("Training Step", fontsize=12)
        ax1.set_ylabel("Loss", fontsize=12)
        ax1.set_title("Training Loss per Step", fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Per-epoch loss
        ax2 = axes[1]
        epochs = np.arange(1, len(self.history["epoch_loss"]) + 1)
        ax2.plot(epochs, self.history["epoch_loss"], marker='o', linewidth=2, color='teal')
        ax2.set_xlabel("Epoch", fontsize=12)
        ax2.set_ylabel("Loss", fontsize=12)
        ax2.set_title("Training Loss per Epoch", fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        fig.suptitle(f"{EXP_ID} - Training Loss Curve", fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        path = self.figures_dir / "diff_bounded_loss_curve.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Saved: {path}")
    
    def plot_mse_comparison(self, results: Dict):
        """Plot MSE comparison at different lambda values (fig_2)."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        lambdas = sorted(results.keys())
        mse_noisy = [results[lam]["mse_noisy"] for lam in lambdas]
        mse_denoised = [results[lam]["mse_denoised"] for lam in lambdas]
        
        ax.plot(lambdas, mse_noisy, 'o-', linewidth=2, markersize=8, 
                label='MSE(noisy, clean)', color='gray')
        ax.plot(lambdas, mse_denoised, 's-', linewidth=2, markersize=8, 
                label='MSE(denoised, clean)', color='forestgreen')
        
        ax.set_xlabel("λ (Noise Level)", fontsize=12)
        ax.set_ylabel("MSE", fontsize=12)
        ax.set_title(f"{EXP_ID}\nMSE Comparison: Noisy vs Denoised", fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Add improvement annotations
        for lam in lambdas:
            improvement = results[lam]["improvement"] * 100
            y_pos = results[lam]["mse_denoised"]
            ax.annotate(f'{improvement:.1f}%↓', 
                       (lam, y_pos), 
                       textcoords="offset points", 
                       xytext=(0, -15),
                       ha='center',
                       fontsize=9,
                       color='forestgreen')
        
        plt.tight_layout()
        
        path = self.figures_dir / "diff_bounded_mse_comparison.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Saved: {path}")
    
    def plot_denoising_samples(self, epoch: int):
        """Plot denoising comparison samples (fig_3)."""
        self.ema.apply_shadow()
        self.model.eval()
        
        n_samples = 3
        n_lambdas = 2  # Show results for 2 different lambda values
        
        fig, axes = plt.subplots(n_samples, 3, figsize=(18, 4 * n_samples))
        
        indices = np.random.choice(len(self.train_dataset), n_samples, replace=False)
        wave = self.wave.numpy()
        
        lambdas_to_show = [0.3, 0.5]  # Show medium and high noise
        
        for i, idx in enumerate(indices):
            clean = self.train_dataset.clean_flux[idx].unsqueeze(0).unsqueeze(0)
            sigma = self.train_dataset.sigma[idx].unsqueeze(0).unsqueeze(0)
            
            lam = lambdas_to_show[i % len(lambdas_to_show)]
            
            # Create noisy observation
            noisy, _ = self.train_dataset.add_bounded_noise(clean.squeeze(0), sigma.squeeze(0), lam)
            noisy = noisy.unsqueeze(0)
            
            clean_gpu = clean.to(self.device)
            noisy_gpu = noisy.to(self.device)
            sigma_gpu = sigma.to(self.device)
            
            # Denoise
            with torch.no_grad():
                denoised = self.diffusion.single_step_denoise(noisy_gpu, sigma_gpu, lam)
            
            clean_np = clean.squeeze().numpy()
            noisy_np = noisy.squeeze().numpy()
            denoised_np = denoised.cpu().squeeze().numpy()
            
            # Column 1: Clean (Ground Truth)
            axes[i, 0].plot(wave, clean_np, color='royalblue', linewidth=0.8, label='Ground Truth')
            axes[i, 0].set_ylabel(f"λ={lam}", fontsize=11, fontweight='bold')
            if i == 0:
                axes[i, 0].set_title("Clean (Ground Truth)", fontsize=12)
            axes[i, 0].set_ylim(-1.5, 1.5)
            axes[i, 0].grid(True, alpha=0.3)
            
            # Column 2: Noisy
            axes[i, 1].plot(wave, noisy_np, color='gray', linewidth=0.8, alpha=0.8, label='Noisy')
            axes[i, 1].plot(wave, clean_np, color='royalblue', linewidth=0.5, alpha=0.5)
            if i == 0:
                axes[i, 1].set_title("Noisy Input", fontsize=12)
            axes[i, 1].set_ylim(-1.5, 1.5)
            axes[i, 1].grid(True, alpha=0.3)
            
            # Column 3: Denoised
            axes[i, 2].plot(wave, denoised_np, color='forestgreen', linewidth=0.8, label='Denoised')
            axes[i, 2].plot(wave, clean_np, color='royalblue', linewidth=0.5, alpha=0.5)
            mse = np.mean((denoised_np - clean_np) ** 2)
            if i == 0:
                axes[i, 2].set_title(f"Denoised (MSE={mse:.5f})", fontsize=12)
            else:
                axes[i, 2].set_title(f"MSE={mse:.5f}", fontsize=11)
            axes[i, 2].set_ylim(-1.5, 1.5)
            axes[i, 2].grid(True, alpha=0.3)
        
        # Set x labels on bottom row
        for j in range(3):
            axes[-1, j].set_xlabel("Wavelength (Å)", fontsize=11)
        
        fig.suptitle(f"{EXP_ID} - Denoising Results (Epoch {epoch+1})", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        path = self.figures_dir / "diff_bounded_denoising_samples.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Saved: {path}")
        
        self.ema.restore()
    
    def plot_improvement_bars(self, results: Dict):
        """Plot improvement percentage as bar chart (fig_5)."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        lambdas = sorted(results.keys())
        improvements = [results[lam]["improvement"] * 100 for lam in lambdas]
        
        colors = ['#2ecc71' if imp > 30 else '#f39c12' if imp > 0 else '#e74c3c' 
                  for imp in improvements]
        
        bars = ax.bar([str(lam) for lam in lambdas], improvements, color=colors, edgecolor='black')
        
        # Add value labels on bars
        for bar, imp in zip(bars, improvements):
            height = bar.get_height()
            ax.annotate(f'{imp:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=11, fontweight='bold')
        
        # Add threshold line
        ax.axhline(y=30, color='red', linestyle='--', linewidth=2, label='30% threshold')
        
        ax.set_xlabel("λ (Noise Level)", fontsize=12)
        ax.set_ylabel("Denoising Improvement (%)", fontsize=12)
        ax.set_title(f"{EXP_ID}\nDenoising Improvement by Noise Level", fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        path = self.figures_dir / "diff_bounded_improvement.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Saved: {path}")
    
    def plot_flux_distribution(self):
        """Plot flux distribution comparison (fig_4)."""
        self.ema.apply_shadow()
        self.model.eval()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Collect flux values
        n_samples = 100
        clean_flux = []
        noisy_flux = []
        denoised_flux = []
        
        indices = np.random.choice(len(self.train_dataset), n_samples, replace=False)
        lam = 0.5  # Use max noise level
        
        for idx in indices:
            clean = self.train_dataset.clean_flux[idx].unsqueeze(0).unsqueeze(0)
            sigma = self.train_dataset.sigma[idx].unsqueeze(0).unsqueeze(0)
            
            noisy, _ = self.train_dataset.add_bounded_noise(clean.squeeze(0), sigma.squeeze(0), lam)
            noisy = noisy.unsqueeze(0)
            
            clean_gpu = clean.to(self.device)
            noisy_gpu = noisy.to(self.device)
            sigma_gpu = sigma.to(self.device)
            
            with torch.no_grad():
                denoised = self.diffusion.single_step_denoise(noisy_gpu, sigma_gpu, lam)
            
            clean_flux.extend(clean.flatten().numpy())
            noisy_flux.extend(noisy.flatten().numpy())
            denoised_flux.extend(denoised.cpu().flatten().numpy())
        
        # Sample for KDE (avoid memory issues)
        n_kde = 50000
        clean_flux = np.random.choice(clean_flux, min(n_kde, len(clean_flux)), replace=False)
        noisy_flux = np.random.choice(noisy_flux, min(n_kde, len(noisy_flux)), replace=False)
        denoised_flux = np.random.choice(denoised_flux, min(n_kde, len(denoised_flux)), replace=False)
        
        # Plot KDE
        sns.kdeplot(clean_flux, ax=ax, label='Clean (Ground Truth)', color='royalblue', linewidth=2)
        sns.kdeplot(noisy_flux, ax=ax, label=f'Noisy (λ={lam})', color='gray', linewidth=2, linestyle='--')
        sns.kdeplot(denoised_flux, ax=ax, label='Denoised', color='forestgreen', linewidth=2)
        
        ax.set_xlabel("Flux Value (normalized)", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title(f"{EXP_ID}\nFlux Distribution Comparison (λ={lam})\n(Verify: NOT Gaussian noise!)", 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Add annotation about distribution shape
        ax.text(0.02, 0.98, 
                "✓ Denoised distribution should match Clean\n✗ NOT symmetric Gaussian",
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        path = self.figures_dir / "diff_bounded_flux_distribution.png"
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
                "type": "ConditionalUNet1D",
                "parameters": count_parameters(self.model),
                "prediction_target": self.diffusion.prediction_target,
            },
            "training": {
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "learning_rate": self.lr,
                "final_loss": self.history["epoch_loss"][-1] if self.history["epoch_loss"] else None,
                "best_loss": min(self.history["epoch_loss"]) if self.history["epoch_loss"] else None,
            },
            "data": {
                "num_samples": len(self.train_dataset),
                "spectrum_length": self.spectrum_length,
                "lambda_values": self.lambda_values,
            },
            "results": {
                str(lam): metrics for lam, metrics in results.items()
            },
        }
        
        path = self.save_dir / "training_summary.json"
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"📝 Saved: {path}")
        
        # Also save to knowledge center
        kc_path = Path(KNOWLEDGE_CENTER) / "logg/diffusion/img/diff_bounded_eval_results.json"
        with open(kc_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"📝 Saved: {kc_path}")


# ============================================
# Main
# ============================================

def main():
    parser = argparse.ArgumentParser(
        description="Train Bounded Noise Denoiser (MVP-0.5)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--lambda-values",
        type=float,
        nargs='+',
        default=[0.1, 0.2, 0.3, 0.4, 0.5],
        help="Lambda values for noise levels"
    )
    parser.add_argument(
        "--target",
        type=str,
        default="x0",
        choices=["x0", "eps"],
        help="Prediction target: x0 (direct) or eps (noise)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Override data root directory"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10000,
        help="Number of training samples"
    )
    args = parser.parse_args()
    
    # Set data root
    data_root = args.data_root or os.environ.get("DATA_ROOT", DEFAULT_DATA_ROOT)
    # Try different possible file locations
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
        data_file = data_file_options[0]  # Default for error message
    
    print(f"Data file: {data_file}")
    
    # Device
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    dataset = BoundedNoiseSpectraDataset(
        file_path=data_file,
        num_samples=args.num_samples,
        normalization="minmax",
        snr_threshold=50.0,
        lambda_values=args.lambda_values,
        sigma_scale=1.0,  # Increased from 0.1 to make noise meaningful
    )
    
    # Create model
    model = ConditionalUNet1D(
        in_channels=1,      # x_t (noisy)
        cond_channels=1,    # sigma
        out_channels=1,     # prediction (x0 or eps)
        base_channels=32,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=2,
        attention_resolutions=(3,),
        dropout=0.1,
        time_emb_dim=256,
    )
    
    # Create diffusion
    diffusion = BoundedNoiseDiffusion(
        model=model,
        lambda_values=args.lambda_values,
        prediction_target=args.target,
        loss_type="l2",
    )
    
    # Create trainer
    trainer = BoundedDenoiserTrainer(
        model=model,
        diffusion=diffusion,
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
    for lam, metrics in sorted(results.items()):
        print(f"λ={lam:.1f}:")
        print(f"  MSE(noisy, clean):    {metrics['mse_noisy']:.6f}")
        print(f"  MSE(denoised, clean): {metrics['mse_denoised']:.6f}")
        print(f"  Improvement:          {metrics['improvement']*100:.1f}%")
        print(f"  wMSE:                 {metrics['wmse']:.6f}")
    print("="*60)
    
    # Check acceptance criteria
    max_lambda_results = results[max(args.lambda_values)]
    improvement = max_lambda_results["improvement"] * 100
    
    print(f"\n🎯 ACCEPTANCE CRITERIA CHECK:")
    print(f"  Improvement @ λ={max(args.lambda_values)}: {improvement:.1f}%")
    if improvement > 30:
        print(f"  ✅ PASSED: Improvement > 30%")
    else:
        print(f"  ❌ FAILED: Improvement <= 30%")
    
    return history, results


if __name__ == "__main__":
    main()

