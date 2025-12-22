#!/usr/bin/env python3
"""
Training Script for Supervised Conditional DDPM Spectrum Denoising

Experiment ID: SD-20251203-diff-supervised-01
MVP Source: MVP-1.0
Description: Train conditional DDPM to denoise spectra conditioned on noisy observations

Usage:
    python scripts/train_supervised.py --config configs/supervised.yaml
    python scripts/train_supervised.py --config configs/supervised.yaml --epochs 50

Author: Viska Wei
Date: 2025-12-03
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
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.diffusion.conditional_unet_1d import ConditionalUNet1D, count_parameters
from models.diffusion.conditional_ddpm import ConditionalGaussianDiffusion, ConditionalDDPM
from models.diffusion.utils import EMA, normalize_spectrum, denormalize_spectrum


# ============================================
# Configuration
# ============================================

DEFAULT_DATA_ROOT = "/srv/local/tmp/swei20/data/bosz50000/z0"

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Get data root from environment or use default
    data_root = os.environ.get("DATA_ROOT", DEFAULT_DATA_ROOT)
    
    # Expand environment variables in paths
    if "data" in config:
        for key in ["file_path", "val_path", "test_path"]:
            if key in config["data"]:
                path = config["data"][key]
                path = path.replace("$DATA_ROOT", data_root)
                path = os.path.expandvars(path)
                config["data"][key] = path
    
    return config


# ============================================
# Dataset
# ============================================

class SupervisedSpectraDataset(Dataset):
    """
    Dataset for supervised spectrum denoising.
    
    Loads clean spectra and generates noisy observations on-the-fly
    with configurable SNR levels.
    """
    
    def __init__(
        self,
        file_path: str,
        num_samples: int = 10000,
        normalization: str = "minmax",
        snr_threshold: float = 50.0,
        snr_levels: List[float] = [5, 10, 20, 50],
        primary_snr: float = 10.0,
        use_random_snr: bool = True,
    ):
        self.file_path = file_path
        self.num_samples = num_samples
        self.normalization = normalization
        self.snr_threshold = snr_threshold
        self.snr_levels = snr_levels
        self.primary_snr = primary_snr
        self.use_random_snr = use_random_snr
        
        self.clean_flux = None
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
        
        # Normalize to [0, 1] range
        self.clean_flux, self.norm_params = normalize_spectrum(flux, method=self.normalization)
        
        # Scale from [0, 1] to [-1, 1] for diffusion
        self.clean_flux = self.clean_flux * 2 - 1
        
        self.num_samples = len(self.clean_flux)
        self.spectrum_length = self.clean_flux.shape[1]
        
        print(f"Loaded {self.num_samples} clean spectra with {self.spectrum_length} wavelength points")
        print(f"Flux range: [{self.clean_flux.min():.3f}, {self.clean_flux.max():.3f}]")
    
    def add_noise(self, clean_spectrum: torch.Tensor, snr: float) -> torch.Tensor:
        """
        Add Gaussian noise to create noisy observation.
        
        Args:
            clean_spectrum: Clean spectrum in [-1, 1] range
            snr: Target signal-to-noise ratio
            
        Returns:
            Noisy spectrum
        """
        # Estimate signal level (use RMS of spectrum)
        signal_rms = torch.sqrt(torch.mean(clean_spectrum ** 2))
        
        # Calculate noise level for target SNR
        noise_std = signal_rms / snr
        
        # Add Gaussian noise
        noise = torch.randn_like(clean_spectrum) * noise_std
        noisy_spectrum = clean_spectrum + noise
        
        # Clip to valid range
        noisy_spectrum = torch.clamp(noisy_spectrum, -1.5, 1.5)
        
        return noisy_spectrum
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Return clean spectrum and noisy observation.
        
        Returns:
            Tuple of (clean_spectrum, noisy_observation, snr)
        """
        clean = self.clean_flux[idx].unsqueeze(0)  # (1, L)
        
        # Select SNR (random or primary)
        if self.use_random_snr:
            snr = np.random.choice(self.snr_levels)
        else:
            snr = self.primary_snr
        
        # Create noisy observation
        noisy = self.add_noise(clean, snr)
        
        return clean, noisy, snr
    
    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Convert from [-1, 1] back to original flux scale."""
        x = (x + 1) / 2
        return denormalize_spectrum(x, self.norm_params, self.normalization)


# ============================================
# Training
# ============================================

class SupervisedDiffusionTrainer:
    """Trainer class for supervised conditional diffusion model."""
    
    def __init__(
        self,
        model: nn.Module,
        diffusion: ConditionalGaussianDiffusion,
        train_dataset: SupervisedSpectraDataset,
        config: Dict[str, Any],
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.diffusion = diffusion.to(device)
        self.train_dataset = train_dataset
        self.config = config
        self.device = device
        
        # Training config
        train_cfg = config.get("training", {})
        self.epochs = train_cfg.get("epochs", 50)
        self.batch_size = train_cfg.get("batch_size", 32)
        self.lr = train_cfg.get("learning_rate", 1e-4)
        self.weight_decay = train_cfg.get("weight_decay", 0.01)
        self.warmup_epochs = train_cfg.get("warmup_epochs", 5)
        self.gradient_clip = train_cfg.get("gradient_clip", 1.0)
        self.ema_decay = train_cfg.get("ema_decay", 0.9999)
        
        # Sampling config
        sample_cfg = config.get("sampling", {})
        self.num_samples = sample_cfg.get("num_samples", 16)
        self.sample_every_n_epochs = sample_cfg.get("sample_every_n_epochs", 10)
        
        # Output config
        output_cfg = config.get("output", {})
        self.save_dir = Path(output_cfg.get("save_dir", "lightning_logs/supervised"))
        self.checkpoint_dir = self.save_dir / "checkpoints"
        self.figures_dir = Path(output_cfg.get("figures_dir", 
            "/home/swei20/Physics_Informed_AI/logg/diffusion/img"))
        
        # Create directories
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup data loader
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=train_cfg.get("num_workers", 4),
            pin_memory=train_cfg.get("pin_memory", True),
            drop_last=True,
        )
        
        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        
        # Setup scheduler with warmup
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
        
        # Setup EMA
        self.ema = EMA(self.model, decay=self.ema_decay)
        
        # Training history
        self.history = {
            "train_loss": [],
            "epoch_loss": [],
            "lr": [],
        }
        
        # Keep reference to wavelength for plotting
        self.wave = train_dataset.wave
        self.spectrum_length = train_dataset.spectrum_length
        
        # SNR levels for evaluation
        self.snr_levels = config.get("data", {}).get("noise_injection", {}).get(
            "snr_levels", [5, 10, 20, 50])
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
        for batch_idx, (clean, noisy, snr) in enumerate(pbar):
            clean = clean.to(self.device)
            noisy = noisy.to(self.device)
            
            # Forward pass: predict noise from x_t conditioned on noisy observation
            loss = self.diffusion.training_loss(clean, noisy)
            
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
    def denoise(self, noisy_obs: torch.Tensor) -> torch.Tensor:
        """Denoise using EMA model."""
        self.ema.apply_shadow()
        self.model.eval()
        
        denoised = self.diffusion.p_sample_loop(
            cond=noisy_obs,
            progress=False,
        )
        
        self.ema.restore()
        return denoised
    
    def save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "ema_shadow": self.ema.shadow,
            "loss": loss,
            "config": self.config,
            "history": self.history,
        }
        
        path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")
        
        # Also save as latest
        latest_path = self.checkpoint_dir / "checkpoint_latest.pt"
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best.ckpt"
            torch.save(checkpoint, best_path)
            print(f"Saved best checkpoint: {best_path}")
    
    def train(self):
        """Full training loop."""
        exp_cfg = self.config.get("experiment", {})
        print(f"\n{'='*60}")
        print(f"Starting training: {exp_cfg.get('id', 'supervised-ddpm')}")
        print(f"{'='*60}")
        print(f"Model parameters: {count_parameters(self.model):,}")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.epochs}")
        print(f"Batch size: {self.batch_size}")
        print(f"Learning rate: {self.lr}")
        print(f"SNR levels: {self.snr_levels}")
        print(f"Output dir: {self.save_dir}")
        print(f"Figures dir: {self.figures_dir}")
        print(f"{'='*60}\n")
        
        best_loss = float("inf")
        
        for epoch in range(self.epochs):
            # Train
            avg_loss = self.train_epoch(epoch)
            
            print(f"\nEpoch {epoch+1}/{self.epochs} - Loss: {avg_loss:.6f}")
            
            # Save checkpoint
            is_best = avg_loss < best_loss
            if is_best:
                best_loss = avg_loss
            self.save_checkpoint(epoch, avg_loss, is_best=is_best)
            
            # Sample and visualize
            if (epoch + 1) % self.sample_every_n_epochs == 0 or epoch == self.epochs - 1:
                print("Generating denoising samples...")
                self.plot_denoising_samples(epoch)
        
        # Final visualizations
        print("\nGenerating final visualizations...")
        self.plot_loss_curve()
        
        # Save final model
        self.save_checkpoint(self.epochs - 1, self.history["epoch_loss"][-1], 
                           is_best=avg_loss <= best_loss)
        
        # Save training summary
        self.save_summary()
        
        print(f"\n{'='*60}")
        print("Training completed!")
        print(f"Best loss: {best_loss:.6f}")
        print(f"Figures saved to: {self.figures_dir}")
        print(f"Checkpoints saved to: {self.checkpoint_dir}")
        print(f"{'='*60}\n")
        
        return self.history
    
    # ============================================
    # Visualization Methods
    # ============================================
    
    def plot_loss_curve(self):
        """Plot training loss curve."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Per-step loss
        ax1 = axes[0]
        steps = np.arange(len(self.history["train_loss"]))
        ax1.plot(steps, self.history["train_loss"], alpha=0.5, linewidth=0.5)
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
                color='red', linewidth=2, label='Smoothed'
            )
        ax1.set_xlabel("Training Step")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training Loss per Step")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Per-epoch loss
        ax2 = axes[1]
        epochs = np.arange(1, len(self.history["epoch_loss"]) + 1)
        ax2.plot(epochs, self.history["epoch_loss"], marker='o', linewidth=2)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.set_title("Training Loss per Epoch")
        ax2.grid(True, alpha=0.3)
        
        exp_id = self.config.get('experiment', {}).get('id', 'supervised-ddpm')
        fig.suptitle(f"{exp_id} - Training Loss Curve", fontsize=14)
        plt.tight_layout()
        
        path = self.figures_dir / "diff_supervised_loss_curve.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {path}")
    
    def plot_denoising_samples(self, epoch: int):
        """Plot denoising comparison grid (4x2: clean, noisy, denoised, residual)."""
        n_samples = 4
        
        # Get random samples
        indices = np.random.choice(len(self.train_dataset), n_samples, replace=False)
        
        wave = self.wave.numpy()
        
        fig, axes = plt.subplots(n_samples, 4, figsize=(20, 4 * n_samples))
        
        snrs = [5, 10, 20, 50]
        
        for i, idx in enumerate(indices):
            clean = self.train_dataset.clean_flux[idx].unsqueeze(0).unsqueeze(0)  # (1, 1, L)
            snr = snrs[i % len(snrs)]
            
            # Add noise
            noisy = self.train_dataset.add_noise(clean.squeeze(0), snr).unsqueeze(0)
            
            # Denoise
            noisy_gpu = noisy.to(self.device)
            denoised = self.denoise(noisy_gpu).cpu()
            
            clean_np = clean.squeeze().numpy()
            noisy_np = noisy.squeeze().numpy()
            denoised_np = denoised.squeeze().numpy()
            residual_np = clean_np - denoised_np
            
            # Column 1: Clean
            axes[i, 0].plot(wave, clean_np, color='blue', linewidth=0.8)
            axes[i, 0].set_ylabel(f"SNR={snr}")
            if i == 0:
                axes[i, 0].set_title("Clean (Ground Truth)")
            axes[i, 0].set_ylim(-1.5, 1.5)
            
            # Column 2: Noisy
            axes[i, 1].plot(wave, noisy_np, color='gray', linewidth=0.8)
            if i == 0:
                axes[i, 1].set_title("Noisy Input")
            axes[i, 1].set_ylim(-1.5, 1.5)
            
            # Column 3: Denoised
            axes[i, 2].plot(wave, denoised_np, color='orange', linewidth=0.8)
            if i == 0:
                axes[i, 2].set_title("Denoised (DDPM)")
            axes[i, 2].set_ylim(-1.5, 1.5)
            
            # Column 4: Residual
            axes[i, 3].plot(wave, residual_np, color='red', linewidth=0.8)
            mse = np.mean(residual_np ** 2)
            axes[i, 3].set_title(f"MSE={mse:.4f}" if i == 0 else f"MSE={mse:.4f}")
            if i == 0:
                axes[i, 3].set_title(f"Residual (MSE={mse:.4f})")
            axes[i, 3].set_ylim(-0.5, 0.5)
        
        # Set x labels on bottom row
        for j in range(4):
            axes[-1, j].set_xlabel("Wavelength (Å)")
        
        exp_id = self.config.get('experiment', {}).get('id', 'supervised-ddpm')
        fig.suptitle(f"{exp_id} - Denoising Results (Epoch {epoch+1})", fontsize=14)
        plt.tight_layout()
        
        path = self.figures_dir / f"diff_supervised_denoising_samples.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {path}")
    
    def save_summary(self):
        """Save training summary."""
        exp_cfg = self.config.get("experiment", {})
        summary = {
            "experiment_id": exp_cfg.get("id", "supervised-ddpm"),
            "experiment_name": exp_cfg.get("name", "Supervised DDPM"),
            "mvp_source": exp_cfg.get("mvp_source", "MVP-1.0"),
            "date": datetime.now().isoformat(),
            "model": {
                "type": "ConditionalUNet1D",
                "parameters": count_parameters(self.model),
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
                "snr_levels": self.snr_levels,
            },
        }
        
        path = self.save_dir / "training_summary.json"
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved: {path}")


# ============================================
# Main
# ============================================

def main():
    parser = argparse.ArgumentParser(
        description="Train Supervised Conditional DDPM for Spectrum Denoising"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/supervised.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size"
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
    args = parser.parse_args()
    
    # Set data root from command line if provided
    if args.data_root:
        os.environ["DATA_ROOT"] = args.data_root
    elif "DATA_ROOT" not in os.environ:
        os.environ["DATA_ROOT"] = DEFAULT_DATA_ROOT
    
    print(f"DATA_ROOT: {os.environ.get('DATA_ROOT')}")
    
    # Load config
    config = load_config(args.config)
    
    # Override config with command line args
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    
    # Device
    if args.device is not None:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {device}")
    
    # Load data
    data_cfg = config.get("data", {})
    noise_cfg = data_cfg.get("noise_injection", {})
    
    dataset = SupervisedSpectraDataset(
        file_path=data_cfg.get("file_path"),
        num_samples=data_cfg.get("train_size", 10000),
        normalization=data_cfg.get("normalization", "minmax"),
        snr_threshold=data_cfg.get("snr_threshold", 50.0),
        snr_levels=noise_cfg.get("snr_levels", [5, 10, 20, 50]),
        primary_snr=noise_cfg.get("primary_snr", 10.0),
        use_random_snr=True,
    )
    
    # Create model
    model_cfg = config.get("model", {})
    model = ConditionalUNet1D(
        in_channels=model_cfg.get("in_channels", 1),
        cond_channels=model_cfg.get("cond_channels", 1),
        out_channels=model_cfg.get("out_channels", 1),
        base_channels=model_cfg.get("base_channels", 32),
        channel_mults=tuple(model_cfg.get("channel_mults", [1, 2, 4, 8])),
        num_res_blocks=model_cfg.get("num_res_blocks", 2),
        attention_resolutions=tuple(model_cfg.get("attention_resolutions", [3])),
        dropout=model_cfg.get("dropout", 0.1),
        time_emb_dim=model_cfg.get("time_emb_dim", 256),
    )
    
    # Create diffusion
    diff_cfg = config.get("diffusion", {})
    diffusion = ConditionalGaussianDiffusion(
        model=model,
        timesteps=diff_cfg.get("timesteps", 1000),
        beta_schedule=diff_cfg.get("beta_schedule", "linear"),
        beta_start=diff_cfg.get("beta_start", 1e-4),
        beta_end=diff_cfg.get("beta_end", 0.02),
        loss_type=diff_cfg.get("loss_type", "l2"),
    )
    
    # Create trainer
    trainer = SupervisedDiffusionTrainer(
        model=model,
        diffusion=diffusion,
        train_dataset=dataset,
        config=config,
        device=device,
    )
    
    # Train
    history = trainer.train()
    
    return history


if __name__ == "__main__":
    main()

