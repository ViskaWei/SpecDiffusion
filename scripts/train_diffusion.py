#!/usr/bin/env python3
"""
Training Script for 1D Diffusion Model on Stellar Spectra

Usage:
    python scripts/train_diffusion.py --config configs/diffusion/baseline.yaml
    python scripts/train_diffusion.py --config configs/diffusion/baseline.yaml --epochs 100

Data paths (set via environment variables):
    export DATA_ROOT=/srv/local/tmp/swei20/data/bosz50000/z0
"""

import os
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

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

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.diffusion.unet_1d import UNet1D, count_parameters
from models.diffusion.ddpm import GaussianDiffusion, DDPM
from models.diffusion.utils import EMA, normalize_spectrum, denormalize_spectrum


# ============================================
# Configuration
# ============================================

# Default data root
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
                # Replace $DATA_ROOT with actual path
                path = path.replace("$DATA_ROOT", data_root)
                path = os.path.expandvars(path)
                config["data"][key] = path
    
    return config


# ============================================
# Dataset
# ============================================

class StellarSpectraDataset(Dataset):
    """
    Dataset for stellar spectra for diffusion training.
    """
    
    def __init__(
        self,
        file_path: str,
        num_samples: int = 10000,
        normalization: str = "minmax",
        snr_threshold: float = 50.0,
    ):
        self.file_path = file_path
        self.num_samples = num_samples
        self.normalization = normalization
        self.snr_threshold = snr_threshold
        
        self.flux = None
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
        
        # Calculate SNR and filter
        snr = flux.norm(dim=-1) / error.norm(dim=-1)
        print(f"SNR range: {snr.min():.1f} - {snr.max():.1f}, median: {snr.median():.1f}")
        
        # Optionally filter by SNR
        if self.snr_threshold > 0:
            high_snr_mask = snr >= self.snr_threshold
            n_high_snr = high_snr_mask.sum().item()
            print(f"High SNR (>={self.snr_threshold}) samples: {n_high_snr}/{len(flux)}")
            
            if n_high_snr > 1000:
                flux = flux[high_snr_mask]
        
        # Normalize to [0, 1] range
        self.flux, self.norm_params = normalize_spectrum(flux, method=self.normalization)
        
        # Scale from [0, 1] to [-1, 1] for diffusion
        self.flux = self.flux * 2 - 1
        
        self.num_samples = len(self.flux)
        self.spectrum_length = self.flux.shape[1]
        
        print(f"Loaded {self.num_samples} spectra with {self.spectrum_length} wavelength points")
        print(f"Flux range: [{self.flux.min():.3f}, {self.flux.max():.3f}]")
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.flux[idx].unsqueeze(0)
    
    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Convert from [-1, 1] back to original flux scale."""
        x = (x + 1) / 2
        return denormalize_spectrum(x, self.norm_params, self.normalization)


# ============================================
# Training
# ============================================

class DiffusionTrainer:
    """Trainer class for diffusion model."""
    
    def __init__(
        self,
        model: nn.Module,
        diffusion: GaussianDiffusion,
        train_dataset: Dataset,
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
        self.batch_size = train_cfg.get("batch_size", 64)
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
        self.save_dir = Path(output_cfg.get("save_dir", "lightning_logs/diffusion"))
        self.checkpoint_dir = self.save_dir / "checkpoints"
        self.figures_dir = self.save_dir / "img"
        
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
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
        for batch_idx, x in enumerate(pbar):
            x = x.to(self.device)
            
            # Forward pass
            loss = self.diffusion.training_loss(x)
            
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
    def sample(self, num_samples: int = 16) -> torch.Tensor:
        """Generate samples using EMA model."""
        self.ema.apply_shadow()
        self.model.eval()
        
        samples = self.diffusion.p_sample_loop(
            shape=(num_samples, 1, self.spectrum_length),
            device=self.device,
            progress=True,
        )
        
        self.ema.restore()
        return samples
    
    def save_checkpoint(self, epoch: int, loss: float):
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
    
    def train(self):
        """Full training loop."""
        print(f"\n{'='*60}")
        print(f"Starting training: {self.config.get('experiment', {}).get('id', 'diffusion')}")
        print(f"{'='*60}")
        print(f"Model parameters: {count_parameters(self.model):,}")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.epochs}")
        print(f"Batch size: {self.batch_size}")
        print(f"Learning rate: {self.lr}")
        print(f"Output dir: {self.save_dir}")
        print(f"{'='*60}\n")
        
        best_loss = float("inf")
        
        for epoch in range(self.epochs):
            # Train
            avg_loss = self.train_epoch(epoch)
            
            print(f"\nEpoch {epoch+1}/{self.epochs} - Loss: {avg_loss:.6f}")
            
            # Save checkpoint
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_checkpoint(epoch, avg_loss)
            
            # Sample and visualize
            if (epoch + 1) % self.sample_every_n_epochs == 0 or epoch == self.epochs - 1:
                print("Generating samples...")
                samples = self.sample(self.num_samples)
                self.plot_samples(samples, epoch)
        
        # Final visualizations
        print("\nGenerating final visualizations...")
        self.plot_loss_curve()
        
        # Save final model
        self.save_checkpoint(self.epochs - 1, self.history["epoch_loss"][-1])
        
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
        
        exp_id = self.config.get('experiment', {}).get('id', 'diffusion')
        fig.suptitle(f"{exp_id} - Training Loss Curve", fontsize=14)
        plt.tight_layout()
        
        path = self.figures_dir / "loss_curve.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {path}")
    
    def plot_samples(self, samples: torch.Tensor, epoch: int):
        """Plot generated samples grid."""
        samples = samples.cpu().numpy()
        n_samples = min(16, len(samples))
        
        fig, axes = plt.subplots(4, 4, figsize=(16, 12))
        axes = axes.flatten()
        
        # Get real samples for comparison
        real_indices = np.random.choice(len(self.train_dataset), n_samples, replace=False)
        
        wave = self.wave.numpy()
        
        for i in range(n_samples):
            ax = axes[i]
            
            # Generated sample (orange)
            gen = samples[i, 0]
            ax.plot(wave, gen, color='orange', alpha=0.8, linewidth=0.8, label='Generated')
            
            # Real sample (blue)
            real = self.train_dataset[real_indices[i]].numpy()[0]
            ax.plot(wave, real, color='blue', alpha=0.5, linewidth=0.8, label='Real')
            
            ax.set_xlim(wave.min(), wave.max())
            ax.set_ylim(-1.5, 1.5)
            
            if i == 0:
                ax.legend(fontsize=8)
            if i >= 12:
                ax.set_xlabel("Wavelength (Å)")
            if i % 4 == 0:
                ax.set_ylabel("Normalized Flux")
        
        exp_id = self.config.get('experiment', {}).get('id', 'diffusion')
        fig.suptitle(f"{exp_id} - Generated Samples (Epoch {epoch+1})", fontsize=14)
        plt.tight_layout()
        
        path = self.figures_dir / f"samples_epoch_{epoch+1:03d}.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {path}")
    
    def save_summary(self):
        """Save training summary."""
        exp_cfg = self.config.get("experiment", {})
        summary = {
            "experiment_id": exp_cfg.get("id", "diffusion"),
            "experiment_name": exp_cfg.get("name", "1D DDPM"),
            "date": datetime.now().isoformat(),
            "model": {
                "type": "UNet1D",
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
        description="Train 1D Diffusion Model on Stellar Spectra"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/diffusion/baseline.yaml",
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
    dataset = StellarSpectraDataset(
        file_path=data_cfg.get("file_path"),
        num_samples=data_cfg.get("train_size", 10000),
        normalization=data_cfg.get("normalization", "minmax"),
        snr_threshold=data_cfg.get("snr_threshold", 50.0),
    )
    
    # Create model
    model_cfg = config.get("model", {})
    model = UNet1D(
        in_channels=model_cfg.get("in_channels", 1),
        out_channels=model_cfg.get("out_channels", 1),
        base_channels=model_cfg.get("base_channels", 64),
        channel_mults=tuple(model_cfg.get("channel_mults", [1, 2, 4, 8])),
        num_res_blocks=model_cfg.get("num_res_blocks", 2),
        attention_resolutions=tuple(model_cfg.get("attention_resolutions", [16, 8])),
        dropout=model_cfg.get("dropout", 0.1),
        time_emb_dim=model_cfg.get("time_emb_dim", 256),
    )
    
    # Create diffusion
    diff_cfg = config.get("diffusion", {})
    diffusion = GaussianDiffusion(
        model=model,
        timesteps=diff_cfg.get("timesteps", 1000),
        beta_schedule=diff_cfg.get("beta_schedule", "linear"),
        beta_start=diff_cfg.get("beta_start", 1e-4),
        beta_end=diff_cfg.get("beta_end", 0.02),
        loss_type=diff_cfg.get("loss_type", "l2"),
    )
    
    # Create trainer
    trainer = DiffusionTrainer(
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

