#!/usr/bin/env python3
"""
Evaluation Script for Bounded Noise Denoiser (MVP-0.5)

Experiment ID: SD-20251204-diff-bounded-01
MVP Source: MVP-0.5

This script evaluates the trained bounded noise denoiser and generates
comprehensive metrics and visualizations.

Usage:
    python scripts/eval_bounded_denoiser.py --ckpt lightning_logs/diffusion/bounded_noise/checkpoints/best.ckpt
    python scripts/eval_bounded_denoiser.py --ckpt best.ckpt --test-lambdas 0.1 0.2 0.3 0.4 0.5

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
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.diffusion.conditional_unet_1d import ConditionalUNet1D, count_parameters
from models.diffusion.utils import normalize_spectrum, denormalize_spectrum

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
# Dataset (Reuse from training)
# ============================================

class BoundedNoiseSpectraDataset(Dataset):
    """
    Dataset for bounded noise spectrum denoising evaluation.
    """
    
    def __init__(
        self,
        file_path: str,
        num_samples: int = 1000,
        normalization: str = "minmax",
        snr_threshold: float = 50.0,
        lambda_values: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5],
        sigma_scale: float = 0.1,
        offset: int = 10000,  # Start from this index for test set
    ):
        self.file_path = file_path
        self.num_samples = num_samples
        self.normalization = normalization
        self.snr_threshold = snr_threshold
        self.lambda_values = lambda_values
        self.sigma_scale = sigma_scale
        self.offset = offset
        
        self.clean_flux = None
        self.sigma = None
        self.wave = None
        self.norm_params = None
        
        self._load_data()
    
    def _load_data(self):
        """Load and preprocess the data."""
        print(f"Loading test data from: {self.file_path}")
        
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Data file not found: {self.file_path}")
        
        with h5py.File(self.file_path, "r") as f:
            self.wave = torch.tensor(f["spectrumdataset/wave"][()], dtype=torch.float32)
            
            # Load from offset for test set
            flux = torch.tensor(
                f["dataset/arrays/flux/value"][self.offset:self.offset + self.num_samples], 
                dtype=torch.float32
            )
            error = torch.tensor(
                f["dataset/arrays/error/value"][self.offset:self.offset + self.num_samples], 
                dtype=torch.float32
            )
        
        flux = flux.clip(min=0.0)
        flux = torch.nan_to_num(flux, nan=0.0)
        error = torch.nan_to_num(error, nan=1.0)
        
        snr = flux.norm(dim=-1) / error.norm(dim=-1)
        print(f"SNR range: {snr.min():.1f} - {snr.max():.1f}, median: {snr.median():.1f}")
        
        if self.snr_threshold > 0:
            high_snr_mask = snr >= self.snr_threshold
            n_high_snr = high_snr_mask.sum().item()
            print(f"High SNR (>={self.snr_threshold}) samples: {n_high_snr}/{len(flux)}")
            
            if n_high_snr > 100:
                flux = flux[high_snr_mask]
                error = error[high_snr_mask]
        
        self.clean_flux, self.norm_params = normalize_spectrum(flux, method=self.normalization)
        self.clean_flux = self.clean_flux * 2 - 1
        
        if self.normalization == "minmax":
            flux_range = self.norm_params["max"] - self.norm_params["min"]
            flux_range = torch.where(flux_range < 1e-8, torch.ones_like(flux_range), flux_range)
            self.sigma = (error / flux_range) * 2
        else:
            self.sigma = error / (error.mean(dim=-1, keepdim=True) + 1e-8)
        
        self.sigma = self.sigma * self.sigma_scale
        self.sigma = self.sigma.clip(min=0.01, max=1.0)
        
        self.num_samples = len(self.clean_flux)
        self.spectrum_length = self.clean_flux.shape[1]
        
        print(f"Loaded {self.num_samples} test spectra")
    
    def add_bounded_noise(
        self, 
        clean_spectrum: torch.Tensor, 
        sigma: torch.Tensor,
        lam: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add bounded noise."""
        epsilon = torch.randn_like(clean_spectrum)
        noise = lam * sigma * epsilon
        noisy_spectrum = clean_spectrum + noise
        return noisy_spectrum, epsilon
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        clean = self.clean_flux[idx].unsqueeze(0)
        sigma = self.sigma[idx].unsqueeze(0)
        return clean, sigma


# ============================================
# Diffusion (Reuse from training)
# ============================================

class BoundedNoiseDiffusion(nn.Module):
    """Bounded Noise Diffusion for spectrum denoising."""
    
    def __init__(
        self,
        model: nn.Module,
        lambda_values: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5],
        prediction_target: str = "x0",
    ):
        super().__init__()
        
        self.model = model
        self.lambda_values = sorted(lambda_values)
        self.lambda_max = max(lambda_values)
        self.prediction_target = prediction_target
    
    @torch.no_grad()
    def single_step_denoise(
        self,
        noisy: torch.Tensor,
        sigma: torch.Tensor,
        lam: float,
    ) -> torch.Tensor:
        """Single-step denoising."""
        device = noisy.device
        batch_size = noisy.shape[0]
        
        t = torch.full((batch_size,), int((lam - 0.1) / 0.1), device=device, dtype=torch.long)
        t = t.clamp(0, len(self.lambda_values) - 1)
        
        if self.prediction_target == "x0":
            x0_pred = self.model(noisy, t, sigma)
        else:
            eps_pred = self.model(noisy, t, sigma)
            x0_pred = noisy - lam * sigma * eps_pred
        
        return x0_pred.clamp(-1.5, 1.5)
    
    @torch.no_grad()
    def multi_step_denoise(
        self,
        noisy: torch.Tensor,
        sigma: torch.Tensor,
        num_steps: int = 10,
    ) -> torch.Tensor:
        """Multi-step denoising starting from observation."""
        device = noisy.device
        batch_size = noisy.shape[0]
        
        x_t = noisy.clone()
        lambda_schedule = np.linspace(self.lambda_max, 0, num_steps + 1)
        
        for i in range(num_steps):
            lam_t = lambda_schedule[i]
            lam_next = lambda_schedule[i + 1]
            
            t = torch.full((batch_size,), int((lam_t - 0.1) / 0.1), device=device, dtype=torch.long)
            t = t.clamp(0, len(self.lambda_values) - 1)
            
            if self.prediction_target == "x0":
                x0_pred = self.model(x_t, t, sigma)
            else:
                eps_pred = self.model(x_t, t, sigma)
                x0_pred = x_t - lam_t * sigma * eps_pred
            
            x0_pred = x0_pred.clamp(-1.5, 1.5)
            
            if i < num_steps - 1 and lam_next > 0:
                if self.prediction_target == "eps":
                    x_t = x0_pred + lam_next * sigma * eps_pred
                else:
                    eps_est = (x_t - x0_pred) / (lam_t * sigma + 1e-8)
                    x_t = x0_pred + lam_next * sigma * eps_est
            else:
                x_t = x0_pred
        
        return x_t


# ============================================
# Evaluator
# ============================================

class BoundedDenoiserEvaluator:
    """Evaluator for bounded noise denoiser."""
    
    def __init__(
        self,
        model: nn.Module,
        diffusion: BoundedNoiseDiffusion,
        test_dataset: BoundedNoiseSpectraDataset,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.diffusion = diffusion.to(device)
        self.test_dataset = test_dataset
        self.device = device
        
        self.model.eval()
        
        self.figures_dir = Path(KNOWLEDGE_CENTER) / "logg/diffusion/img"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        self.wave = test_dataset.wave
        self.lambda_values = test_dataset.lambda_values
    
    @torch.no_grad()
    def evaluate_all_lambdas(self) -> Dict[float, Dict[str, float]]:
        """Evaluate at all lambda values."""
        results = {}
        
        print("\n📊 Evaluating at all lambda values...")
        
        for lam in tqdm(self.lambda_values, desc="Lambda values"):
            mse_noisy_list = []
            mse_denoised_list = []
            wmse_list = []
            
            for idx in range(len(self.test_dataset)):
                clean = self.test_dataset.clean_flux[idx].unsqueeze(0).unsqueeze(0)
                sigma = self.test_dataset.sigma[idx].unsqueeze(0).unsqueeze(0)
                
                noisy, _ = self.test_dataset.add_bounded_noise(clean.squeeze(0), sigma.squeeze(0), lam)
                noisy = noisy.unsqueeze(0)
                
                clean = clean.to(self.device)
                noisy = noisy.to(self.device)
                sigma = sigma.to(self.device)
                
                denoised = self.diffusion.single_step_denoise(noisy, sigma, lam)
                
                mse_noisy = F.mse_loss(noisy, clean).item()
                mse_denoised = F.mse_loss(denoised, clean).item()
                
                residual = (denoised - clean) / (sigma + 1e-8)
                wmse = (residual ** 2).mean().item()
                
                mse_noisy_list.append(mse_noisy)
                mse_denoised_list.append(mse_denoised)
                wmse_list.append(wmse)
            
            results[lam] = {
                "mse_noisy": np.mean(mse_noisy_list),
                "mse_noisy_std": np.std(mse_noisy_list),
                "mse_denoised": np.mean(mse_denoised_list),
                "mse_denoised_std": np.std(mse_denoised_list),
                "wmse": np.mean(wmse_list),
                "wmse_std": np.std(wmse_list),
                "improvement": 1 - np.mean(mse_denoised_list) / np.mean(mse_noisy_list),
            }
        
        return results
    
    @torch.no_grad()
    def evaluate_multi_step(self, num_steps_list: List[int] = [5, 10, 20]) -> Dict:
        """Evaluate multi-step denoising."""
        results = {}
        lam = self.lambda_values[-1]  # Use max lambda
        
        print(f"\n📊 Evaluating multi-step denoising at λ={lam}...")
        
        for num_steps in tqdm(num_steps_list, desc="Step counts"):
            mse_denoised_list = []
            
            for idx in range(min(100, len(self.test_dataset))):  # Use subset
                clean = self.test_dataset.clean_flux[idx].unsqueeze(0).unsqueeze(0)
                sigma = self.test_dataset.sigma[idx].unsqueeze(0).unsqueeze(0)
                
                noisy, _ = self.test_dataset.add_bounded_noise(clean.squeeze(0), sigma.squeeze(0), lam)
                noisy = noisy.unsqueeze(0)
                
                clean = clean.to(self.device)
                noisy = noisy.to(self.device)
                sigma = sigma.to(self.device)
                
                denoised = self.diffusion.multi_step_denoise(noisy, sigma, num_steps)
                mse_denoised = F.mse_loss(denoised, clean).item()
                mse_denoised_list.append(mse_denoised)
            
            results[num_steps] = {
                "mse_denoised": np.mean(mse_denoised_list),
                "mse_denoised_std": np.std(mse_denoised_list),
            }
        
        return results
    
    def plot_comprehensive_results(self, results: Dict):
        """Generate all plots."""
        self.plot_mse_comparison(results)
        self.plot_improvement_bars(results)
        self.plot_denoising_samples(results)
        self.plot_flux_distribution()
    
    def plot_mse_comparison(self, results: Dict):
        """Plot MSE comparison."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        lambdas = sorted(results.keys())
        mse_noisy = [results[lam]["mse_noisy"] for lam in lambdas]
        mse_noisy_std = [results[lam]["mse_noisy_std"] for lam in lambdas]
        mse_denoised = [results[lam]["mse_denoised"] for lam in lambdas]
        mse_denoised_std = [results[lam]["mse_denoised_std"] for lam in lambdas]
        
        ax.errorbar(lambdas, mse_noisy, yerr=mse_noisy_std, fmt='o-', 
                   linewidth=2, markersize=8, capsize=5,
                   label='MSE(noisy, clean)', color='gray')
        ax.errorbar(lambdas, mse_denoised, yerr=mse_denoised_std, fmt='s-', 
                   linewidth=2, markersize=8, capsize=5,
                   label='MSE(denoised, clean)', color='forestgreen')
        
        ax.set_xlabel("λ (Noise Level)", fontsize=12)
        ax.set_ylabel("MSE", fontsize=12)
        ax.set_title(f"{EXP_ID}\nMSE Comparison: Noisy vs Denoised (Test Set)", 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        path = self.figures_dir / "diff_bounded_mse_comparison.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Saved: {path}")
    
    def plot_improvement_bars(self, results: Dict):
        """Plot improvement bars."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        lambdas = sorted(results.keys())
        improvements = [results[lam]["improvement"] * 100 for lam in lambdas]
        
        colors = ['#2ecc71' if imp > 30 else '#f39c12' if imp > 0 else '#e74c3c' 
                  for imp in improvements]
        
        bars = ax.bar([str(lam) for lam in lambdas], improvements, color=colors, edgecolor='black')
        
        for bar, imp in zip(bars, improvements):
            height = bar.get_height()
            ax.annotate(f'{imp:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=11, fontweight='bold')
        
        ax.axhline(y=30, color='red', linestyle='--', linewidth=2, label='30% threshold')
        
        ax.set_xlabel("λ (Noise Level)", fontsize=12)
        ax.set_ylabel("Denoising Improvement (%)", fontsize=12)
        ax.set_title(f"{EXP_ID}\nDenoising Improvement by Noise Level (Test Set)", 
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        path = self.figures_dir / "diff_bounded_improvement.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Saved: {path}")
    
    def plot_denoising_samples(self, results: Dict):
        """Plot denoising samples."""
        n_samples = 3
        fig, axes = plt.subplots(n_samples, 3, figsize=(18, 4 * n_samples))
        
        indices = np.random.choice(len(self.test_dataset), n_samples, replace=False)
        wave = self.wave.numpy()
        lambdas_to_show = [0.3, 0.5]
        
        for i, idx in enumerate(indices):
            clean = self.test_dataset.clean_flux[idx].unsqueeze(0).unsqueeze(0)
            sigma = self.test_dataset.sigma[idx].unsqueeze(0).unsqueeze(0)
            
            lam = lambdas_to_show[i % len(lambdas_to_show)]
            
            noisy, _ = self.test_dataset.add_bounded_noise(clean.squeeze(0), sigma.squeeze(0), lam)
            noisy = noisy.unsqueeze(0)
            
            clean_gpu = clean.to(self.device)
            noisy_gpu = noisy.to(self.device)
            sigma_gpu = sigma.to(self.device)
            
            with torch.no_grad():
                denoised = self.diffusion.single_step_denoise(noisy_gpu, sigma_gpu, lam)
            
            clean_np = clean.squeeze().numpy()
            noisy_np = noisy.squeeze().numpy()
            denoised_np = denoised.cpu().squeeze().numpy()
            
            axes[i, 0].plot(wave, clean_np, color='royalblue', linewidth=0.8)
            axes[i, 0].set_ylabel(f"λ={lam}", fontsize=11, fontweight='bold')
            if i == 0:
                axes[i, 0].set_title("Clean (Ground Truth)", fontsize=12)
            axes[i, 0].set_ylim(-1.5, 1.5)
            axes[i, 0].grid(True, alpha=0.3)
            
            axes[i, 1].plot(wave, noisy_np, color='gray', linewidth=0.8, alpha=0.8)
            axes[i, 1].plot(wave, clean_np, color='royalblue', linewidth=0.5, alpha=0.5)
            if i == 0:
                axes[i, 1].set_title("Noisy Input", fontsize=12)
            axes[i, 1].set_ylim(-1.5, 1.5)
            axes[i, 1].grid(True, alpha=0.3)
            
            axes[i, 2].plot(wave, denoised_np, color='forestgreen', linewidth=0.8)
            axes[i, 2].plot(wave, clean_np, color='royalblue', linewidth=0.5, alpha=0.5)
            mse = np.mean((denoised_np - clean_np) ** 2)
            axes[i, 2].set_title(f"Denoised (MSE={mse:.5f})" if i == 0 else f"MSE={mse:.5f}", fontsize=12)
            axes[i, 2].set_ylim(-1.5, 1.5)
            axes[i, 2].grid(True, alpha=0.3)
        
        for j in range(3):
            axes[-1, j].set_xlabel("Wavelength (Å)", fontsize=11)
        
        fig.suptitle(f"{EXP_ID} - Denoising Results (Test Set)", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        path = self.figures_dir / "diff_bounded_denoising_samples.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Saved: {path}")
    
    def plot_flux_distribution(self):
        """Plot flux distribution comparison."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        n_samples = 100
        clean_flux = []
        noisy_flux = []
        denoised_flux = []
        
        indices = np.random.choice(len(self.test_dataset), n_samples, replace=False)
        lam = self.lambda_values[-1]
        
        for idx in indices:
            clean = self.test_dataset.clean_flux[idx].unsqueeze(0).unsqueeze(0)
            sigma = self.test_dataset.sigma[idx].unsqueeze(0).unsqueeze(0)
            
            noisy, _ = self.test_dataset.add_bounded_noise(clean.squeeze(0), sigma.squeeze(0), lam)
            noisy = noisy.unsqueeze(0)
            
            clean_gpu = clean.to(self.device)
            noisy_gpu = noisy.to(self.device)
            sigma_gpu = sigma.to(self.device)
            
            with torch.no_grad():
                denoised = self.diffusion.single_step_denoise(noisy_gpu, sigma_gpu, lam)
            
            clean_flux.extend(clean.flatten().numpy())
            noisy_flux.extend(noisy.flatten().numpy())
            denoised_flux.extend(denoised.cpu().flatten().numpy())
        
        n_kde = 50000
        clean_flux = np.random.choice(clean_flux, min(n_kde, len(clean_flux)), replace=False)
        noisy_flux = np.random.choice(noisy_flux, min(n_kde, len(noisy_flux)), replace=False)
        denoised_flux = np.random.choice(denoised_flux, min(n_kde, len(denoised_flux)), replace=False)
        
        sns.kdeplot(clean_flux, ax=ax, label='Clean (Ground Truth)', color='royalblue', linewidth=2)
        sns.kdeplot(noisy_flux, ax=ax, label=f'Noisy (λ={lam})', color='gray', linewidth=2, linestyle='--')
        sns.kdeplot(denoised_flux, ax=ax, label='Denoised', color='forestgreen', linewidth=2)
        
        ax.set_xlabel("Flux Value (normalized)", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title(f"{EXP_ID}\nFlux Distribution Comparison (Test Set, λ={lam})\n(Verify: NOT Gaussian noise!)", 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
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
    
    def save_results(self, results: Dict, output_path: str = None):
        """Save evaluation results."""
        output = {
            "experiment_id": EXP_ID,
            "experiment_name": EXP_NAME,
            "date": datetime.now().isoformat(),
            "test_samples": len(self.test_dataset),
            "results": {str(k): v for k, v in results.items()},
        }
        
        if output_path is None:
            output_path = self.figures_dir / "diff_bounded_eval_results.json"
        
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"📝 Saved: {output_path}")
        
        return output


# ============================================
# Main
# ============================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Bounded Noise Denoiser (MVP-0.5)"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="lightning_logs/diffusion/bounded_noise/checkpoints/best.ckpt",
        help="Path to checkpoint file"
    )
    parser.add_argument(
        "--test-lambdas",
        type=float,
        nargs='+',
        default=[0.1, 0.2, 0.3, 0.4, 0.5],
        help="Lambda values to evaluate"
    )
    parser.add_argument(
        "--num-test-samples",
        type=int,
        default=1000,
        help="Number of test samples"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Override data root directory"
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
    
    # Device
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load checkpoint
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        # Try relative to project
        ckpt_path = Path(__file__).parent.parent / args.ckpt
    
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
    
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # Get config from checkpoint
    diffusion_config = checkpoint.get("diffusion_config", {})
    lambda_values = diffusion_config.get("lambda_values", args.test_lambdas)
    prediction_target = diffusion_config.get("prediction_target", "x0")
    
    print(f"Lambda values: {lambda_values}")
    print(f"Prediction target: {prediction_target}")
    
    # Load test dataset
    dataset = BoundedNoiseSpectraDataset(
        file_path=data_file,
        num_samples=args.num_test_samples,
        normalization="minmax",
        snr_threshold=50.0,
        lambda_values=lambda_values,
        sigma_scale=1.0,  # Match training sigma_scale
        offset=10000,  # Use different data than training
    )
    
    # Create model
    model = ConditionalUNet1D(
        in_channels=1,
        cond_channels=1,
        out_channels=1,
        base_channels=32,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=2,
        attention_resolutions=(3,),
        dropout=0.1,
        time_emb_dim=256,
    )
    
    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Model parameters: {count_parameters(model):,}")
    
    # If EMA weights available, use them
    if "ema_shadow" in checkpoint:
        print("Using EMA weights")
        for name, param in model.named_parameters():
            if name in checkpoint["ema_shadow"]:
                param.data = checkpoint["ema_shadow"][name]
    
    # Create diffusion
    diffusion = BoundedNoiseDiffusion(
        model=model,
        lambda_values=lambda_values,
        prediction_target=prediction_target,
    )
    
    # Create evaluator
    evaluator = BoundedDenoiserEvaluator(
        model=model,
        diffusion=diffusion,
        test_dataset=dataset,
        device=device,
    )
    
    # Evaluate
    results = evaluator.evaluate_all_lambdas()
    
    # Generate plots
    evaluator.plot_comprehensive_results(results)
    
    # Save results
    evaluator.save_results(results)
    
    # Print results
    print("\n" + "="*60)
    print("📊 EVALUATION RESULTS")
    print("="*60)
    for lam, metrics in sorted(results.items()):
        print(f"λ={lam:.1f}:")
        print(f"  MSE(noisy, clean):    {metrics['mse_noisy']:.6f} ± {metrics['mse_noisy_std']:.6f}")
        print(f"  MSE(denoised, clean): {metrics['mse_denoised']:.6f} ± {metrics['mse_denoised_std']:.6f}")
        print(f"  Improvement:          {metrics['improvement']*100:.1f}%")
        print(f"  wMSE:                 {metrics['wmse']:.6f}")
    print("="*60)
    
    # Check acceptance criteria
    max_lambda_results = results[max(lambda_values)]
    improvement = max_lambda_results["improvement"] * 100
    
    print(f"\n🎯 ACCEPTANCE CRITERIA CHECK:")
    print(f"  Improvement @ λ={max(lambda_values)}: {improvement:.1f}%")
    if improvement > 30:
        print(f"  ✅ PASSED: Improvement > 30%")
    else:
        print(f"  ❌ FAILED: Improvement <= 30%")
    
    return results


if __name__ == "__main__":
    main()

