#!/usr/bin/env python3
"""
Evaluation Script for Supervised Conditional DDPM Spectrum Denoising

Generates all required figures:
1. Loss curve (diff_supervised_loss_curve.png)
2. Denoising comparison grid (diff_supervised_denoising_samples.png)
3. MSE comparison bar chart (diff_supervised_mse_comparison.png)
4. SSIM-1D comparison (diff_supervised_ssim_comparison.png)
5. SNR sweep curves (diff_supervised_snr_sweep.png)

Experiment ID: SD-20251203-diff-supervised-01
MVP Source: MVP-1.0

Usage:
    python scripts/eval_supervised.py --ckpt lightning_logs/supervised/checkpoints/best.ckpt
    python scripts/eval_supervised.py --ckpt lightning_logs/supervised/checkpoints/best.ckpt \
        --output /home/swei20/Physics_Informed_AI/logg/diffusion/img/

Author: Viska Wei
Date: 2025-12-03
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.diffusion.conditional_unet_1d import ConditionalUNet1D, count_parameters
from models.diffusion.conditional_ddpm import ConditionalGaussianDiffusion
from models.diffusion.utils import normalize_spectrum, denormalize_spectrum


# ============================================
# Metrics
# ============================================

def compute_mse(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute Mean Squared Error."""
    return np.mean((pred - target) ** 2)


def compute_mae(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute Mean Absolute Error."""
    return np.mean(np.abs(pred - target))


def compute_ssim_1d(pred: np.ndarray, target: np.ndarray, 
                    window_size: int = 11) -> float:
    """
    Compute 1D Structural Similarity Index (SSIM) for spectra.
    
    Adapted from 2D SSIM to 1D signals.
    """
    # Constants for stability
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # Gaussian window
    gauss = signal.windows.gaussian(window_size, std=1.5)
    gauss = gauss / gauss.sum()
    
    # Local means
    mu_pred = np.convolve(pred, gauss, mode='valid')
    mu_target = np.convolve(target, gauss, mode='valid')
    
    # Local variances and covariance
    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_pred_target = mu_pred * mu_target
    
    sigma_pred_sq = np.convolve(pred ** 2, gauss, mode='valid') - mu_pred_sq
    sigma_target_sq = np.convolve(target ** 2, gauss, mode='valid') - mu_target_sq
    sigma_pred_target = np.convolve(pred * target, gauss, mode='valid') - mu_pred_target
    
    # SSIM
    numerator = (2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)
    denominator = (mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2)
    
    ssim_map = numerator / denominator
    return float(np.mean(ssim_map))


def compute_psnr(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio."""
    mse = compute_mse(pred, target)
    if mse == 0:
        return float('inf')
    max_val = max(np.max(np.abs(target)), np.max(np.abs(pred)))
    return 20 * np.log10(max_val / np.sqrt(mse))


# ============================================
# Data Loading
# ============================================

DEFAULT_DATA_ROOT = "/srv/local/tmp/swei20/data/bosz50000/z0"


def load_test_data(data_path: str, num_samples: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load test data from HDF5 file."""
    print(f"Loading test data from: {data_path}")
    
    with h5py.File(data_path, "r") as f:
        wave = torch.tensor(f["spectrumdataset/wave"][()], dtype=torch.float32)
        flux = torch.tensor(
            f["dataset/arrays/flux/value"][:num_samples], 
            dtype=torch.float32
        )
    
    # Clean and normalize
    flux = flux.clip(min=0.0)
    flux = torch.nan_to_num(flux, nan=0.0)
    
    flux_normalized, _ = normalize_spectrum(flux, method="minmax")
    flux_normalized = flux_normalized * 2 - 1  # Scale to [-1, 1]
    
    print(f"Loaded {len(flux_normalized)} test spectra")
    return wave, flux_normalized


def add_noise(clean: torch.Tensor, snr: float) -> torch.Tensor:
    """Add Gaussian noise to spectrum."""
    signal_rms = torch.sqrt(torch.mean(clean ** 2))
    noise_std = signal_rms / snr
    noise = torch.randn_like(clean) * noise_std
    return torch.clamp(clean + noise, -1.5, 1.5)


# ============================================
# Model Loading
# ============================================

def load_model(checkpoint_path: str, device: str = "cuda") -> Tuple[ConditionalUNet1D, ConditionalGaussianDiffusion, Dict]:
    """Load trained model from checkpoint."""
    print(f"Loading model from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    
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
    
    # Load weights (prefer EMA if available)
    if "ema_shadow" in checkpoint and checkpoint["ema_shadow"]:
        print("Loading EMA weights...")
        for name, param in model.named_parameters():
            if name in checkpoint["ema_shadow"]:
                param.data = checkpoint["ema_shadow"][name]
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
    
    model = model.to(device)
    model.eval()
    
    # Create diffusion
    diff_cfg = config.get("diffusion", {})
    diffusion = ConditionalGaussianDiffusion(
        model=model,
        timesteps=diff_cfg.get("timesteps", 1000),
        beta_schedule=diff_cfg.get("beta_schedule", "linear"),
        beta_start=diff_cfg.get("beta_start", 1e-4),
        beta_end=diff_cfg.get("beta_end", 0.02),
        loss_type=diff_cfg.get("loss_type", "l2"),
    ).to(device)
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    return model, diffusion, config


# ============================================
# Evaluation
# ============================================

@torch.no_grad()
def evaluate_snr_sweep(
    diffusion: ConditionalGaussianDiffusion,
    clean_spectra: torch.Tensor,
    snr_levels: List[float],
    device: str,
    n_samples: int = 100,
) -> Dict[str, Dict[str, float]]:
    """Evaluate model across different SNR levels."""
    results = {}
    
    for snr in snr_levels:
        print(f"\nEvaluating SNR={snr}...")
        
        mse_list, mae_list, ssim_list, psnr_list = [], [], [], []
        mse_noisy_list = []
        
        for i in tqdm(range(min(n_samples, len(clean_spectra)))):
            clean = clean_spectra[i:i+1].unsqueeze(1).to(device)  # (1, 1, L)
            noisy = add_noise(clean.squeeze(1), snr).unsqueeze(1).to(device)
            
            # Denoise
            denoised = diffusion.p_sample_loop(noisy, progress=False)
            
            # Convert to numpy
            clean_np = clean.squeeze().cpu().numpy()
            noisy_np = noisy.squeeze().cpu().numpy()
            denoised_np = denoised.squeeze().cpu().numpy()
            
            # Compute metrics
            mse_list.append(compute_mse(denoised_np, clean_np))
            mae_list.append(compute_mae(denoised_np, clean_np))
            ssim_list.append(compute_ssim_1d(denoised_np, clean_np))
            psnr_list.append(compute_psnr(denoised_np, clean_np))
            mse_noisy_list.append(compute_mse(noisy_np, clean_np))
        
        results[f"snr_{snr}"] = {
            "snr": snr,
            "mse_mean": np.mean(mse_list),
            "mse_std": np.std(mse_list),
            "mae_mean": np.mean(mae_list),
            "mae_std": np.std(mae_list),
            "ssim_mean": np.mean(ssim_list),
            "ssim_std": np.std(ssim_list),
            "psnr_mean": np.mean(psnr_list),
            "psnr_std": np.std(psnr_list),
            "mse_noisy_mean": np.mean(mse_noisy_list),
        }
        
        print(f"  MSE: {results[f'snr_{snr}']['mse_mean']:.6f} ± {results[f'snr_{snr}']['mse_std']:.6f}")
        print(f"  SSIM: {results[f'snr_{snr}']['ssim_mean']:.4f} ± {results[f'snr_{snr}']['ssim_std']:.4f}")
    
    return results


# ============================================
# Plotting Functions
# ============================================

def plot_loss_curve(history: Dict, output_dir: Path, exp_id: str):
    """Plot training loss curve (Figure 1)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Per-step loss
    ax1 = axes[0]
    train_loss = history.get("train_loss", [])
    if train_loss:
        steps = np.arange(len(train_loss))
        ax1.plot(steps, train_loss, alpha=0.5, linewidth=0.5)
        # Smoothed
        window = min(100, len(train_loss) // 10)
        if window > 1:
            smoothed = np.convolve(train_loss, np.ones(window)/window, mode='valid')
            ax1.plot(steps[window-1:], smoothed, color='red', linewidth=2, label='Smoothed')
        ax1.set_xlabel("Training Step")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training Loss per Step")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Per-epoch loss
    ax2 = axes[1]
    epoch_loss = history.get("epoch_loss", [])
    if epoch_loss:
        epochs = np.arange(1, len(epoch_loss) + 1)
        ax2.plot(epochs, epoch_loss, marker='o', linewidth=2, color='blue')
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.set_title("Training Loss per Epoch")
        ax2.grid(True, alpha=0.3)
    
    fig.suptitle(f"{exp_id} - Training Loss Curve", fontsize=14)
    plt.tight_layout()
    
    path = output_dir / "diff_supervised_loss_curve.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_denoising_samples(
    diffusion: ConditionalGaussianDiffusion,
    clean_spectra: torch.Tensor,
    wave: torch.Tensor,
    output_dir: Path,
    exp_id: str,
    device: str,
):
    """Plot denoising comparison grid (4x2 - Figure 2)."""
    n_samples = 4
    snrs = [5, 10, 20, 50]
    
    wave_np = wave.numpy()
    
    fig, axes = plt.subplots(n_samples, 4, figsize=(20, 4 * n_samples))
    
    for i in range(n_samples):
        clean = clean_spectra[i:i+1].unsqueeze(1).to(device)
        snr = snrs[i]
        noisy = add_noise(clean.squeeze(1), snr).unsqueeze(1).to(device)
        
        with torch.no_grad():
            denoised = diffusion.p_sample_loop(noisy, progress=False)
        
        clean_np = clean.squeeze().cpu().numpy()
        noisy_np = noisy.squeeze().cpu().numpy()
        denoised_np = denoised.squeeze().cpu().numpy()
        residual_np = clean_np - denoised_np
        
        mse = compute_mse(denoised_np, clean_np)
        
        # Column 1: Clean
        axes[i, 0].plot(wave_np, clean_np, color='blue', linewidth=0.8)
        axes[i, 0].set_ylabel(f"SNR={snr}")
        if i == 0:
            axes[i, 0].set_title("Clean (Ground Truth)")
        axes[i, 0].set_ylim(-1.5, 1.5)
        
        # Column 2: Noisy
        axes[i, 1].plot(wave_np, noisy_np, color='gray', linewidth=0.8)
        if i == 0:
            axes[i, 1].set_title("Noisy Input")
        axes[i, 1].set_ylim(-1.5, 1.5)
        
        # Column 3: Denoised
        axes[i, 2].plot(wave_np, denoised_np, color='orange', linewidth=0.8)
        if i == 0:
            axes[i, 2].set_title("Denoised (DDPM)")
        axes[i, 2].set_ylim(-1.5, 1.5)
        
        # Column 4: Residual
        axes[i, 3].plot(wave_np, residual_np, color='red', linewidth=0.8)
        axes[i, 3].set_title(f"Residual (MSE={mse:.4f})")
        axes[i, 3].set_ylim(-0.5, 0.5)
    
    for j in range(4):
        axes[-1, j].set_xlabel("Wavelength (Å)")
    
    fig.suptitle(f"{exp_id} - Denoising Results", fontsize=14)
    plt.tight_layout()
    
    path = output_dir / "diff_supervised_denoising_samples.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_mse_comparison(results: Dict, output_dir: Path, exp_id: str):
    """Plot MSE comparison bar chart (Figure 3)."""
    snrs = []
    mse_denoised = []
    mse_noisy = []
    
    for key, val in sorted(results.items(), key=lambda x: x[1]["snr"]):
        snrs.append(val["snr"])
        mse_denoised.append(val["mse_mean"])
        mse_noisy.append(val["mse_noisy_mean"])
    
    x = np.arange(len(snrs))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, mse_noisy, width, label='Noisy Input', color='gray', alpha=0.7)
    bars2 = ax.bar(x + width/2, mse_denoised, width, label='Denoised (DDPM)', color='orange')
    
    ax.set_xlabel('SNR')
    ax.set_ylabel('MSE')
    ax.set_title(f'{exp_id} - MSE Comparison: Noisy vs Denoised')
    ax.set_xticks(x)
    ax.set_xticklabels([f'SNR={s}' for s in snrs])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    path = output_dir / "diff_supervised_mse_comparison.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_ssim_comparison(results: Dict, output_dir: Path, exp_id: str):
    """Plot SSIM comparison (Figure 4)."""
    snrs = []
    ssim_vals = []
    ssim_stds = []
    
    for key, val in sorted(results.items(), key=lambda x: x[1]["snr"]):
        snrs.append(val["snr"])
        ssim_vals.append(val["ssim_mean"])
        ssim_stds.append(val["ssim_std"])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.bar(range(len(snrs)), ssim_vals, yerr=ssim_stds, capsize=5, 
           color='green', alpha=0.7, edgecolor='darkgreen')
    
    ax.set_xlabel('SNR')
    ax.set_ylabel('SSIM')
    ax.set_title(f'{exp_id} - SSIM-1D Score by SNR Level')
    ax.set_xticks(range(len(snrs)))
    ax.set_xticklabels([f'SNR={s}' for s in snrs])
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (val, std) in enumerate(zip(ssim_vals, ssim_stds)):
        ax.annotate(f'{val:.3f}±{std:.3f}',
                    xy=(i, val + std + 0.02),
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    path = output_dir / "diff_supervised_ssim_comparison.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_snr_sweep(results: Dict, output_dir: Path, exp_id: str):
    """Plot SNR sweep curves (Figure 5)."""
    snrs = []
    mse_denoised = []
    mse_noisy = []
    ssim_vals = []
    psnr_vals = []
    
    for key, val in sorted(results.items(), key=lambda x: x[1]["snr"]):
        snrs.append(val["snr"])
        mse_denoised.append(val["mse_mean"])
        mse_noisy.append(val["mse_noisy_mean"])
        ssim_vals.append(val["ssim_mean"])
        psnr_vals.append(val["psnr_mean"])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # MSE vs SNR
    ax1 = axes[0, 0]
    ax1.plot(snrs, mse_noisy, 'o-', label='Noisy Input', color='gray', linewidth=2, markersize=8)
    ax1.plot(snrs, mse_denoised, 's-', label='Denoised (DDPM)', color='orange', linewidth=2, markersize=8)
    ax1.set_xlabel('SNR')
    ax1.set_ylabel('MSE')
    ax1.set_title('MSE vs SNR')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Improvement ratio
    ax2 = axes[0, 1]
    improvement = [mn / md if md > 0 else 0 for mn, md in zip(mse_noisy, mse_denoised)]
    ax2.bar(range(len(snrs)), improvement, color='purple', alpha=0.7)
    ax2.set_xlabel('SNR')
    ax2.set_ylabel('MSE Improvement Ratio (Noisy/Denoised)')
    ax2.set_title('Denoising Improvement by SNR')
    ax2.set_xticks(range(len(snrs)))
    ax2.set_xticklabels([f'{s}' for s in snrs])
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=1, color='red', linestyle='--', label='No improvement')
    ax2.legend()
    
    # SSIM vs SNR
    ax3 = axes[1, 0]
    ax3.plot(snrs, ssim_vals, 'o-', color='green', linewidth=2, markersize=8)
    ax3.set_xlabel('SNR')
    ax3.set_ylabel('SSIM')
    ax3.set_title('SSIM vs SNR')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)
    
    # PSNR vs SNR
    ax4 = axes[1, 1]
    ax4.plot(snrs, psnr_vals, 'o-', color='blue', linewidth=2, markersize=8)
    ax4.set_xlabel('SNR')
    ax4.set_ylabel('PSNR (dB)')
    ax4.set_title('PSNR vs SNR')
    ax4.grid(True, alpha=0.3)
    
    fig.suptitle(f'{exp_id} - SNR Sweep Analysis', fontsize=14)
    plt.tight_layout()
    
    path = output_dir / "diff_supervised_snr_sweep.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ============================================
# Main
# ============================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Supervised Conditional DDPM for Spectrum Denoising"
    )
    parser.add_argument(
        "--ckpt", 
        type=str, 
        required=True,
        help="Path to checkpoint file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/home/swei20/Physics_Informed_AI/logg/diffusion/img/",
        help="Output directory for figures"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Override data root directory"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of samples for evaluation"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu)"
    )
    args = parser.parse_args()
    
    # Setup
    if args.data_root:
        os.environ["DATA_ROOT"] = args.data_root
    elif "DATA_ROOT" not in os.environ:
        os.environ["DATA_ROOT"] = DEFAULT_DATA_ROOT
    
    data_root = os.environ.get("DATA_ROOT")
    print(f"DATA_ROOT: {data_root}")
    
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model, diffusion, config = load_model(args.ckpt, device)
    exp_id = config.get("experiment", {}).get("id", "supervised-ddpm")
    history = torch.load(args.ckpt, map_location="cpu").get("history", {})
    
    # Load test data
    test_path = config.get("data", {}).get("test_path", 
        f"{data_root}/test_1k/dataset.h5")
    wave, clean_spectra = load_test_data(test_path, num_samples=args.n_samples * 2)
    
    print(f"\n{'='*60}")
    print(f"Evaluating: {exp_id}")
    print(f"{'='*60}")
    
    # Plot loss curve from history
    if history:
        plot_loss_curve(history, output_dir, exp_id)
    
    # Plot denoising samples
    print("\nGenerating denoising samples...")
    plot_denoising_samples(diffusion, clean_spectra, wave, output_dir, exp_id, device)
    
    # Evaluate across SNR levels
    snr_levels = [5, 10, 20, 50]
    results = evaluate_snr_sweep(diffusion, clean_spectra, snr_levels, device, args.n_samples)
    
    # Plot MSE comparison
    plot_mse_comparison(results, output_dir, exp_id)
    
    # Plot SSIM comparison
    plot_ssim_comparison(results, output_dir, exp_id)
    
    # Plot SNR sweep
    plot_snr_sweep(results, output_dir, exp_id)
    
    # Save results
    results_path = output_dir / "diff_supervised_eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results: {results_path}")
    
    print(f"\n{'='*60}")
    print("Evaluation completed!")
    print(f"Figures saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

