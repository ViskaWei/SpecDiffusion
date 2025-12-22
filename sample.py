#!/usr/bin/env python
"""
SpecDiffusion Sampling Script

Generate samples from a trained diffusion model.

Usage:
    # Generate samples with DDIM (fast)
    python sample.py --checkpoint path/to/checkpoint.ckpt --num_samples 16
    
    # Generate samples with DDPM (slow but higher quality)
    python sample.py --checkpoint path/to/checkpoint.ckpt --num_samples 16 --no_ddim
    
    # Save samples to file
    python sample.py --checkpoint path/to/checkpoint.ckpt --output samples.npy
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    parser = argparse.ArgumentParser(description="Generate samples from diffusion model")
    
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config (uses checkpoint config if not provided)"
    )
    parser.add_argument(
        "--num_samples", "-n",
        type=int,
        default=16,
        help="Number of samples to generate"
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=50,
        help="Number of DDIM sampling steps"
    )
    parser.add_argument(
        "--no_ddim",
        action="store_true",
        help="Use DDPM instead of DDIM (slower)"
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="DDIM eta parameter (0=deterministic, 1=DDPM-like)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="samples.npy",
        help="Output file for samples"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot samples (for spectra)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    import numpy as np
    import torch
    
    from src.utils.seed import set_all_seeds
    from src.utils.config import load_config
    from src.nn.lightning_module import DiffusionLightningModule
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set seed
    set_all_seeds(args.seed)
    
    print("=" * 60)
    print("SpecDiffusion Sampling")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Samples: {args.num_samples}")
    print(f"Method: {'DDPM' if args.no_ddim else f'DDIM ({args.num_steps} steps)'}")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Load checkpoint
    print("\nLoading model...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Get config
    if args.config:
        config = load_config(args.config)
    elif 'hyper_parameters' in checkpoint:
        config = checkpoint['hyper_parameters'].get('config', {})
    else:
        raise ValueError("No config found. Please provide --config argument.")
    
    # Create model
    lightning_module = DiffusionLightningModule.from_config(config)
    
    # Load state dict
    if 'state_dict' in checkpoint:
        lightning_module.load_state_dict(checkpoint['state_dict'])
    else:
        lightning_module.load_state_dict(checkpoint)
    
    lightning_module = lightning_module.to(device)
    lightning_module.eval()
    
    # Generate samples
    print(f"\nGenerating {args.num_samples} samples...")
    
    with torch.no_grad():
        if args.no_ddim:
            samples = lightning_module.model.sample(
                shape=lightning_module._get_sample_shape(args.num_samples),
                device=device,
            )
        else:
            samples = lightning_module.model.sample_ddim(
                shape=lightning_module._get_sample_shape(args.num_samples),
                num_inference_steps=args.num_steps,
                eta=args.eta,
                device=device,
            )
    
    # Convert to numpy
    samples_np = samples.cpu().numpy()
    
    print(f"Generated samples shape: {samples_np.shape}")
    print(f"Sample range: [{samples_np.min():.3f}, {samples_np.max():.3f}]")
    
    # Save samples
    np.save(args.output, samples_np)
    print(f"\nSamples saved to: {args.output}")
    
    # Plot if requested
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            axes = axes.flatten()
            
            for i in range(min(4, args.num_samples)):
                ax = axes[i]
                
                if len(samples_np.shape) == 3:
                    # 1D data (spectra)
                    ax.plot(samples_np[i, 0])
                    ax.set_xlabel("Wavelength bin")
                    ax.set_ylabel("Flux")
                elif len(samples_np.shape) == 4:
                    # 2D data (images)
                    img = samples_np[i].transpose(1, 2, 0)
                    if img.shape[-1] == 1:
                        img = img.squeeze(-1)
                    # Denormalize from [-1,1] to [0,1]
                    img = (img + 1) / 2
                    img = np.clip(img, 0, 1)
                    ax.imshow(img)
                    ax.axis('off')
                
                ax.set_title(f"Sample {i+1}")
            
            plt.tight_layout()
            
            plot_path = args.output.replace('.npy', '.png')
            plt.savefig(plot_path, dpi=150)
            print(f"Plot saved to: {plot_path}")
            plt.close()
            
        except ImportError:
            print("matplotlib not installed, skipping plot")


if __name__ == "__main__":
    main()

