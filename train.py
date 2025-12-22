#!/usr/bin/env python
"""
SpecDiffusion Training Script

Usage:
    # Train with config file
    python train.py --config src/configs/spectrum_ddpm.yaml
    
    # Train with overrides
    python train.py --config src/configs/spectrum_ddpm.yaml --epochs 100 --lr 1e-4
    
    # Train with W&B logging
    python train.py --config src/configs/spectrum_ddpm.yaml --wandb --project MyProject
    
    # Quick debug run
    python train.py --config src/configs/spectrum_ddpm.yaml --debug
"""

import argparse
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    parser = argparse.ArgumentParser(description="Train diffusion model")
    
    # Config
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to config YAML file"
    )
    
    # Training overrides
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--precision", type=str, help="Training precision (16-mixed, 32, bf16-mixed)")
    
    # Model overrides
    parser.add_argument("--base_channels", type=int, help="Base channel count")
    parser.add_argument("--num_timesteps", type=int, help="Number of diffusion timesteps")
    
    # Hardware
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--num_workers", type=int, help="Number of data loading workers")
    
    # Logging
    parser.add_argument("--wandb", action="store_true", help="Use W&B logging")
    parser.add_argument("--project", type=str, default="SpecDiffusion", help="W&B project name")
    parser.add_argument("--run_name", type=str, help="W&B run name")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--debug", action="store_true", help="Debug mode (fast_dev_run)")
    parser.add_argument("--no_compile", action="store_true", help="Disable torch.compile")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Import after parsing to show help faster
    import torch
    import lightning as L
    
    from src.utils.config import load_config, merge_configs
    from src.utils.seed import set_all_seeds
    from src.nn.trainer import DiffusionTrainer
    from src.nn.lightning_module import DiffusionLightningModule
    from src.dataloader.datamodule import BaseDataModule
    from src.dataloader.base import Spectrum1DDiffusionDataset, ImageDiffusionDataset
    
    # Load config
    config = load_config(args.config)
    
    # Apply command-line overrides
    overrides = {}
    
    if args.epochs:
        overrides.setdefault('train', {})['epochs'] = args.epochs
    if args.batch_size:
        overrides.setdefault('train', {})['batch_size'] = args.batch_size
    if args.lr:
        overrides.setdefault('opt', {})['lr'] = args.lr
    if args.precision:
        overrides.setdefault('train', {})['precision'] = args.precision
    if args.base_channels:
        overrides.setdefault('model', {})['base_channels'] = args.base_channels
    if args.num_timesteps:
        overrides.setdefault('diffusion', {})['num_timesteps'] = args.num_timesteps
    if args.num_workers:
        overrides.setdefault('train', {})['num_workers'] = args.num_workers
    if args.debug:
        overrides.setdefault('train', {})['debug'] = True
    if args.no_compile:
        overrides.setdefault('train', {})['compile'] = False
    
    config = merge_configs(config, overrides)
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    # Set seeds
    set_all_seeds(args.seed)
    L.seed_everything(args.seed, workers=True)
    
    print("=" * 60)
    print("SpecDiffusion Training")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"GPU: {args.gpu}")
    print(f"Seed: {args.seed}")
    print(f"Model type: {config.get('model', {}).get('type', '1d')}")
    print(f"Epochs: {config.get('train', {}).get('epochs', 100)}")
    print(f"Batch size: {config.get('train', {}).get('batch_size', 64)}")
    print(f"Learning rate: {config.get('opt', {}).get('lr', 1e-4)}")
    print("=" * 60)
    
    # Create logger
    logger = None
    if args.wandb:
        from lightning.pytorch.loggers import WandbLogger
        logger = WandbLogger(
            project=args.project,
            name=args.run_name,
            config=config,
        )
    
    # Select dataset class based on model type
    model_type = config.get('model', {}).get('type', '1d')
    if model_type == '1d':
        dataset_cls = Spectrum1DDiffusionDataset
    else:
        dataset_cls = ImageDiffusionDataset
    
    # Create data module
    data_module = BaseDataModule.from_config(config, dataset_cls=dataset_cls)
    
    # Create model
    lightning_module = DiffusionLightningModule.from_config(config)
    
    # Print model info
    num_params = sum(p.numel() for p in lightning_module.model.parameters())
    print(f"Model parameters: {num_params / 1e6:.2f}M")
    
    # Enable torch.compile if available
    if hasattr(torch, "compile") and config.get('train', {}).get('compile', True):
        try:
            lightning_module.model.model = torch.compile(
                lightning_module.model.model,
                mode="reduce-overhead",
            )
            print("torch.compile enabled")
        except Exception as e:
            print(f"torch.compile not available: {e}")
    
    # Create trainer
    trainer = DiffusionTrainer(config, logger=logger)
    
    # Train
    print("\nStarting training...")
    trainer.fit(lightning_module, datamodule=data_module)
    
    # Test
    if not args.debug:
        print("\nRunning test...")
        trainer.test(lightning_module, datamodule=data_module)
    
    print("\nTraining complete!")
    
    # Print final metrics
    if trainer.callback_metrics:
        print("\nFinal metrics:")
        for key, value in trainer.callback_metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            print(f"  {key}: {value:.6f}")


if __name__ == "__main__":
    main()

