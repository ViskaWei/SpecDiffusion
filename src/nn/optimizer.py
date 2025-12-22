"""Optimizer and learning rate scheduler configuration.

Provides flexible optimizer creation with:
- Multiple optimizer types (Adam, AdamW, SGD, etc.)
- Various LR schedulers (Cosine, OneCycle, Plateau, etc.)
- Optional warmup
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class OptModule:
    """Optimizer and scheduler configuration module.
    
    Usage:
        opt = OptModule.from_config(config['opt'])
        optimizer_config = opt(model)  # Returns optimizer or dict with scheduler
    """
    
    OPTIMIZERS = {
        'adam': torch.optim.Adam,
        'adamw': torch.optim.AdamW,
        'sgd': torch.optim.SGD,
        'rmsprop': torch.optim.RMSprop,
    }
    
    SCHEDULERS = {
        'cosine': torch.optim.lr_scheduler.CosineAnnealingLR,
        'onecycle': torch.optim.lr_scheduler.OneCycleLR,
        'plateau': torch.optim.lr_scheduler.ReduceLROnPlateau,
        'constant': torch.optim.lr_scheduler.ConstantLR,
        'exponential': torch.optim.lr_scheduler.ExponentialLR,
        'step': torch.optim.lr_scheduler.StepLR,
    }
    
    def __init__(
        self,
        lr: float = 1e-4,
        opt_type: str = 'adamw',
        weight_decay: float = 1e-4,
        betas: tuple = (0.9, 0.999),
        lr_scheduler_name: Optional[str] = None,
        warmup_epochs: int = 0,
        monitor_metric: str = 'loss',
        **kwargs,
    ):
        self.lr = float(lr)
        self.opt_type = opt_type.lower()
        self.weight_decay = weight_decay
        self.betas = betas
        self.lr_scheduler_name = lr_scheduler_name
        self.warmup_epochs = warmup_epochs
        self.monitor_metric = monitor_metric
        self.kwargs = kwargs

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        """Create from configuration dictionary."""
        lr = config.get('lr', 1e-4)
        opt_type = config.get('type', 'adamw').lower()
        weight_decay = config.get('weight_decay', 1e-4)
        betas = tuple(config.get('betas', (0.9, 0.999)))
        monitor_metric = config.get('monitor_metric', 'loss')
        
        # LR scheduler
        lr_sch = config.get('lr_sch', None)
        if lr_sch and str(lr_sch).lower() in ('none', 'null', 'false', ''):
            lr_sch = None
        
        # Warmup
        warmup_epochs = config.get('warmup_epochs', 0)
        
        # Collect scheduler-specific kwargs
        kwargs = {}
        
        if lr_sch:
            lr_sch = str(lr_sch).lower()
            
            if 'cosine' in lr_sch:
                kwargs['T_max'] = config.get('T_max', config.get('epochs', 100))
                kwargs['eta_min'] = config.get('eta_min', 0)
            
            elif 'onecycle' in lr_sch:
                kwargs['max_lr'] = lr
                kwargs['steps_per_epoch'] = config.get('steps_per_epoch', 1000)
                kwargs['epochs'] = config.get('epochs', 100)
                kwargs['pct_start'] = config.get('pct_start', 0.3)
            
            elif 'plateau' in lr_sch:
                kwargs['factor'] = config.get('factor', 0.5)
                kwargs['patience'] = config.get('patience', 10)
                kwargs['mode'] = config.get('mode', 'min')
            
            elif 'exponential' in lr_sch:
                kwargs['gamma'] = config.get('gamma', 0.99)
            
            elif 'step' in lr_sch:
                kwargs['step_size'] = config.get('step_size', 30)
                kwargs['gamma'] = config.get('gamma', 0.1)
        
        return cls(
            lr=lr,
            opt_type=opt_type,
            weight_decay=weight_decay,
            betas=betas,
            lr_scheduler_name=lr_sch,
            warmup_epochs=warmup_epochs,
            monitor_metric=monitor_metric,
            **kwargs,
        )

    def __call__(self, model: nn.Module) -> Dict[str, Any]:
        """Create optimizer and optionally scheduler.
        
        Args:
            model: PyTorch model
            
        Returns:
            Optimizer or dict with optimizer and lr_scheduler
        """
        # Create optimizer
        opt_cls = self.OPTIMIZERS.get(self.opt_type, torch.optim.AdamW)
        
        opt_kwargs = {'lr': self.lr, 'weight_decay': self.weight_decay}
        if self.opt_type in ('adam', 'adamw'):
            opt_kwargs['betas'] = self.betas
        
        optimizer = opt_cls(model.parameters(), **opt_kwargs)
        
        # No scheduler
        if self.lr_scheduler_name is None:
            return optimizer
        
        # Create scheduler
        scheduler = self._create_scheduler(optimizer)
        
        # Create warmup if needed
        if self.warmup_epochs > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,
                total_iters=self.warmup_epochs,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, scheduler],
                milestones=[self.warmup_epochs],
            )
            print(f"[Optimizer] Using {self.warmup_epochs} warmup epochs")
        
        # Configure scheduler for Lightning
        scheduler_config = {
            "scheduler": scheduler,
            "monitor": f"val_{self.monitor_metric}",
        }
        
        if 'plateau' in (self.lr_scheduler_name or ''):
            scheduler_config["reduce_on_plateau"] = True
            scheduler_config["strict"] = False
        elif 'onecycle' in (self.lr_scheduler_name or ''):
            scheduler_config["interval"] = "step"
            scheduler_config["frequency"] = 1
        else:
            scheduler_config["interval"] = "epoch"
            scheduler_config["frequency"] = 1
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_config,
        }

    def _create_scheduler(self, optimizer: torch.optim.Optimizer):
        """Create learning rate scheduler."""
        sch_name = self.lr_scheduler_name.lower()
        
        if 'cosine' in sch_name:
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.kwargs.get('T_max', 100),
                eta_min=self.kwargs.get('eta_min', 0),
            )
        
        elif 'onecycle' in sch_name:
            return torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.kwargs.get('max_lr', self.lr),
                steps_per_epoch=self.kwargs.get('steps_per_epoch', 1000),
                epochs=self.kwargs.get('epochs', 100),
                pct_start=self.kwargs.get('pct_start', 0.3),
            )
        
        elif 'plateau' in sch_name:
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=self.kwargs.get('factor', 0.5),
                patience=self.kwargs.get('patience', 10),
                mode=self.kwargs.get('mode', 'min'),
            )
        
        elif 'exponential' in sch_name:
            return torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=self.kwargs.get('gamma', 0.99),
            )
        
        elif 'step' in sch_name:
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.kwargs.get('step_size', 30),
                gamma=self.kwargs.get('gamma', 0.1),
            )
        
        else:
            raise ValueError(f"Unknown scheduler: {sch_name}")


__all__ = ["OptModule"]

