"""
Conditional Denoising Diffusion Probabilistic Models (DDPM) for Supervised Spectrum Denoising

This implements conditional DDPM where the model learns to denoise spectra
conditioned on noisy observations. Key difference from standard DDPM:
- Training: x_0 is clean spectrum, condition is noisy observation
- Inference: start from noise, iteratively denoise conditioned on noisy observation

Experiment ID: SD-20251203-diff-supervised-01
MVP Source: MVP-1.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable, Tuple
from tqdm import tqdm

from .utils import linear_beta_schedule, cosine_beta_schedule, extract


class ConditionalGaussianDiffusion(nn.Module):
    """
    Conditional Gaussian Diffusion process for supervised denoising.
    
    Args:
        model: Conditional neural network (e.g., ConditionalUNet1D)
        timesteps: Number of diffusion steps (T)
        beta_schedule: Type of beta schedule ('linear' or 'cosine')
        beta_start: Starting beta value (for linear schedule)
        beta_end: Ending beta value (for linear schedule)
        loss_type: Loss function type ('l1', 'l2', 'huber')
    """
    
    def __init__(
        self,
        model: nn.Module,
        timesteps: int = 1000,
        beta_schedule: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        loss_type: str = "l2",
    ):
        super().__init__()
        
        self.model = model
        self.timesteps = timesteps
        self.loss_type = loss_type
        
        # Setup beta schedule
        if beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        # Precompute diffusion constants
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Register as buffers (not parameters)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        # Calculations for diffusion q(x_t | x_0)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1))
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', 
                           torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                           betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                           (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))
    
    def q_sample(
        self, 
        x_start: torch.Tensor, 
        t: torch.Tensor, 
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward diffusion process: sample x_t given x_0.
        
        q(x_t | x_0) = N(x_t; √ᾱ_t x_0, (1-ᾱ_t) I)
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alpha_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise
    
    def predict_start_from_noise(
        self, 
        x_t: torch.Tensor, 
        t: torch.Tensor, 
        noise: torch.Tensor
    ) -> torch.Tensor:
        """Predict x_0 from x_t and predicted noise."""
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
    
    def q_posterior_mean_variance(
        self, 
        x_start: torch.Tensor, 
        x_t: torch.Tensor, 
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute posterior q(x_{t-1} | x_t, x_0)."""
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        
        return posterior_mean, posterior_variance, posterior_log_variance
    
    def p_mean_variance(
        self, 
        x_t: torch.Tensor, 
        t: torch.Tensor,
        cond: torch.Tensor,
        clip_denoised: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute mean and variance for reverse process p(x_{t-1} | x_t, cond)."""
        # Predict noise conditioned on noisy observation
        pred_noise = self.model(x_t, t, cond)
        
        # Predict x_0
        x_start = self.predict_start_from_noise(x_t, t, pred_noise)
        
        if clip_denoised:
            x_start = torch.clamp(x_start, -1.0, 1.0)
        
        # Get posterior parameters
        model_mean, posterior_variance, posterior_log_variance = \
            self.q_posterior_mean_variance(x_start, x_t, t)
        
        return model_mean, posterior_variance, posterior_log_variance
    
    @torch.no_grad()
    def p_sample(
        self, 
        x_t: torch.Tensor, 
        t: int,
        cond: torch.Tensor,
        clip_denoised: bool = True
    ) -> torch.Tensor:
        """Single reverse diffusion step: sample x_{t-1} from x_t conditioned on cond."""
        batch_size = x_t.shape[0]
        device = x_t.device
        
        # Create batch of timesteps
        t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
        
        # Get mean and variance
        model_mean, _, posterior_log_variance = self.p_mean_variance(
            x_t, t_batch, cond, clip_denoised=clip_denoised
        )
        
        # Sample
        noise = torch.randn_like(x_t) if t > 0 else 0
        
        return model_mean + (0.5 * posterior_log_variance).exp() * noise
    
    @torch.no_grad()
    def p_sample_loop(
        self, 
        cond: torch.Tensor,
        progress: bool = True,
        return_intermediates: bool = False,
        intermediate_steps: Optional[list] = None,
    ):
        """
        Full reverse diffusion sampling loop conditioned on noisy observation.
        
        Args:
            cond: (B, C, L) conditioning signal (noisy observation)
            progress: Whether to show progress bar
            return_intermediates: Whether to return intermediate states
            intermediate_steps: Timesteps at which to save intermediates
        """
        device = cond.device
        shape = cond.shape
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        intermediates = []
        if intermediate_steps is None:
            intermediate_steps = [0, 250, 500, 750, 999]
        
        # Reverse diffusion
        timesteps = list(reversed(range(self.timesteps)))
        if progress:
            timesteps = tqdm(timesteps, desc="Denoising", leave=False)
        
        for t in timesteps:
            x = self.p_sample(x, t, cond)
            
            if return_intermediates and t in intermediate_steps:
                intermediates.append((t, x.clone()))
        
        if return_intermediates:
            return x, intermediates
        return x
    
    def training_loss(
        self, 
        x_start: torch.Tensor,
        cond: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute training loss for a batch.
        
        Args:
            x_start: (B, C, L) clean spectra
            cond: (B, C, L) noisy observations (conditioning signal)
            t: Optional timesteps
            noise: Optional noise to add
        """
        batch_size = x_start.shape[0]
        device = x_start.device
        
        # Sample timesteps uniformly
        if t is None:
            t = torch.randint(0, self.timesteps, (batch_size,), device=device, dtype=torch.long)
        
        # Sample noise
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # Get noisy samples (diffusion noise, not observation noise)
        x_t = self.q_sample(x_start, t, noise)
        
        # Predict noise conditioned on noisy observation
        pred_noise = self.model(x_t, t, cond)
        
        # Compute loss
        if self.loss_type == "l1":
            loss = F.l1_loss(pred_noise, noise)
        elif self.loss_type == "l2":
            loss = F.mse_loss(pred_noise, noise)
        elif self.loss_type == "huber":
            loss = F.smooth_l1_loss(pred_noise, noise)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return loss
    
    def forward(self, x_start: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Forward pass returns training loss."""
        return self.training_loss(x_start, cond)


class ConditionalDDPM:
    """
    High-level Conditional DDPM wrapper for training and sampling.
    """
    
    def __init__(
        self,
        model: nn.Module,
        timesteps: int = 1000,
        beta_schedule: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        loss_type: str = "l2",
        device: str = "cuda",
    ):
        self.device = device
        self.model = model.to(device)
        
        self.diffusion = ConditionalGaussianDiffusion(
            model=model,
            timesteps=timesteps,
            beta_schedule=beta_schedule,
            beta_start=beta_start,
            beta_end=beta_end,
            loss_type=loss_type,
        ).to(device)
    
    def training_step(
        self, 
        x_0: torch.Tensor, 
        cond: torch.Tensor
    ) -> torch.Tensor:
        """Single training step."""
        x_0 = x_0.to(self.device)
        cond = cond.to(self.device)
        return self.diffusion.training_loss(x_0, cond)
    
    @torch.no_grad()
    def denoise(
        self, 
        noisy_obs: torch.Tensor,
        progress: bool = True,
    ) -> torch.Tensor:
        """
        Denoise a noisy observation using the trained model.
        
        Args:
            noisy_obs: (B, C, L) or (B, L) noisy observation
            progress: Whether to show progress bar
            
        Returns:
            (B, C, L) denoised spectrum
        """
        self.model.eval()
        
        # Ensure correct shape
        if noisy_obs.dim() == 2:
            noisy_obs = noisy_obs.unsqueeze(1)
        
        noisy_obs = noisy_obs.to(self.device)
        return self.diffusion.p_sample_loop(noisy_obs, progress=progress)
    
    @torch.no_grad()
    def denoise_with_intermediates(
        self,
        noisy_obs: torch.Tensor,
        intermediate_steps: Optional[list] = None,
    ):
        """Denoise and return intermediate steps for visualization."""
        self.model.eval()
        
        if noisy_obs.dim() == 2:
            noisy_obs = noisy_obs.unsqueeze(1)
        
        noisy_obs = noisy_obs.to(self.device)
        return self.diffusion.p_sample_loop(
            noisy_obs, 
            progress=True,
            return_intermediates=True,
            intermediate_steps=intermediate_steps,
        )


if __name__ == "__main__":
    from .conditional_unet_1d import ConditionalUNet1D
    
    # Quick test
    print("Testing ConditionalDDPM...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = ConditionalUNet1D(
        in_channels=1,
        cond_channels=1,
        out_channels=1,
        base_channels=32,
        channel_mults=(1, 2, 4),
        num_res_blocks=1,
        attention_resolutions=(2,),
        dropout=0.1,
        time_emb_dim=128,
    )
    
    ddpm = ConditionalDDPM(
        model=model,
        timesteps=100,  # Small for testing
        beta_schedule="linear",
        device=device,
    )
    
    # Test training step
    batch_size = 4
    length = 512
    
    x_0 = torch.randn(batch_size, 1, length)  # Clean spectra
    cond = x_0 + 0.1 * torch.randn_like(x_0)  # Noisy observations
    
    loss = ddpm.training_step(x_0, cond)
    print(f"Training loss: {loss.item():.6f}")
    
    # Test denoising
    print("Testing denoising...")
    denoised = ddpm.denoise(cond, progress=True)
    print(f"Denoised shape: {denoised.shape}")
    print("✓ ConditionalDDPM test successful!")

