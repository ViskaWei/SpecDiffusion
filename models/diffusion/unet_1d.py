"""
1D U-Net for Diffusion Models on Stellar Spectra

Simplified architecture following standard U-Net pattern.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from .utils import SinusoidalPositionEmbeddings


class GroupNorm32(nn.GroupNorm):
    """GroupNorm with 32 groups, commonly used in diffusion models."""
    
    def __init__(self, num_channels: int):
        num_groups = min(32, num_channels)
        while num_channels % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        super().__init__(num_groups, num_channels)


class Swish(nn.Module):
    """Swish activation: x * sigmoid(x), also known as SiLU."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    """Time embedding: Sinusoidal encoding → MLP."""
    
    def __init__(self, time_channels: int, emb_channels: int):
        super().__init__()
        self.sinusoidal = SinusoidalPositionEmbeddings(time_channels)
        self.mlp = nn.Sequential(
            nn.Linear(time_channels, emb_channels),
            Swish(),
            nn.Linear(emb_channels, emb_channels),
        )
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = self.sinusoidal(t)
        return self.mlp(emb)


class ResBlock1D(nn.Module):
    """1D Residual block with time embedding injection."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.norm1 = GroupNorm32(in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.time_mlp = nn.Sequential(
            Swish(),
            nn.Linear(time_emb_dim, out_channels),
        )
        
        self.norm2 = GroupNorm32(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        
        if in_channels != out_channels:
            self.skip_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_conv = nn.Identity()
        
        self.act = Swish()
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        
        h = h + self.time_mlp(t_emb)[:, :, None]
        
        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.skip_conv(x)


class Attention1D(nn.Module):
    """1D Self-Attention module."""
    
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        self.norm = GroupNorm32(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv1d(channels, channels, kernel_size=1)
        
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        
        q = q.view(B, self.num_heads, self.head_dim, L).permute(0, 1, 3, 2)
        k = k.view(B, self.num_heads, self.head_dim, L).permute(0, 1, 3, 2)
        v = v.view(B, self.num_heads, self.head_dim, L).permute(0, 1, 3, 2)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 3, 2).contiguous().view(B, C, L)
        
        return x + self.proj(out)


class Downsample1D(nn.Module):
    """1D Downsampling using strided convolution."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1D(nn.Module):
    """1D Upsampling using nearest interpolation + convolution."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class UNet1D(nn.Module):
    """
    Simple 1D U-Net for Diffusion Models on Stellar Spectra.
    
    Args:
        in_channels: Number of input channels (1 for single-channel spectrum)
        out_channels: Number of output channels (1 for noise prediction)
        base_channels: Base channel count (doubled at each downsample)
        channel_mults: Channel multipliers for each stage
        num_res_blocks: Number of residual blocks per stage
        attention_resolutions: Stages at which to add attention (0-indexed)
        dropout: Dropout rate
        time_emb_dim: Time embedding dimension
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (2, 3),
        dropout: float = 0.1,
        time_emb_dim: int = 256,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.channel_mults = channel_mults
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        
        num_levels = len(channel_mults)
        
        # Time embedding
        self.time_embedding = TimeEmbedding(base_channels, time_emb_dim)
        
        # Initial convolution
        self.conv_in = nn.Conv1d(in_channels, base_channels, kernel_size=3, padding=1)
        
        # Build encoder (downsampling) path
        self.encoder_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        
        ch = base_channels
        self.skip_channels = []  # Track channels for skip connections
        
        for level in range(num_levels):
            out_ch = base_channels * channel_mults[level]
            add_attn = level in attention_resolutions
            
            # ResBlocks for this level
            blocks = nn.ModuleList()
            for i in range(num_res_blocks):
                blocks.append(ResBlock1D(ch, out_ch, time_emb_dim, dropout))
                ch = out_ch
                if add_attn:
                    blocks.append(Attention1D(ch))
            
            self.encoder_blocks.append(blocks)
            self.skip_channels.append(ch)
            
            # Downsample (except at last level)
            if level < num_levels - 1:
                self.downsamplers.append(Downsample1D(ch))
            else:
                self.downsamplers.append(nn.Identity())
        
        # Middle block
        self.middle_res1 = ResBlock1D(ch, ch, time_emb_dim, dropout)
        self.middle_attn = Attention1D(ch)
        self.middle_res2 = ResBlock1D(ch, ch, time_emb_dim, dropout)
        
        # Build decoder (upsampling) path
        self.decoder_blocks = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        
        for level in reversed(range(num_levels)):
            out_ch = base_channels * channel_mults[level]
            skip_ch = self.skip_channels[level]
            add_attn = level in attention_resolutions
            
            # ResBlocks for this level (including one for skip connection)
            blocks = nn.ModuleList()
            for i in range(num_res_blocks + 1):
                # First block takes concatenated channels (h + skip)
                in_ch = ch + skip_ch if i == 0 else out_ch
                blocks.append(ResBlock1D(in_ch, out_ch, time_emb_dim, dropout))
                if i == 0:
                    ch = out_ch
                if add_attn and i < num_res_blocks:
                    blocks.append(Attention1D(out_ch))
            
            self.decoder_blocks.append(blocks)
            ch = out_ch
            
            # Upsample (except at first level)
            if level > 0:
                self.upsamplers.append(Upsample1D(ch))
            else:
                self.upsamplers.append(nn.Identity())
        
        # Output
        self.norm_out = GroupNorm32(ch)
        self.act_out = Swish()
        self.conv_out = nn.Conv1d(ch, out_channels, kernel_size=3, padding=1)
        
        # Initialize output conv to zero for residual learning
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of U-Net.
        
        Args:
            x: (B, C, L) noisy input spectrum
            t: (B,) diffusion timesteps
            
        Returns:
            (B, C, L) predicted noise
        """
        # Time embedding
        t_emb = self.time_embedding(t)
        
        # Initial conv
        h = self.conv_in(x)
        
        # Encoder path with skip connections
        skips = []
        
        for blocks, downsample in zip(self.encoder_blocks, self.downsamplers):
            for block in blocks:
                if isinstance(block, ResBlock1D):
                    h = block(h, t_emb)
                else:
                    h = block(h)
            skips.append(h)
            h = downsample(h)
        
        # Middle
        h = self.middle_res1(h, t_emb)
        h = self.middle_attn(h)
        h = self.middle_res2(h, t_emb)
        
        # Decoder path
        for blocks, upsample in zip(self.decoder_blocks, self.upsamplers):
            skip = skips.pop()
            
            # Handle size mismatch
            if h.shape[-1] != skip.shape[-1]:
                h = F.interpolate(h, size=skip.shape[-1], mode='nearest')
            
            # Concatenate with skip
            h = torch.cat([h, skip], dim=1)
            
            # Process blocks
            for block in blocks:
                if isinstance(block, ResBlock1D):
                    h = block(h, t_emb)
                else:
                    h = block(h)
            
            h = upsample(h)
        
        # Handle final size matching with input
        if h.shape[-1] != x.shape[-1]:
            h = F.interpolate(h, size=x.shape[-1], mode='nearest')
        
        # Output
        h = self.norm_out(h)
        h = self.act_out(h)
        h = self.conv_out(h)
        
        return h


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick test
    model = UNet1D(
        in_channels=1,
        out_channels=1,
        base_channels=64,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=2,
        attention_resolutions=(2, 3),
        dropout=0.1,
        time_emb_dim=256,
    )
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Test forward pass
    x = torch.randn(4, 1, 4096)  # Batch of 4, 1 channel, 4096 wavelength points
    t = torch.randint(0, 1000, (4,))
    
    with torch.no_grad():
        out = model(x, t)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == x.shape, "Output shape should match input shape"
    print("✓ Forward pass successful!")

