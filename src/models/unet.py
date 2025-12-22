"""UNet architectures for diffusion models.

Implements:
- UNet1D: For 1D signals (spectra, audio, time series)
- UNet2D: For 2D data (images)
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal position embedding for timesteps."""
    
    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: [B] tensor of timestep values
            
        Returns:
            [B, dim] embedding tensor
        """
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half_dim, device=timesteps.device) / half_dim
        )
        args = timesteps[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        # Handle odd dimensions
        if self.dim % 2:
            embedding = F.pad(embedding, (0, 1))
        
        return embedding


class TimeEmbedding(nn.Module):
    """Timestep embedding with projection layers."""
    
    def __init__(self, dim: int, time_embed_dim: int = None):
        super().__init__()
        time_embed_dim = time_embed_dim or dim * 4
        
        self.sinusoidal = SinusoidalPositionEmbedding(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        x = self.sinusoidal(timesteps)
        return self.mlp(x)


class ConvBlock1D(nn.Module):
    """Convolutional block for 1D UNet."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        time_embed_dim: Optional[int] = None,
        groups: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Time embedding projection
        if time_embed_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_embed_dim, out_channels),
            )
        else:
            self.time_mlp = None
        
        # Residual connection
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        
        if self.time_mlp is not None and time_emb is not None:
            h = h + self.time_mlp(time_emb)[:, :, None]
        
        h = self.conv2(self.dropout(self.act(self.norm2(h))))
        
        return h + self.skip(x)


class ConvBlock2D(nn.Module):
    """Convolutional block for 2D UNet."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        time_embed_dim: Optional[int] = None,
        groups: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        if time_embed_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_embed_dim, out_channels),
            )
        else:
            self.time_mlp = None
        
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        
        if self.time_mlp is not None and time_emb is not None:
            h = h + self.time_mlp(time_emb)[:, :, None, None]
        
        h = self.conv2(self.dropout(self.act(self.norm2(h))))
        
        return h + self.skip(x)


class Attention1D(nn.Module):
    """Self-attention for 1D data."""
    
    def __init__(self, channels: int, num_heads: int = 4, head_dim: int = 32):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim
        
        self.norm = nn.GroupNorm(8, channels)
        self.to_qkv = nn.Conv1d(channels, inner_dim * 3, 1)
        self.to_out = nn.Conv1d(inner_dim, channels, 1)
        self.scale = head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, l = x.shape
        
        h = self.norm(x)
        qkv = self.to_qkv(h).chunk(3, dim=1)
        q, k, v = [rearrange(t, 'b (h d) l -> b h l d', h=self.num_heads) for t in qkv]
        
        attn = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h l d -> b (h d) l')
        
        return x + self.to_out(out)


class Attention2D(nn.Module):
    """Self-attention for 2D data."""
    
    def __init__(self, channels: int, num_heads: int = 4, head_dim: int = 32):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim
        
        self.norm = nn.GroupNorm(8, channels)
        self.to_qkv = nn.Conv2d(channels, inner_dim * 3, 1)
        self.to_out = nn.Conv2d(inner_dim, channels, 1)
        self.scale = head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        
        norm_x = self.norm(x)
        qkv = self.to_qkv(norm_x)
        qkv = rearrange(qkv, 'b (n h d) x y -> n b h (x y) d', n=3, h=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        
        return x + self.to_out(out)


class Downsample1D(nn.Module):
    """Downsample 1D feature map by factor of 2."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1D(nn.Module):
    """Upsample 1D feature map by factor of 2."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(channels, channels, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Downsample2D(nn.Module):
    """Downsample 2D feature map by factor of 2."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample2D(nn.Module):
    """Upsample 2D feature map by factor of 2."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet1D(nn.Module):
    """1D UNet for diffusion models (spectra, audio, time series).
    
    Architecture:
    - Encoder: Conv blocks with downsampling
    - Middle: Conv blocks with attention
    - Decoder: Conv blocks with upsampling and skip connections
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (4,),
        num_heads: int = 4,
        dropout: float = 0.0,
        time_embed_dim: Optional[int] = None,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Time embedding
        time_embed_dim = time_embed_dim or base_channels * 4
        self.time_embed = TimeEmbedding(base_channels, time_embed_dim)
        
        # Initial projection
        self.init_conv = nn.Conv1d(in_channels, base_channels, 3, padding=1)
        
        # Encoder
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        
        channels = [base_channels]
        ch = base_channels
        
        for level, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            
            for _ in range(num_res_blocks):
                block = ConvBlock1D(ch, out_ch, time_embed_dim=time_embed_dim, dropout=dropout)
                self.down_blocks.append(block)
                ch = out_ch
                channels.append(ch)
            
            # Add attention at specified resolutions
            if level in attention_resolutions:
                self.down_blocks.append(Attention1D(ch, num_heads=num_heads))
                channels.append(ch)
            
            # Downsample (except last level)
            if level < len(channel_mults) - 1:
                self.down_samples.append(Downsample1D(ch))
                channels.append(ch)
        
        # Middle
        self.mid_block1 = ConvBlock1D(ch, ch, time_embed_dim=time_embed_dim, dropout=dropout)
        self.mid_attn = Attention1D(ch, num_heads=num_heads)
        self.mid_block2 = ConvBlock1D(ch, ch, time_embed_dim=time_embed_dim, dropout=dropout)
        
        # Decoder
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        
        for level, mult in enumerate(reversed(channel_mults)):
            out_ch = base_channels * mult
            
            for i in range(num_res_blocks + 1):
                skip_ch = channels.pop()
                block = ConvBlock1D(ch + skip_ch, out_ch, time_embed_dim=time_embed_dim, dropout=dropout)
                self.up_blocks.append(block)
                ch = out_ch
            
            # Add attention at specified resolutions
            rev_level = len(channel_mults) - 1 - level
            if rev_level in attention_resolutions:
                self.up_blocks.append(Attention1D(ch, num_heads=num_heads))
            
            # Upsample (except first level)
            if level < len(channel_mults) - 1:
                self.up_samples.append(Upsample1D(ch))
        
        # Output
        self.out_norm = nn.GroupNorm(8, base_channels)
        self.out_conv = nn.Conv1d(base_channels, out_channels, 3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, C, L] noisy input
            timesteps: [B] timestep values
            cond: Optional conditioning tensor
            
        Returns:
            [B, C, L] predicted noise or sample
        """
        # Time embedding
        t_emb = self.time_embed(timesteps)
        
        # Initial convolution
        h = self.init_conv(x)
        
        # Encoder
        skips = [h]
        down_idx = 0
        
        for block in self.down_blocks:
            if isinstance(block, ConvBlock1D):
                h = block(h, t_emb)
            else:  # Attention
                h = block(h)
            skips.append(h)
            
            # Check for downsample
            if down_idx < len(self.down_samples):
                # Check if this is the end of a level
                h = self.down_samples[down_idx](h)
                down_idx += 1
                skips.append(h)
        
        # Middle
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)
        
        # Decoder
        up_idx = 0
        
        for block in self.up_blocks:
            if isinstance(block, ConvBlock1D):
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = block(h, t_emb)
            else:  # Attention
                h = block(h)
            
            # Check for upsample
            if up_idx < len(self.up_samples) and isinstance(block, ConvBlock1D):
                h = self.up_samples[up_idx](h)
                up_idx += 1
        
        # Output
        h = self.out_conv(F.silu(self.out_norm(h)))
        
        return h


class UNet2D(nn.Module):
    """2D UNet for diffusion models (images).
    
    Similar architecture to UNet1D but with 2D operations.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (2, 3),
        num_heads: int = 4,
        dropout: float = 0.0,
        time_embed_dim: Optional[int] = None,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Time embedding
        time_embed_dim = time_embed_dim or base_channels * 4
        self.time_embed = TimeEmbedding(base_channels, time_embed_dim)
        
        # Initial projection
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # Encoder
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        
        channels = [base_channels]
        ch = base_channels
        
        for level, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            
            for _ in range(num_res_blocks):
                block = ConvBlock2D(ch, out_ch, time_embed_dim=time_embed_dim, dropout=dropout)
                self.down_blocks.append(block)
                ch = out_ch
                channels.append(ch)
            
            if level in attention_resolutions:
                self.down_blocks.append(Attention2D(ch, num_heads=num_heads))
                channels.append(ch)
            
            if level < len(channel_mults) - 1:
                self.down_samples.append(Downsample2D(ch))
                channels.append(ch)
        
        # Middle
        self.mid_block1 = ConvBlock2D(ch, ch, time_embed_dim=time_embed_dim, dropout=dropout)
        self.mid_attn = Attention2D(ch, num_heads=num_heads)
        self.mid_block2 = ConvBlock2D(ch, ch, time_embed_dim=time_embed_dim, dropout=dropout)
        
        # Decoder
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        
        for level, mult in enumerate(reversed(channel_mults)):
            out_ch = base_channels * mult
            
            for i in range(num_res_blocks + 1):
                skip_ch = channels.pop()
                block = ConvBlock2D(ch + skip_ch, out_ch, time_embed_dim=time_embed_dim, dropout=dropout)
                self.up_blocks.append(block)
                ch = out_ch
            
            rev_level = len(channel_mults) - 1 - level
            if rev_level in attention_resolutions:
                self.up_blocks.append(Attention2D(ch, num_heads=num_heads))
            
            if level < len(channel_mults) - 1:
                self.up_samples.append(Upsample2D(ch))
        
        # Output
        self.out_norm = nn.GroupNorm(8, base_channels)
        self.out_conv = nn.Conv2d(base_channels, out_channels, 3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] noisy input
            timesteps: [B] timestep values
            cond: Optional conditioning tensor
            
        Returns:
            [B, C, H, W] predicted noise or sample
        """
        t_emb = self.time_embed(timesteps)
        h = self.init_conv(x)
        
        # Encoder
        skips = [h]
        down_idx = 0
        
        for block in self.down_blocks:
            if isinstance(block, ConvBlock2D):
                h = block(h, t_emb)
            else:
                h = block(h)
            skips.append(h)
            
            if down_idx < len(self.down_samples):
                h = self.down_samples[down_idx](h)
                down_idx += 1
                skips.append(h)
        
        # Middle
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)
        
        # Decoder
        up_idx = 0
        
        for block in self.up_blocks:
            if isinstance(block, ConvBlock2D):
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = block(h, t_emb)
            else:
                h = block(h)
            
            if up_idx < len(self.up_samples) and isinstance(block, ConvBlock2D):
                h = self.up_samples[up_idx](h)
                up_idx += 1
        
        h = self.out_conv(F.silu(self.out_norm(h)))
        
        return h


__all__ = [
    "SinusoidalPositionEmbedding",
    "TimeEmbedding",
    "UNet1D",
    "UNet2D",
]

