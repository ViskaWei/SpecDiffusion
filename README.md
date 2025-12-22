# SpecDiffusion

<p align="center">
  <strong>Diffusion Models for 1D Stellar Spectra Denoising and Parameter Inference</strong>
</p>

<p align="center">
  A research framework for applying diffusion-based generative models to astronomical spectral analysis,<br>
  with a focus on denoising low-SNR stellar spectra and enabling robust parameter inference.
</p>

---

## Overview

**SpecDiffusion** explores the application of Denoising Diffusion Probabilistic Models (DDPM) to 1D stellar spectra. The project aims to:

1. **Denoise** low signal-to-noise ratio (SNR) spectra from astronomical surveys
2. **Preserve scientific validity** - avoiding "hallucination" artifacts that could bias stellar parameter estimates
3. **Provide uncertainty quantification** through posterior sampling

### Research Questions

> *Can diffusion models effectively denoise stellar spectra while maintaining scientific credibility for parameter inference?*

Key investigations:
- Does supervised DDPM denoising introduce parameter biases?
- Can Diffusion Posterior Sampling (DPS) suppress hallucination artifacts?
- Are the resulting uncertainty estimates well-calibrated?

---

## Key Features

- **1D Diffusion Models**: Specialized U-Net architectures for spectral data (4096+ wavelength points)
- **Multiple Denoising Approaches**:
  - Standard DDPM (ε-prediction / x₀-prediction)
  - Bounded Noise Denoiser (MVP-0.5) - doesn't require going to pure noise
  - Residual Denoiser with weighted MAE (MVP-0.6) - identity-preserving at low noise
- **PyTorch 2.x Optimizations**: `torch.compile`, mixed precision training
- **Flexible Training**: YAML-based configuration with CLI overrides
- **Comprehensive Evaluation**: MSE, wMAE, flux distribution analysis

---

## Installation

### Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/username/SpecDiffusion.git
cd SpecDiffusion

# Install dependencies
pip install -r requirements.txt

# Set data path (for BOSZ dataset)
export DATA_ROOT=/path/to/your/spectral/data
```

### Dependencies

```
torch>=2.0.0
lightning>=2.0.0
numpy>=1.21.0
h5py>=3.0.0
einops>=0.6.0
omegaconf>=2.3.0
matplotlib>=3.5.0
tqdm>=4.60.0
```

---

## Quick Start

### Training a Diffusion Model

```bash
# Train baseline DDPM (1000 steps, ε-prediction)
python scripts/train_diffusion.py --config configs/diffusion/baseline.yaml

# Quick test with fewer epochs
python scripts/train_diffusion.py --config configs/diffusion/baseline.yaml --epochs 10

# Train bounded noise denoiser (recommended for practical use)
python scripts/train_bounded_denoiser.py --epochs 50 --target x0
```

### Using the Lightning-based Trainer

```bash
# Train with full framework
python train.py --config src/configs/spectrum_ddpm.yaml

# With W&B logging
python train.py --config src/configs/spectrum_ddpm.yaml --wandb --project SpecDiffusion

# Debug mode
python train.py --config src/configs/spectrum_ddpm.yaml --debug
```

---

## Project Structure

```
SpecDiffusion/
├── configs/                    # Experiment configurations
│   ├── diffusion/
│   │   ├── baseline.yaml       # Standard DDPM config
│   │   └── bounded_noise.yaml  # Bounded denoiser config
│   └── supervised.yaml         # Supervised denoising config
│
├── models/                     # Core model implementations
│   └── diffusion/
│       ├── ddpm.py             # Gaussian Diffusion (DDPM)
│       ├── unet_1d.py          # 1D U-Net architecture
│       ├── conditional_unet_1d.py  # U-Net with σ conditioning
│       ├── conditional_ddpm.py # Conditional diffusion
│       └── utils.py            # Beta schedules, EMA, normalization
│
├── scripts/                    # Training scripts
│   ├── train_diffusion.py      # Standard DDPM training
│   ├── train_bounded_denoiser.py   # MVP-0.5: Bounded noise
│   ├── train_wmae_residual_denoiser.py  # MVP-0.6: wMAE + residual
│   ├── train_supervised.py     # Supervised conditional DDPM
│   └── eval_*.py               # Evaluation scripts
│
├── src/                        # Extended framework (Lightning-based)
│   ├── dataloader/             # Data loading utilities
│   ├── models/                 # Alternative model implementations
│   ├── nn/                     # Lightning modules & trainers
│   └── utils/                  # Config, seeds, hardware detection
│
├── train.py                    # Main training entry point
├── sample.py                   # Sampling/inference script
└── requirements.txt
```

---

## Model Architectures

### 1D U-Net

The core architecture is a 1D U-Net adapted for spectral data:

```
Input (1, 4096) → Encoder → Middle Block → Decoder → Output (1, 4096)
       ↓              ↓           ↓            ↓
   Time Embed    Skip Connections    Attention (low-res only)
```

**Key Design Choices**:
- Base channels: 32-64 (memory efficient for long sequences)
- Channel multipliers: [1, 2, 4, 8] → 32→64→128→256
- Attention: Only at lowest resolution (level 3) to save memory
- Sinusoidal time embeddings with MLP projection

### Conditional U-Net

For the bounded noise denoiser, the U-Net accepts additional conditioning:

```
Inputs:
  - x_t: Noisy spectrum (1, L)
  - σ: Per-pixel error vector (1, L)  
  - t: Timestep / noise level embedding

Output:
  - x₀ or ε prediction
```

---

## Experiment Results

### MVP-0.5: Bounded Noise Denoiser ✅

**Noise Model**: y = x₀ + λ · σ ⊙ ε, where λ ∈ [0.1, 0.5]

| λ (Noise Level) | MSE (noisy) | MSE (denoised) | Improvement |
|-----------------|-------------|----------------|-------------|
| 0.5 | 0.0846 | 0.0342 | **59.5%** |
| 0.4 | 0.0538 | 0.0384 | 28.5% |
| 0.3 | 0.0303 | 0.0261 | 13.9% |

**Key Finding**: Effective for high-noise scenarios (SNR < 10), but may over-smooth at low noise levels.

### MVP-0.6: wMAE Residual Denoiser ✅

**Architecture**: x̂₀ = y + s · g_θ(y, s, σ)

| Noise Level (s) | wMAE (noisy) | wMAE (denoised) | Improvement |
|-----------------|--------------|-----------------|-------------|
| 0.00 | 0.0000 | 0.0000 | **Identity** ✓ |
| 0.05 | 0.0399 | 0.0320 | **19.8%** |
| 0.10 | 0.0798 | 0.0527 | **33.9%** |
| 0.20 | 0.1596 | 0.0854 | **46.5%** |

**Key Findings**:
- Residual structure guarantees identity at s=0 (critical for preservation)
- wMAE loss protects high-SNR regions (1/σ weighting)
- Consistent improvement across all noise levels

### Lessons Learned

| Principle | Recommendation | Evidence |
|-----------|---------------|----------|
| **Validate Sampling** | Loss convergence ≠ sampling success; always visualize | MVP-0.0 failure |
| **Interface Consistency** | ε-prediction vs x₀-prediction must align in train/sample | MVP-0.0 failure |
| **Residual Structure** | Use x̂₀ = y + s · g_θ for controllable denoising | MVP-0.6 success |
| **Weighted Loss** | wMAE (1/σ) > MSE for spectral denoising | MVP-0.6 |

---

## Configuration

### YAML Configuration

```yaml
# configs/diffusion/baseline.yaml

experiment:
  id: "SPEC-DIFF-baseline-01"
  name: "1D U-Net DDPM Baseline"

data:
  file_path: "$DATA_ROOT/train_100k/dataset.h5"
  train_size: 10000
  snr_threshold: 50
  normalization: "minmax"

model:
  type: "UNet1D"
  base_channels: 32
  channel_mults: [1, 2, 4, 8]
  num_res_blocks: 2
  attention_resolutions: [3]

diffusion:
  timesteps: 1000
  beta_schedule: "linear"
  beta_start: 0.0001
  beta_end: 0.02
  prediction_type: "epsilon"

training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.0001
  warmup_epochs: 5
  ema_decay: 0.9999
```

### Command-Line Overrides

```bash
python scripts/train_diffusion.py \
    --config configs/diffusion/baseline.yaml \
    --epochs 100 \
    --batch-size 64 \
    --device cuda
```

---

## API Usage

### Training from Python

```python
from models.diffusion.unet_1d import UNet1D
from models.diffusion.ddpm import DDPM

# Create model
model = UNet1D(
    in_channels=1,
    out_channels=1,
    base_channels=64,
    channel_mults=(1, 2, 4, 8),
    attention_resolutions=(3,),
)

# Create DDPM
ddpm = DDPM(
    model=model,
    timesteps=1000,
    beta_schedule="linear",
    device="cuda",
)

# Training step
x_0 = torch.randn(32, 1, 4096).to("cuda")  # Batch of spectra
loss = ddpm.training_step(x_0)
loss.backward()
```

### Sampling

```python
# Generate samples
samples = ddpm.sample(
    num_samples=16,
    length=4096,
    channels=1,
    progress=True,
)  # Returns (16, 1, 4096) tensor
```

### Bounded Noise Denoising

```python
from scripts.train_bounded_denoiser import BoundedNoiseDiffusion
from models.diffusion.conditional_unet_1d import ConditionalUNet1D

# Create conditional model
model = ConditionalUNet1D(
    in_channels=1,
    cond_channels=1,  # σ conditioning
    out_channels=1,
    base_channels=32,
)

diffusion = BoundedNoiseDiffusion(
    model=model,
    lambda_values=[0.1, 0.2, 0.3, 0.4, 0.5],
    prediction_target="x0",
)

# Denoise
denoised = diffusion.single_step_denoise(noisy, sigma, lam=0.5)
```

---

## References

### Core Methods

1. **DDPM**: Ho et al., "Denoising Diffusion Probabilistic Models" (NeurIPS 2020)
2. **DDIM**: Song et al., "Denoising Diffusion Implicit Models" (ICLR 2021)
3. **DPS**: Chung et al., "Diffusion Posterior Sampling" (ICLR 2023)

### Astronomical Applications

4. **spec-DDPM**: RAA 2025 - Stellar spectra denoising with DDPM on LAMOST DR10
5. **Cosmological Diffusion**: arXiv:2312.07534 - Field generation + parameter inference

### Data

- **BOSZ**: Bohlin et al., "New Grids of ATLAS9 Model Atmospheres" - Synthetic stellar spectra
- **LAMOST DR10**: 22.29 million stellar spectra

---

## Roadmap

### Phase 0: Sanity Check ✅
- [x] MVP-0.0: Standard DDPM baseline (❌ sampling failed)
- [x] MVP-0.5: Bounded noise denoiser (✅ 59.5% improvement)
- [x] MVP-0.6: wMAE residual denoiser (✅ identity + improvement)

### Phase 1: Denoising Comparison (In Progress)
- [ ] MVP-1.0: Supervised DDPM (spec-DDPM reproduction)
- [ ] MVP-1.1: DPS posterior sampling
- [ ] MVP-1.2: +ivar conditioning for heteroscedastic noise

### Phase 2: Parameter Inference
- [ ] MVP-2.0: Sample propagation for parameter posteriors
- [ ] MVP-3.0: Spectral line evaluation (EW, line depth, RV bias)
- [ ] MVP-3.1: Coverage rate testing (PIT calibration)

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{specdiffusion2025,
  author = {Wei, Viska},
  title = {SpecDiffusion: Diffusion Models for Stellar Spectra},
  year = {2025},
  url = {https://github.com/username/SpecDiffusion}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- Architecture inspired by the [VIT project](https://github.com/) data loading patterns
- BOSZ synthetic spectra from STScI
- LAMOST collaboration for DR10 data
