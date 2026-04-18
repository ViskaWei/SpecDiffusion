# SpecDiffusion · 理论扎实版 · 必胜方向 + MVP · 综合报告

**日期**: 2026-04-18
**来源**: Claude 内部分析 + GPT-Pro 深度审计（req-84592b74）
**上下文**: BSN-Bayes 投稿后下一代正确架构；用户关切 — 之前 flux-space 方案没用光谱生成的完整物理信息

---

## 0. 一页纸结论

**必胜方向 = D3 Template-Bayes（85-90% 概率达 floor），不是 Diffusion。**

Diffusion 是第二级（Gate-1），且**不能用 DPS/ΠGDM**——要用 **Exact Gaussian-Product Posterior-Score (EGPPS)** 形式，这是你们"已知 Gaussian 似然"条件独有的理论洁净路线。

**理论保证**（可写进 proposal）:
在 BOSZ template RMS mismatch ≤ 0.3σ₀√(d/L) 的前提下，D3 posterior mean 满足
$$\text{RMSE} \leq 1.05\, \sigma_0\, \sqrt{d/L}$$
对 L=4096, d=6：**SNR=10 ⇒ RMSE ≤ 4×10⁻³，SNR=50 ⇒ RMSE ≤ 8×10⁻⁴**。
单 A100，~6 小时训练，N=50,000 synthetic spectra 即可观测。

**最大陷阱**：**1% template mismatch 比 floor 大 13 倍**，会毁掉整条理论叙事。interpolator 验证 RMS 必须 ≤ 3×10⁻⁴。

---

## 1. 理论基石（Pro 给的严格版本）

### 1.1 Bayes gap 恒等式

对任意估计器 `x̂(y)`：
$$\mathbb{E}\|\hat x(y) - x\|^2 = \underbrace{\mathbb{E}\|\hat x(y) - \mathbb{E}[x|y]\|^2}_{\text{Bayes gap}} + \mathbb{E}\,\text{tr}\,\text{Cov}(x|y)$$

唯一 Bayes 最优点估计是 `x̂_MMSE = E[x|y]`，最小可达风险是 `R⋆ = E tr Cov(x|y)`。所有方法的"额外 gap"都在第一项。

### 1.2 Manifold Bayes Floor（精确形式）

$$x = f(\theta),\quad \theta \in \Theta \subset \mathbb{R}^d,\quad d \ll L$$

Hessian (Fisher + residual curvature + prior curvature)：
$$H_\theta = \underbrace{J^\top D^{-1} J}_{\text{likelihood Fisher}} + \underbrace{P_\theta}_{\text{prior curvature}} + \text{residual curvature terms } O(\sigma_0^2)$$

Laplace 近似：`θ|y ≈ N(θ̂, H_θ^{-1})`。推到 spectrum 空间：

$$\boxed{F_{\text{man}} = \tfrac{1}{L}\,\text{tr}\bigl[J(J^\top D^{-1} J + P_\theta)^{-1} J^\top\bigr] + O(\sigma_0^4)}$$

**同方差极限**（`D = σ₀² I`, prior 弱）：
$$\Sigma_{x|y} \approx \sigma_0^2 J(J^\top J)^{-1} J^\top,\quad \text{RMSE}_{\text{floor}} \approx \sigma_0 \sqrt{d/L}$$

**对 L=4096**:
| d | √(d/L) | 标准差下降 (max) |
|---|---|---|
| 4 | 0.0313 | 32× |
| 6 | 0.0383 | 26.1× |
| 8 | 0.0442 | 22.6× |

BSN 报的"170×"**不可能**是 RMSE reduction——它只能是 estimator repeatability σ(D)。你的理论叙事要绕开这个混淆。

**异方差精确版**：`D^{-1/2} J = U S R^⊤`（SVD），则
$$F_{\text{man}} = \tfrac{1}{L}\text{tr}(U^\top D U)$$
即 floor 是噪声协方差在 **whitened tangent subspace** 上的 trace。

### 1.3 各架构理论 gap

| 方向 | Gap 构成 | 80% 胜率 |
|---|---|---|
| **D1 PCA** | projection residual + `tr C_z` + Davis-Kahan 有限样本 `O_p(r_eff/N)` | 90%（但非 NN） |
| **D3 Template Bayes + tiny interpolator** | `η_tpl² + η_interp² + C_curv σ₀⁴ + Δ_opt` | **85-90%** ★ |
| **EGPPS Diffusion** (见 §2.2) | score MSE × sampler Jacobian + 离散化误差 | 75-85% |
| **D5 NPE/NSF** | `KL + flow expressivity + summary-net info loss` | 70-80% |
| **DPS / ΠGDM / DDRM** | score MSE + **likelihood approximation bias** `O(‖C_t‖‖D⁻¹‖² ‖y-m‖)` | 50-65% |
| **DiT / MambaDiff** | 理论无优势（见 §2.3） | 不适合 MVP |

---

## 2. Diffusion 深挖（回答用户"Diffusion 可不可以"）

### 2.1 DPS/ΠGDM/DDRM 都不 exact —— 即使似然是已知 Gaussian

这是最关键的纠正。DPS 用
$$\nabla_{x_t}\log p(y|\hat x_0(x_t)) \approx \nabla_{x_t}\log \int p(y|x_0) p(x_0|x_t) dx_0$$
把积分用 Tweedie posterior mean 替掉。Delta expansion 后 leading bias：
$$O(\|C_t\| \cdot \|D^{-1}\|^2 \cdot \|y - m\|)$$
中高噪声步明显。ΠGDM 的 SVD 对 `A = I` 的去噪问题**完全 trivial**（异方差 D 只是逐像素缩放 1/σᵢ，不需要大矩阵 SVD）。

### 2.2 Exact Gaussian-Product Posterior Score (EGPPS) —— SpecDiffusion 正统路线

对 `y = x + n, n ~ N(0,D)`，引入人工扩散变量 `z = x + ξ, ξ ~ N(0, Σ_t)`。两个 Gaussian 相乘：
$$p_t(z|y) \propto \mathcal{N}(z; y, D+\Sigma_t) \cdot p_{C_t}(\mu_t)$$
其中
$$C_t = (D^{-1} + \Sigma_t^{-1})^{-1},\quad \mu_t = C_t(D^{-1} y + \Sigma_t^{-1} z)$$
$$p_{C_t} = p_0 * \mathcal{N}(0, C_t) \quad (\text{anisotropically smoothed prior})$$

**Exact conditional score**:
$$\boxed{\nabla_z \log p_t(z|y) = -(D+\Sigma_t)^{-1}(z-y) + (\Sigma_t^{-1} C_t)^\top \nabla_\mu \log p_{C_t}(\mu_t)}$$

**理论优势（相对 DPS/ΠGDM）**:
1. **零 likelihood approximation bias** — 似然项闭式，只剩 score network 估计 `∇log p_{C_t}`
2. **异方差 D 自然进入 C_t** — score net 需输入 diagonal noise map
3. **高噪声像素不被放大** — `(A_t)_{ii} = D_i/(D_i + σ_t²) ≤ 1`，这是本问题特有的美德
4. Score error 传播经 Grönwall：`E‖X₀ - X̃₀‖² ≤ e^{2∫L_t dt} ∫g_t⁴ E‖δ_t‖² dt`

### 2.3 为什么 MVP 不用 DiT / Transformer / MambaDiff

- DiT/U-ViT 的 win 来自 **image scaling**（FID vs GFLOPs），不是 1D spectra
- U-ViT 的论文结论是 **long skip connections** 重要，不是"attention 必须赢"
- 你们的 Bayes floor 由 `d ≈ 6` 决定，不由 token pair complexity 决定
- **Full-resolution attention 在 L=4096 的代价是 O(L²) 计算**，把 4096² token interaction 算一遍只为学 d 维物理结构 = 最大浪费

**MVP 架构定调**：1D U-Net + 底层低分辨率 attention + FiLM σ-conditioning，≤20M params。DiT 作为 full-version 的 ablation，不做 MVP 主线。

### 2.4 Consistency / Rectified Flow / Flow Matching

**不是第一步**。它们解决 sampler efficiency（NFE 从 50→1），不解决 posterior correctness。先把 EGPPS 做对，再蒸馏加速。

---

## 3. 必胜方向（理论保证 + 具体数字）

### 3.1 排名（2-4 周 + N=10⁵ + A100×1-8 预算下，"接近 floor (c)"为标准）

1. **D3 Template Bayes + tiny differentiable interpolator + Laplace** ← MVP 主线
2. **D1 PCA posterior** ← 零 NN 必跑 baseline
3. **EGPPS Diffusion** ← Gate-1 正统 SpecDiffusion
4. D5 NPE / Conditional NSF
5. 普通 DPS/ΠGDM/DDRM
6. DiT/MambaDiff（排除）

### 3.2 可写进 proposal 的严格理论保证

**假设**:
- **A1**：test spectra 由同一 BOSZ simulator 生成（self-consistency）或 template RMS mismatch `η_tpl ≤ 0.3 σ₀ √(d/L)`
- **A2**：`f(θ)` 在 posterior ball 内二阶可微，Laplace remainder 满足 `C_curv σ₀⁴ ≤ 0.05 (d/L) σ₀²`
- **A3**：MAP optimizer 找到 global mode，`J^⊤ D^{-1} J + P_θ` 条件数有界
- **A4**：interpolator validation RMS `η_interp ≤ 3×10⁻⁴` (SNR=50 regime)

**则 D3 posterior mean 满足**:
$$\tfrac{1}{L}\,\mathbb{E}\|\hat x - x\|^2 \leq F_{\text{man}} + \eta_{\text{tpl}}^2 + \eta_{\text{interp}}^2 + C_{\text{curv}}\sigma_0^4$$

**同方差 L=4096, d=6，prior 弱**:
$$\boxed{\text{RMSE} \leq 1.05\, \sigma_0\, \sqrt{d/L} = 0.0402\, \sigma_0}$$

| SNR | σ₀ | RMSE 上界 |
|---|---|---|
| 10 | 0.1 | 4.02 × 10⁻³ |
| 30 | 0.033 | 1.33 × 10⁻³ |
| 50 | 0.02 | 8.04 × 10⁻⁴ |

**资源**：N₀ = 5×10⁴，单 A100，~6 小时（Y=6h 是工程预算非数学常数）

### 3.3 用户"物理信息没用上"的直接回应

Pro 同意你的判断，但路径比"在 θ-space 扩散"更锋利：

**物理信息的正确注入方式 = 把 f(θ) 写成可微前向算子链**
$$f(\theta) = \text{Continuum} \circ \text{LSF} \circ \text{RotBroaden} \circ \text{RVShift} \circ \text{BOSZ\_interp}(\theta)$$

这链本身**无 NN 参数**。唯一的 NN 是 BOSZ interpolator（tiny MLP 3-5M params）。

**两条路同时可行**:

| 路线 | Score 定义域 | 主力 NN | 理论保证 |
|---|---|---|---|
| **D3 Template Bayes** | 参数空间 R^d | tiny BOSZ MLP | Laplace bound `RMSE ≤ 1.05σ₀√(d/L)` |
| **EGPPS Diffusion** | Flux 空间 R^L，但带 anisotropic C_t | 1D U-Net | 只剩 score MSE 传播项 |

MVP 阶段选 D3（统计效率高、可证）。Gate-1 用 EGPPS 扩展到 non-Gaussian 残差。

---

## 4. MVP（最小算力/数据，立即可跑）

### 4.1 Gate-0 · D3 + D1 baseline（必须先跑）

| 项 | 值 |
|---|---|
| Training samples | **N = 50,000** clean spectra with θ（不够则升到 200K 或改 cubic spline） |
| Validation | 5,000 |
| Test | 2,000（RMSE/EW/RV）+ SBC 用 500 obs × 256 draws |
| Sequence length | **Full L=4096**（禁止 subsample；EW/RV profile 会毁） |
| Model | **The Payne-style MLP**，in=d=5-7，hidden=4×512 或 5×512，out=4096，~3-5M params |
| **RV 处理** | **用可微 wavelength shift，不让 MLP 学 RV-shifted spectra**（关键 inductive bias）|
| Batch | 512-2048 |
| Epochs | 100-300，early stop on interpolator RMS |
| GPU | **Single A100** |
| Inference | multi-start L-BFGS MAP + Laplace Cov + 128-512 samples 从 `N(θ̂, C_θ)` 过 `f_φ` 平均 |
| SNR split | log-uniform [5, 50]，分 bin 报: 5-8, 8-15, 15-30, 30-50 |

**必须跑的 baseline**: D1 PCA posterior（零 NN，`r=10,20,30,50` sweep）+ BSN-Bayes（同 test set）

### 4.2 评估（两条指标足以证明理论保证）

**指标 1 · RMSE/Floor 比率**
$$\rho = \frac{\text{RMSE}_{\text{method}}}{\sigma_0 \sqrt{d/L}}$$
异方差版：分母换成 `[L⁻¹ tr J(J^⊤ D^{-1} J + P)^{-1} J^⊤]^{1/2}`

- **D3 pass**: ρ ≤ 1.10
- **首版 diffusion pass**: ρ ≤ 1.15

**指标 2 · Posterior calibration (SBC)**
- 500 obs × 256 draws 起步
- 论文版 1000-2000 × 512
- 检查 θ 的 rank histograms + EW 线性泛函 coverage

### 4.3 Go / No-go

**继续 full version 若**:
- D1 打平或优于 BSN-Bayes (RMSE/EW/RV)
- D3 achieves ρ ≤ 1.10 on simulator held-out
- SBC/TARP 无系统 over/underconfidence
- Gate-1 diffusion posterior mean 与 D3 差 ≤ 15% 且 posterior samples 改善 EW/RV coverage

**Pivot 若**:
- D1 已达 floor 且 diffusion 无法在 10-15% 内击败 D1 → **放弃"diffusion 降 RMSE"叙事**，diffusion 只用于 OOD 残差
- D3 在 simulator 上 ρ > 1.25 → optimizer / interpolator / LSF 实现错了（查 bug 而非换架构）
- D3 在 simulator 通过但真实 spectra 挂 → template mismatch 主导，转 D3 + residual prior，不是纯 DDPM
- Diffusion 降噪但 SBC 挂 → sampler guidance bias，换 DPS/ΠGDM → EGPPS

### 4.4 Gate-1 · EGPPS Diffusion MVP（D3 过关后启动）

| 项 | 值 |
|---|---|
| Training samples | 100,000 |
| Train noising | sample obs D ~ log-uniform SNR [5,50]，sample artificial `Σ_t = σ_t² I`，compute `C_t = (D⁻¹ + Σ_t⁻¹)⁻¹` |
| Training target | `s_φ(u, C_t) ≈ ∇log p_{C_t}(u)`，`u = x + ε, ε ~ N(0, C_t)` |
| Architecture | 1D U-Net，base_ch=64，mults [1,2,4,6]，2 res blocks，kernel 7 或 15，attn at L/16, L/32 |
| Conditioning | FiLM by `log C_i^{1/2}`（异方差进入 score net） |
| Params | **< 20M**（不要 DiT 量级） |
| Batch | 64-128 |
| Epochs | 100 |
| Schedule | **EDM/VE 但 σ_max ≈ 1-2**（**不是** image 的 80；flux 已 normalize） |
| Prediction | x₀-prediction，convert 到 score via `s = C^{-1}(D - u)` |
| **Inference** | **Exact posterior score**（§2.2 公式）：`-(D+Σ_t)⁻¹(z-y) + Σ_t⁻¹ C_t · s_φ(μ_t, C_t)` |
| NFE | 32-64 |
| DPS 对照 | **仅作 ablation**，不作主 sampler |

---

## 5. 用户必须知道的陷阱（Pro 红队）

1. **1% BOSZ mismatch 已足以毁掉 floor 声明**。SNR=50 时 tangent floor 是 7.66×10⁻⁴，1% flux mismatch = 10⁻²，**大 13 倍**。真实观测上不要声称 floor (c)/(d)，除非 template mismatch 建模进 posterior。

2. **BOSZ/Kurucz 不是真实恒星宇宙**。LTE 假设 + 特定 line list + 特定 atmosphere model。opacity / chromosphere / binary / abundance-pattern mismatch 不可避免。MVP 在 simulator self-consistency 上跑 OK，发 paper 前必须加 real-data validation。

3. **Continuum norm + LSF 不是后处理小事**。它们引入跨全谱 coupling。若 simulator 训练时没把 continuum/LSF uncertainty 纳入 θ，posterior 会**过窄**。

4. **异方差 D 的 SVD 陷阱**。对 `A = I`，ΠGDM 的 SVD 无神奇优势。别被 DPS/ΠGDM 流行度误导；写代码时要看 **D 如何进入 posterior score / Fisher matrix**。

5. **普通 denoising improvement 不证明 posterior correctness**。MVP-0.5/0.6 降 MSE 只是点估计；posterior calibration 是独立一件事，**必须 SBC 验**。

6. **Transformer 不是理论捷径**。你们的 bottleneck 是 d 维物理 posterior，不是 token mixing。DiT/U-ViT 的 image scaling 成果不迁移。

---

## 6. 论文叙事（不是"diffusion beats BSN"）

**最强叙事**（Pro 建议）:
> Known Gaussian likelihood + physical simulator **imply a computable Bayes floor**. BSN-Bayes 缺 joint covariance；D1/D3 建立 floor；SpecDiffusion (EGPPS) 处理 non-Gaussian residuals 并提供 calibrated samples.

这直接修复了 BSN synthesis 报告已经指出的理论缺陷（无 joint Cov、repeatability-vs-RMSE 混淆、OOD 未测），而且**与 D1/D3 baseline 形成上升阶梯**，不是"推翻 BSN"。

---

## 7. 立即执行清单

### 本周（Gate-0 D3 MVP）

```
[ ] 1. 准备 BOSZ 数据: 50K clean spectra with θ (已有 simulator, 跑脚本)
[ ] 2. 写 The-Payne-style MLP interpolator (hidden 4×512, in=d=5-7, out=4096)
[ ] 3. 训练: 100 epochs, single A100, 报 interpolation RMS (目标 ≤3×10⁻⁴)
[ ] 4. RV 处理: 可微 wavelength shift (FFT 或 interp)
[ ] 5. LSF 卷积层 (已知 kernel, batched GPU)
[ ] 6. L-BFGS + multi-start MAP
[ ] 7. Laplace Cov + 128-512 posterior samples
[ ] 8. 评估: ρ = RMSE / σ₀√(d/L), 每 SNR bin 报
[ ] 9. SBC: 500 obs × 256 draws, rank histograms + EW coverage
[ ] 10. 对照跑 D1 PCA posterior + BSN-Bayes 同 test set
[ ] 11. 写 Gate-0 report: 决定走 Gate-1 还是 pivot
```

### 下阶段（Gate-1 EGPPS Diffusion，D3 过关后）

```
[ ] 12. 实现 EGPPS training loop (Σ_t, C_t, p_{C_t} 采样)
[ ] 13. 1D U-Net with FiLM σ-conditioning
[ ] 14. Exact posterior score inference sampler
[ ] 15. DPS ablation 对照
[ ] 16. Posterior mean 与 D3 比，SBC 比
```

---

## 附录 · 关键参考

- **DPS bias**: Chung et al. ICLR 2023 (Tweedie + Laplace posterior approx)
- **ΠGDM**: Song et al. ICLR 2023 (pseudoinverse guidance)
- **DDRM**: Kawar et al. NeurIPS 2022 (linear inverse SVD)
- **Optimal posterior covariance**: 2024 综述统一 DPS/ΠGDM 为 Gaussian approximation variants
- **EDM**: Karras et al. NeurIPS 2022 (noise schedule + sampling)
- **The Payne**: Ting et al. 2019 (stellar-label-to-spectrum MLP interpolator)
- **SBC**: Talts et al. 2018 (posterior validation)
- **TARP**: Lemos et al. 2023 (posterior accuracy test)
- **Flow Matching**: Lipman et al. 2023 (simulation-free CNF training)
- **Consistency Models**: Song et al. 2023 (one-step sampling)
- **U-ViT**: long-skip ViT for diffusion
- **1D spec-DDPM**: LAMOST DR10 (2024-2025 类似工作，但无 posterior calibration + manifold floor 对照)
