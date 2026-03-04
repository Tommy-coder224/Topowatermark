"""
稳定性定理验证与界估计
========================

本模块提供定理 B, C, D 的实证验证工具：
  · VAE round-trip 拓扑保持度
  · 微调前后拓扑分布距离
  · 检测阈值的理论与实证估计

══════════════════════════════════════════════════════════════
核心定理链（审稿人可逐步手推）
══════════════════════════════════════════════════════════════

**定理 B (Cohen-Steiner et al. 2007):**

  d_B(Dgm(f), Dgm(g)) ≤ ‖f − g‖_∞

  验证方法：计算 VAE 重建误差 ε_VAE = ‖x − Dec(Enc(x))‖_∞,
  与 bottleneck 距离对比，确认不等式成立。

**引理 1 (VAE 拓扑保持):**

  设 VAE 编码-解码满足 ‖x − Dec(Enc(x))‖_∞ ≤ ε_VAE，则

      d_B(Dgm(x), Dgm(Dec(Enc(x)))) ≤ ε_VAE

  对每个通道、每个同调维度分别成立。

  这是定理 B 的直接应用——持久图的 bottleneck 距离不超过
  原函数的 L^∞ 扰动。

**引理 2 (持久图像稳定性, Adams et al. 2017):**

      ‖PI(D₁) − PI(D₂)‖₂ ≤ L_σ · W₁(D₁, D₂)

  验证方法：对多组 (D₁, D₂) 经验估计 L_σ。

**定理 D (本文 — 拓扑放射性):**

  设：
    · p_w = 水印图像分布，φ = 拓扑特征映射（Lipschitz, 常数 L_φ）
    · p_θ = 微调模型分布，满足 W₁(p_θ, p_w) ≤ ε_train
    · p₀ = 未水印图像分布

  则：
      W₁(φ_#p_θ, φ_#p₀) ≥ W₁(φ_#p_w, φ_#p₀) − L_φ · ε_train

  其中 φ_# 表示 pushforward 测度。

  直觉：如果水印引入的拓扑偏移 W₁(φ_#p_w, φ_#p₀) 足够大，
  即使微调有近似误差 ε_train，偏移仍可检测。

  证明（3 行）：
    W₁(φ_#p_θ, φ_#p₀) ≥ W₁(φ_#p_w, φ_#p₀) − W₁(φ_#p_θ, φ_#p_w)
                        ≥ W₁(φ_#p_w, φ_#p₀) − L_φ · W₁(p_θ, p_w)
                        ≥ W₁(φ_#p_w, φ_#p₀) − L_φ · ε_train    □

  其中第一步用三角不等式，第二步用 φ 的 Lipschitz 性质。

**定理 E (本文 — 检测保证):**

  在 Neyman-Pearson 框架下：
    H₀: 模型未使用水印数据训练
    H₁: 模型使用水印数据训练

  检测统计量：T(X) = (1/N) Σᵢ ⟨φ(xᵢ), μ_reg⟩

  其中 μ_reg 是注册拓扑指纹方向。

  在 H₀ 下 E[T] = 0, Var[T] = σ₀²/N（可从 clean 数据估计）。
  在 H₁ 下 E[T] ≥ δ > 0（拓扑偏移）。

  给定虚警率 α，阈值 τ = Φ⁻¹(1−α) · σ₀/√N。
  检测功效 β = P(T > τ | H₁) = Φ(δ√N/σ₀ − Φ⁻¹(1−α))。

  即：样本量 N → ∞ 时，β → 1（检测功效趋于 1）。

══════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .gudhi_persistence import (
    bottleneck_distance_dgm,
    channel_persistence,
    multi_channel_wasserstein,
    wasserstein_distance_dgm,
)
from .topo_vectorize import (
    TopoSignature,
    extract_topo_signature,
    signature_from_diagrams,
)


@dataclass
class StabilityReport:
    """VAE 拓扑稳定性报告。"""
    n_samples: int = 0
    l_inf_errors: List[float] = field(default_factory=list)
    bottleneck_distances: List[float] = field(default_factory=list)
    wasserstein_distances: List[float] = field(default_factory=list)
    pi_l2_distances: List[float] = field(default_factory=list)
    theorem_b_satisfied: List[bool] = field(default_factory=list)

    @property
    def mean_l_inf(self) -> float:
        return float(np.mean(self.l_inf_errors)) if self.l_inf_errors else 0.0

    @property
    def mean_bottleneck(self) -> float:
        return float(np.mean(self.bottleneck_distances)) if self.bottleneck_distances else 0.0

    @property
    def mean_wasserstein(self) -> float:
        return float(np.mean(self.wasserstein_distances)) if self.wasserstein_distances else 0.0

    @property
    def mean_pi_l2(self) -> float:
        return float(np.mean(self.pi_l2_distances)) if self.pi_l2_distances else 0.0

    @property
    def theorem_b_rate(self) -> float:
        return float(np.mean(self.theorem_b_satisfied)) if self.theorem_b_satisfied else 0.0

    def summary(self) -> str:
        return (
            f"StabilityReport(n={self.n_samples})\n"
            f"  L_inf error:      {self.mean_l_inf:.6f}\n"
            f"  Bottleneck d_B:   {self.mean_bottleneck:.6f}\n"
            f"  Wasserstein W_2:  {self.mean_wasserstein:.6f}\n"
            f"  PI L2 distance:   {self.mean_pi_l2:.6f}\n"
            f"  Thm B satisfied:  {self.theorem_b_rate:.2%}\n"
            f"  Thm B check:     d_B <= L_inf  for {self.theorem_b_rate:.0%} samples"
        )


class StabilityVerifier:
    """
    稳定性定理的实证验证器。

    验证：
      1. 定理 B: d_B(Dgm(x), Dgm(x')) ≤ ‖x − x'‖_∞
      2. 引理 2: PI Lipschitz 常数估计
      3. 定理 D: 拓扑偏移量的经验估计
    """

    def __init__(
        self,
        max_dim: int = 1,
        pi_resolution: int = 20,
        pi_bandwidth: float = 0.05,
    ):
        self.max_dim = max_dim
        self.pi_resolution = pi_resolution
        self.pi_bandwidth = pi_bandwidth

    # ── 1. VAE round-trip 拓扑稳定性 ──────────────────────

    def verify_vae_stability(
        self,
        originals: np.ndarray,
        reconstructed: np.ndarray,
        verbose: bool = False,
        log_fn=None,
        **kwargs,
    ) -> StabilityReport:
        """
        验证引理 1：VAE 编码-解码对拓扑的影响有界。

        对每对 (x, x') = (原始, 重建)：
          1. 计算 ‖x − x'‖_∞
          2. 计算逐通道 bottleneck 距离
          3. 验证 d_B ≤ ‖x − x'‖_∞（定理 B）
          4. 计算 PI 向量 L₂ 距离

        Args:
            originals:     shape (N, C, H, W)
            reconstructed: shape (N, C, H, W)

        Returns:
            StabilityReport
        """
        assert originals.shape == reconstructed.shape
        N = originals.shape[0]
        report = StabilityReport(n_samples=N)
        verbose = verbose or kwargs.get("verbose", False)
        if log_fn is None:
            log_fn = kwargs.get("log_fn", lambda msg: print(msg, flush=True))

        for i in range(N):
            x = originals[i]
            x_prime = reconstructed[i]

            l_inf = float(np.max(np.abs(x - x_prime)))
            report.l_inf_errors.append(l_inf)

            if verbose:
                log_fn("      sample %d/%d: persistence orig (GUDHI cubical)..." % (i + 1, N))
            dgms_x = channel_persistence(x, max_dim=self.max_dim)
            if verbose:
                log_fn("      sample %d/%d: persistence recon (GUDHI cubical)..." % (i + 1, N))
            dgms_xp = channel_persistence(x_prime, max_dim=self.max_dim)

            if verbose:
                log_fn("      sample %d/%d: bottleneck + Wasserstein..." % (i + 1, N))
            d_bottle = 0.0
            for c in range(len(dgms_x)):
                for dim in range(self.max_dim + 1):
                    d1 = dgms_x[c].get(dim, np.empty((0, 2)))
                    d2 = dgms_xp[c].get(dim, np.empty((0, 2)))
                    db = bottleneck_distance_dgm(d1, d2)
                    d_bottle = max(d_bottle, db)
            report.bottleneck_distances.append(d_bottle)

            d_wass = multi_channel_wasserstein(dgms_x, dgms_xp)
            report.wasserstein_distances.append(d_wass)

            # Reuse dgms for PI (no second persistence pass)
            if verbose:
                log_fn("      sample %d/%d: PI from diagrams..." % (i + 1, N))
            sig_x = signature_from_diagrams(
                dgms_x, max_dim=self.max_dim,
                pi_resolution=self.pi_resolution,
                pi_bandwidth=self.pi_bandwidth,
            )
            sig_xp = signature_from_diagrams(
                dgms_xp, max_dim=self.max_dim,
                pi_resolution=self.pi_resolution,
                pi_bandwidth=self.pi_bandwidth,
            )
            pi_dist = float(np.linalg.norm(sig_x.vector - sig_xp.vector))
            report.pi_l2_distances.append(pi_dist)

            thm_ok = d_bottle <= l_inf + 1e-8
            report.theorem_b_satisfied.append(thm_ok)

            if verbose:
                log_fn("      sample %d/%d: L_inf=%.6f  d_B=%.6f  W_2=%.6f  PI_L2=%.4f  Thm_B=%s" % (
                    i + 1, N, l_inf, d_bottle, d_wass, pi_dist, "OK" if thm_ok else "FAIL"))

        return report

    # ── 2. Lipschitz 常数经验估计 ─────────────────────────

    def estimate_pi_lipschitz(
        self,
        originals: np.ndarray,
        perturbed: np.ndarray,
    ) -> float:
        """
        经验估计持久图像映射的 Lipschitz 常数。

        L_σ ≈ max_i ‖PI(x_i) − PI(x'_i)‖₂ / W₁(Dgm(x_i), Dgm(x'_i))

        这验证引理 2 的常数在实际数据上的大小。
        """
        N = originals.shape[0]
        ratios = []

        for i in range(N):
            sig_x = extract_topo_signature(
                originals[i], max_dim=self.max_dim,
                pi_resolution=self.pi_resolution,
                pi_bandwidth=self.pi_bandwidth,
            )
            sig_xp = extract_topo_signature(
                perturbed[i], max_dim=self.max_dim,
                pi_resolution=self.pi_resolution,
                pi_bandwidth=self.pi_bandwidth,
            )

            pi_dist = float(np.linalg.norm(sig_x.vector - sig_xp.vector))

            dgms_x = channel_persistence(originals[i], max_dim=self.max_dim)
            dgms_xp = channel_persistence(perturbed[i], max_dim=self.max_dim)
            w_dist = multi_channel_wasserstein(dgms_x, dgms_xp, order=1.0)

            if w_dist > 1e-10:
                ratios.append(pi_dist / w_dist)

        return float(np.max(ratios)) if ratios else 0.0

    # ── 3. 拓扑偏移量估计（定理 D） ──────────────────────

    def estimate_topo_shift(
        self,
        watermarked_sigs: List[TopoSignature],
        clean_sigs: List[TopoSignature],
    ) -> Dict[str, float]:
        """
        估计水印引入的拓扑偏移 W₁(φ_#p_w, φ_#p₀)。

        通过比较水印图像与 clean 图像的拓扑向量分布，
        计算经验 MMD、均值偏移、以及方向一致性。

        定理 D 的实证支撑：偏移量越大，放射性检测越容易。
        """
        wm_vecs = np.stack([s.vector for s in watermarked_sigs])
        cl_vecs = np.stack([s.vector for s in clean_sigs])

        mean_wm = wm_vecs.mean(axis=0)
        mean_cl = cl_vecs.mean(axis=0)
        shift_norm = float(np.linalg.norm(mean_wm - mean_cl))

        shift_dir = mean_wm - mean_cl
        shift_dir_norm = shift_dir / (np.linalg.norm(shift_dir) + 1e-12)

        proj_wm = wm_vecs @ shift_dir_norm
        proj_cl = cl_vecs @ shift_dir_norm
        cohen_d = (proj_wm.mean() - proj_cl.mean()) / (
            np.sqrt((proj_wm.var() + proj_cl.var()) / 2) + 1e-12
        )

        mmd = self._linear_mmd(wm_vecs, cl_vecs)

        return {
            "shift_l2_norm": shift_norm,
            "cohen_d": float(cohen_d),
            "linear_mmd": mmd,
            "n_watermarked": len(watermarked_sigs),
            "n_clean": len(clean_sigs),
        }

    @staticmethod
    def _linear_mmd(X: np.ndarray, Y: np.ndarray) -> float:
        """
        线性 MMD 估计：

          MMD²(P, Q) = ‖E_P[φ(x)] − E_Q[φ(y)]‖²

        这里直接在拓扑向量空间中计算（kernel = linear）。
        """
        diff = X.mean(axis=0) - Y.mean(axis=0)
        return float(np.dot(diff, diff))

    # ── 4. 检测阈值计算（定理 E） ────────────────────────

    @staticmethod
    def compute_detection_threshold(
        clean_scores: np.ndarray,
        fpr: float = 0.01,
    ) -> float:
        """
        根据 H₀ 下的分数分布和目标虚警率 α 计算检测阈值。

        定理 E：τ = Φ⁻¹(1−α) · σ₀/√N

        实际实现使用经验百分位数（更鲁棒）。
        """
        threshold = float(np.quantile(clean_scores, 1.0 - fpr))
        return threshold

    @staticmethod
    def compute_detection_power(
        delta: float,
        sigma0: float,
        n_samples: int,
        fpr: float = 0.01,
    ) -> float:
        """
        计算检测功效 β（定理 E 公式）。

        β = Φ(δ√N/σ₀ − Φ⁻¹(1−α))

        Args:
            delta:     拓扑偏移量 E[T|H₁]
            sigma0:    H₀ 下标准差
            n_samples: 样本量 N
            fpr:       虚警率 α
        """
        from scipy.stats import norm
        z_alpha = norm.ppf(1.0 - fpr)
        z_power = delta * np.sqrt(n_samples) / (sigma0 + 1e-12) - z_alpha
        return float(norm.cdf(z_power))
