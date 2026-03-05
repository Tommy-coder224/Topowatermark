"""
RadioactiveDetector — 放射性水印检测与假设检验
===============================================

核心数学（定理 E + 定理 F，审稿人可手推）：
═══════════════════════════════════════════════════════════

**Neyman-Pearson 假设检验 + Sablayrolles 载波检测：**

  H₀: 被检测模型未使用水印数据训练
  H₁: 被检测模型使用了用户 u 的水印数据训练

**检测方式 1 — 载波投影 + incomplete beta p-value（推荐）：**

  统计量（Sablayrolles ICML 2020 §3.1，适配拓扑空间）：
    φ'_mean = mean(λ(x_i)) − μ_clean   （中心化拓扑均值）
    cos_u = ⟨φ'_mean, u_u⟩ / ‖φ'_mean‖   （载波余弦）
    p_u = ½ I_{1−cos²}((d−1)/2, 1/2)   （精确 p-value）

  在 H₀ 下：u_u 与 φ'_mean 独立 → cos 服从 beta-incomplete → p ≈ U[0,1]
  在 H₁ 下：φ'_mean 沿 u_u 偏移 → cos > 0 → p << α

  用户溯源：û = argmin_u p_u = argmax_u cos_u

**检测方式 2 — 旧逻辑（兼容无载波情况）：**
  用均值投影 + 正态近似，参见旧版。

参考文献：
  [1] Sablayrolles et al. "Radioactive data: tracing through training."
      ICML 2020. (载波 + 余弦检验 + incomplete beta)
  [2] Gretton et al. "A kernel two-sample test." JMLR 2012. (MMD)
  [3] Lehmann & Romano. "Testing Statistical Hypotheses." Springer 2005.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as sp_stats

from ..core.carrier import CarrierManager
from ..core.topo_vectorize import (
    TopoSignature,
    extract_topo_signature,
)
from .embedder import WatermarkRegistry


@dataclass
class DetectionResult:
    """单次检测结果。"""
    is_radioactive: bool
    p_value: float
    test_statistic: float
    threshold: float
    attributed_user: Optional[int] = None
    user_scores: Dict[int, float] = field(default_factory=dict)
    mmd_score: float = 0.0
    n_samples: int = 0

    def summary(self) -> str:
        status = "RADIOACTIVE" if self.is_radioactive else "CLEAN"
        lines = [
            f"Detection: {status}",
            f"  p-value:    {self.p_value:.6f}",
            f"  statistic:  {self.test_statistic:.6f}",
            f"  threshold:  {self.threshold:.6f}",
            f"  MMD^2:      {self.mmd_score:.6f}",
            f"  N samples:  {self.n_samples}",
        ]
        if self.attributed_user is not None:
            lines.append(f"  Attributed: user {self.attributed_user}")
        if self.user_scores:
            top3 = sorted(self.user_scores.items(), key=lambda x: -x[1])[:3]
            lines.append(f"  Top users:  {top3}")
        return "\n".join(lines)


class RadioactiveDetector:
    """
    拓扑放射性水印检测器。

    用法：
        detector = RadioactiveDetector(config)
        result = detector.detect(test_images, registry)
    """

    def __init__(
        self,
        max_dim: int = 1,
        pi_resolution: int = 20,
        pi_bandwidth: float = 0.05,
        fpr: float = 0.01,
    ):
        self.max_dim = max_dim
        self.pi_resolution = pi_resolution
        self.pi_bandwidth = pi_bandwidth
        self.fpr = fpr

    def _extract_signature(self, image: Any) -> TopoSignature:
        """从 PIL Image 或 numpy array 提取拓扑签名。"""
        if hasattr(image, "convert"):
            img_np = np.array(image).astype(np.float64) / 255.0
            if img_np.ndim == 3:
                img_np = img_np.transpose(2, 0, 1)
        else:
            img_np = np.asarray(image, dtype=np.float64)
            if img_np.ndim == 4:
                img_np = img_np[0]
        return extract_topo_signature(
            img_np,
            max_dim=self.max_dim,
            pi_resolution=self.pi_resolution,
            pi_bandwidth=self.pi_bandwidth,
        )

    # ── Layer 1: 拓扑投影检测 ─────────────────────────────

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-12 or nb < 1e-12:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def compute_user_scores(
        self,
        test_sigs: List[TopoSignature],
        registry: WatermarkRegistry,
    ) -> Dict[int, float]:
        """
        对每个用户计算 centered 拓扑投影分数。
        先将 test 均值减去 clean 均值（center），再与偏移方向做余弦。
        """
        test_vecs = np.stack([s.vector for s in test_sigs])
        test_mean = test_vecs.mean(axis=0)

        if registry.global_clean_mean is not None:
            test_centered = test_mean - registry.global_clean_mean
        else:
            test_centered = test_mean

        scores = {}
        for uid, profile in registry.user_profiles.items():
            mu_u = profile["mean_vector"]
            if registry.global_clean_mean is not None:
                direction = mu_u - registry.global_clean_mean
            else:
                direction = mu_u
            scores[uid] = self._cosine_similarity(test_centered, direction)

        return scores

    # ── Layer 2: MMD 分布偏移检测 ─────────────────────────

    def compute_mmd(
        self,
        test_sigs: List[TopoSignature],
        clean_sigs: List[TopoSignature],
    ) -> float:
        """
        计算拓扑向量空间中的线性 MMD²。

        MMD²(P, Q) = ‖E_P[φ(x)] − E_Q[φ(y)]‖²
        """
        test_vecs = np.stack([s.vector for s in test_sigs])
        clean_vecs = np.stack([s.vector for s in clean_sigs])
        diff = test_vecs.mean(axis=0) - clean_vecs.mean(axis=0)
        return float(np.dot(diff, diff))

    # ── 综合检测 ──────────────────────────────────────────

    def detect(
        self,
        test_images: list,
        registry: WatermarkRegistry,
        clean_images: Optional[list] = None,
        carrier_manager: Optional[CarrierManager] = None,
        clean_baseline_sigs: Optional[List[TopoSignature]] = None,
    ) -> DetectionResult:
        """
        综合放射性检测。

        当 carrier_manager 已校准时，使用 Sablayrolles 式载波检测；
        若同时提供 clean_baseline_sigs，则用其 bootstrap 得到经验 H₀ 阈值，
        判放射性 = (max_u cos > 阈值)，从而降低干净图误判（FPR）。

        Args:
            test_images: 待检测图像列表（PIL 或 numpy）
            registry:    水印注册表
            clean_images: clean 基线图像（可选，用于 MMD）
            carrier_manager: 已校准的载波管理器（可选）
            clean_baseline_sigs: 干净基线拓扑签名（可选，用于经验 H₀ 阈值）

        Returns:
            DetectionResult
        """
        print(f"[RadioactiveDetector] Extracting topology from {len(test_images)} images...")
        test_sigs = [self._extract_signature(img) for img in test_images]
        N = len(test_sigs)

        use_carrier = (
            carrier_manager is not None
            and carrier_manager._calibrated
            and carrier_manager.carriers is not None
        )

        if use_carrier:
            return self._detect_carrier(
                test_sigs, carrier_manager, N, clean_images, clean_baseline_sigs
            )

        user_scores = self.compute_user_scores(test_sigs, registry)

        if not user_scores:
            return DetectionResult(
                is_radioactive=False,
                p_value=1.0,
                test_statistic=0.0,
                threshold=0.0,
                n_samples=N,
            )

        best_user = max(user_scores, key=user_scores.get)
        T = user_scores[best_user]

        mmd = 0.0
        if clean_images is not None:
            clean_sigs = [self._extract_signature(img) for img in clean_images]
            mmd = self.compute_mmd(test_sigs, clean_sigs)

        p_value, threshold = self._hypothesis_test(
            test_sigs, registry, best_user,
        )

        is_radioactive = p_value < self.fpr

        return DetectionResult(
            is_radioactive=is_radioactive,
            p_value=p_value,
            test_statistic=T,
            threshold=threshold,
            attributed_user=best_user if is_radioactive else None,
            user_scores=user_scores,
            mmd_score=mmd,
            n_samples=N,
        )

    def _detect_carrier(
        self,
        test_sigs: List[TopoSignature],
        carrier_manager: CarrierManager,
        N: int,
        clean_images: Optional[list] = None,
        clean_baseline_sigs: Optional[List[TopoSignature]] = None,
    ) -> DetectionResult:
        """
        w_u 归属驱动检测（命题 10(d) 修正版 + 定理 F + Bonferroni + 经验 H₀）。

        检测逻辑：
          对每用户 u: T_u = cos(φ'_mean, w_u)，p_u = cosine_p_value(T_u, d)
          Bonferroni: reject if min_u p_u < α/U  [Lehmann & Romano 2005]
          经验 H₀: bootstrap max_u cos from clean → reject if exceeds threshold [Efron 2004]

        若提供 clean_baseline_sigs → 用经验阈值（更鲁棒）
        否则 → 用 Bonferroni 分析阈值
        """
        test_vecs = np.stack([s.vector for s in test_sigs])

        clean_vecs = None
        if clean_baseline_sigs is not None and len(clean_baseline_sigs) >= 5:
            clean_vecs = np.stack([s.vector for s in clean_baseline_sigs])

        result = carrier_manager.detect_two_stage(
            test_vecs, alpha=self.fpr, clean_baseline=clean_vecs,
        )

        user_scores_cos = {uid: info["cosine"] for uid, info in result["user_scores"].items()}

        mmd = 0.0
        if clean_images is not None:
            clean_sigs = [self._extract_signature(img) for img in clean_images]
            mmd = self.compute_mmd(test_sigs, clean_sigs)

        return DetectionResult(
            is_radioactive=result["is_watermarked"],
            p_value=result["min_p_wu"],
            test_statistic=result.get("best_t_wu", result["best_cos_wu"]),
            threshold=result.get("empirical_threshold") or self.fpr,
            attributed_user=result["attributed_user"],
            user_scores=user_scores_cos,
            mmd_score=mmd,
            n_samples=N,
        )

    def _hypothesis_test(
        self,
        test_sigs: List[TopoSignature],
        registry: WatermarkRegistry,
        user_id: int,
    ) -> Tuple[float, float]:
        """
        Neyman-Pearson 假设检验。

        H0: test images 的拓扑分布 = clean 分布（未使用水印数据）
        H1: test images 的拓扑分布向注册用户 u 偏移

        统计量：
          direction = mu_u - mu_clean   （水印用户 vs clean 的偏移方向）
          对每个 test image: proj_i = <(phi(x_i) - mu_clean), direction_unit>
          T = mean(proj_i)

        在 H0 下 E[proj_i] = 0（因为 clean 减去自身均值后在任意方向期望为 0）。
        在 H1 下 E[proj_i] > 0（水印图像朝 mu_u 偏移）。

        p-value = 1 - Phi(T / (sigma_hat / sqrt(N)))   单侧检验。
        """
        profile = registry.user_profiles[user_id]
        mu_u = profile["mean_vector"]

        if registry.global_clean_mean is not None:
            direction = mu_u - registry.global_clean_mean
            center = registry.global_clean_mean
        else:
            direction = mu_u
            center = np.zeros_like(mu_u)

        dir_norm = np.linalg.norm(direction)
        if dir_norm < 1e-12:
            return 1.0, 0.0
        direction_unit = direction / dir_norm

        projections = []
        for sig in test_sigs:
            centered = sig.vector - center
            proj = float(np.dot(centered, direction_unit))
            projections.append(proj)

        projections = np.array(projections)
        N = len(projections)
        T_mean = projections.mean()
        T_std = projections.std(ddof=1) if N > 1 else 1.0

        se = T_std / np.sqrt(N) if N > 0 else 1.0

        if se < 1e-12:
            p_value = 0.0 if T_mean > 0 else 1.0
        else:
            z = T_mean / se
            p_value = 1.0 - sp_stats.norm.cdf(z)

        threshold = sp_stats.norm.ppf(1.0 - self.fpr) * se

        return float(p_value), float(threshold)

    # ── 多用户溯源 ────────────────────────────────────────

    def attribute(
        self,
        test_images: list,
        registry: WatermarkRegistry,
    ) -> Tuple[int, Dict[int, float]]:
        """
        用户溯源：û = argmax_u T₁(X; μ_u)

        Returns:
            (attributed_user_id, all_user_scores)
        """
        test_sigs = [self._extract_signature(img) for img in test_images]
        scores = self.compute_user_scores(test_sigs, registry)
        if not scores:
            return -1, {}
        best_user = max(scores, key=scores.get)
        return best_user, scores
