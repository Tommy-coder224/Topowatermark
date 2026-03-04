"""
载波管理器（CarrierManager）— w_u 归属驱动 + 经验 H₀ 校准
=============================================================

理论基础（审稿人可手推，所有定理均有文献原型）
═══════════════════════════════════════════════════════════

**核心洞察：w₀（第一主成分）捕获的是干净图像共有结构，
  不适合做检测统计量；w_u（次要主成分空间）对干净图近似各向同性，
  适合做 Sablayrolles 式 p-value 检测。**

  实验证据（benchmark 结果）：
    · 干净图 cos(w₀) ≈ 0.69  →  Sablayrolles p ≈ 0  →  FPR 失控
    · 干净图 cos(w_u) ≈ 0     →  Sablayrolles p ≈ 1  →  FPR 可控

  理论解释：
    · w₀ = v₁（第一主成分），解释 40%+ 方差
    · w_u ⊥ w₀，在次要主成分空间内，解释方差 << 2%
    · 干净图在 w_u 方向近似各向同性 → 满足 Sablayrolles 的 H₀ 假设

**定理 F (Sablayrolles et al., ICML 2020 §3.1):**

  P(cos(w_u, φ'_mean) ≥ τ | H₀) = ½ I_{1−τ²}((d−1)/2, 1/2)

  此公式对 w_u 有效（干净图在 w_u 方向近似各向同性），
  对 w₀ 无效（干净图在 w₀ 方向有系统性偏移）。

**命题 10 (本文 — 修正版):**

  (a) w₀ = v₁ / ‖v₁‖  — 保留用于分析，不用于检测判决

  (b) w_u ⊥ w₀，从 v₂...v_r 子空间生成 + Gram-Schmidt 正交化

  (c) 生成时选择准则（仅 w_u）：
        i* = argmax_i ⟨λ_i − μ, w_u⟩
      理由：w₀ 方向信号被干净图共性淹没，仅优化 w_u 归属信号

  (d) 检测（Bonferroni + 经验 H₀）：
      对每用户 u：T_u = cos(φ'_mean, w_u)，p_u = cosine_p_value(T_u, d)
      Bonferroni：reject H₀ if min_u p_u < α / U
      [Lehmann & Romano 2005, §9.1]

      经验 H₀（可选，Efron 2004）：
        从干净基线 bootstrap max_u cos → 阈值 τ_emp
        reject if max_u T_u > τ_emp
        — 不依赖分布假设，对非各向同性也鲁棒

  (e) LDA 精炼（可选，Fisher 1936）：
      用已标注数据计算类间/类内散度比，求判别方向，
      替换 PCA 载波为判别性更强的 LDA 载波。

参考文献：
  [1] Sablayrolles et al. "Radioactive data." ICML 2020.
  [2] Jolliffe. "Principal Component Analysis." Springer 2002.
  [3] Lehmann & Romano. "Testing Statistical Hypotheses." Springer 2005.
  [4] Efron. "Large-Scale Simultaneous Hypothesis Testing." JASA 2004.
  [5] Fisher. "The Use of Multiple Measurements in Taxonomic Problems."
      Annals of Eugenics 1936.
"""
from __future__ import annotations

import hashlib
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.special import betainc


class CarrierManager:
    """
    载波管理器：w_u 归属驱动选择 + Bonferroni/经验 H₀ 检测。
    """

    def __init__(
        self,
        num_users: int = 10,
        n_components: int = 50,
        carrier_seed: int = 2025,
    ):
        self.num_users = num_users
        self.n_components = n_components
        self.carrier_seed = carrier_seed

        self.mu_clean: Optional[np.ndarray] = None
        self.components: Optional[np.ndarray] = None
        self.explained_variance: Optional[np.ndarray] = None

        self.w0: Optional[np.ndarray] = None
        self.carriers: Optional[np.ndarray] = None
        self._calibrated = False

    # ═══ 1. PCA 校准 ═════════════════════════════════════════

    def calibrate(
        self,
        topo_vectors: np.ndarray,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """PCA 校准（Jolliffe 2002）。"""
        Phi = np.asarray(topo_vectors, dtype=np.float64)
        N, d = Phi.shape
        r = min(self.n_components, N - 1, d)

        self.mu_clean = Phi.mean(axis=0)
        Phi_cent = Phi - self.mu_clean
        _, S, Vt = np.linalg.svd(Phi_cent, full_matrices=False)

        self.components = Vt[:r]
        total_var = (S ** 2).sum()
        self.explained_variance = (S[:r] ** 2) / (total_var + 1e-12)
        cumulative = self.explained_variance.cumsum()
        self._calibrated = True

        stats = {
            "n_samples": N, "topo_dim": d, "n_components": r,
            "explained_variance_top1": float(self.explained_variance[0]),
            "cumulative_variance": float(cumulative[-1]),
        }
        if verbose:
            print(f"[CarrierManager] Calibrated: N={N}, d={d}, r={r}")
            print(f"  Top-1 explained var: {stats['explained_variance_top1']:.4f}")
            print(f"  Cumulative ({r} comp): {stats['cumulative_variance']:.4f}")
        return stats

    # ═══ 2. 载波生成 ═════════════════════════════════════════

    def generate_carriers(
        self,
        num_users: Optional[int] = None,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        命题 10(a)(b)：w₀ + 正交化 w_u 生成。

        Returns: (w0, carriers)
        """
        if not self._calibrated:
            raise RuntimeError("Must call calibrate() first")

        U = num_users or self.num_users
        r = self.components.shape[0]
        d = self.components.shape[1]

        self.w0 = self.components[0].copy()
        self.w0 /= np.linalg.norm(self.w0)

        remaining = self.components[1:]
        r_remain = remaining.shape[0]

        raw_vectors = []
        for u in range(U):
            seed = self._user_seed(u)
            rng = np.random.RandomState(seed)
            z_u = rng.randn(r_remain).astype(np.float64)
            v_u = remaining.T @ z_u
            raw_vectors.append(v_u)

        carriers = self._gram_schmidt(raw_vectors)

        for i in range(len(carriers)):
            dot_w0 = np.dot(carriers[i], self.w0)
            carriers[i] -= dot_w0 * self.w0
            norm = np.linalg.norm(carriers[i])
            if norm > 1e-10:
                carriers[i] /= norm

        self.carriers = carriers

        if verbose:
            max_cos_uu = self._max_pairwise_cosine(carriers)
            max_cos_w0 = max(abs(float(np.dot(c, self.w0))) for c in carriers)
            print(f"[CarrierManager] w0 + {U} user carriers generated")
            print(f"  max|cos(w_u, w_v)|: {max_cos_uu:.2e}")
            print(f"  max|cos(w_u, w0)|:  {max_cos_w0:.2e}")

        return self.w0, carriers

    def _gram_schmidt(self, vectors: List[np.ndarray]) -> np.ndarray:
        result = []
        for v in vectors:
            v = v.copy().astype(np.float64)
            for u in result:
                v -= np.dot(v, u) * u
            norm = np.linalg.norm(v)
            if norm < 1e-10:
                rng = np.random.RandomState(len(result) + 9999)
                v = rng.randn(len(v))
                for u in result:
                    v -= np.dot(v, u) * u
                norm = np.linalg.norm(v)
            v /= norm
            result.append(v)
        return np.stack(result)

    def _user_seed(self, user_id: int) -> int:
        data = f"carrier_{self.carrier_seed}_{user_id}".encode()
        return int(hashlib.sha256(data).hexdigest()[:8], 16) % (2 ** 31)

    def _max_pairwise_cosine(self, carriers: np.ndarray) -> float:
        U = carriers.shape[0]
        m = 0.0
        for i in range(U):
            for j in range(i + 1, U):
                m = max(m, abs(float(np.dot(carriers[i], carriers[j]))))
        return m

    # ═══ 3. w_u 驱动选择 ═════════════════════════════════════

    def select_best_candidate(
        self,
        candidate_vectors: List[np.ndarray],
        user_id: int,
    ) -> Tuple[int, float, float]:
        """
        w_u 归属驱动选择（命题 10(c) 修正版）：

          i* = argmax_i ⟨λ_i − μ, w_u⟩

        仅优化用户归属信号。w₀ 投影仅记录，不参与选择。

        理由（基于实验 + 理论）：
          · w₀ 投影量级 >> w_u 投影量级（~10-100x）
          · 联合选择 score = proj(w₀) + proj(w_u) 被 w₀ 主导
          · w_u 信号被淹没 → 用户不可分
          · 仅优化 w_u → 全部选择压力集中于归属信号

        Returns:
            (best_idx, proj_wu, proj_w0)
        """
        if self.carriers is None:
            raise RuntimeError("Must call generate_carriers() first")

        u_u = self.carriers[user_id % len(self.carriers)]
        mu = self.mu_clean if self.mu_clean is not None else 0.0

        best_idx = 0
        best_score = -np.inf
        best_p0 = 0.0
        best_pu = 0.0

        for i, lam in enumerate(candidate_vectors):
            centered = np.asarray(lam, dtype=np.float64) - mu
            pu = float(np.dot(centered, u_u))
            if pu > best_score:
                best_score = pu
                best_idx = i
                best_pu = pu
                best_p0 = float(np.dot(centered, self.w0)) if self.w0 is not None else 0.0

        return best_idx, best_pu, best_p0

    # ═══ 4. 检测 ═════════════════════════════════════════════

    @staticmethod
    def cosine_p_value(cos_sim: float, dim: int) -> float:
        """
        Sablayrolles (ICML 2020) §3.1:
        P(cos ≥ τ | H₀) = ½ I_{1−τ²}((d−1)/2, 1/2)
        """
        if cos_sim <= 0:
            return 1.0
        tau_sq = min(cos_sim ** 2, 1.0 - 1e-15)
        a = (dim - 1) / 2.0
        b = 0.5
        return 0.5 * float(betainc(a, b, 1.0 - tau_sq))

    def _bootstrap_null_threshold(
        self,
        clean_vectors: np.ndarray,
        batch_size: int = 8,
        n_bootstrap: int = 1000,
        quantile: float = 0.99,
    ) -> float:
        """
        经验 H₀ 阈值（Efron 2004）。

        Bootstrap 干净数据的 max_u cos(batch_mean, w_u) 分布，
        取 quantile 分位数作为检测阈值。

        此方法不依赖分布假设，对 w_u 方向的非各向同性也鲁棒。
        """
        vecs = np.asarray(clean_vectors, dtype=np.float64)
        N = len(vecs)
        centered = vecs - self.mu_clean
        bs = min(batch_size, N)
        U = len(self.carriers)

        rng = np.random.default_rng(42)
        null_stats = []

        for _ in range(n_bootstrap):
            idx = rng.choice(N, size=bs, replace=True)
            batch_mean = centered[idx].mean(axis=0)
            batch_norm = np.linalg.norm(batch_mean)
            if batch_norm < 1e-12:
                null_stats.append(0.0)
                continue
            max_cos = max(
                float(np.dot(batch_mean, self.carriers[u])) / batch_norm
                for u in range(U)
            )
            null_stats.append(max_cos)

        return float(np.quantile(null_stats, quantile))

    def detect_two_stage(
        self,
        test_vectors: np.ndarray,
        alpha: float = 0.01,
        clean_baseline: Optional[np.ndarray] = None,
    ) -> Dict[str, object]:
        """
        w_u 归属驱动检测（命题 10(d) 修正版）：

        分析路径（Sablayrolles ICML 2020 + Bonferroni [Lehmann & Romano 2005]）：
          对每用户 u：p_u = cosine_p_value(cos(φ'_mean, w_u), d)
          Reject H₀ if min_u p_u < α / U

        经验路径（Efron 2004, 若提供 clean_baseline）：
          Bootstrap max_u cos from clean → threshold τ_emp
          Reject H₀ if max_u cos(φ'_mean, w_u) > τ_emp

        判定逻辑：
          若有 clean_baseline → 用经验阈值（更鲁棒）
          否则 → 用 Bonferroni 分析阈值

        Returns:
            dict with detection result and diagnostics
        """
        if self.carriers is None or self.mu_clean is None:
            raise RuntimeError("Not calibrated / carriers not generated")

        vecs = np.asarray(test_vectors, dtype=np.float64)
        d = vecs.shape[1]
        N = len(vecs)
        U = len(self.carriers)
        centered = vecs - self.mu_clean
        mean_cent = centered.mean(axis=0)
        mean_norm = np.linalg.norm(mean_cent)

        cos_w0 = float(np.dot(mean_cent, self.w0)) / (mean_norm + 1e-12) if self.w0 is not None else 0.0

        user_scores = {}
        best_user = 0
        best_cos = -np.inf
        min_p = 1.0

        for uid in range(U):
            cos_u = float(np.dot(mean_cent, self.carriers[uid])) / (mean_norm + 1e-12)
            p_u = self.cosine_p_value(cos_u, d)
            user_scores[uid] = {"cosine": cos_u, "p_value": p_u}
            if cos_u > best_cos:
                best_cos = cos_u
                best_user = uid
            min_p = min(min_p, p_u)

        bonferroni_wm = min_p < alpha / U
        emp_threshold = None

        if clean_baseline is not None and len(clean_baseline) >= 5:
            emp_threshold = self._bootstrap_null_threshold(
                clean_baseline, batch_size=N, n_bootstrap=1000, quantile=0.99,
            )
            is_wm = best_cos > emp_threshold
        else:
            is_wm = bonferroni_wm

        return {
            "is_watermarked": is_wm,
            "attributed_user": best_user if is_wm else None,
            "best_cos_wu": best_cos,
            "min_p_wu": min_p,
            "bonferroni_wm": bonferroni_wm,
            "cos_w0": cos_w0,
            "user_scores": user_scores,
            "empirical_threshold": emp_threshold,
        }

    # ═══ 5. LDA 精炼 ═════════════════════════════════════════

    def refine_carriers_lda(
        self,
        labeled_vectors: np.ndarray,
        labels: np.ndarray,
        verbose: bool = True,
    ) -> np.ndarray:
        """
        LDA 精炼（Fisher 1936）：用判别方向替换 PCA 载波。

        在 PCA 子空间内做 LDA（避免 d >> N 的奇异性）：
          1. 投影到 PCA 空间：X_pca = (X − μ) @ V_r^T  ∈ R^{N×r}
          2. 计算类间散度 S_b 和类内散度 S_w
          3. 求解 S_w^{-1} S_b v = λ v
          4. 取前 min(C−1, U) 个特征向量映射回原空间
          5. Gram-Schmidt 正交化 + 投影掉 w₀

        Args:
            labeled_vectors: (N, d) 拓扑向量
            labels: (N,) 用户 ID

        Returns:
            new carriers (U, d)
        """
        if not self._calibrated:
            raise RuntimeError("Must calibrate first")

        X = np.asarray(labeled_vectors, dtype=np.float64)
        y = np.asarray(labels).astype(int)
        N, d = X.shape

        X_pca = (X - self.mu_clean) @ self.components.T
        r = X_pca.shape[1]

        classes = np.unique(y)
        C = len(classes)
        mu_all = X_pca.mean(axis=0)

        S_b = np.zeros((r, r))
        S_w = np.zeros((r, r))

        for c in classes:
            X_c = X_pca[y == c]
            n_c = len(X_c)
            if n_c == 0:
                continue
            mu_c = X_c.mean(axis=0)
            diff_b = (mu_c - mu_all).reshape(-1, 1)
            S_b += n_c * (diff_b @ diff_b.T)
            X_c_cent = X_c - mu_c
            S_w += X_c_cent.T @ X_c_cent

        S_w += 1e-6 * np.eye(r)

        M = np.linalg.solve(S_w, S_b)
        eigenvalues, eigenvectors = np.linalg.eigh(M)

        idx_sorted = np.argsort(-eigenvalues)
        eigenvalues = eigenvalues[idx_sorted]
        eigenvectors = eigenvectors[:, idx_sorted]

        n_lda = min(C - 1, r)
        lda_dirs_pca = eigenvectors[:, :n_lda]
        lda_dirs_orig = self.components.T @ lda_dirs_pca

        U = len(self.carriers)
        raw_vectors = []
        for u in range(U):
            if u < n_lda:
                raw_vectors.append(lda_dirs_orig[:, u].copy())
            else:
                seed = self._user_seed(u) + 7777
                rng = np.random.RandomState(seed)
                coeffs = rng.randn(n_lda)
                raw_vectors.append((lda_dirs_orig @ coeffs).copy())

        new_carriers = self._gram_schmidt(raw_vectors)

        if self.w0 is not None:
            for i in range(len(new_carriers)):
                dot_w0 = np.dot(new_carriers[i], self.w0)
                new_carriers[i] -= dot_w0 * self.w0
                norm = np.linalg.norm(new_carriers[i])
                if norm > 1e-10:
                    new_carriers[i] /= norm

        old_carriers = self.carriers.copy()
        self.carriers = new_carriers

        if verbose:
            print(f"[CarrierManager] LDA refinement: {n_lda} discriminant directions")
            top_eig = eigenvalues[:min(5, len(eigenvalues))]
            print(f"  Top eigenvalues: {['%.3f' % e for e in top_eig]}")
            max_cos = self._max_pairwise_cosine(new_carriers)
            print(f"  max|cos(w_u, w_v)|: {max_cos:.2e}")

        return new_carriers

    # ═══ 6. 持久化 ═══════════════════════════════════════════

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {
            "mu_clean": self.mu_clean, "components": self.components,
            "explained_variance": self.explained_variance,
            "w0": self.w0, "carriers": self.carriers,
            "num_users": self.num_users, "n_components": self.n_components,
            "carrier_seed": self.carrier_seed,
        }
        np.savez_compressed(path, **{k: v for k, v in data.items() if v is not None})

    @classmethod
    def load(cls, path: str) -> "CarrierManager":
        data = np.load(path, allow_pickle=True)
        mgr = cls(
            num_users=int(data.get("num_users", 10)),
            n_components=int(data.get("n_components", 50)),
            carrier_seed=int(data.get("carrier_seed", 2025)),
        )
        if "mu_clean" in data:
            mgr.mu_clean = data["mu_clean"]
            mgr.components = data["components"]
            mgr.explained_variance = data["explained_variance"]
            mgr._calibrated = True
        if "w0" in data:
            mgr.w0 = data["w0"]
        if "carriers" in data:
            mgr.carriers = data["carriers"]
        return mgr
