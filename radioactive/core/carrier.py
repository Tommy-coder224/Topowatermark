"""
载波管理器（CarrierManager）— per-image t-test 检测
=====================================================

核心改进（相对于 cosine 版本）：
  旧版：cos(φ'_mean, w_u) / ‖φ'_mean‖  → 被 ‖φ'_mean‖ 稀释信号
  新版：t_u = mean(⟨φ_i − μ, w_u⟩) / SE  → 标准 t 检验，√N 放大

理论基础：
═══════════════════════════════════════════════════════════

**定理 F' (Neyman-Pearson t-检验，本文修正):**

  对 N 张测试图像 {x_i}，定义逐图投影：
    p_i^u = ⟨λ(x_i) − μ_clean, w_u⟩

  H₀：E[p_i^u] = 0  （图像拓扑与 w_u 无关）
  H₁：E[p_i^u] > 0  （图像拓扑沿 w_u 偏移）

  检验统计量：t_u = mean(p_i^u) / (std(p_i^u) / √N)

  在 H₀ 下：t_u ~ t(N−1) ≈ N(0,1)（CLT, Lehmann & Romano 2005 §5.1）
  在 H₁ 下：t_u >> 0，功效 ∝ √N · (信号/噪声)

  用户归属：û = argmax_u t_u
  检测判据：max_u t_u > τ_emp（bootstrap 经验阈值）
  或 Bonferroni: min_u p_u^{t-test} < α/U

**优势（相对 cosine 检测）：**
  · 不除以 ‖φ'_mean‖ → 避免高维均值范数稀释信号
  · 功效 ∝ √N → 更多图像 = 更强检测力
  · t-test 对非正态投影分布鲁棒（CLT）

**命题 10 (本文):**
  (a)-(c) 同前（w₀ + w_u 生成 + Gram-Schmidt + 选择）
  (d) 检测改用逐图 t-test（见上）

参考文献：
  [1] Sablayrolles et al. "Radioactive data." ICML 2020.
  [2] Lehmann & Romano. "Testing Statistical Hypotheses." Springer 2005.
  [3] Efron. "Large-Scale Simultaneous Hypothesis Testing." JASA 2004.
  [4] Fisher. "The Use of Multiple Measurements." Ann. Eugenics 1936.
  [5] Jolliffe. "Principal Component Analysis." Springer 2002.
"""
from __future__ import annotations

import hashlib
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.special import betainc
from scipy import stats as sp_stats


class CarrierManager:
    """
    载波管理器：PCA 校准 → 正交化 w_u → per-image t-test 检测。
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
        self.boot_quantile: float = 0.95

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
        命题 10(a)(b)：w₀ + 正交化 w_u。
        """
        if not self._calibrated:
            raise RuntimeError("Must call calibrate() first")

        U = num_users or self.num_users
        r = self.components.shape[0]

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
        w_u 归属驱动选择（命题 10(c)）：

          i* = argmax_i ⟨λ_i − μ, w_u⟩

        Returns: (best_idx, proj_wu, proj_w0)
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

    # ═══ 4. Per-image t-test 检测 ═════════════════════════════

    @staticmethod
    def cosine_p_value(cos_sim: float, dim: int) -> float:
        """Sablayrolles (ICML 2020) §3.1: P(cos >= tau | H0)."""
        if cos_sim <= 0:
            return 1.0
        tau_sq = min(cos_sim ** 2, 1.0 - 1e-15)
        a = (dim - 1) / 2.0
        b = 0.5
        return 0.5 * float(betainc(a, b, 1.0 - tau_sq))

    def _bootstrap_null_threshold_t(
        self,
        clean_vectors: np.ndarray,
        batch_size: int = 50,
        n_bootstrap: int = 2000,
        quantile: float = 0.95,
    ) -> float:
        """
        经验 H₀ 阈值 — 基于 per-image t-test（Efron 2004）。

        Bootstrap 干净数据的 max_u t_u 分布，取 quantile 分位数。

        与 cosine 版本的区别：
          cosine: max_u cos(batch_mean, w_u)    → 被 ||mean|| 稀释
          t-test: max_u mean(proj) / SE(proj)   → 标准化，不受 ||mean|| 影响
        """
        vecs = np.asarray(clean_vectors, dtype=np.float64)
        N_clean = len(vecs)
        centered = vecs - self.mu_clean
        bs = min(batch_size, N_clean)
        U = len(self.carriers)

        carrier_matrix = np.stack(self.carriers)  # (U, d)

        rng = np.random.default_rng(42)
        null_stats = []

        for _ in range(n_bootstrap):
            idx = rng.choice(N_clean, size=bs, replace=True)
            batch = centered[idx]  # (bs, d)
            all_proj = batch @ carrier_matrix.T  # (bs, U)

            max_t = -np.inf
            for u in range(U):
                proj_u = all_proj[:, u]
                mean_p = proj_u.mean()
                std_p = proj_u.std(ddof=1) if bs > 1 else 1.0
                se = std_p / np.sqrt(bs)
                t_val = mean_p / (se + 1e-12)
                if t_val > max_t:
                    max_t = t_val
            null_stats.append(max_t)

        return float(np.quantile(null_stats, quantile))

    def detect_two_stage(
        self,
        test_vectors: np.ndarray,
        alpha: float = 0.01,
        clean_baseline: Optional[np.ndarray] = None,
    ) -> Dict[str, object]:
        """
        Per-image t-test 检测（定理 F' + Bonferroni + 经验 H₀）。

        核心改进：用逐图投影的 t 检验替代 batch cosine。

        对每用户 u：
          proj_i^u = ⟨φ_i − μ, w_u⟩  （逐图投影）
          t_u = mean(proj_i^u) / SE(proj_i^u)  （t 统计量）

        判据（经验 H₀）：max_u t_u > τ_emp  [Efron 2004]
        判据（Bonferroni）：min_u p_u < α/U  [Lehmann & Romano 2005]
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

        carrier_matrix = np.stack(self.carriers)  # (U, d)
        all_proj = centered @ carrier_matrix.T   # (N, U)

        user_scores = {}
        best_user = 0
        best_t = -np.inf
        best_cos = -np.inf
        min_p_sab = 1.0

        for uid in range(U):
            proj_u = all_proj[:, uid]
            mean_proj = float(proj_u.mean())
            std_proj = float(proj_u.std(ddof=1)) if N > 1 else 1.0
            se = std_proj / np.sqrt(N)
            t_u = mean_proj / (se + 1e-12)
            p_t = float(1.0 - sp_stats.t.cdf(t_u, N - 1)) if N > 1 else 1.0

            cos_u = float(np.dot(mean_cent, self.carriers[uid])) / (mean_norm + 1e-12)
            p_sab = self.cosine_p_value(cos_u, d)

            user_scores[uid] = {
                "cosine": cos_u, "p_sablayrolles": p_sab,
                "t_stat": t_u, "p_ttest": p_t,
                "mean_proj": mean_proj, "se": se,
            }
            if t_u > best_t:
                best_t = t_u
                best_user = uid
            best_cos = max(best_cos, cos_u)
            min_p_sab = min(min_p_sab, p_sab)

        bonferroni_t_wm = any(
            s["p_ttest"] < alpha / U for s in user_scores.values()
        )

        emp_threshold = None
        if clean_baseline is not None and len(clean_baseline) >= 5:
            emp_threshold = self._bootstrap_null_threshold_t(
                clean_baseline,
                batch_size=N,
                n_bootstrap=2000,
                quantile=self.boot_quantile,
            )
            is_wm = best_t > emp_threshold
        else:
            is_wm = bonferroni_t_wm

        return {
            "is_watermarked": is_wm,
            "attributed_user": best_user if is_wm else None,
            "best_cos_wu": best_cos,
            "best_t_wu": best_t,
            "min_p_wu": min_p_sab,
            "bonferroni_t_wm": bonferroni_t_wm,
            "cos_w0": cos_w0,
            "user_scores": user_scores,
            "empirical_threshold": emp_threshold,
        }

    # ═══ 5. 两层检测协议 ═══════════════════════════════════════

    def detect_model_level(
        self,
        test_vectors: np.ndarray,
        alpha: float = 0.01,
        clean_baseline: Optional[np.ndarray] = None,
    ) -> Dict[str, object]:
        """
        Layer 1: Model Detection — "Was this model trained on watermarked data?"

        Protocol (Theorem F', Layer 1):
          T_model = max_u t_u
          Under H0: T_model follows the maximum of U independent t(N-1) distributions
          Under H1: at least one t_u >> 0 ⟹ T_model >> threshold
          Decision: T_model > τ_emp → model is watermarked

        Reference: Sablayrolles ICML 2020 §3 + Bonferroni (Lehmann & Romano 2005 §9.1)
        """
        result = self.detect_two_stage(test_vectors, alpha, clean_baseline)
        min_p = min(s["p_ttest"] for s in result["user_scores"].values())
        return {
            "is_watermarked": result["is_watermarked"],
            "test_statistic": result["best_t_wu"],
            "threshold": result.get("empirical_threshold"),
            "p_value": min_p,
        }

    def detect_user_level(
        self,
        test_vectors: np.ndarray,
        alpha: float = 0.01,
        clean_baseline: Optional[np.ndarray] = None,
    ) -> Dict[str, object]:
        """
        Layer 2: User Attribution — "Which user's watermarked data was used?"

        Protocol (Theorem F', Layer 2):
          û = argmax_u t_u
          Under H1 with true user u*: E[t_{u*}] >> 0, E[t_v] ≈ 0 for v ≠ u*
          ⟹ argmax_u t_u = u* with high probability
          Confidence = t_{û} − max_{v≠û} t_v (decision margin)

        Reference: Fisher 1936 (LDA principle applied to carrier space)
        """
        result = self.detect_two_stage(test_vectors, alpha, clean_baseline)
        user_t = {uid: s["t_stat"] for uid, s in result["user_scores"].items()}
        best = result["attributed_user"]
        if best is not None:
            second_best = max(t for uid, t in user_t.items() if uid != best)
            margin = user_t[best] - second_best
        else:
            margin = 0.0
        return {
            "attributed_user": result["attributed_user"],
            "user_t_stats": user_t,
            "confidence": result["best_t_wu"],
            "margin": margin,
        }

    # ═══ 6. LDA 精炼 ═════════════════════════════════════════

    def refine_carriers_lda(
        self,
        labeled_vectors: np.ndarray,
        labels: np.ndarray,
        verbose: bool = True,
    ) -> np.ndarray:
        """
        LDA 精炼（Fisher 1936）：用判别方向替换 PCA 载波。
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

        self.carriers = new_carriers

        if verbose:
            print(f"[CarrierManager] LDA refinement: {n_lda} discriminant directions")
            top_eig = eigenvalues[:min(5, len(eigenvalues))]
            print(f"  Top eigenvalues: {['%.3f' % e for e in top_eig]}")
            max_cos = self._max_pairwise_cosine(new_carriers)
            print(f"  max|cos(w_u, w_v)|: {max_cos:.2e}")

        return new_carriers

    # ═══ 7. 持久化 ═══════════════════════════════════════════

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
