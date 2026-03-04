"""
持久图向量化：从持久图到固定维度特征
======================================

本模块将 GUDHI 输出的持久图 {(b_i, d_i)} 转化为固定维度向量，
供后续拓扑-角度耦合与假设检验使用。

核心定理（审稿人可手推）
--------------------------

**定理 C (Adams et al., JMLR 2017 — 持久图像稳定性):**

  设 PI_σ: Dgm → R^{r²} 为带宽为 σ 的持久图像映射，
  权重函数 w(b,d) = (d−b)^p，则

      ‖PI_σ(D₁) − PI_σ(D₂)‖₂ ≤ L_σ · W₁(D₁, D₂)

  其中 L_σ 是依赖于 σ, r, w 的 Lipschitz 常数。

  直觉：持久图的小扰动只会导致持久图像的小变化。
  这保证了拓扑向量化对噪声/VAE重建误差的鲁棒性。

**推论（本文定理 D 的关键引理）:**

  若 VAE 编码-解码满足 ‖x − Dec(Enc(x))‖_∞ ≤ ε_VAE，
  则由定理 B + 定理 C：

      ‖PI(x) − PI(Dec(Enc(x)))‖₂ ≤ L_σ · C_p · ε_VAE

  即拓扑向量在 VAE round-trip 下的变化有界。

参考文献：
  [1] Adams, Emerson, Kirby, et al.
      "Persistence images: a stable vector representation of persistent
       homology." JMLR 18(8):1–35, 2017.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .gudhi_persistence import channel_persistence


# ── 标量拓扑不变量 ─────────────────────────────────────────


def total_persistence(dgm: np.ndarray, p: float = 1.0) -> float:
    """
    总持久性：TP_p(D) = Σ_i (d_i − b_i)^p

    衡量持久图中所有特征的"生命期"之和，
    p=1 时即 L^1 总持久性，p=2 时即 L^2。
    """
    if dgm.shape[0] == 0:
        return 0.0
    lifetimes = dgm[:, 1] - dgm[:, 0]
    return float(np.sum(np.power(np.abs(lifetimes), p)))


def persistence_entropy(dgm: np.ndarray) -> float:
    """
    持久熵：PE(D) = − Σ_i p_i log(p_i)

    其中 p_i = ℓ_i / TP₁，ℓ_i = d_i − b_i。
    衡量拓扑特征分布的均匀程度。
    """
    if dgm.shape[0] == 0:
        return 0.0
    lifetimes = dgm[:, 1] - dgm[:, 0]
    lifetimes = np.abs(lifetimes)
    total = lifetimes.sum()
    if total < 1e-12:
        return 0.0
    probs = lifetimes / total
    probs = probs[probs > 1e-15]
    return float(-np.sum(probs * np.log(probs)))


# ── 持久图像（Persistence Image） ─────────────────────────


def _birth_death_to_birth_persistence(dgm: np.ndarray) -> np.ndarray:
    """(b, d) → (b, d−b)，转到 birth-persistence 坐标。"""
    if dgm.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64)
    bp = np.column_stack([dgm[:, 0], dgm[:, 1] - dgm[:, 0]])
    return bp


def persistence_image_vector(
    dgm: np.ndarray,
    resolution: int = 20,
    bandwidth: float = 0.05,
    weight_power: float = 1.0,
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """
    将单个持久图转化为持久图像向量 PI ∈ R^{r²}。

    算法（Adams et al. 2017 定义 4）：
      1. 将 (b_i, d_i) 转到 birth-persistence 坐标 (b_i, ℓ_i)
      2. 权重：w_i = ℓ_i^p （短命特征权重低）
      3. 对每个网格点 (x_j, y_j)：
           ρ(x_j, y_j) = Σ_i w_i · φ_σ(x_j − b_i, y_j − ℓ_i)
         其中 φ_σ 是标准差为 σ 的高斯核。
      4. 展平为向量 PI ∈ R^{r²}。

    Args:
        dgm:          shape (n, 2)，(birth, death) 对
        resolution:   网格分辨率 r
        bandwidth:    高斯核标准差 σ
        weight_power: 权重指数 p
        x_range:      birth 轴范围；None 则自动
        y_range:      persistence 轴范围；None 则自动

    Returns:
        shape (r*r,) 的向量
    """
    bp = _birth_death_to_birth_persistence(dgm)
    r = resolution
    img = np.zeros((r, r), dtype=np.float64)

    if bp.shape[0] == 0:
        return img.flatten()

    births = bp[:, 0]
    pers = bp[:, 1]

    if x_range is None:
        x_min = births.min() - 3 * bandwidth
        x_max = births.max() + 3 * bandwidth
    else:
        x_min, x_max = x_range
    if y_range is None:
        y_min = max(0, pers.min() - 3 * bandwidth)
        y_max = pers.max() + 3 * bandwidth
    else:
        y_min, y_max = y_range

    if x_max - x_min < 1e-10:
        x_max = x_min + 1.0
    if y_max - y_min < 1e-10:
        y_max = y_min + 1.0

    xs = np.linspace(x_min, x_max, r)
    ys = np.linspace(y_min, y_max, r)
    X, Y = np.meshgrid(xs, ys)

    weights = np.power(np.abs(pers), weight_power)
    inv_2sigma2 = 1.0 / (2.0 * bandwidth * bandwidth)

    for k in range(bp.shape[0]):
        dx = X - births[k]
        dy = Y - pers[k]
        gauss = np.exp(-(dx * dx + dy * dy) * inv_2sigma2)
        img += weights[k] * gauss

    return img.flatten()


# ── 完整拓扑签名 ──────────────────────────────────────────


@dataclass
class TopoSignature:
    """
    单个样本的完整拓扑签名。

    包含：
      vector:       PI 向量拼接，shape (2*C*r², )
      tp_per_ch:    逐通道总持久性 [TP_0(c), TP_1(c)] × C
      entropy:      逐通道持久熵
      raw_diagrams: 原始持久图（用于精确距离计算）
    """
    vector: np.ndarray
    tp_per_channel: np.ndarray
    entropy_per_channel: np.ndarray
    raw_diagrams: List[Dict[int, np.ndarray]]


def extract_topo_signature(
    tensor_chw: np.ndarray,
    max_dim: int = 1,
    pi_resolution: int = 20,
    pi_bandwidth: float = 0.05,
    pi_weight_power: float = 1.0,
) -> TopoSignature:
    """
    对 C×H×W 张量提取完整拓扑签名。

    流程：
      1. 逐通道 cubical PH (GUDHI)
      2. 逐通道逐维度持久图像向量化
      3. 计算标量不变量（总持久性、持久熵）
      4. 拼接为统一向量

    Args:
        tensor_chw: shape (C, H, W)
        max_dim:    PH 最高维度
        pi_resolution:   持久图像分辨率
        pi_bandwidth:    高斯带宽
        pi_weight_power: 权重指数

    Returns:
        TopoSignature
    """
    arr = np.asarray(tensor_chw, dtype=np.float64)
    C = arr.shape[0]

    dgms_list = channel_persistence(arr, max_dim=max_dim)

    pi_vectors = []
    tp_values = []
    ent_values = []

    for c in range(C):
        dgms_c = dgms_list[c]
        for dim in range(max_dim + 1):
            dgm = dgms_c.get(dim, np.empty((0, 2)))
            pi = persistence_image_vector(
                dgm,
                resolution=pi_resolution,
                bandwidth=pi_bandwidth,
                weight_power=pi_weight_power,
            )
            pi_vectors.append(pi)
            tp_values.append(total_persistence(dgm, p=1.0))
            ent_values.append(persistence_entropy(dgm))

    return TopoSignature(
        vector=np.concatenate(pi_vectors),
        tp_per_channel=np.array(tp_values, dtype=np.float64),
        entropy_per_channel=np.array(ent_values, dtype=np.float64),
        raw_diagrams=dgms_list,
    )


def signature_from_diagrams(
    dgms_list: List[Dict[int, np.ndarray]],
    max_dim: int = 1,
    pi_resolution: int = 20,
    pi_bandwidth: float = 0.05,
    pi_weight_power: float = 1.0,
) -> TopoSignature:
    """
    从已计算好的持久图列表构建 TopoSignature，不再调用 channel_persistence。

    用于避免重复计算：当已有 dgms（如 stability 中先算了 d_B/W）时，
    用本函数直接得到 PI 向量，无需再次做 cubical persistence。
    """
    pi_vectors = []
    tp_values = []
    ent_values = []
    for dgms_c in dgms_list:
        for dim in range(max_dim + 1):
            dgm = dgms_c.get(dim, np.empty((0, 2)))
            pi = persistence_image_vector(
                dgm,
                resolution=pi_resolution,
                bandwidth=pi_bandwidth,
                weight_power=pi_weight_power,
            )
            pi_vectors.append(pi)
            tp_values.append(total_persistence(dgm, p=1.0))
            ent_values.append(persistence_entropy(dgm))
    return TopoSignature(
        vector=np.concatenate(pi_vectors),
        tp_per_channel=np.array(tp_values, dtype=np.float64),
        entropy_per_channel=np.array(ent_values, dtype=np.float64),
        raw_diagrams=dgms_list,
    )


def batch_extract_topo_signatures(
    tensor_bchw: np.ndarray,
    max_dim: int = 1,
    pi_resolution: int = 20,
    pi_bandwidth: float = 0.05,
    pi_weight_power: float = 1.0,
) -> List[TopoSignature]:
    """批量版：对 B×C×H×W 张量逐样本提取拓扑签名。"""
    arr = np.asarray(tensor_bchw, dtype=np.float64)
    return [
        extract_topo_signature(
            arr[b], max_dim=max_dim,
            pi_resolution=pi_resolution,
            pi_bandwidth=pi_bandwidth,
            pi_weight_power=pi_weight_power,
        )
        for b in range(arr.shape[0])
    ]
