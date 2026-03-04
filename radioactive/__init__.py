"""
TopoRad — 拓扑放射性水印框架
=================================

首个在潜扩散模型（LDM）中实现放射性水印的理论驱动框架。

核心思想：
  现有水印（Tree-Ring, Stable Signature 等）在 LDM 中丧失放射性，
  因为 VAE autoencoder 过滤高频细节（Dubiński et al., ICLR 2026）。
  我们将水印耦合到持续同调（persistent homology）提取的拓扑特征上——
  即图像的语义骨架。VAE 必须保留语义 → 拓扑特征存活 →
  下游微调模型继承拓扑分布 → 水印具有放射性。

理论链（审稿人可手推）：
  定理 A  Anderson (2003) 球面-卡方分解 → 严格无损
  定理 B  Cohen-Steiner et al. (2007) 持续同调稳定性 → VAE 下拓扑保持
  定理 C  Adams et al. (2017) 持久图像 Lipschitz 稳定性 → 向量化保真
  定理 D  [本文] 拓扑放射性 → 微调模型继承拓扑分布
  定理 E  [本文] 检测保证 → Neyman-Pearson 假设检验
  定理 F  Sablayrolles et al. (2020) 载波余弦检验 → 精确 p-value (incomplete beta)
  命题 10 [本文] 双载波 (Dual Carrier):
    (a) w₀ = 第一主成分 → 公共零位检测（FPR 与用户数 U 无关）
    (b) w_u ⊥ w₀ → 用户归属（Gram-Schmidt 正交化）
    (c) 选择 argmax [proj(w₀) + proj(w_u)] → 同时推检测+归属信号
    (d) 两阶段检测：先 w₀ 判水印，再 w_u 判用户

参考文献：
  [1] Sablayrolles et al. "Radioactive data: tracing through training." ICML 2020.
  [2] Dubiński et al. "Are Watermarks For Diffusion Models Radioactive?" ICLR 2026.
  [3] HMark: arXiv 2512.00094 — h-space 语义水印
  [4] Cohen-Steiner, Edelsbrunner, Harer. "Stability of persistence diagrams." DCG 2007.
  [5] Adams et al. "Persistence images: a stable vector representation." JMLR 2017.
"""

from .config import RadioactiveConfig

__all__ = ["RadioactiveConfig"]
__version__ = "0.1.0"
