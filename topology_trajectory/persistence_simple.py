"""
可微持久表示：Phase 1 用简单可微代理，保证梯度存在、损失可下降。
后续可替换为 TopologyLayer / PersLay / 持久图像 等真实可微 TDA 层。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .interfaces import PersistenceLayer


class SimplePersistenceProxy(nn.Module, PersistenceLayer):
    """
    可微代理：对轨迹的展平表示做线性投影 + 池化 → R^m。
    不计算真实持久同调，但梯度存在，用于验证「拓扑引导能训练」；
    替换为真实可微持久层后接口不变。
    """

    def __init__(self, input_dim: int, repr_dim: int = 64):
        super().__init__()
        self._repr_dim = repr_dim
        # 输入 [B, T+1, D] -> 先池化再线性，或先线性再池化
        self.proj = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, repr_dim),
        )

    @property
    def repr_dim(self) -> int:
        return self._repr_dim

    def forward(self, filtration_input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            filtration_input: [B, T+1, D] 来自 FlattenTrajectoryFiltration
        Returns:
            [B, repr_dim]
        """
        # 沿时间维 dim=1 取 mean + std，再投影
        x = filtration_input
        mean = x.mean(dim=1)
        if x.size(1) > 1:
            std = x.std(dim=1, unbiased=False).clamp(min=1e-6)
        else:
            std = torch.zeros_like(mean)
        feat = torch.cat([mean, std], dim=-1)
        return self.proj(feat)
