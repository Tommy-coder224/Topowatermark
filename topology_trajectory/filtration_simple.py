"""
FiltrationBuilder：轨迹 → 供持久层使用的输入。
本模块实现「展平为点云」的简单方案；可扩展为 1D 时间函数、自定义点云等。
"""
from __future__ import annotations

import torch

from .interfaces import FiltrationBuilder


class FlattenTrajectoryFiltration(FiltrationBuilder):
    """
    将轨迹 [T+1, B, C, H, W] 展平为 [B, T+1, D]，D = C*H*W。
    作为「点云」时，每步 latent 为一个点（B 个 batch 各自一条轨迹）。
    """

    def build(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Args:
            trajectory: [T+1, B, C, H, W]
        Returns:
            [B, T+1, D]，D = C*H*W，可传入 PersistenceLayer（如 SimplePersistenceProxy）
        """
        T1, B, C, H, W = trajectory.shape
        x = trajectory.permute(1, 0, 2, 3, 4).reshape(B, T1, -1)
        return x
