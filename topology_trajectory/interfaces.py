"""
拓扑引导轨迹水印：抽象接口。
其他研究者可替换实现（不同扩散模型、不同 filtration、不同可微持久层）而不改上层逻辑。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import torch


class TrajectoryGenerator(ABC):
    """z → 轨迹 γ(z)。可替换：DDIM/DDPM、不同步数、不同模型。"""

    @abstractmethod
    def generate(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: 噪声参数，形状 [B, ...]（可为初始噪声或整条 z_1:T）
        Returns:
            trajectory: [T+1, B, D] 或 [T+1, B, C, H, W]，依实现而定
        """
        pass

    @property
    @abstractmethod
    def trajectory_shape(self) -> Tuple[int, ...]:
        """单步轨迹的形状，例如 (B, D) 或 (B, C, H, W)。"""
        pass


class FiltrationBuilder(ABC):
    """轨迹 γ → 供持久层使用的输入（点云、函数值等）。可替换：点云 / 1D 函数 / 自定义。"""

    @abstractmethod
    def build(self, trajectory: torch.Tensor) -> Any:
        """
        Args:
            trajectory: [T+1, B, ...] 来自 TrajectoryGenerator
        Returns:
            filtration_input: 供 PersistenceLayer.forward 使用的格式（实现自定）
        """
        pass


class PersistenceLayer(ABC):
    """filtration 输入 → 可微持久表示 D̃ ∈ R^m。可替换：TopologyLayer / PersLay / 持久图像 / 自定义。"""

    @property
    @abstractmethod
    def repr_dim(self) -> int:
        """表示维度 m。"""
        pass

    @abstractmethod
    def forward(self, filtration_input: Any) -> torch.Tensor:
        """
        Args:
            filtration_input: 来自 FiltrationBuilder.build
        Returns:
            repr: [B, m]，可微
        """
        pass


class Embedder(ABC):
    """给定目标 λ_target，优化 z 使 D̃(z) 接近 λ_target。"""

    @abstractmethod
    def embed(
        self,
        target_lambda: torch.Tensor,
        num_steps: int = 500,
        lr: float = 1e-2,
        z_init: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            target_lambda: [B, m] 或 [m]
            num_steps: 优化步数
            lr: 学习率
            z_init: 初始 z；None 则随机
        Returns:
            z_opt: 优化后的 z
            info: 含 loss_history, final_loss, D̃(z_opt) 等
        """
        pass


class Detector(ABC):
    """给定 z 或图像，计算表示并判断是否在 Λ_u 内。"""

    @abstractmethod
    def compute_representation(self, z_or_image: torch.Tensor) -> torch.Tensor:
        """返回 D̃，形状 [B, m]。"""
        pass

    def in_region(
        self,
        repr: torch.Tensor,
        lambda_u: torch.Tensor,
        rho: float,
    ) -> torch.Tensor:
        """repr 是否在 Λ_u = { λ : ‖λ − λ_u‖ ≤ ρ } 内。"""
        dist = (repr - lambda_u).norm(dim=-1)
        return dist <= rho
