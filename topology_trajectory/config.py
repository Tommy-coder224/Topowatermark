"""
配置：便于复现与消融；单文件对应一次实验。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TopologyTrajectoryConfig:
    """拓扑引导轨迹水印实验配置。"""

    # 扩散
    model_id: str = "runwayml/stable-diffusion-v1-5"
    num_inference_steps: int = 50
    use_ddim: bool = True  # 确定性轨迹，便于可复现

    # 轨迹 → 表示
    trajectory_repr: str = "flatten"  # "flatten" | "pointcloud" | "time_series"
    persistence_layer: str = "simple"  # "simple" | "topologylayer" | "perslay"
    repr_dim: int = 64  # m

    # 优化（Phase 1）
    embed_steps: int = 500
    embed_lr: float = 1e-2

    # 复现
    seed: Optional[int] = 42

    # 设备
    device: str = "cuda"

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "num_inference_steps": self.num_inference_steps,
            "use_ddim": self.use_ddim,
            "trajectory_repr": self.trajectory_repr,
            "persistence_layer": self.persistence_layer,
            "repr_dim": self.repr_dim,
            "embed_steps": self.embed_steps,
            "embed_lr": self.embed_lr,
            "seed": self.seed,
            "device": self.device,
        }
