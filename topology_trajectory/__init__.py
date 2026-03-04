"""
拓扑引导轨迹水印：水印编在轨迹的持久表示 D̃(z) 中。
理论见 THEORY_TOPOLOGY_GUIDED_TRAJECTORY_WATERMARK.md；
实验顺序与设计见 TOPOLOGY_TRAJECTORY_EXPERIMENT_DESIGN.md。
"""
from .interfaces import (
    TrajectoryGenerator,
    FiltrationBuilder,
    PersistenceLayer,
    Embedder,
    Detector,
)
from .config import TopologyTrajectoryConfig

__all__ = [
    "TrajectoryGenerator",
    "FiltrationBuilder",
    "PersistenceLayer",
    "Embedder",
    "Detector",
    "TopologyTrajectoryConfig",
]
