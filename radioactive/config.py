"""
TopoRad 全局配置
================

所有超参数、路径、实验设置集中管理，便于复现与消融。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _get_model_dir() -> str:
    for name in ["stable-diffusion-v1-4", "stable-diffusion-v1-5-main"]:
        p = os.path.join(_PROJECT_ROOT, "model", name)
        if os.path.exists(p) and os.path.exists(os.path.join(p, "vae")):
            return p
    return "runwayml/stable-diffusion-v1-5"


@dataclass
class RadioactiveConfig:
    """拓扑放射性水印框架的完整配置。"""

    # ── 模型 ──────────────────────────────────────────────
    model_id: str = field(default_factory=_get_model_dir)
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    image_size: int = 512

    # ── 用户与球形编码 ────────────────────────────────────
    num_users: int = 10
    embed_dim: int = 64

    # ── 拓扑（GUDHI） ────────────────────────────────────
    ph_max_dim: int = 1                     # 持续同调最高维度：H_0 + H_1
    pi_resolution: int = 20                 # 持久图像分辨率 r×r
    pi_bandwidth: float = 0.05              # 持久图像高斯带宽 σ
    pi_weight_power: float = 1.0            # 持久图像权重 w(b,d) = (d-b)^p

    # ── 拓扑-角度耦合 ────────────────────────────────────
    coupling_strength: float = 0.3          # 拓扑调制强度 η ∈ [0,1]
    coupling_temperature: float = 50.0       # softmax 温度（通道权重，值越大越均匀）
    coupling_hash_seed: int = 2025          # 确定性哈希基础种子
    max_topo_dist_fallback: Optional[float] = None

    # ── 载波 (Carrier) — Sablayrolles 式拓扑流形载波 ──────
    use_carrier: bool = True                 # 是否使用载波选择（False 则退回旧逻辑）
    carrier_n_components: int = 50           # PCA 保留主成分数 r（需 ≥ num_users）
    carrier_seed: int = 2025                 # 载波生成种子
    carrier_n_calib: int = 30                # 校准用干净图像数量（benchmark 内自动生成）

    # ── 放射性实验 ────────────────────────────────────────
    num_watermarked_images: int = 1000      # 水印图像数量
    num_test_images: int = 500              # 检测用生成图像数
    radioactive_candidates_k: int = 8       # 拓扑选择候选数 K
    detection_fpr: float = 0.01             # 目标虚警率 α

    # ── LoRA 微调 ─────────────────────────────────────────
    lora_rank: int = 8
    lora_alpha: int = 8
    lora_lr: float = 1e-4
    lora_epochs: int = 50
    lora_batch_size: int = 2
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["to_q", "to_k", "to_v", "to_out.0"]
    )

    # ── 训练提示词 ────────────────────────────────────────
    prompts: List[str] = field(default_factory=lambda: [
        "a photo of a cat sitting on a windowsill",
        "a beautiful sunset over the ocean",
        "a portrait of a young woman, studio lighting",
        "a futuristic cityscape at night",
        "a bowl of fresh fruit on a wooden table",
        "a snowy mountain landscape with pine trees",
        "an astronaut riding a horse on mars",
        "a cozy cabin in the woods during autumn",
        "a detailed painting of a medieval castle",
        "a close-up photo of colorful flowers",
    ])

    # ── 设备与复现 ────────────────────────────────────────
    device: str = "cuda"
    seed: int = 42
    output_dir: str = field(
        default_factory=lambda: os.path.join(_PROJECT_ROOT, "output_radioactive")
    )

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)
