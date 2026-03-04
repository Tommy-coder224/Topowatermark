"""
Phase 1：验证「拓扑引导能训练」——最小化 ‖D̃(z) − λ_target‖²，看损失是否下降。
用法：
  # 干跑（不加载 SD，用假轨迹验证梯度与流程）
  python -m topology_trajectory.run_phase1 --dry_run --steps 200

  # 真实运行（需 diffusers + 模型）
  python -m topology_trajectory.run_phase1 --steps 500 --lr 1e-2
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from .config import TopologyTrajectoryConfig
from .embedder import TopologyGuidedEmbedder
from .filtration_simple import FlattenTrajectoryFiltration
from .interfaces import FiltrationBuilder, PersistenceLayer, TrajectoryGenerator
from .persistence_simple import SimplePersistenceProxy


def get_dry_run_components(repr_dim: int, batch_size: int = 1, num_steps: int = 10, device: torch.device = None) -> tuple:
    """不加载扩散模型：用线性变换模拟 z→轨迹→可微表示，用于验证优化流程与梯度。"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T = num_steps
    D = 64

    class DryRunTrajectoryGenerator(TrajectoryGenerator):
        def __init__(self):
            self._shape = (batch_size, D)
            self.num_steps = T

        @property
        def trajectory_shape(self):
            return self._shape

        def generate(self, z: torch.Tensor) -> torch.Tensor:
            # 假轨迹 [T+1, B, D]：随 t 与 z 变化，可微
            t_coef = torch.arange(T + 1, dtype=z.dtype, device=z.device).view(-1, 1, 1)
            return z.unsqueeze(0) + 0.01 * t_coef * z.unsqueeze(0)

    class DryRunFiltration(FiltrationBuilder):
        def build(self, trajectory: torch.Tensor) -> torch.Tensor:
            # 轨迹 [T+1, B, D] -> 持久层期望 [B, T+1, D]
            return trajectory.permute(1, 0, 2)

    # SimplePersistenceProxy 输入为 [B, T+1, D]，输出 [B, repr_dim]；其内部 concat mean+std -> 2*D
    persistence = SimplePersistenceProxy(input_dim=2 * D, repr_dim=repr_dim).to(device)

    gen = DryRunTrajectoryGenerator()
    filt = DryRunFiltration()
    return gen, filt, persistence


def _resolve_model_path(model_id: str) -> str:
    """若 model_id 为本地存在的目录则返回其绝对路径，否则返回原字符串（HF id）。"""
    if not model_id:
        return model_id
    for path in (model_id, os.path.join(os.getcwd(), model_id)):
        if os.path.isdir(path):
            # 确认为 diffusers 格式（有 config.json 或 model_index.json）
            if os.path.isfile(os.path.join(path, "model_index.json")) or os.path.isfile(
                os.path.join(path, "unet", "config.json")
            ):
                return os.path.abspath(path)
    return model_id


def get_real_components(
    cfg: TopologyTrajectoryConfig,
    height: int = 512,
    width: int = 512,
    num_inference_steps: int = 20,
    use_fp16: bool = True,
) -> tuple:
    """加载 diffusers pipeline 与真实 TrajectoryGenerator + Filtration + Persistence。"""
    from .trajectory_diffusers import DiffusersTrajectoryGenerator

    try:
        from diffusers import StableDiffusionPipeline
    except ImportError:
        raise ImportError("请安装 diffusers: pip install diffusers")

    load_path = _resolve_model_path(cfg.model_id)
    is_local = os.path.isdir(load_path)
    if is_local:
        print(f"从本地加载模型: {load_path}")
    dtype = torch.float16 if (use_fp16 and torch.cuda.is_available()) else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        load_path,
        safety_checker=None,
        torch_dtype=dtype,
        local_files_only=is_local,
    )
    pipe.set_progress_bar_config(disable=True)
    if use_fp16 and torch.cuda.is_available():
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass
    try:
        pipe.unet.enable_gradient_checkpointing()
    except Exception:
        pass

    device = torch.device(cfg.device)
    pipe = pipe.to(device)

    B = 1
    if height >= 512 or width >= 512:
        num_inference_steps = min(num_inference_steps, 10)
        print(f"  [512+] 分辨率较重，去噪步数已限制为 {num_inference_steps}；若仍卡住请用 --height 256 --width 256")
    num_steps = min(num_inference_steps, 50)
    gen = DiffusersTrajectoryGenerator(
        pipe=pipe,
        num_steps=num_steps,
        height=height,
        width=width,
        batch_size=B,
        device=device,
        prompt="a photo",
    )
    filt = FlattenTrajectoryFiltration()
    C, H, W = gen.channels, gen.latent_h, gen.latent_w
    D = C * H * W
    persistence = SimplePersistenceProxy(input_dim=2 * D, repr_dim=cfg.repr_dim).to(device)
    if use_fp16 and torch.cuda.is_available():
        persistence = persistence.half()
    return gen, filt, persistence


def main():
    p = argparse.ArgumentParser(description="Phase 1: 拓扑引导可训练性验证")
    p.add_argument("--dry_run", action="store_true", help="不加载 SD，用假轨迹跑通流程")
    p.add_argument("--steps", type=int, default=500, help="嵌入优化步数")
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--repr_dim", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="output_topology_phase1")
    # 真实扩散模型
    p.add_argument("--height", type=int, default=512, help="图像高（真实模式）")
    p.add_argument("--width", type=int, default=512, help="图像宽（真实模式）")
    p.add_argument("--num_inference_steps", type=int, default=20, help="去噪步数（真实模式）")
    p.add_argument("--fp16", action="store_true", help="使用 FP16 加速（默认 FP32 更稳定，不易 nan）")
    p.add_argument(
        "--model_id",
        type=str,
        default="model/stable-diffusion-v1-4",
        help="本地路径如 model/stable-diffusion-v1-4 或 HF id；本地优先，避免联网",
    )
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"使用 GPU: {torch.cuda.get_device_name(0)}")

    if args.dry_run:
        gen, filt, persistence = get_dry_run_components(
            repr_dim=args.repr_dim, batch_size=1, num_steps=10, device=device
        )
    else:
        cfg = TopologyTrajectoryConfig(
            model_id=args.model_id,
            repr_dim=args.repr_dim,
            embed_steps=args.steps,
            embed_lr=args.lr,
            seed=args.seed,
            device=str(device),
        )
        gen, filt, persistence = get_real_components(
            cfg,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            use_fp16=args.fp16,
        )

    embedder = TopologyGuidedEmbedder(gen, filt, persistence, device=device)
    dtype = next(persistence.parameters()).dtype if hasattr(persistence, "parameters") else torch.float32
    target_lambda = torch.randn(1, persistence.repr_dim, device=device, dtype=dtype)

    z_opt, info = embedder.embed(
        target_lambda,
        num_steps=args.steps,
        lr=args.lr,
        log_every=max(1, args.steps // 10),
        grad_clip=1.0,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "phase1_result.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "final_loss": info["final_loss"],
                "loss_history": [l for l in info["loss_history"] if l == l],
                "steps": args.steps,
                "lr": args.lr,
                "dry_run": args.dry_run,
            },
            f,
            indent=2,
        )
    fl = info["final_loss"]
    print(f"Phase 1 完成. final_loss={fl} 已写入 {out_dir}")

    if info["loss_history"]:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.plot(info["loss_history"])
            plt.xlabel("step")
            plt.ylabel("loss")
            plt.title("Phase 1: ||D̃(z) − λ_target||²")
            plt.savefig(out_dir / "phase1_loss.png")
            plt.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
