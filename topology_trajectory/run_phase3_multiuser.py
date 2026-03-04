"""
Phase 3：多用户溯源/可扩展性（在表示空间 Λ_u 上做最近邻归属）

流程：
1) 取参考噪声 z0，计算 λ0 = D̃(z0)
2) 构造 users 个目标 λ_u = λ0 + sigma * ε_u
3) 对每个 u，优化得到 z_u，使 D̃(z_u) 接近 λ_u
4) 归属：对每个 z_u 计算 repr_u，与所有 λ_v 做距离，取最近邻 v_hat；报告准确率

输出：
- output_topology_phase3/phase3_result.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import torch

from .config import TopologyTrajectoryConfig
from .embedder import TopologyGuidedEmbedder
from .run_phase1 import get_dry_run_components, get_real_components


def _pairwise_l2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a:[U,m], b:[U,m] -> dist:[U,U]"""
    aa = (a * a).sum(dim=1, keepdim=True)
    bb = (b * b).sum(dim=1, keepdim=True).t()
    return (aa + bb - 2 * (a @ b.t())).clamp(min=0).sqrt()


def main():
    p = argparse.ArgumentParser(description="Phase 3: 多用户溯源/可扩展（最近邻归属）")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--users", type=int, default=8)
    p.add_argument("--sigma", type=float, default=0.5, help="λ_u = λ0 + sigma * N(0,I)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--repr_dim", type=int, default=64)
    p.add_argument("--out_dir", type=str, default="output_topology_phase3")
    # real
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--num_inference_steps", type=int, default=15)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--model_id", type=str, default="model/stable-diffusion-v1-4")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"使用 GPU: {torch.cuda.get_device_name(0)}")

    torch.manual_seed(args.seed)

    if args.dry_run:
        gen, filt, persistence = get_dry_run_components(repr_dim=args.repr_dim, batch_size=1, num_steps=10, device=device)
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

    # 参考 z0 与 λ0
    z0 = torch.randn(1, *embedder.trajectory_gen.trajectory_shape[1:], device=device, dtype=dtype)
    with torch.no_grad():
        traj0 = embedder.trajectory_gen.generate(z0)
        fil0 = embedder.filtration_builder.build(traj0)
        lambda0 = embedder.persistence_layer.forward(fil0).detach()  # [1,m]

    U = int(args.users)
    eps = torch.randn(U, persistence.repr_dim, device=device, dtype=dtype)
    lambdas = lambda0.repeat(U, 1) + float(args.sigma) * eps  # [U,m]

    z_opts: List[torch.Tensor] = []
    reprs: List[torch.Tensor] = []
    final_losses: List[float] = []

    for u in range(U):
        print(f"\n== user {u}/{U-1} ==")
        target_u = lambdas[u : u + 1]
        z_init = torch.randn_like(z0)
        z_u, info = embedder.embed(
            target_u,
            num_steps=args.steps,
            lr=args.lr,
            z_init=z_init,
            log_every=max(1, args.steps // 5),
            grad_clip=1.0,
        )
        z_opts.append(z_u.detach().cpu())
        final_losses.append(float(info["final_loss"]) if isinstance(info["final_loss"], (int, float)) and math.isfinite(info["final_loss"]) else float("nan"))
        reprs.append(info["repr_final"].to(device))

    reprs_t = torch.cat(reprs, dim=0)  # [U,m]
    dist = _pairwise_l2(reprs_t, lambdas)  # [U,U]
    pred = dist.argmin(dim=1)  # [U]
    gt = torch.arange(U, device=device)
    acc = (pred == gt).float().mean().item()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "users": U,
        "sigma": args.sigma,
        "steps": args.steps,
        "lr": args.lr,
        "acc": acc,
        "final_loss_mean": float(torch.tensor(final_losses).nanmean().item()) if final_losses else None,
        "final_losses": final_losses,
        "args": vars(args),
    }
    with open(out_dir / "phase3_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    # 保存距离矩阵（便于画热力图）
    torch.save({"dist": dist.detach().cpu(), "lambdas": lambdas.detach().cpu(), "reprs": reprs_t.detach().cpu()}, out_dir / "phase3_tensors.pt")

    print(f"\nPhase 3 完成. acc={acc:.3f} 结果写入 {out_dir}")


if __name__ == "__main__":
    main()

