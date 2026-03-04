"""
Phase 2：稳定性/效率评估（多 seed 重复 Phase 1）

输出：
- output_topology_phase2/phase2_runs.jsonl  每次运行一行（seed、final_loss、time、peak_mem）
- output_topology_phase2/phase2_summary.json 汇总统计
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

from .config import TopologyTrajectoryConfig
from .embedder import TopologyGuidedEmbedder
from .run_phase1 import get_dry_run_components, get_real_components


def _safe_float(x: float) -> float | None:
    return x if isinstance(x, (int, float)) and math.isfinite(x) else None


def main():
    p = argparse.ArgumentParser(description="Phase 2: 稳定性/效率（多 seed）")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--target_seed", type=int, default=123, help="固定目标 λ_target 的随机种子")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--repr_dim", type=int, default=64)
    p.add_argument("--out_dir", type=str, default="output_topology_phase2")
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

    # 只加载一次组件（避免每个 seed 重新加载模型）
    if args.dry_run:
        gen, filt, persistence = get_dry_run_components(repr_dim=args.repr_dim, batch_size=1, num_steps=10, device=device)
    else:
        cfg = TopologyTrajectoryConfig(
            model_id=args.model_id,
            repr_dim=args.repr_dim,
            embed_steps=args.steps,
            embed_lr=args.lr,
            seed=args.target_seed,
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

    # 固定目标 λ_target，便于比较不同初始化/seed 的稳定性
    torch.manual_seed(args.target_seed)
    dtype = next(persistence.parameters()).dtype if hasattr(persistence, "parameters") else torch.float32
    target_lambda = torch.randn(1, persistence.repr_dim, device=device, dtype=dtype)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "phase2_runs.jsonl"

    runs: List[Dict[str, Any]] = []
    for s in args.seeds:
        print(f"\n== seed {s} ==")
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.reset_max_memory_allocated()
        z_init = torch.randn(1, *embedder.trajectory_gen.trajectory_shape[1:], device=device, dtype=dtype)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        z_opt, info = embedder.embed(
            target_lambda,
            num_steps=args.steps,
            lr=args.lr,
            z_init=z_init,
            log_every=max(1, args.steps // 5),
            grad_clip=1.0,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        peak_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        rec = {
            "seed": s,
            "final_loss": _safe_float(info["final_loss"]),
            "time_sec": t1 - t0,
            "time_per_step_sec": (t1 - t0) / max(1, args.steps),
            "peak_mem_bytes": int(peak_mem),
        }
        runs.append(rec)
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 汇总
    losses = [r["final_loss"] for r in runs if r["final_loss"] is not None]
    times = [r["time_sec"] for r in runs]
    summary = {
        "num_runs": len(runs),
        "loss_mean": (sum(losses) / len(losses)) if losses else None,
        "loss_min": min(losses) if losses else None,
        "loss_max": max(losses) if losses else None,
        "time_mean_sec": sum(times) / len(times) if times else None,
        "time_per_step_mean_sec": (sum(r["time_per_step_sec"] for r in runs) / len(runs)) if runs else None,
        "peak_mem_max_bytes": max((r["peak_mem_bytes"] for r in runs), default=0),
        "args": vars(args),
    }
    with open(out_dir / "phase2_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nPhase 2 完成. 结果写入 {out_dir}")


if __name__ == "__main__":
    main()

