"""
完整放射性水印实验管线
========================

一键运行：从生成 → 微调 → 检测 → 消融，
验证定理 A–E 的全部预测。

实验设计（5 组）：
═══════════════════════════════════════════════════════════

E1: VAE 拓扑稳定性验证（支撑定理 B / 引理 1）
  · 生成图像 → VAE 编码 → 解码 → 比较拓扑
  · 预期：bottleneck 距离 ≤ L∞ 误差

E2: 拓扑耦合有效性（支撑构造 2 / 命题 8-9）
  · 不同用户的水印图像 → 提取拓扑 → 比较用户间可分性
  · 预期：用户间拓扑距离 >> 用户内拓扑距离

E3: 放射性验证（核心，支撑定理 D）
  · 水印图像 → LoRA 微调 → 新模型生成 → 检测
  · 预期：TPR@1%FPR >> 随机基线

E4: 无损性验证（支撑定理 A / 命题 7）
  · 水印噪声统计量检验
  · 预期：均值 ≈ 0，方差 ≈ 1，KS 检验不拒绝

E5: 消融实验
  · 无拓扑耦合 vs 有拓扑耦合
  · 不同 K (候选数) 对放射性的影响
  · 不同 LoRA rank 对放射性的影响
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from radioactive.config import RadioactiveConfig
from radioactive.pipeline.embedder import RadioactiveEmbedder, WatermarkRegistry
from radioactive.pipeline.detector import RadioactiveDetector
from radioactive.train.finetune import LoRAFineTuner
from radioactive.core.stability import StabilityVerifier
from radioactive.core.topo_vectorize import extract_topo_signature


def setup_pipe(config: RadioactiveConfig, warmup: bool = True):
    """
    加载 Stable Diffusion pipeline。
    warmup=True 时在首次推理前做一次短步数生成，触发 CUDA 预热（约 30–90s），
    避免用户误以为程序在「加载完成后停住」。
    """
    from diffusers import StableDiffusionPipeline, DDIMScheduler

    pipe = StableDiffusionPipeline.from_pretrained(
        config.model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(config.device)
    pipe.set_progress_bar_config(disable=True)

    if warmup:
        from radioactive.utils.warmup import warmup_pipeline
        warmup_pipeline(pipe, device=config.device, steps=2)
    return pipe


def run_e1_vae_stability(pipe, config: RadioactiveConfig):
    """
    E1: VAE 拓扑稳定性验证。

    验证引理 1：d_B(Dgm(x), Dgm(Dec(Enc(x)))) ≤ ‖x − Dec(Enc(x))‖_∞
    """
    print("\n" + "="*60)
    print("E1: VAE 拓扑稳定性验证 (定理 B / 引理 1)")
    print("="*60)

    verifier = StabilityVerifier(
        max_dim=config.ph_max_dim,
        pi_resolution=config.pi_resolution,
        pi_bandwidth=config.pi_bandwidth,
    )

    n_test = 20
    originals = []
    reconstructed = []

    vae = pipe.vae
    vae.eval()

    with torch.no_grad():
        for i in range(n_test):
            prompt = config.prompts[i % len(config.prompts)]
            result = pipe(prompt, num_inference_steps=config.num_inference_steps)
            img = result.images[0]

            img_np = np.array(img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
            img_tensor = (img_tensor * 2.0 - 1.0).to(pipe.device, dtype=vae.dtype)

            z = vae.encode(img_tensor).latent_dist.sample()
            recon = vae.decode(z).sample
            recon = ((recon.clamp(-1, 1) + 1) / 2).cpu().numpy()[0]

            originals.append(img_np.transpose(2, 0, 1))
            reconstructed.append(recon)

            if (i+1) % 5 == 0:
                print(f"  [{i+1}/{n_test}] VAE round-trip done")

    originals = np.stack(originals)
    reconstructed = np.stack(reconstructed)

    report = verifier.verify_vae_stability(originals, reconstructed)
    print(report.summary())

    return report


def run_e2_coupling_separability(embedder: RadioactiveEmbedder, config: RadioactiveConfig):
    """
    E2: 拓扑耦合用户间可分性。

    验证命题 8-9：不同用户的拓扑签名是否可区分。
    """
    print("\n" + "="*60)
    print("E2: 拓扑耦合用户间可分性 (命题 8-9)")
    print("="*60)

    n_per_user = 5
    user_sigs = {}

    for uid in range(min(config.num_users, 5)):
        sigs = []
        for j in range(n_per_user):
            prompt = config.prompts[j % len(config.prompts)]
            _, sig, _ = embedder.generate_single_watermarked(
                user_id=uid, prompt=prompt, K=1,
            )
            sigs.append(sig)
            print(f"  user {uid}, img {j+1}/{n_per_user}")
        user_sigs[uid] = sigs

    intra_dists = []
    inter_dists = []

    uids = list(user_sigs.keys())
    for uid in uids:
        vecs = [s.vector for s in user_sigs[uid]]
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                intra_dists.append(float(np.linalg.norm(vecs[i] - vecs[j])))

    for i, u1 in enumerate(uids):
        for u2 in uids[i+1:]:
            vecs1 = [s.vector for s in user_sigs[u1]]
            vecs2 = [s.vector for s in user_sigs[u2]]
            mean1 = np.mean(vecs1, axis=0)
            mean2 = np.mean(vecs2, axis=0)
            inter_dists.append(float(np.linalg.norm(mean1 - mean2)))

    intra_mean = np.mean(intra_dists) if intra_dists else 0
    inter_mean = np.mean(inter_dists) if inter_dists else 0
    ratio = inter_mean / (intra_mean + 1e-12)

    print(f"\n  Intra-user mean distance: {intra_mean:.4f}")
    print(f"  Inter-user mean distance: {inter_mean:.4f}")
    print(f"  Separability ratio:       {ratio:.4f}")
    print(f"  (ratio >> 1 → 拓扑耦合有效)")

    return {"intra": intra_mean, "inter": inter_mean, "ratio": ratio}


def run_e3_radioactivity(
    pipe,
    embedder: RadioactiveEmbedder,
    config: RadioactiveConfig,
):
    """
    E3: 核心放射性验证（定理 D）。

    流程：
      1. 生成水印图像
      2. 生成 clean 基线
      3. LoRA 微调新模型
      4. 新模型生成图像
      5. 检测放射性
    """
    print("\n" + "="*60)
    print("E3: 放射性验证 (定理 D) — 核心实验")
    print("="*60)

    N_wm = min(config.num_watermarked_images, 200)
    N_test = min(config.num_test_images, 100)
    N_clean = 50

    user_ids = [i % config.num_users for i in range(N_wm)]
    prompts = [config.prompts[i % len(config.prompts)] for i in range(N_wm)]

    print(f"\n--- Phase 1: 生成 {N_wm} 张水印图像 ---")
    images, registry = embedder.generate_batch(
        user_ids=user_ids,
        prompts=prompts,
        use_topo_coupling=True,
        K=config.radioactive_candidates_k,
    )

    print(f"\n--- Phase 2: 生成 {N_clean} 张 clean 基线 ---")
    clean_sigs = embedder.generate_clean_baseline(config.prompts, n_images=N_clean)
    registry.register_clean_baseline(clean_sigs)

    reg_path = os.path.join(config.output_dir, "registry_with_baseline.json")
    registry.save(reg_path)

    print(f"\n--- Phase 3: LoRA 微调 ---")
    finetuner = LoRAFineTuner(config)

    from copy import deepcopy
    from diffusers import StableDiffusionPipeline, DDIMScheduler

    pipe_ft = StableDiffusionPipeline.from_pretrained(
        config.model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipe_ft.scheduler = DDIMScheduler.from_config(pipe_ft.scheduler.config)
    pipe_ft = pipe_ft.to(config.device)
    pipe_ft.set_progress_bar_config(disable=True)

    wm_dir = os.path.join(config.output_dir, "watermarked_images")
    pipe_ft = finetuner.finetune(
        pipe_ft,
        images=images,
        prompts=prompts,
    )

    print(f"\n--- Phase 4: 微调模型生成 {N_test} 张图像 ---")
    gen_dir = os.path.join(config.output_dir, "finetuned_generations")
    test_prompts = [config.prompts[i % len(config.prompts)] for i in range(N_test)]
    gen_images = finetuner.generate(pipe_ft, test_prompts, n_images=N_test, save_dir=gen_dir)

    print(f"\n--- Phase 5: 放射性检测 ---")
    detector = RadioactiveDetector(
        max_dim=config.ph_max_dim,
        pi_resolution=config.pi_resolution,
        pi_bandwidth=config.pi_bandwidth,
        fpr=config.detection_fpr,
    )

    result = detector.detect(gen_images, registry)
    print("\n" + result.summary())

    del pipe_ft
    gc.collect()
    torch.cuda.empty_cache()

    return result


def run_e4_lossless(embedder: RadioactiveEmbedder, config: RadioactiveConfig):
    """
    E4: 无损性验证（定理 A / 命题 7）。
    """
    print("\n" + "="*60)
    print("E4: 无损性验证 (定理 A / 命题 7)")
    print("="*60)

    from neurips_core.lossless import verify_lossless
    from neurips_core.lossless import sample_watermarked_noise

    C, H, W = embedder._get_latent_shape()
    shape = (100, C, H, W)
    user_ids = torch.zeros(100, dtype=torch.long, device=embedder.device)

    eps = sample_watermarked_noise(
        user_ids=user_ids,
        spherical=embedder.spherical,
        shape=shape,
        device=embedder.device,
    )
    stats = verify_lossless(eps)
    print(f"  Mean:     {stats['mean']:.6f}  (≈0: {stats['is_approx_zero_mean']})")
    print(f"  Variance: {stats['var']:.6f}   (≈1: {stats['is_approx_unit_var']})")

    from scipy.stats import kstest, normaltest
    flat = eps.cpu().numpy().flatten()
    subset = flat[np.random.choice(len(flat), size=min(10000, len(flat)), replace=False)]
    ks_stat, ks_p = kstest(subset, "norm")
    print(f"  KS test:  stat={ks_stat:.6f}, p={ks_p:.6f}")
    print(f"  (p > 0.05 → 不拒绝 N(0,1) 假设)")

    return stats


def run_e5_ablation(pipe, embedder, config):
    """
    E5: 消融实验。
    """
    print("\n" + "="*60)
    print("E5: 消融实验")
    print("="*60)
    print("  (完整消融需要多次微调，此处给出实验框架)")
    print("  消融变量：")
    print("    A1: 无拓扑耦合 (K=1, coupling=0)")
    print("    A2: 拓扑耦合 K=1")
    print("    A3: 拓扑耦合 K=4")
    print("    A4: 拓扑耦合 K=8")
    print("    A5: 不同 LoRA rank (2, 4, 8, 16)")
    return {}


import gc


def main():
    parser = argparse.ArgumentParser(description="TopoRad 放射性水印完整实验")
    parser.add_argument("--experiments", nargs="+", default=["e1", "e2", "e3", "e4"],
                        choices=["e1", "e2", "e3", "e4", "e5", "all"])
    parser.add_argument("--num-users", type=int, default=10)
    parser.add_argument("--num-wm-images", type=int, default=200)
    parser.add_argument("--num-test-images", type=int, default=100)
    parser.add_argument("--lora-epochs", type=int, default=50)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--candidates-k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    config = RadioactiveConfig(
        num_users=args.num_users,
        num_watermarked_images=args.num_wm_images,
        num_test_images=args.num_test_images,
        lora_epochs=args.lora_epochs,
        lora_rank=args.lora_rank,
        radioactive_candidates_k=args.candidates_k,
        seed=args.seed,
        device=args.device,
    )

    os.makedirs(config.output_dir, exist_ok=True)

    config_path = os.path.join(config.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    print(f"Config saved to {config_path}")

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    exps = args.experiments
    if "all" in exps:
        exps = ["e1", "e2", "e3", "e4", "e5"]

    results = {}

    pipe = setup_pipe(config)
    embedder = RadioactiveEmbedder(pipe, config)

    if "e1" in exps:
        results["e1"] = run_e1_vae_stability(pipe, config)

    if "e2" in exps:
        results["e2"] = run_e2_coupling_separability(embedder, config)

    if "e3" in exps:
        results["e3"] = run_e3_radioactivity(pipe, embedder, config)

    if "e4" in exps:
        results["e4"] = run_e4_lossless(embedder, config)

    if "e5" in exps:
        results["e5"] = run_e5_ablation(pipe, embedder, config)

    results_path = os.path.join(config.output_dir, "results_summary.json")
    serializable = {}
    for k, v in results.items():
        if hasattr(v, "summary"):
            serializable[k] = v.summary()
        elif isinstance(v, dict):
            serializable[k] = {
                kk: (vv if isinstance(vv, (int, float, str, bool)) else str(vv))
                for kk, vv in v.items()
            }
        else:
            serializable[k] = str(v)

    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nAll results saved to {results_path}")


if __name__ == "__main__":
    main()
