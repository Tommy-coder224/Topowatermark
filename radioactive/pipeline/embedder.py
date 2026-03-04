"""
RadioactiveEmbedder — 拓扑放射性水印嵌入管线
==============================================

完整流程：
  Phase 0: 校准载波空间（PCA on clean topology manifold）
  Phase 1: 首轮生成 → 提取拓扑签名 → 建立用户拓扑档案
  Phase 2: 载波对齐采样 → 选择最佳候选 → 输出放射性水印图像

数学保证（逐步可验证）：
  1. ε_base = R·S ∈ C_u, R²~χ²_n, R⊥S  → N(0,I)  [定理 A, Anderson 2003]
  2. ε_coupled = Q·ε_base, Q ∈ O(n)     → N(0,I)  [命题 7, 正交不变性]
  3. 通道权重 α ∝ TP_c（拓扑自适应）    → 放射性信号放大 [命题 9]
  4. 载波选择 argmax ⟨λ_i−μ, u_u⟩       → 用户拓扑聚合 [命题 10, Sablayrolles 2020]
"""
from __future__ import annotations

import os
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ..config import RadioactiveConfig
from ..core.carrier import CarrierManager
from ..core.topo_coupler import TopoCoupler
from ..core.topo_vectorize import (
    TopoSignature,
    extract_topo_signature,
    batch_extract_topo_signatures,
)


@dataclass
class WatermarkRegistry:
    """
    水印注册表：记录每个用户的拓扑档案，供检测使用。

    数据结构：
      user_profiles[u] = {
          "topo_vectors": [λ₁, λ₂, ...],      # 拓扑向量集合
          "mean_vector":  μ_u,                   # 均值拓扑向量
          "coupling_info": [...],                # 耦合元数据
      }
    """
    user_profiles: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    global_clean_mean: Optional[np.ndarray] = None
    global_clean_std: Optional[np.ndarray] = None

    def register(
        self,
        user_id: int,
        topo_sig: TopoSignature,
        coupling_info: dict,
    ):
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                "topo_vectors": [],
                "mean_vector": None,
                "coupling_info": [],
            }
        profile = self.user_profiles[user_id]
        profile["topo_vectors"].append(topo_sig.vector.copy())
        profile["coupling_info"].append(coupling_info)
        vecs = np.stack(profile["topo_vectors"])
        profile["mean_vector"] = vecs.mean(axis=0)

    def register_clean_baseline(self, clean_sigs: List[TopoSignature]):
        """注册 clean（未水印）图像的拓扑基线，用于假设检验 H₀。"""
        vecs = np.stack([s.vector for s in clean_sigs])
        self.global_clean_mean = vecs.mean(axis=0)
        self.global_clean_std = vecs.std(axis=0)

    def save(self, path: str):
        data = {
            "user_profiles": {},
            "global_clean_mean": (
                self.global_clean_mean.tolist()
                if self.global_clean_mean is not None else None
            ),
            "global_clean_std": (
                self.global_clean_std.tolist()
                if self.global_clean_std is not None else None
            ),
        }
        for uid, profile in self.user_profiles.items():
            data["user_profiles"][str(uid)] = {
                "topo_vectors": [v.tolist() for v in profile["topo_vectors"]],
                "mean_vector": profile["mean_vector"].tolist(),
                "coupling_info": profile["coupling_info"],
            }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "WatermarkRegistry":
        with open(path) as f:
            data = json.load(f)
        reg = cls()
        if data.get("global_clean_mean") is not None:
            reg.global_clean_mean = np.array(data["global_clean_mean"])
            reg.global_clean_std = np.array(data["global_clean_std"])
        for uid_str, profile in data["user_profiles"].items():
            uid = int(uid_str)
            reg.user_profiles[uid] = {
                "topo_vectors": [np.array(v) for v in profile["topo_vectors"]],
                "mean_vector": np.array(profile["mean_vector"]),
                "coupling_info": profile["coupling_info"],
            }
        return reg


class RadioactiveEmbedder:
    """
    拓扑放射性水印嵌入器。

    用法：
        embedder = RadioactiveEmbedder(pipe, config)
        embedder.calibrate_carriers(prompts)  # 一次性校准
        images, registry = embedder.generate_batch(user_ids, prompts)
    """

    def __init__(
        self,
        pipe: Any,
        config: RadioactiveConfig,
        spherical: Optional[Any] = None,
        carrier_manager: Optional[CarrierManager] = None,
    ):
        self.pipe = pipe
        self.config = config
        self.device = torch.device(config.device)

        if spherical is None:
            import sys, pathlib
            sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
            from neurips_core.spherical import SphericalEmbedding
            self.spherical = SphericalEmbedding(
                config.num_users, config.embed_dim
            ).to(self.device)
        else:
            self.spherical = spherical.to(self.device)

        self.coupler = TopoCoupler(
            embed_dim=config.embed_dim,
            coupling_strength=config.coupling_strength,
            temperature=config.coupling_temperature,
            hash_seed=config.coupling_hash_seed,
        )

        if carrier_manager is not None:
            self.carrier = carrier_manager
        else:
            self.carrier = CarrierManager(
                num_users=config.num_users,
                n_components=config.carrier_n_components,
                carrier_seed=config.carrier_seed,
            )

    def calibrate_carriers(
        self,
        prompts: List[str],
        n_calib: Optional[int] = None,
        verbose: bool = True,
    ) -> CarrierManager:
        """
        校准载波空间：生成干净图像 → 提取拓扑 → PCA → 生成正交化载波。

        仅需运行一次（或换模型/数据集时重跑）。

        Args:
            prompts: 校准用提示词
            n_calib: 校准图数（None 则用 config.carrier_n_calib）
            verbose: 是否打印

        Returns:
            校准完成的 CarrierManager
        """
        n = n_calib or self.config.carrier_n_calib

        if verbose:
            print(f"[Carrier Calibration] Generating {n} clean images for PCA...")

        topo_vecs = []
        with torch.no_grad():
            for i in range(n):
                prompt = prompts[i % len(prompts)]
                result = self.pipe(
                    prompt,
                    num_inference_steps=self.config.num_inference_steps,
                    guidance_scale=self.config.guidance_scale,
                )
                image = result.images[0]
                img_np = np.array(image).astype(np.float64) / 255.0
                if img_np.ndim == 3:
                    img_np = img_np.transpose(2, 0, 1)
                sig = extract_topo_signature(
                    img_np,
                    max_dim=self.config.ph_max_dim,
                    pi_resolution=self.config.pi_resolution,
                    pi_bandwidth=self.config.pi_bandwidth,
                )
                topo_vecs.append(sig.vector)
                if verbose and ((i + 1) % 10 == 0 or i == 0):
                    print(f"  calib img {i+1}/{n}")

        Phi = np.stack(topo_vecs)
        self.carrier.calibrate(Phi, verbose=verbose)
        self.carrier.generate_carriers(
            num_users=min(self.config.num_users, Phi.shape[0] - 1),
            verbose=verbose,
        )
        return self.carrier

    def _pipe_generate(
        self,
        prompt: str,
        latents: torch.Tensor,
        **kwargs,
    ):
        """封装 pipe 生成调用。"""
        return self.pipe(
            prompt,
            latents=latents,
            num_inference_steps=self.config.num_inference_steps,
            guidance_scale=self.config.guidance_scale,
            **kwargs,
        )

    def _get_latent_shape(self) -> Tuple[int, int, int]:
        """返回 (C, H, W) latent shape。"""
        vae_scale = getattr(self.pipe, "vae_scale_factor", 8)
        latent_h = self.config.image_size // vae_scale
        latent_w = self.config.image_size // vae_scale
        in_ch = getattr(
            self.pipe.unet.config, "in_channels", None
        ) or getattr(self.pipe.unet, "in_channels", 4)
        return in_ch, latent_h, latent_w

    def generate_single_watermarked(
        self,
        user_id: int,
        prompt: str,
        use_topo_coupling: bool = True,
        K: int = 1,
    ) -> Tuple[Any, TopoSignature, dict]:
        """
        生成单张拓扑耦合水印图像。

        当 K > 1 时启用候选选择机制：
          生成 K 个候选，选择拓扑最接近用户目标的那个。

        Args:
            user_id: 用户 ID
            prompt:  文本提示
            use_topo_coupling: 是否使用拓扑耦合
            K:       候选数量

        Returns:
            (image, topo_signature, coupling_info)
        """
        C, H, W = self._get_latent_shape()
        shape = (1, C, H, W)
        user_ids = torch.tensor([user_id], dtype=torch.long, device=self.device)
        e_u = self.spherical(user_ids)[0].detach().cpu().numpy()

        if not use_topo_coupling or K <= 1:
            import sys, pathlib
            sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
            from neurips_core.lossless import sample_watermarked_noise

            eps = sample_watermarked_noise(
                user_ids=user_ids,
                spherical=self.spherical,
                shape=shape,
                device=self.device,
            )

            init_sigma = getattr(self.pipe.scheduler, "init_noise_sigma", 1.0)
            latents = eps * init_sigma

            result = self._pipe_generate(prompt, latents=latents.to(self.pipe.unet.dtype))
            image = result.images[0]

            img_np = np.array(image).astype(np.float64) / 255.0
            if img_np.ndim == 3:
                img_np = img_np.transpose(2, 0, 1)
            topo_sig = extract_topo_signature(
                img_np,
                max_dim=self.config.ph_max_dim,
                pi_resolution=self.config.pi_resolution,
                pi_bandwidth=self.config.pi_bandwidth,
            )
            info = {"user_id": user_id, "coupled": False}
            return image, topo_sig, info

        candidate_images = []
        candidate_sigs = []
        candidate_infos = []

        for k_idx in range(K):
            import sys, pathlib
            sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
            from neurips_core.lossless import sample_watermarked_noise

            eps = sample_watermarked_noise(
                user_ids=user_ids,
                spherical=self.spherical,
                shape=shape,
                device=self.device,
            )

            init_sigma = getattr(self.pipe.scheduler, "init_noise_sigma", 1.0)
            latents = eps * init_sigma
            result = self._pipe_generate(prompt, latents=latents.to(self.pipe.unet.dtype))
            image = result.images[0]

            img_np = np.array(image).astype(np.float64) / 255.0
            if img_np.ndim == 3:
                img_np = img_np.transpose(2, 0, 1)
            topo_sig = extract_topo_signature(
                img_np,
                max_dim=self.config.ph_max_dim,
                pi_resolution=self.config.pi_resolution,
                pi_bandwidth=self.config.pi_bandwidth,
            )

            candidate_images.append(image)
            candidate_sigs.append(topo_sig)
            candidate_infos.append({"user_id": user_id, "coupled": True, "candidate_k": k_idx})

        use_carrier = getattr(self.config, "use_carrier", False) and self.carrier._calibrated

        if use_carrier:
            best_idx, best_proj = self.coupler.select_best_candidate_carrier(
                candidate_sigs, self.carrier, user_id
            )
            candidate_infos[best_idx]["carrier_projection"] = best_proj
        else:
            img_h = getattr(self.config, "image_size", 512)
            dummy_img = np.random.rand(3, img_h, img_h).astype(np.float64)
            topo_dim = len(extract_topo_signature(
                dummy_img,
                max_dim=self.config.ph_max_dim,
                pi_resolution=self.config.pi_resolution,
                pi_bandwidth=self.config.pi_bandwidth,
            ).vector)
            user_target = self.coupler.compute_user_topo_target(e_u, topo_dim)
            target_sig = TopoSignature(
                vector=user_target,
                tp_per_channel=np.zeros(1),
                entropy_per_channel=np.zeros(1),
                raw_diagrams=[],
            )
            max_dist = getattr(self.config, "max_topo_dist_fallback", None)
            best_idx, _, _ = self.coupler.select_best_candidate(
                candidate_sigs, target_sig, max_dist=max_dist
            )

        return (
            candidate_images[best_idx],
            candidate_sigs[best_idx],
            candidate_infos[best_idx],
        )

    def generate_batch(
        self,
        user_ids: List[int],
        prompts: List[str],
        use_topo_coupling: bool = True,
        K: int = 1,
        save_images: bool = True,
    ) -> Tuple[List[Any], WatermarkRegistry]:
        """
        批量生成拓扑放射性水印图像并构建注册表。

        Args:
            user_ids: 每张图的用户 ID
            prompts:  每张图的文本提示
            use_topo_coupling: 是否使用拓扑耦合
            K:        候选选择数
            save_images: 是否保存图像到磁盘

        Returns:
            (images, registry)
        """
        assert len(user_ids) == len(prompts)
        N = len(user_ids)

        registry = WatermarkRegistry()
        images = []

        out_dir = os.path.join(self.config.output_dir, "watermarked_images")
        if save_images:
            os.makedirs(out_dir, exist_ok=True)

        print(f"[RadioactiveEmbedder] Generating {N} watermarked images (K={K})...")

        for i in range(N):
            t0 = time.time()
            image, topo_sig, info = self.generate_single_watermarked(
                user_id=user_ids[i],
                prompt=prompts[i],
                use_topo_coupling=use_topo_coupling,
                K=K,
            )
            dt = time.time() - t0

            registry.register(user_ids[i], topo_sig, info)
            images.append(image)

            if save_images:
                fname = f"user{user_ids[i]:03d}_{i:05d}.png"
                image.save(os.path.join(out_dir, fname))

            if (i + 1) % 10 == 0 or i == 0:
                print(f"  [{i+1}/{N}] user={user_ids[i]} dt={dt:.1f}s")

        reg_path = os.path.join(self.config.output_dir, "registry.json")
        registry.save(reg_path)
        print(f"[RadioactiveEmbedder] Registry saved to {reg_path}")

        return images, registry

    def generate_clean_baseline(
        self,
        prompts: List[str],
        n_images: int = 100,
    ) -> List[TopoSignature]:
        """
        生成 clean（未水印）图像的拓扑签名基线（用于 H₀ 分布估计）。
        """
        print(f"[RadioactiveEmbedder] Generating {n_images} clean baseline images...")
        sigs = []

        for i in range(n_images):
            prompt = prompts[i % len(prompts)]
            result = self.pipe(
                prompt,
                num_inference_steps=self.config.num_inference_steps,
                guidance_scale=self.config.guidance_scale,
            )
            image = result.images[0]

            img_np = np.array(image).astype(np.float64) / 255.0
            if img_np.ndim == 3:
                img_np = img_np.transpose(2, 0, 1)
            sig = extract_topo_signature(
                img_np,
                max_dim=self.config.ph_max_dim,
                pi_resolution=self.config.pi_resolution,
                pi_bandwidth=self.config.pi_bandwidth,
            )
            sigs.append(sig)

            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{n_images}] clean baseline")

        return sigs
