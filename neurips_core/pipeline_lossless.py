"""
严格无损生成 Pipeline：球面+卡方初始噪声
理论：构造1，ε_w ~ N(0,I)
用法：替换 pipe 的 latents 参数
"""
from __future__ import annotations

import torch
from typing import Optional, Tuple, Any

from .lossless import sample_watermarked_noise, verify_lossless
from .spherical import SphericalEmbedding


def get_watermarked_latents(
    pipe: Any,
    user_ids: torch.Tensor,
    spherical: SphericalEmbedding,
    height: int = 512,
    width: int = 512,
    batch_size: int = 1,
    device: torch.device = None,
    dtype: torch.dtype = None,
    generator: Optional[torch.Generator] = None,
    use_voronoi: bool = True,
) -> torch.Tensor:
    """
    生成严格无损带水印初始噪声，供 pipe(prompt, latents=...) 使用。

    理论：定理1+推论1，ε_w = R·S ~ N(0,I)。

    Args:
        pipe: diffusers StableDiffusionPipeline
        user_ids: [B] 用户 ID
        spherical: SphericalEmbedding
        height, width: 图像尺寸
        batch_size: 批大小
        device, dtype: 与 pipe 一致
        generator: 可选，用于可复现

    Returns:
        latents: [B, 4, H, W]，~ N(0,I)，可直接传入 pipe
    """
    if device is None:
        device = next(pipe.unet.parameters()).device
    if dtype is None:
        dtype = pipe.unet.dtype

    vae_scale = getattr(pipe, "vae_scale_factor", 8)
    latent_h = height // vae_scale
    latent_w = width // vae_scale
    in_ch = getattr(pipe.unet.config, "in_channels", None) or getattr(pipe.unet, "in_channels", 4)
    shape = (batch_size, in_ch, latent_h, latent_w)

    eps = sample_watermarked_noise(
        user_ids=user_ids,
        spherical=spherical,
        shape=shape,
        device=device,
        method="spherical_chi2",
        use_voronoi=use_voronoi,
    )
    # SD 初始 latent 需乘 scaling（部分 scheduler 有 init_noise_sigma）
    init_sigma = getattr(pipe.scheduler, "init_noise_sigma", 1.0)
    latents = eps * init_sigma
    return latents.to(dtype)


def generate_watermarked(
    pipe: Any,
    prompt: str,
    user_id: int,
    spherical: SphericalEmbedding,
    **pipe_kwargs,
) -> Any:
    """
    生成带水印图像（严格无损初始噪声）。

    Args:
        pipe: StableDiffusionPipeline
        prompt: 文本提示
        user_id: 用户 ID
        spherical: SphericalEmbedding
        **pipe_kwargs: 传给 pipe 的额外参数

    Returns:
        pipe 的返回值（通常 .images[0]）
    """
    device = next(pipe.unet.parameters()).device
    user_ids = torch.tensor([user_id], dtype=torch.long, device=device)
    latents = get_watermarked_latents(
        pipe=pipe,
        user_ids=user_ids,
        spherical=spherical,
        height=pipe_kwargs.get("height", 512),
        width=pipe_kwargs.get("width", 512),
        batch_size=1,
        device=device,
        dtype=pipe.unet.dtype,
        generator=pipe_kwargs.get("generator"),
    )
    return pipe(prompt, latents=latents, **pipe_kwargs)
