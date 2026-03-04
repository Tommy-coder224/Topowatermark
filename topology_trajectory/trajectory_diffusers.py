"""
TrajectoryGenerator：用 diffusers 从初始噪声生成整条去噪轨迹并返回每步 latent。
便于替换为其他 scheduler / 模型。
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import torch

from .interfaces import TrajectoryGenerator


class DiffusersTrajectoryGenerator(TrajectoryGenerator):
    """z = 初始噪声 [B,C,H,W]；生成轨迹 = 每步 latent 的序列 [T+1, B, C, H, W]。"""

    def __init__(
        self,
        pipe: Any,
        num_steps: int = 50,
        height: int = 512,
        width: int = 512,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        prompt: str = "",
    ):
        self.pipe = pipe
        self.num_steps = num_steps
        self.height = height
        self.width = width
        self.batch_size = batch_size
        self.device = device or next(pipe.unet.parameters()).device
        self.pipe = self.pipe.to(self.device)
        self.prompt = prompt
        vae_scale = getattr(pipe, "vae_scale_factor", 8)
        in_ch = getattr(pipe.unet.config, "in_channels", 4) or 4
        self.latent_h = height // vae_scale
        self.latent_w = width // vae_scale
        self.channels = in_ch
        self._traj_shape = (batch_size, in_ch, self.latent_h, self.latent_w)
        self._prompt_embeds: Optional[torch.Tensor] = None

    @property
    def trajectory_shape(self) -> Tuple[int, ...]:
        return self._traj_shape

    def generate(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: 初始噪声 [B, C, H, W]，需与 pipe 的 latent 尺寸一致。
        Returns: trajectory [T+1, B, C, H, W]，第 0 步为 z，第 T 步为最终 latent。
        """
        if z.dim() == 4:
            # 单组初始噪声
            latents = z.to(self.device)
        else:
            raise ValueError("z 应为 [B, C, H, W]")

        scheduler = self.pipe.scheduler
        scheduler.set_timesteps(self.num_steps, device=self.device)
        timesteps = scheduler.timesteps

        trajectory = [latents]
        for t in timesteps:
            latent_model_input = scheduler.scale_model_input(latents, t)
            if self._prompt_embeds is None:
                try:
                    out = self.pipe.encode_prompt(
                        self.prompt,
                        self.device,
                        1,
                        do_classifier_free_guidance=False,
                    )
                    pe = out[0] if isinstance(out, (list, tuple)) else out
                except TypeError:
                    out = self.pipe.encode_prompt(
                        self.prompt,
                        self.device,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=False,
                    )
                    pe = out[0] if isinstance(out, (list, tuple)) else out
                self._prompt_embeds = pe.detach()
            noise_pred = self.pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=self._prompt_embeds,
            ).sample
            latents = scheduler.step(noise_pred, t, latents).prev_sample.clone()
            trajectory.append(latents)

        return torch.stack(trajectory, dim=0)

    @staticmethod
    def default_latent_shape(pipe: Any, height: int = 512, width: int = 512, batch_size: int = 1) -> Tuple[int, ...]:
        vae_scale = getattr(pipe, "vae_scale_factor", 8)
        in_ch = getattr(pipe.unet.config, "in_channels", 4) or 4
        return (batch_size, in_ch, height // vae_scale, width // vae_scale)
