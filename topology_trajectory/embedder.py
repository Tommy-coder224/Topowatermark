"""
Embedder：给定 λ_target，优化 z 使 ‖D̃(z) − λ_target‖² 最小。
实现 TrajectoryGenerator + FiltrationBuilder + PersistenceLayer 的串联与梯度下降。
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from .interfaces import Embedder, FiltrationBuilder, PersistenceLayer, TrajectoryGenerator


class TopologyGuidedEmbedder(Embedder):
    """拓扑引导嵌入：min_z ‖D̃(z) − λ_target‖²。"""

    def __init__(
        self,
        trajectory_gen: TrajectoryGenerator,
        filtration_builder: FiltrationBuilder,
        persistence_layer: PersistenceLayer,
        device: Optional[torch.device] = None,
    ):
        self.trajectory_gen = trajectory_gen
        self.filtration_builder = filtration_builder
        self.persistence_layer = persistence_layer
        self.device = device
        if self.device is None and hasattr(trajectory_gen, "device"):
            self.device = trajectory_gen.device
        if self.device is None and hasattr(getattr(trajectory_gen, "pipe", None), "unet"):
            self.device = next(trajectory_gen.pipe.unet.parameters()).device
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def embed(
        self,
        target_lambda: torch.Tensor,
        num_steps: int = 500,
        lr: float = 1e-2,
        z_init: Optional[torch.Tensor] = None,
        log_every: int = 50,
        grad_clip: Optional[float] = 1.0,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        target_lambda: [B, m] 或 [m]；若 [m] 则 broadcast 到 batch。
        Returns:
            z_opt: 优化后的初始噪声
            info: loss_history, final_loss, repr_final
        """
        if target_lambda.dim() == 1:
            target_lambda = target_lambda.unsqueeze(0)
        target_lambda = target_lambda.to(self.device)
        B = target_lambda.shape[0]

        if z_init is not None:
            z = z_init.detach().clone().to(self.device).requires_grad_(True)
        else:
            # 与 TrajectoryGenerator 的 latent shape 一致
            shp = self.trajectory_gen.trajectory_shape
            z = torch.randn(B, *shp[1:], device=self.device, dtype=target_lambda.dtype, requires_grad=True)

        opt = torch.optim.Adam([z], lr=lr)
        loss_history = []

        for step in range(num_steps):
            opt.zero_grad()
            trajectory = self.trajectory_gen.generate(z)
            filtration_input = self.filtration_builder.build(trajectory)
            repr_z = self.persistence_layer.forward(filtration_input)
            loss = ((repr_z - target_lambda) ** 2).sum(dim=-1).mean()
            if not torch.isfinite(loss).all():
                loss_history.append(float("nan"))
                if log_every and (step + 1) % log_every == 0:
                    print(f"  step {step+1}/{num_steps} loss=nan (skip backward)")
                continue
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_([z], max_norm=grad_clip)
            opt.step()
            loss_history.append(loss.item())
            if log_every and (step + 1) % log_every == 0:
                print(f"  step {step+1}/{num_steps} loss={loss.item():.6f}")

        with torch.no_grad():
            trajectory_f = self.trajectory_gen.generate(z)
            filtration_f = self.filtration_builder.build(trajectory_f)
            repr_final = self.persistence_layer.forward(filtration_f)

        info = {
            "loss_history": loss_history,
            "final_loss": loss_history[-1] if loss_history else float("nan"),
            "repr_final": repr_final.detach().cpu(),
            "target_lambda": target_lambda.cpu(),
        }
        return z.detach(), info
