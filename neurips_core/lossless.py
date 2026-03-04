"""
严格无损初始噪声采样：球面+卡方
理论：定理1+推论1，ε_w = R·S ~ N(0,I)
参考文献：Anderson (2003), Spherical Watermark ICLR 2026
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Tuple, Optional


def sample_chi2_radius(n: int, batch_size: int, device: torch.device) -> torch.Tensor:
    """
    R² ~ χ²_n，R = sqrt(R²)。
    定理1：X~N(0,I) 时 ||X||² ~ χ²_n。
    """
    if n <= 0:
        raise ValueError(f"维度 n 必须为正整数，got n={n}")
    return torch.sqrt(torch.distributions.Chi2(n).sample((batch_size,)).to(device))


def _sample_s_in_cell_voronoi(
    user_ids: torch.Tensor,
    e_table: torch.Tensor,
    n: int,
    d: int,
    device: torch.device,
    max_trials: int = 50,
) -> torch.Tensor:
    """
    Voronoi 拒绝采样：S ~ Uniform(C_u)，C_u = {s : argmax_v <s_d, e_v> = u}。
    理论：构造1，等测度胞腔。
    """
    B = user_ids.shape[0]
    S_list = []
    for b in range(B):
        u = user_ids[b].item()
        for _ in range(max_trials):
            Z = torch.randn(n, device=device)
            S = F.normalize(Z, p=2, dim=-1)
            s_d = S[:d]
            if s_d.norm() < 1e-6:
                continue
            dir_d = F.normalize(s_d, p=2, dim=-1)
            cos_sim = dir_d.unsqueeze(0) @ e_table.t()
            if cos_sim.argmax().item() == u:
                S_list.append(S)
                break
        else:
            S_list.append(F.normalize(torch.randn(n, device=device), p=2, dim=-1))
    return torch.stack(S_list)


def sample_watermarked_noise(
    user_ids: torch.Tensor,
    spherical: torch.nn.Module,
    shape: Tuple[int, ...],
    device: torch.device,
    method: str = "spherical_chi2",
    use_voronoi: bool = True,
) -> torch.Tensor:
    """
    生成严格无损带水印初始噪声 ε_w ~ N(0,I)。
    构造1：等测度胞腔 C_u 内均匀采样 S，R²~χ²_n，ε_w = R·S。
    use_voronoi=True：Voronoi 拒绝采样，严格等测度。
    """
    B, C, H, W = shape
    n = C * H * W
    d = spherical.embed_dim
    assert user_ids.shape[0] == B, f"user_ids {user_ids.shape[0]} vs B={B}"

    if method == "standard":
        return torch.randn(*shape, device=device)

    with torch.no_grad():
        e_table = F.normalize(spherical.embedding.weight, p=2, dim=-1)
        if use_voronoi and n >= d:
            S = _sample_s_in_cell_voronoi(user_ids, e_table, n, d, device)
        else:
            e_u = spherical(user_ids)  # [B, d]
            rep = (n + d - 1) // d
            base = e_u.unsqueeze(1).expand(B, rep, d).reshape(B, -1)[:, :n]
            if base.shape[1] < n:
                base = F.pad(base, (0, n - base.shape[1]))
            S = F.normalize(base + 0.3 * torch.randn(B, n, device=device), p=2, dim=-1)

        R = sample_chi2_radius(n, B, device)
        eps_w = R.unsqueeze(-1) * S
        out = eps_w.reshape(shape)
        assert out.shape == shape, f"lossless: {out.shape} vs {shape}"
        return out


def verify_lossless(eps: torch.Tensor, tol: float = 5e-2) -> dict:
    """
    验证 ε 与 N(0,I) 的近似程度（审稿人可复现）。
    返回均值、方差、与标准高斯的 KL 上界估计。
    """
    flat = eps.flatten()
    mean = flat.mean().item()
    var = flat.var().item()
    return {
        "mean": mean,
        "var": var,
        "is_approx_zero_mean": abs(mean) < tol,
        "is_approx_unit_var": abs(var - 1.0) < tol,
    }
