"""
Path B 解码器：z_final -> ê_pred ∈ S^{d-1}
理论：命题 4 可证鲁棒半径 r_adv = √(2(1-ε))/L，L 为 Decoder Lipschitz 常数。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _spectral_norm_conv(conv: nn.Conv2d) -> nn.Conv2d:
    """对 Conv2d 做谱归一化，有助于控制 Lipschitz 常数 L。"""
    return nn.utils.spectral_norm(conv)


class Decoder(nn.Module):
    """
    z_final -> ê_pred ∈ S^{d-1}
    从扩散最终 latent 提取用户嵌入估计，溯源规则 argmax_v <ê, e_v>。
    """
    def __init__(
        self,
        latent_dim: int = 4,
        embed_dim: int = 64,
        hidden: int = 128,
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        c1 = nn.Conv2d(latent_dim, hidden, 3, padding=1)
        c2 = nn.Conv2d(hidden, hidden, 3, padding=1)
        if use_spectral_norm:
            c1, c2 = _spectral_norm_conv(c1), _spectral_norm_conv(c2)
        self.conv = nn.Sequential(
            c1,
            nn.ReLU(),
            c2,
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, z_w: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.conv(z_w), p=2, dim=-1)
