"""
球形编码：user_id -> e_u ∈ S^{d-1}
理论：Delsarte 球面码，容量 O(2^d)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SphericalEmbedding(nn.Module):
    """user_id -> e_u ∈ S^{d-1}，单位球面"""
    def __init__(self, num_users: int, embed_dim: int):
        super().__init__()
        self.num_users = num_users
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(num_users, embed_dim)
        with torch.no_grad():
            w = torch.randn(num_users, embed_dim)
            w = F.normalize(w, p=2, dim=-1)
            if num_users <= embed_dim:
                for i in range(num_users):
                    for j in range(i):
                        w[i] = w[i] - (w[i] @ w[j]) * w[j]
                    w[i] = F.normalize(w[i].unsqueeze(0), p=2, dim=-1).squeeze(0)
            self.embedding.weight.data = w

    def forward(self, user_ids: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.embedding(user_ids), p=2, dim=-1)
