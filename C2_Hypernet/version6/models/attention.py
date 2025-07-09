import torch
import torch.nn as nn
from models.hyperlayer import HyperLayer

class TargetSelfAttention(nn.Module):
    def __init__(self, genome_dim=64, hidden_dim=128, embed_dim=28, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # 3 HyperLayers: Q, K, V projection matrices
        self.q_proj = HyperLayer(genome_dim, hidden_dim, embed_dim, embed_dim)
        self.k_proj = HyperLayer(genome_dim, hidden_dim, embed_dim, embed_dim)
        self.v_proj = HyperLayer(genome_dim, hidden_dim, embed_dim, embed_dim)

        # Output projection
        self.out_proj = HyperLayer(genome_dim, hidden_dim, embed_dim, embed_dim)

    def forward(self, x, z):
        B, T, E = x.size()  # [batch, tokens, embed]
        z = z.expand(B, -1)

        def apply_proj(hyper, x):
            W, b = hyper(z)
            x_proj = torch.einsum("boi,bti->bto", W, x) + b.unsqueeze(1)
            return x_proj

        Q = apply_proj(self.q_proj, x)
        K = apply_proj(self.k_proj, x)
        V = apply_proj(self.v_proj, x)

        # Split into heads
        def split_heads(x):
            return x.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, D]

        Q, K, V = map(split_heads, (Q, K, V))

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)  # [B, H, T, D]

        # Merge heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, E)

        # Final projection
        out_W, out_b = self.out_proj(z)
        out = torch.einsum("boi,bti->bto", out_W, attn_output) + out_b.unsqueeze(1)
        return out
