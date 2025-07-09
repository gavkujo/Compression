import torch
import torch.nn as nn
import torch.nn.functional as F

# ==== Vectorized BasisHyperLayer ====
class BasisHyperLayer(nn.Module):
    def __init__(self, genome_dim, hidden_dim, out_dim, in_dim, M=128):
        super().__init__()
        self.out_dim, self.in_dim, self.M = out_dim, in_dim, M

        # stack bases into one tensor [M, out, in]
        self.bases = nn.Parameter(torch.randn(M, out_dim, in_dim) * 0.02)

        # hypernet to compute M mixing coeffs
        self.fc1     = nn.Linear(genome_dim, hidden_dim)
        self.fc_coef = nn.Linear(hidden_dim, M)
        self.fc_bias = nn.Linear(hidden_dim, out_dim)

    def forward(self, z):
        # z: [B, genome_dim]
        h      = F.relu(self.fc1(z))       # [B, hidden_dim]
        coeffs = self.fc_coef(h)           # [B, M]
        bias   = self.fc_bias(h)           # [B, out_dim]

        # vectorized mix: coeffs [B,M] × bases [M,out,in] → W [B,out,in]
        W = torch.einsum("bm,moi->boi", coeffs, self.bases)

        return W, bias


# ==== Scalable Self-Attn with correct hidden_dim arg ====
class ScalableSelfAttention(nn.Module):
    def __init__(self, genome_dim, embed_dim=512, num_heads=8, M=128, hidden_dim=256):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim, self.num_heads = embed_dim, num_heads
        self.head_dim = embed_dim // num_heads

        # now pass hidden_dim!
        self.hyper_q = BasisHyperLayer(genome_dim, hidden_dim, embed_dim, embed_dim, M)
        self.hyper_k = BasisHyperLayer(genome_dim, hidden_dim, embed_dim, embed_dim, M)
        self.hyper_v = BasisHyperLayer(genome_dim, hidden_dim, embed_dim, embed_dim, M)
        self.hyper_o = BasisHyperLayer(genome_dim, hidden_dim, embed_dim, embed_dim, M)

    def forward(self, x, z):
        B, T, E = x.size()

        Wq, bq = self.hyper_q(z)
        Wk, bk = self.hyper_k(z)
        Wv, bv = self.hyper_v(z)
        Wo, bo = self.hyper_o(z)

        # einsum does batch‑matmul
        Q = torch.einsum("bei,bte->bti", Wq, x) + bq.unsqueeze(1)
        K = torch.einsum("bei,bte->bti", Wk, x) + bk.unsqueeze(1)
        V = torch.einsum("bei,bte->bti", Wv, x) + bv.unsqueeze(1)

        # split heads
        Qh = Q.view(B, T, self.num_heads, self.head_dim).transpose(1,2)
        Kh = K.view(B, T, self.num_heads, self.head_dim).transpose(1,2)
        Vh = V.view(B, T, self.num_heads, self.head_dim).transpose(1,2)

        scores  = (Qh @ Kh.transpose(-2,-1)) / (self.head_dim ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        out_h   = weights @ Vh

        out = out_h.transpose(1,2).reshape(B, T, E)
        out = torch.einsum("bei,bte->bti", Wo, out) + bo.unsqueeze(1)
        return out


# ==== TinyTransformerLayer stays same ====
class TinyTransformerLayer(nn.Module):
    def __init__(self, genome_dim, embed_dim=512, num_heads=8, M=128, hidden_dim=256):
        super().__init__()
        # pass hidden_dim along!
        self.self_attn = ScalableSelfAttention(genome_dim, embed_dim, num_heads, M, hidden_dim)
        self.norm1     = nn.LayerNorm(embed_dim)
        self.ff        = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.norm2     = nn.LayerNorm(embed_dim)

    def forward(self, x, z):
        a = self.self_attn(x, z)
        x = self.norm1(x + a)
        f = self.ff(x)
        x = self.norm2(x + f)
        return x
