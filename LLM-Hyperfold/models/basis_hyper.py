import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class FactorizedBasisHyperLayer(nn.Module):
    def __init__(
        self,
        genome_dim: int,
        hidden_dim: int,
        out_dim: int,
        in_dim: int,
        M: int = 32,
        rank: int = 64,
        top_k: int = 4  # New: sparsity control
    ) -> None:
        super().__init__()
        self.M = M
        self.rank = rank
        self.top_k = top_k
        
        # Factorized basis tensors
        self.U = nn.Parameter(torch.randn(M, out_dim, rank) * 0.01)
        self.V = nn.Parameter(torch.randn(M, rank, in_dim) * 0.01)
        
        # Efficient hypernetwork
        self.fc1 = nn.Linear(genome_dim, hidden_dim)
        self.gate_network = nn.Linear(hidden_dim, M)  # Replaces fc_coef
        
        # Quantization parameters
        self.quant_scale = nn.Parameter(torch.ones(1))
        self.quant_zero = nn.Parameter(torch.zeros(1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        single = z.dim() == 1
        if single:
            z = z.unsqueeze(0)

        h = F.gelu(self.fc1(z))
        
        # Sparse gating mechanism
        gate_logits = self.gate_network(h)
        topk_indices = torch.topk(gate_logits, self.top_k, dim=-1).indices
        sparse_mask = torch.zeros_like(gate_logits).scatter_(-1, topk_indices, 1.0)
        coeffs = F.softmax(gate_logits, dim=-1) * sparse_mask
        
        # Factorized reconstruction with sparse bases
        U_combined = torch.einsum('bm, mor -> bor', coeffs, self.U)
        V_combined = torch.einsum('bm, mri -> bri', coeffs, self.V)
        W = torch.bmm(U_combined, V_combined)
        
        if single:
            W = W.squeeze(0)
            
        return W

    '''
    def quantize(self, W, bits=8):
        """Improved quantization with min-max scaling"""
        min_val = W.min()
        max_val = W.max()
        scale = (max_val - min_val) / (2**bits - 1) if max_val > min_val else 1.0
        zero_point = torch.round(-min_val / scale)
        W_q = torch.clamp(torch.round((W - min_val) / scale), 0, 2**bits-1)
        return W_q, (scale, zero_point)
    '''
    def quantize(self, W, bits=8):
        """Improved quantization with min-max scaling"""
        min_val = W.min()
        max_val = W.max()
        scale = (max_val - min_val) / (2**bits - 1) if max_val > min_val else 1.0
        zero_point = torch.round(-min_val / scale)
        W_q = torch.clamp(torch.round(W / scale), -2**(bits-1), 2**(bits-1)-1)  # FIXED RANGE
        return W_q, (scale, zero_point)

    def set_rank(self, new_rank):
        """Dynamically adjust factorization rank"""
        self.rank = new_rank

# Test
if __name__ == "__main__":
    B, G, H, O, I, M = 4, 32, 64, 128, 128, 16
    z = torch.randn(B, G)
    layer = FactorizedBasisHyperLayer(G, H, O, I, M, rank=32)
    W, b = layer(z)
    assert W.shape == (B, O, I)
    assert b.shape == (B, O)
    print("✅ FactorizedBasisHyperLayer test passed!")