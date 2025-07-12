import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class FactorizedBasisHyperLayer(nn.Module):
    """Factorized basis-decomposition hypernetwork with reduced parameters"""
    def __init__(
        self,
        genome_dim: int,
        hidden_dim: int,
        out_dim: int,
        in_dim: int,
        M: int = 16, #32
        rank: int = 32 #64
    ) -> None:
        super().__init__()
        self.M = M
        self.rank = rank
        
        # Factorized basis tensors
        self.U = nn.Parameter(torch.randn(M, out_dim, rank) * 0.01) #0.02
        self.V = nn.Parameter(torch.randn(M, rank, in_dim) * 0.01) #0.02
        
        # Efficient hypernetwork
        self.fc1 = nn.Linear(genome_dim, hidden_dim)
        self.fc_coef = nn.Linear(hidden_dim, M)
        # self.fc_bias = nn.Linear(hidden_dim, out_dim)
        
        # Quantization parameters
        self.quant_scale = nn.Parameter(torch.ones(1))
        self.quant_zero = nn.Parameter(torch.zeros(1))

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        single = z.dim() == 1
        if single:
            z = z.unsqueeze(0)

        h = F.gelu(self.fc1(z))
        coeffs = self.fc_coef(h)
        #bias = self.fc_bias(h)
        
        # Factorized reconstruction
        U_combined = torch.einsum('bm, mor -> bor', coeffs, self.U)
        V_combined = torch.einsum('bm, mri -> bri', coeffs, self.V)
        W = torch.bmm(U_combined, V_combined)
        
        if single:
            W = W.squeeze(0)
            #bias = bias.squeeze(0)
            
        return W, #bias

    def quantize(self, W, bits=8):
        """Improved quantization with min-max scaling"""
        min_val = W.min()
        max_val = W.max()
        scale = (max_val - min_val) / (2**bits - 1) if max_val > min_val else 1.0
        zero_point = torch.round(-min_val / scale)
        W_q = torch.clamp(torch.round((W - min_val) / scale), 0, 2**bits-1)
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