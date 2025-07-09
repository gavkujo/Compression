# models/basis_hyper.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BasisHyperLayer(nn.Module):
    """
    A basis‑decomposition hypernetwork layer that generates a weight matrix W and bias vector b 
    from a low‑dimensional genome vector z.

    Attributes:
        bases (nn.Parameter): A learnable tensor of shape (M, out_dim, in_dim) containing M basis matrices.
        fc1 (nn.Linear): Maps genome vector z to a hidden representation of size hidden_dim.
        fc_coef (nn.Linear): Maps hidden representation to M mixing coefficients.
        fc_bias (nn.Linear): Maps hidden representation to an output bias vector of size out_dim.
    """

    def __init__(
        self,
        genome_dim: int,
        hidden_dim: int,
        out_dim: int,
        in_dim: int,
        M: int = 64
    ) -> None:
        """
        Args:
            genome_dim: Size of the input genome vector z.
            hidden_dim: Size of the intermediate hidden layer.
            out_dim: Number of output features (rows of W).
            in_dim: Number of input features  (columns of W).
            M: Number of basis matrices to learn.
        """
        super().__init__()
        self.bases = nn.Parameter(
            torch.randn(M, out_dim, in_dim) * 0.02,
            requires_grad=True
        )
        self.fc1     = nn.Linear(genome_dim, hidden_dim)
        self.fc_coef = nn.Linear(hidden_dim, M)
        self.fc_bias = nn.Linear(hidden_dim, out_dim)


    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            z: [B, genome_dim] or [genome_dim]
        Returns:
            W: [B, out_dim, in_dim] or [out_dim, in_dim]
            b: [B, out_dim] or [out_dim]
        """
        # 1) Handle single-vector input by batching it
        single = False
        if z.dim() == 1:
            single = True
            z = z.unsqueeze(0)             # become [1, genome_dim]

        # 2) Forward through hypernet
        h      = F.relu(self.fc1(z))      # [B, hidden_dim]
        coeffs = self.fc_coef(h)          # [B, M]
        bias   = self.fc_bias(h)          # [B, out_dim]
        W      = torch.einsum("bm,moi->boi", coeffs, self.bases)

        # 3) If single-vector, squeeze back
        if single:
            W    = W.squeeze(0)           # [out_dim, in_dim]
            bias = bias.squeeze(0)        # [out_dim]

        return W, bias

# === Self‑test ===
if __name__ == "__main__":
    # Quick sanity check
    B, G, H, O, I, M = 4, 16, 32, 64, 64, 8
    z = torch.randn(B, G)
    layer = BasisHyperLayer(genome_dim=G, hidden_dim=H, out_dim=O, in_dim=I, M=M)

    W, b = layer(z)
    assert W.shape == (B, O, I), f"Expected W shape {(B,O,I)}, got {W.shape}"
    assert b.shape == (B, O),      f"Expected b shape {(B,O)}, got {b.shape}"

    # Single‑vector inference
    z0 = torch.randn(G)
    W0, b0 = layer(z0)
    assert W0.shape == (O, I),     f"Expected W0 shape {(O,I)}, got {W0.shape}"
    assert b0.shape == (O,),       f"Expected b0 shape {(O,)}, got {b0.shape}"

    print("✅ BasisHyperLayer smoke‑test passed!")
