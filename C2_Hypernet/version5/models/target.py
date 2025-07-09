# models/target_mlp_perlayer.py
import torch
import torch.nn as nn
from models.hyperlayer import HyperLayer

class TargetMLP_PerLayer(nn.Module):
    def __init__(self, genome_dim=64, hidden_dim=128):
        super().__init__()
        self.genome_dim = genome_dim

        # Define each layer’s dimensions
        self.layer_dims = [
            (256, 784),   # Layer 1
            (128, 256),   # Layer 2
            (10, 128)     # Output
        ]

        # One HyperLayer per actual layer
        self.hyperlayers = nn.ModuleList([
            HyperLayer(genome_dim, hidden_dim, out_dim, in_dim)
            for (out_dim, in_dim) in self.layer_dims
        ])

    def forward(self, x, z):
        B = x.size(0)
        h = x

        for i, hyper in enumerate(self.hyperlayers):
            W, b = hyper(z.expand(B, -1))  # [B, out, in], [B, out]
            h = torch.einsum('boi,bi->bo', W, h) + b
            if i < len(self.hyperlayers) - 1:
                h = torch.relu(h)

        return h
