import torch
import torch.nn as nn
from models.hypernet import HyperNetwork, TargetMLP

class MetaLearner(nn.Module):
    def __init__(self, genome_dim=128):
        super().__init__()
        self.z = nn.Parameter(torch.randn(1, genome_dim))
        self.hypernet = HyperNetwork(genome_dim)
        self.target = TargetMLP()

    def forward(self, x):
        B = x.size(0)
        z_batch = self.z.expand(B, -1)
        W, b = self.hypernet(z_batch)
        return self.target(x, W, b)
