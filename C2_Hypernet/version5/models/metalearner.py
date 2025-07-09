# models/metalearner_perlayer.py
import torch
import torch.nn as nn
from models.target import TargetMLP_PerLayer

class MetaLearnerPerLayer(nn.Module):
    def __init__(self, genome_dim=64):
        super().__init__()
        self.z = nn.Parameter(torch.randn(1, genome_dim))
        self.target = TargetMLP_PerLayer(genome_dim)

    def forward(self, x):
        B = x.size(0)
        z_batch = self.z.expand(B, -1)
        return self.target(x, z_batch)
