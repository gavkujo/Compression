import torch
import torch.nn as nn
from models.hypernet import TargetMLP_Large
from models.hypernet import HyperNetworkLarge

class MetaLearnerLarge(nn.Module):
    def __init__(self, genome_dim=64):
        super().__init__()
        self.z = nn.Parameter(torch.randn(1, genome_dim))
        self.hypernet = HyperNetworkLarge(genome_dim)
        self.target = TargetMLP_Large()

    def forward(self, x):
        B = x.size(0)
        z_batch = self.z.expand(B, -1)
        weights, biases = self.hypernet(z_batch)
        return self.target(x, weights, biases)