import torch
import torch.nn as nn
from models.hypernet import UniversalHyperNet
from models.target import TargetMLP_Generic

class MetaLearner(nn.Module):
    def __init__(self, genome_dim=128, layer_dims=None):
        super().__init__()
        self.z = nn.Parameter(torch.randn(1, genome_dim))
        self.hypernet = UniversalHyperNet(genome_dim)
        self.target = TargetMLP_Generic(layer_dims, self.hypernet)

    def forward(self, x):
        z = self.z.expand(x.size(0), -1)
        return self.target(x, z)
