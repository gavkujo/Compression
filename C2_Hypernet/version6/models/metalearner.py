import torch
import torch.nn as nn
from models.attention import TargetSelfAttention

class MetaLearnerTransformer(nn.Module):
    def __init__(self, genome_dim=64):
        super().__init__()
        self.z = nn.Parameter(torch.randn(1, genome_dim))
        self.target = TargetSelfAttention(genome_dim=genome_dim,embed_dim=28)
        self.classifier = nn.Linear(28, 10)

    def forward(self, x):
        B = x.size(0)
        z_batch = self.z.expand(B, -1)
        x_out =  self.target(x, z_batch)
        logits = self.classifier(x_out[:, -1, :])
        return logits
