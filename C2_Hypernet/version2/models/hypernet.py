import torch
import torch.nn as nn

class HyperNetwork(nn.Module):
    def __init__(self, genome_dim=128, hidden_dim=64, target_in=784, target_out=10):
        super().__init__()
        self.fc1 = nn.Linear(genome_dim, hidden_dim)
        self.fc_w = nn.Linear(hidden_dim, target_out * target_in)
        self.fc_b = nn.Linear(hidden_dim, target_out)

    def forward(self, z):
        h = torch.relu(self.fc1(z))
        w_flat = self.fc_w(h)
        b = self.fc_b(h)
        W = w_flat.view(-1, 10, 784)
        return W, b

class TargetMLP(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, W, b):
        return torch.einsum('boi,bi->bo', W, x) + b
