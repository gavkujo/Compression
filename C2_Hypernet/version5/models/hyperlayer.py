# models/hyperlayer.py
import torch
import torch.nn as nn

class HyperLayer(nn.Module):
    def __init__(self, genome_dim, hidden_dim, out_dim, in_dim):
        super().__init__()
        self.out_dim = out_dim
        self.in_dim = in_dim

        self.fc1 = nn.Linear(genome_dim, hidden_dim)
        self.fc_w = nn.Linear(hidden_dim, out_dim * in_dim)
        self.fc_b = nn.Linear(hidden_dim, out_dim)

    def forward(self, z):
        h = torch.relu(self.fc1(z))
        w = self.fc_w(h).view(-1, self.out_dim, self.in_dim)
        b = self.fc_b(h)
        return w, b
