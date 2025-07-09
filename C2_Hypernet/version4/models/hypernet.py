import torch
import torch.nn as nn

class UniversalHyperNet(nn.Module):
    def __init__(self, genome_dim=128, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(genome_dim, hidden_dim)

    def fold(self, z, out_dim, in_dim):
        h = torch.relu(self.fc1(z))
        fc_w = nn.Linear(h.size(1), out_dim * in_dim).to(z.device)
        W = fc_w(h).view(-1, out_dim, in_dim)
        return W

    def fold_bias(self, z, out_dim):
        h = torch.relu(self.fc1(z))
        fc_b = nn.Linear(h.size(1), out_dim).to(z.device)
        b = fc_b(h)
        return b
