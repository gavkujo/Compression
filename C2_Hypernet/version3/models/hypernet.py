import torch
import torch.nn as nn

class HyperNetworkLarge(nn.Module):
    def __init__(self, genome_dim=128, hidden_dim=128, shapes=None):
        super().__init__()
        self.shapes = shapes or [
            (256, 784),   # W1
            (128, 256),   # W2
            (10, 128)     # W3
        ]
        self.bias_shapes = [ (s[0],) for s in self.shapes ]

        total_params = sum(o*i for o, i in self.shapes)
        total_bias = sum(o for o, _ in self.shapes)

        self.fc1 = nn.Linear(genome_dim, hidden_dim)
        self.fc_w = nn.Linear(hidden_dim, total_params)
        self.fc_b = nn.Linear(hidden_dim, total_bias)

    def forward(self, z):
        B = z.size(0)
        h = torch.relu(self.fc1(z))
        w_flat = self.fc_w(h)
        b_flat = self.fc_b(h)

        weights = []
        biases = []

        # reshape
        idx = 0
        for out_dim, in_dim in self.shapes:
            num_params = out_dim * in_dim
            w = w_flat[:, idx:idx+num_params].view(B, out_dim, in_dim)
            weights.append(w)
            idx += num_params

        idx = 0
        for b_shape in self.bias_shapes:
            num_params = b_shape[0]
            b = b_flat[:, idx:idx+num_params]
            biases.append(b)
            idx += num_params

        return weights, biases

class TargetMLP_Large(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x, weights, biases):
        # weights: [B, L, out, in], biases: [B, L, out]
        B = x.size(0)
        h = x

        for i in range(len(weights)):
            w = weights[i]  # shape: [B, out, in]
            b = biases[i]   # shape: [B, out]
            h = torch.einsum('boi,bi->bo', w, h) + b
            if i < len(weights) - 1:
                h = torch.relu(h)
        return h
