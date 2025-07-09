import torch
import torch.nn as nn

class TargetMLP_Generic(nn.Module):
    def __init__(self, layer_dims, hypernet):
        super().__init__()
        self.layer_dims = layer_dims
        self.hypernet = hypernet

    def forward(self, x, z):
        h = x
        for i in range(len(self.layer_dims) - 1):
            in_dim = self.layer_dims[i]
            out_dim = self.layer_dims[i + 1]
            W = self.hypernet.fold(z, out_dim, in_dim)
            b = self.hypernet.fold_bias(z, out_dim)
            h = torch.einsum('boi,bi->bo', W, h) + b
            if i < len(self.layer_dims) - 2:
                h = torch.relu(h)
            del W, b
        return h
