import torch
import torch.nn as nn

class HyperNetwork(nn.Module):
    def __init__(self, genome_dim=128, hidden_dim=64, target_in=32, target_out=16):
        super().__init__()
        self.genome_dim = genome_dim
        self.hidden_dim = hidden_dim
        self.target_in = target_in
        self.target_out = target_out
        
        self.fc1 = nn.Linear(genome_dim, hidden_dim)
        self.fc_w = nn.Linear(hidden_dim, target_out * target_in)
        self.fc_b = nn.Linear(hidden_dim, target_out)

    def forward(self, z):
        # Debug: input genome
        print(f"[HyperNetwork] z.shape = {z.shape}")  # (B, 128)
        
        h_lin = self.fc1(z)
        print(f"[HyperNetwork] after fc1 (linear) h_lin.shape = {h_lin.shape}")  # (B, hidden_dim)
        
        h = torch.relu(h_lin)
        print(f"[HyperNetwork] after ReLU h.shape = {h.shape}")  # (B, hidden_dim)
        
        w_flat = self.fc_w(h)
        print(f"[HyperNetwork] w_flat.shape = {w_flat.shape}")  # (B, target_out*target_in)
        print(f"[HyperNetwork] sample w_flat[0,:5] = {w_flat[0, :5].detach().cpu().numpy()}")  # peek some values
        
        b = self.fc_b(h)
        print(f"[HyperNetwork] b.shape = {b.shape}")  # (B, target_out)
        print(f"[HyperNetwork] sample b[0,:5] = {b[0, :5].detach().cpu().numpy()}")  # peek some values
        
        W = w_flat.view(-1, self.target_out, self.target_in)
        print(f"[HyperNetwork] reshaped W.shape = {W.shape}")  # (B, target_out, target_in)
        
        return W, b


class TargetMLP(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # dummy linear; we override its weights anyway
        self.linear = nn.Linear(in_features, out_features, bias=True)

    def forward(self, x, W, b):
        print(f"[TargetMLP] x.shape = {x.shape}")            # (B, in_features)
        print(f"[TargetMLP] W.shape = {W.shape}, b.shape = {b.shape}")
        
        # batched matmul + bias
        y = torch.einsum('boi,bi->bo', W, x) + b
        
        print(f"[TargetMLP] output y.shape = {y.shape}")
        print(f"[TargetMLP] sample y[0,:5] = {y[0, :5].detach().cpu().numpy()}")
        return y


def unit_test():
    B = 5
    genome_dim = 128
    in_dim = 32
    out_dim = 16
    hidden_dim = 64

    z = torch.randn(B, genome_dim)
    x = torch.randn(B, in_dim)

    hypernet = HyperNetwork(genome_dim, hidden_dim, in_dim, out_dim)
    target = TargetMLP(in_dim, out_dim)

    print("\n=== Running HyperNetwork ===")
    W, b = hypernet(z)

    print("\n=== Running TargetMLP ===")
    y = target(x, W, b)

    assert W.shape == (B, out_dim, in_dim), f"W shape mismatch: got {W.shape}"
    assert b.shape == (B, out_dim), f"b shape mismatch: got {b.shape}"
    assert y.shape == (B, out_dim), f"y shape mismatch: got {y.shape}"
    print("\n✅ Unit test passed: shapes all good!")

if __name__ == "__main__":
    unit_test()
