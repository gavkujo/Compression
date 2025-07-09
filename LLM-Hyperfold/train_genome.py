# train_genome.py

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import LlamaModel
from models.basis_hyper import BasisHyperLayer

# ─── CONFIG ────────────────────────────────────────────────────────────────
PRETRAINED_NAME = "EleutherAI/gpt-j-6b"
GENOME_DIM      = 64
M_BASIS         = 128
HIDDEN_DIM      = 256
LR              = 1e-3
EPOCHS          = 5
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH       = "checkpoints/genome_meta.pth"

# ─── LOAD PRETRAINED LLaMA ─────────────────────────────────────────────────
base = LlamaModel.from_pretrained(PRETRAINED_NAME, torch_dtype=torch.float32)
base.eval().to(DEVICE)

# 1) Extract original weights
orig_attn, orig_mlp = [], []
for layer in base.model.layers:
    a = layer.self_attn
    orig_attn.append({
        'q': (a.q_proj.weight.data.clone(), a.q_proj.bias.data.clone()),
        'k': (a.k_proj.weight.data.clone(), a.k_proj.bias.data.clone()),
        'v': (a.v_proj.weight.data.clone(), a.v_proj.bias.data.clone()),
        'o': (a.o_proj.weight.data.clone(), a.o_proj.bias.data.clone()),
    })
    m = layer.mlp
    orig_mlp.append({
        'up':   (m.up_proj.weight.data.clone(),   m.up_proj.bias.data.clone()),
        'down': (m.down_proj.weight.data.clone(), m.down_proj.bias.data.clone()),
    })

num_layers = len(orig_attn)

# 2) Initialize genome + hypernets
z = nn.Parameter(torch.randn(num_layers, GENOME_DIM, device=DEVICE))
hyper_attn = nn.ModuleList([
    nn.ModuleDict({
        p: BasisHyperLayer(GENOME_DIM, HIDDEN_DIM,
                           orig_attn[i][p][0].size(0),
                           orig_attn[i][p][0].size(1), M=M_BASIS)
        for p in orig_attn[i]
    })
    for i in range(num_layers)
]).to(DEVICE)

hyper_mlp = nn.ModuleList([
    nn.ModuleDict({
        p: BasisHyperLayer(GENOME_DIM, HIDDEN_DIM,
                           orig_mlp[i][p][0].size(0),
                           orig_mlp[i][p][0].size(1), M=M_BASIS)
        for p in orig_mlp[i]
    })
    for i in range(num_layers)
]).to(DEVICE)

# 3) Optimizer & loss
opt = optim.Adam([z] + list(hyper_attn.parameters()) + list(hyper_mlp.parameters()), lr=LR)
crit = nn.MSELoss()

# 4) Meta-training loop
print("🔨 Meta-training genome...")
for epoch in range(1, EPOCHS+1):
    total_loss = 0.0

    for i in range(num_layers):
        zi = z[i].unsqueeze(0)  # shape [1, GENOME_DIM]

        # reconstruct attn
        for p, (W_true, b_true) in orig_attn[i].items():
            W_gen, b_gen = hyper_attn[i][p](zi)
            total_loss += crit(W_gen, W_true.unsqueeze(0).to(DEVICE))
            total_loss += crit(b_gen, b_true.unsqueeze(0).to(DEVICE))

        # reconstruct mlp
        for p, (W_true, b_true) in orig_mlp[i].items():
            W_gen, b_gen = hyper_mlp[i][p](zi)
            total_loss += crit(W_gen, W_true.unsqueeze(0).to(DEVICE))
            total_loss += crit(b_gen, b_true.unsqueeze(0).to(DEVICE))

    opt.zero_grad()
    total_loss.backward()
    opt.step()

    print(f"Epoch {epoch}/{EPOCHS} — Loss: {total_loss.item():.4f}")

# 5) Save genome + hypernet
torch.save({
    'z': z.detach().cpu(),
    'hyper_attn': hyper_attn.state_dict(),
    'hyper_mlp':  hyper_mlp.state_dict(),
}, SAVE_PATH)
print("✅ Genome meta-training complete. Saved to", SAVE_PATH)
