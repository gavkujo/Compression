import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time, torch, torch.nn as nn

from models import TinyTransformerLayer  # make sure your __init__ exposes it

# ─── CONFIG ────────────────────────────────────────────────────────────────
DATA_DIR    = "C2_Hypernet/version7/data"
BATCH_SIZE  = 4
EPOCHS      = 5
LR          = 1e-3
GENOME_DIM  = 16
EMBED_DIM   = 64
NUM_HEADS   = 2
M_BASIS     = 8
HIDDEN_DIM  = 32
SAVE_DIR    = "C2_Hypernet/version7/checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── DATA ──────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.squeeze(0).T)  # [28,28] → [T,E]
])
train_ds   = datasets.MNIST(DATA_DIR, train=True,  download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)


# ─── MODEL ─────────────────────────────────────────────────────────────────
class HypernetClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # learnable genome vector
        self.z = nn.Parameter(torch.randn(1, GENOME_DIM))
        self.input_proj = nn.Linear(28, EMBED_DIM)
        self.block = TinyTransformerLayer(
            genome_dim=GENOME_DIM,
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            M=M_BASIS
        )
        # final classifier on last token
        self.classifier = nn.Linear(EMBED_DIM, 10)

    def forward(self, x):
        # x: [B, 28, 28]  → project to E=512 dims
        B, T, Ein = x.size()
        if Ein != EMBED_DIM:
            # embed 28→512 with linear
            x = x.reshape(B, T, Ein)
            if not hasattr(self, "input_proj"):
                self.input_proj = nn.Linear(Ein, EMBED_DIM).to(x.device)
            x = self.input_proj(x)

        z_batch = self.z.expand(B, -1)
        x = self.block(x, z_batch)        # [B, T, E]
        # classification on last token
        logits = self.classifier(x[:, -1, :])  # [B,10]
        return logits

model = HypernetClassifier().to(DEVICE)
opt   = optim.Adam(model.parameters(), lr=LR)
crit  = nn.CrossEntropyLoss()


# ─── TRAIN LOOP ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🔨 Starting training…")

    '''
    B, T, E, G = 4, 28, 512, 64
    x = torch.randn(B, T, E)
    z = torch.randn(B, G)

    model = TinyTransformerLayer(genome_dim=G, embed_dim=E, num_heads=8, M=128, hidden_dim=256)
    crit  = nn.CrossEntropyLoss()
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    y     = torch.randint(0, 10, (B,))

    t0 = time.time()
    logits = model(x, z)
    loss   = crit(logits[:,0,:], y)   # pick token 0
    loss.backward()
    opt.step()
    t1 = time.time()

    print(f"⏱️ 1 step time: {(t1-t0)*1000:.1f} ms")
    exit()
    '''

    for epoch in range(1, EPOCHS+1):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logits = model(x)                   # [B,10]
            loss   = crit(logits, y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)
        avg_loss = total_loss / len(train_loader.dataset)
        print(f"🚧 Epoch {epoch}/{EPOCHS} — Loss: {avg_loss:.4f}")

        # optional: save intermediate checkpoint
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"epoch{epoch}.pt"))

    # ─── FINAL SAVE ───────────────────────────────────────────────────────────
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "final_model.pt"))
    print("✅ Training done! Models in", SAVE_DIR)
