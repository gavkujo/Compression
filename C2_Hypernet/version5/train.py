# C2_Hypernet/version4/train.py
import torch
import torch.nn as nn
import torch.optim as optim
import os
from models.metalearner import MetaLearnerPerLayer
from data import get_mnist_loaders
from utils import save_genome

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader, test_loader = get_mnist_loaders()

model = MetaLearnerPerLayer(genome_dim=64).to(device)
opt = optim.Adam(model.parameters(), lr=1e-3)
crit = nn.CrossEntropyLoss()

print("🔨 Training MetaLearner with per-layer HyperNet...")
for epoch in range(5):
    total_loss = 0
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = crit(logits, y)
        loss.backward()
        opt.step()
        total_loss += loss.item() * x.size(0)
    print(f"📈 Epoch {epoch+1}: Loss = {total_loss / len(train_loader.dataset):.4f}")

save_genome(model)
os.makedirs("C2_Hypernet/version5/checkpoints", exist_ok=True)
torch.save(model.state_dict(), "C2_Hypernet/version5/checkpoints/full_model.pt")
print("✅ Training complete! Genome + model saved.")
