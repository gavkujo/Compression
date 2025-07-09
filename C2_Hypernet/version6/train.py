# C2_Hypernet/version6/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os
from torch.utils.data import DataLoader
from models.metalearner import MetaLearnerTransformer
from utils import save_genome

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare data: [B, 1, 28, 28] → [B, 28, 28]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.squeeze(0).T)  # [28, 28]
])
train_ds = datasets.MNIST(root="C2_Hypernet/version6/data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)

model = MetaLearnerTransformer(genome_dim=64).to(device)
opt = optim.Adam(model.parameters(), lr=1e-3)
crit = nn.MSELoss()  # or CrossEntropyLoss with flattening at the end

print("🔨 Training MetaLearner (self-attention)...")
for epoch in range(5):
    total_loss = 0
    for x, y in train_loader:
        x = x.to(device)  # [B, 28, 28]
        y = nn.functional.one_hot(y, num_classes=10).float().to(device)  # [B, 10]

        opt.zero_grad()
        out = model(x) # take last token as classifier output
        loss = crit(out, y)
        loss.backward()
        opt.step()
        total_loss += loss.item() * x.size(0)

    avg_loss = total_loss / len(train_loader.dataset)
    print(f"📈 Epoch {epoch+1}: Loss = {avg_loss:.4f}")

# Save model + genome
save_genome(model)
os.makedirs("C2_Hypernet/version6/checkpoints", exist_ok=True)
torch.save(model.state_dict(), "C2_Hypernet/version6/checkpoints/full_model.pt")
print("✅ Training complete! 🎉")
