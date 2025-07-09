import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os
from torch.utils.data import DataLoader
from models.metalearner import MetaLearner

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

layer_dims = [784, 256, 128, 10]  # Match your VanillaTargetMLP_Large

# Load MNIST
transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
train_ds = datasets.MNIST(root="C2_Hypernet/version4/data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

model = MetaLearner(genome_dim=128, layer_dims=layer_dims).to(device)
opt = optim.Adam(model.parameters(), lr=1e-3)
crit = nn.CrossEntropyLoss()

print("🔨 Training MetaLearner...")
for epoch in range(5):
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = crit(logits, y)
        loss.backward()
        opt.step()
        total_loss += loss.item() * x.size(0)
    print(f"Epoch {epoch+1}: Loss = {total_loss/len(train_ds):.4f}")

os.makedirs("C2_Hypernet/version4/checkpoints", exist_ok=True)
torch.save(model.state_dict(), "C2_Hypernet/version4/checkpoints/universal_hyper.pt")
torch.save(model.z.detach().cpu(), "C2_Hypernet/version4/checkpoints/universal_genome.pt")
