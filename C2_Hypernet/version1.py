'''
🔨 Training...
Epoch 1: Loss = 0.3854
Epoch 2: Loss = 0.3021
Epoch 3: Loss = 0.2872
✅ Trained model accuracy: 92.01%
✅ Fresh model w/ loaded genome accuracy: 10.51%

📊 Accuracy Comparison:
                 Model  Accuracy (%)
               Trained         92.01
Fresh w/ Loaded Genome         10.51
'''



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pandas as pd
import os

# --- 1. Define HyperNetwork and TargetMLP ---

class HyperNetwork(nn.Module):
    def __init__(self, genome_dim=128, hidden_dim=64, target_in=784, target_out=10):
        super().__init__()
        self.fc1 = nn.Linear(genome_dim, hidden_dim)
        self.fc_w = nn.Linear(hidden_dim, target_out * target_in)
        self.fc_b = nn.Linear(hidden_dim, target_out)

    def forward(self, z):
        h = torch.relu(self.fc1(z))
        w_flat = self.fc_w(h)
        b = self.fc_b(h)
        W = w_flat.view(-1, 10, 784)
        return W, b

class TargetMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(784, 10)

    def forward(self, x, W, b):
        return torch.einsum('boi,bi->bo', W, x) + b

# --- 2. MetaLearner ---
class MetaLearner(nn.Module):
    def __init__(self, genome_dim=128):
        super().__init__()
        self.z = nn.Parameter(torch.randn(1, genome_dim))
        self.hypernet = HyperNetwork(genome_dim)
        self.target = TargetMLP()

    def forward(self, x):
        B = x.size(0)
        z_batch = self.z.expand(B, -1)
        W, b = self.hypernet(z_batch)
        return self.target(x, W, b)

# --- 3. Data Loaders ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda t: t.view(-1))
])
train_ds = datasets.MNIST(root='C2_Hypernet/mnist_data', train=True, download=True, transform=transform)
test_ds  = datasets.MNIST(root='C2_Hypernet/mnist_data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=1000)

# --- 4. Train Model ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MetaLearner().to(device)
opt = optim.Adam(model.parameters(), lr=1e-3)
crit = nn.CrossEntropyLoss()

print("🔨 Training...")
for epoch in range(3):
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

# --- 5. Evaluate After Training ---
model.eval()
correct = 0
for x, y in test_loader:
    x, y = x.to(device), y.to(device)
    preds = model(x).argmax(dim=1)
    correct += (preds == y).sum().item()
acc_trained = correct / len(test_ds) * 100
print(f"✅ Trained model accuracy: {acc_trained:.2f}%")

# --- 6. Save Genome ---
os.makedirs('C2_Hypernet/checkpoints_2', exist_ok=True)
torch.save(model.z.detach().cpu(), 'C2_Hypernet/checkpoints_2/genome.pt')

# --- 7. Load & Evaluate Fresh Model ---
fresh = MetaLearner().to(device)
loaded_z = torch.load('C2_Hypernet/checkpoints_2/genome.pt').to(device)
with torch.no_grad():
    fresh.z.copy_(loaded_z)
fresh.eval()

correct = 0
for x, y in test_loader:
    x, y = x.to(device), y.to(device)
    preds = fresh(x).argmax(dim=1)
    correct += (preds == y).sum().item()
acc_loaded = correct / len(test_ds) * 100
print(f"✅ Fresh model w/ loaded genome accuracy: {acc_loaded:.2f}%")

# --- 8. Side-by-Side Comparison ---
import pandas as pd
df = pd.DataFrame({
    'Model': ['Trained', 'Fresh w/ Loaded Genome'],
    'Accuracy (%)': [acc_trained, acc_loaded]
})
print("\n📊 Accuracy Comparison:")
print(df.to_string(index=False))
