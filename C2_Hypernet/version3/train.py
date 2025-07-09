import torch
import torch.nn as nn
import torch.optim as optim
from models.metalearner import MetaLearnerLarge as MetaLearner
from data import get_mnist_loaders
from utils import save_genome

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader, test_loader = get_mnist_loaders()
model = MetaLearner().to(device)

opt = optim.Adam(model.parameters(), lr=1e-3)
crit = nn.CrossEntropyLoss()

print("🔨 Training MetaLearner...")
for epoch in range(5):  # try 5 epochs for better accuracy
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = crit(logits, y)
        loss.backward()
        opt.step()
        total_loss += loss.item() * x.size(0)
    print(f"Epoch {epoch+1}: Loss = {total_loss/len(train_loader.dataset):.4f}")

# Save the trained genome
save_genome(model)
torch.save(model.state_dict(), 'C2_Hypernet/version3/checkpoints/full_model.pt')
print("✅ Training complete. Model and genome saved.")
