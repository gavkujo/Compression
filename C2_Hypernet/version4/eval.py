import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.metalearner import MetaLearner

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
layer_dims = [784, 256, 128, 10]

# Data
transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
test_ds = datasets.MNIST(root="C2_Hypernet/version4/data", train=False, download=True, transform=transform)
test_loader = DataLoader(test_ds, batch_size=1000)

# Model
model = MetaLearner(genome_dim=128, layer_dims=layer_dims).to(device)
model.load_state_dict(torch.load("C2_Hypernet/version4/checkpoints/universal_hyper.pt"))
model.z.data = torch.load("C2_Hypernet/version4/checkpoints/universal_genome.pt").to(device)
model.eval()

# Accuracy
correct = 0
for x, y in test_loader:
    x, y = x.to(device), y.to(device)
    preds = model(x).argmax(dim=1)
    correct += (preds == y).sum().item()

acc = correct / len(test_loader.dataset) * 100
print(f"📊 Accuracy of Universal HyperNet w/ TargetMLP_Generic: {acc:.2f}%")
