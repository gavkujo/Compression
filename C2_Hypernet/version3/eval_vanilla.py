import torch
import tracemalloc
from vanillatargetmlp import VanillaTargetMLP_Large
from data import get_mnist_loaders
from metrics import count_parameters, get_memory_usage, time_inference

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_, test_loader = get_mnist_loaders()

model = VanillaTargetMLP_Large().to(device)
weights = torch.load('vanilla_target_weights.pt')
model.load_state_dict(weights)
model.eval()

tracemalloc.start()
correct = 0

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()

acc = correct / len(test_loader.dataset) * 100
params = count_parameters(model)
mem = get_memory_usage()
x_sample = next(iter(test_loader))[0][:1].to(device)
inf_time = time_inference(model, x_sample, device)

current, peak = tracemalloc.get_traced_memory()
print(f"\nVanilla TargetMLP Accuracy: {acc:.2f}%")
print(f"Param count: {params}")
print(f"RAM used: {mem:.2f} MB")
print(f"Inference time (1 sample): {inf_time:.2f} ms")
print(f"Tracemalloc current: {current/(1024**2):.2f} MB")
print(f"Tracemalloc peak: {peak/(1024**2):.2f} MB")

tracemalloc.stop()
