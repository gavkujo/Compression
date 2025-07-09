# C2_Hypernet/version4/eval.py
import torch
import tracemalloc
from models.metalearner import MetaLearnerPerLayer
from data import get_mnist_loaders
from utils import load_genome
from metrics import count_parameters, get_memory_usage, time_inference

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_, test_loader = get_mnist_loaders()
model = MetaLearnerPerLayer(genome_dim=64).to(device)
load_genome(model)
model.load_state_dict(torch.load("C2_Hypernet/version5/checkpoints/full_model.pt"))

tracemalloc.start()
model.eval()

correct = 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()

acc = correct / len(test_loader.dataset) * 100

# --- Stats
hypernet_params = sum(p.numel() for p in model.target.hyperlayers.parameters())
target_shape = [(256, 784), (128, 256), (10, 128)]
target_params = sum(o * i + o for o, i in target_shape)
genome_size = model.z.numel()
compression_ratio = target_params / genome_size

x_sample = next(iter(test_loader))[0][:1].to(device)
ram_used = get_memory_usage()
infer_time = time_inference(model, x_sample, device)

# --- Output
print(f"\nAccuracy of fresh model w/ loaded genome: {acc:.2f}%")
print(f"Target model param count: {target_params}")
print(f"HyperNetwork param count: {hypernet_params}")
print(f"Genome vector size: {genome_size}")
print(f"Compression ratio (Target / Genome): {compression_ratio:.2f}x")
print(f"Inference time (1 sample): {infer_time:.2f} ms")
print(f"RAM used: {ram_used:.2f} MB")

current, peak = tracemalloc.get_traced_memory()
print(f"Tracemalloc current: {current / 1024**2:.2f} MB")
print(f"Tracemalloc peak: {peak / 1024**2:.2f} MB")
tracemalloc.stop()
