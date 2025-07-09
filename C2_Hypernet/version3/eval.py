import torch
import tracemalloc
from models.metalearner import MetaLearnerLarge as MetaLearner
from data import get_mnist_loaders
from utils import load_genome
from metrics import count_parameters, get_memory_usage, time_inference

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data + model
_, test_loader = get_mnist_loaders()
model = MetaLearner().to(device)
load_genome(model)
model.load_state_dict(torch.load("C2_Hypernet/version3/checkpoints/full_model.pt"))
tracemalloc.start()
model.eval()

# Evaluate accuracy
correct = 0
for x, y in test_loader:
    x, y = x.to(device), y.to(device)
    preds = model(x).argmax(dim=1)
    correct += (preds == y).sum().item()

acc = correct / len(test_loader.dataset) * 100

# Resource + Compression Stats
target_params = 0
for p in model.target.parameters():
    target_params += p.numel()

hypernet_params = count_parameters(model.hypernet)
genome_size = model.z.numel()

compression_ratio = target_params / genome_size

# Memory & inference
x_sample = next(iter(test_loader))[0][:1].to(device)
mem = get_memory_usage()
inference_time = time_inference(model, x_sample, device)

# Print Results
print(f"\n Accuracy of fresh model w/ loaded genome: {acc:.2f}%")
print(f" Target model param count: {target_params}")
print(f" HyperNetwork param count: {hypernet_params}")
print(f" Genome vector size: {genome_size}")
print(f" Compression ratio (Target / Genome): {compression_ratio:.2f}x")
print(f" Inference time (1 sample): {inference_time:.2f} ms")
print(f" RAM used: {mem:.2f} MB")

current, peak = tracemalloc.get_traced_memory()
print(f"Current RAM usage: {current / (1024 ** 2):.2f} MB")
print(f"Peak RAM usage: {peak / (1024 ** 2):.2f} MB")

tracemalloc.stop()
