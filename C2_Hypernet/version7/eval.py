import os
import time
import torch
import tracemalloc
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from train import HypernetClassifier  # make sure this is fixed as per earlier

# ─── CONFIG ────────────────────────────────────────────────────────────────
DATA_DIR   = "C2_Hypernet/version7/data"
BATCH_SIZE = 64
CKPT_PATH  = "C2_Hypernet/version7/checkpoints/final_model.pt"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── DATA ──────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.squeeze(0).T)
])
test_ds    = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
test_loader= DataLoader(test_ds, batch_size=BATCH_SIZE, pin_memory=True)

# ─── LOAD MODEL ────────────────────────────────────────────────────────────
model = HypernetClassifier().to(DEVICE)
model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
model.eval()

# ─── 1) Accuracy ───────────────────────────────────────────────────────────
correct = total = 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        preds = model(x).argmax(dim=-1)
        correct += (preds == y).sum().item()
        total   += y.size(0)
acc = 100 * correct / total

# ─── 2) Inference latency & throughput ────────────────────────────────────
# measure one batch for micro-latency
x0 = next(iter(test_loader))[0].to(DEVICE)
torch.cuda.synchronize() if DEVICE.type=="cuda" else None
tracemalloc.start()
t0 = time.time()
_  = model(x0)
torch.cuda.synchronize() if DEVICE.type=="cuda" else None
t1 = time.time()
cur_mem, peak_mem = tracemalloc.get_traced_memory()
tracemalloc.stop()

batch_latency_ms = (t1 - t0) * 1000

# measure full-test throughput
start_full = time.time()
with torch.no_grad():
    for x, _ in test_loader:
        model(x.to(DEVICE))
torch.cuda.synchronize() if DEVICE.type=="cuda" else None
end_full = time.time()
total_time = end_full - start_full
throughput = total / total_time  # images / sec

# ─── 3) Parameter & compression stats ─────────────────────────────────────
# total params
total_params = sum(p.numel() for p in model.parameters())

# genome size
genome_size = model.z.numel()

# target attention params per layer (Q/K/V/O): out_dim=in_dim=EMBED_DIM
EMBED_DIM   = model.block.self_attn.embed_dim
num_projs   = 4
target_params = num_projs * (EMBED_DIM * EMBED_DIM + EMBED_DIM)

comp_ratio = target_params / genome_size

# ─── REPORT ────────────────────────────────────────────────────────────────
print(f"🏆 Accuracy: {acc:.2f}%")
print(f"⏱️  One-batch latency: {batch_latency_ms:.2f} ms")
print(f"⚡  Full-test throughput: {throughput:.1f} imgs/sec ({total_time:.2f}s total)")
print(f"💾  RAM (one-batch) | current: {cur_mem/1024**2:.2f} MB; peak: {peak_mem/1024**2:.2f} MB")
print(f"⚙️  Total params: {total_params/1e6:.2f} M")
print(f"🧬 Genome size: {genome_size} floats")
print(f"🧩 Target-attn params (per layer): {target_params}")
print(f"📊 Compression ratio (target/genome): {comp_ratio:.2f}×")
