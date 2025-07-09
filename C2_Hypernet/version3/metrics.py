import torch
import os
import psutil
import time

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024  # MB
    return mem

def time_inference(model, x, device='cpu'):
    model.eval()
    x = x.to(device)
    with torch.no_grad():
        start = time.time()
        _ = model(x)
        end = time.time()
    return (end - start) * 1000  # milliseconds
