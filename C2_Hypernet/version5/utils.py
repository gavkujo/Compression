import torch
import os

def save_genome(model, path='C2_Hypernet/version3/checkpoints/genome.pt'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.z.detach().cpu(), path)

def load_genome(model, path='C2_Hypernet/version3/checkpoints/genome.pt'):
    genome = torch.load(path).to(next(model.parameters()).device)
    with torch.no_grad():
        model.z.copy_(genome)
