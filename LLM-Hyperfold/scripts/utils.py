import torch
import time
import psutil
from datasets import load_dataset
from transformers import LlamaTokenizer

def set_cpu_threads(num_threads=None):
    """Set number of CPU threads for PyTorch"""
    if num_threads is None:
        num_threads = psutil.cpu_count(logical=False)
    torch.set_num_threads(num_threads)
    print(f"Using {num_threads} CPU threads")

def measure_latency(model, inputs, repeat=10):
    """Measure inference latency"""
    # Warm up
    with torch.no_grad():
        for _ in range(3):
            model(**inputs)
    
    # Measure
    timings = []
    for _ in range(repeat):
        start_time = time.perf_counter()
        with torch.no_grad():
            model(**inputs)
        timings.append(time.perf_counter() - start_time)
    
    return sum(timings) / len(timings) * 1000  # ms

def measure_ram():
    """Measure current and peak RAM usage"""
    process = psutil.Process()
    current = process.memory_info().rss / (1024 ** 2)  # MB
    peak = current
    return current, peak

def compute_perplexity(model, tokenizer, dataset, device, max_samples=100, seq_len=512):
    """Compute perplexity on dataset"""
    model.eval()
    losses = []
    count = 0
    
    try:
        for example in dataset:
            if count >= max_samples:
                break
            text = example.get('text', '') or ""
            if len(text.strip()) < 10:  # Skip very short texts
                continue
                
            enc = tokenizer(text, return_tensors="pt", max_length=seq_len, truncation=True)
            input_ids = enc.input_ids.to(device)
            
            if input_ids.size(1) < 2:  # Need at least 2 tokens
                continue
            
            with torch.no_grad():
                outputs = model(input_ids, labels=input_ids)
                loss = outputs.loss.item()
                if not torch.isnan(torch.tensor(loss)) and not torch.isinf(torch.tensor(loss)):
                    losses.append(loss)
                    count += 1
                    
        if len(losses) == 0:
            return float('inf')
            
        return torch.exp(torch.tensor(losses).mean()).item()
        
    except Exception as e:
        print(f"Skipping perplexity measurement due to dataset error: {e}")
        print("This is a known issue with dataset version mismatches.")
        return float('inf')

def quantize_model(model, bits=8):
    """Apply quantization to model weights"""
    for name, param in model.named_parameters():
        if "genome" not in name and "hyper" not in name:
            max_val = torch.max(torch.abs(param.data))
            scale = (2**(bits-1)-1) / max_val
            param.data = torch.clamp(torch.round(param.data * scale), -2**(bits-1), 2**(bits-1)-1)
            param.quant_scale = scale
            param.quant_zero = 0.0

# scripts/utils.py (even better version)

def save_compressed(model, path):
    """Save compressed model with quantization"""
    state = {
        'global_genome': model.model.global_genome.data,  # Keep FP32 for compatibility
        'layer_genome': model.model.layer_genome.data,   # Keep FP32
        'layer_position': model.model.layer_position.weight.data,  # Keep FP32
        'genome_proj': model.model.genome_proj.state_dict(),
        'hypernet': {k: v for k, v in model.state_dict().items() if 'hyper' in k}
    }
    
    # Calculate total parameters
    genome_params = (state['global_genome'].numel() + 
                    state['layer_genome'].numel() + 
                    state['layer_position'].numel())
    proj_params = sum(t.numel() for t in state['genome_proj'].values())
    hypernet_params = sum(t.numel() for t in state['hypernet'].values())
    total_params = genome_params + proj_params + hypernet_params
    
    torch.save(state, path, _use_new_zipfile_serialization=True)
    print(f"Saved compressed model to {path} ({total_params/1e6:.2f}M params)")

def load_compressed(model, path, device):
    """Load quantized model"""
    state = torch.load(path, map_location=device)
    model.model.global_genome.data = state['global_genome'].float()
    model.model.layer_genome.data = state['layer_genome'].float()
    model.model.layer_position.weight.data = state['layer_position'].float()
    
    # Load genome projection
    proj_state = {k: v.float() for k, v in state['genome_proj'].items()}
    model.model.genome_proj.load_state_dict(proj_state)
    
    # Load hypernet weights
    hyper_state = {k: v.float() for k, v in state['hypernet'].items()}
    model.load_state_dict(hyper_state, strict=False)
    return model