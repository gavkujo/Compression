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
    for i, example in enumerate(dataset):
        if i >= max_samples:
            break
        text = example['text'] or ""
        enc = tokenizer(text, return_tensors="pt", max_length=seq_len, truncation=True)
        input_ids = enc.input_ids.to(device)
        
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss.item()
            losses.append(loss)
    
    return torch.exp(torch.tensor(losses).mean()).item()

def quantize_model(model, bits=8):
    """Apply quantization to model weights"""
    for name, param in model.named_parameters():
        if "genome" not in name and "hyper" not in name:
            max_val = torch.max(torch.abs(param.data))
            scale = (2**(bits-1)-1) / max_val
            param.data = torch.clamp(torch.round(param.data * scale), -2**(bits-1), 2**(bits-1)-1)
            param.quant_scale = scale
            param.quant_zero = 0.0

def save_compressed(model, path):
    """Save model with quantization"""
    state = {
        'genome': model.model.genome.data.half(),  # FP16
        'hypernet': {k: v.half() for k, v in model.state_dict().items()}
    }
    torch.save(state, path, _use_new_zipfile_serialization=True)
    print(f"Saved compressed model to {path} ({sum(t.numel() for t in state.values())/1e6:.2f}M params)")

def load_compressed(model, path, device):
    """Load quantized model"""
    state = torch.load(path, map_location=device)
    model.model.genome.data = state['genome'].float()
    
    # Load hypernet weights
    hyper_state = {}
    for k, v in state['hypernet'].items():
        if "hyper" in k:
            hyper_state[k] = v.float()
    
    model.load_state_dict(hyper_state, strict=False)
    return model