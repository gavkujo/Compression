# scripts/utils.py

import torch
import tracemalloc
import time
from torch import Tensor
from typing import Tuple


def set_cpu_threads(num_threads: int = None) -> None:
    """
    Pin PyTorch to a given number of CPU threads (or all if None).
    """
    if num_threads is None:
        num_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads)
    print(f"🔧 Using {torch.get_num_threads()} CPU threads")


def measure_latency(model: torch.nn.Module,
                    inputs: Tensor,
                    repeat: int = 10) -> Tuple[float, float]:
    """
    Measure average and peak latency for a single forward pass.
    Returns (avg_ms, std_ms).
    """
    # Warm‑up
    with torch.no_grad():
        for _ in range(3):
            _ = model(**inputs)

    timings = []
    with torch.no_grad():
        for _ in range(repeat):
            t0 = time.time()
            _  = model(**inputs)
            t1 = time.time()
            timings.append((t1 - t0) * 1000)

    import statistics
    return statistics.mean(timings), statistics.stdev(timings)


def measure_ram(model: torch.nn.Module, inputs: Tensor) -> Tuple[float, float]:
    """
    Measure current & peak RAM usage (Python heap) during one forward pass.
    """
    tracemalloc.start()
    with torch.no_grad():
        _ = model(**inputs)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    # convert to MB
    return current / 1024**2, peak / 1024**2


def compute_perplexity(model, tokenizer, dataset, device, max_samples: int = 500):
    """
    Compute perplexity (exponentiated cross‑entropy) on a text dataset.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    import math

    for i, example in enumerate(dataset):
        if i >= max_samples:
            break
        enc = tokenizer(example["text"], return_tensors="pt", truncation=True, max_length=512)
        input_ids = enc["input_ids"].to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, return_dict=True)
            # assume causal LM; get logits and shift
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="sum"
            )
            total_loss += loss.item()
            total_tokens += shift_labels.numel()

    ppl = math.exp(total_loss / total_tokens)
    return ppl
