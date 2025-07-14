import torch
import time
import os
from models.hyper_model import HyperLlamaForCausalLM
from transformers import LlamaTokenizer, LlamaConfig
from scripts.utils import set_cpu_threads, measure_latency, measure_ram, compute_perplexity
from datasets import load_dataset

# Config
MODEL_PATH = "hyperllama-init"  # Path to model config
COMPRESSED_PATH = "checkpoints/compressed_model.pt"
TOKENIZER_PATH = "llama-tokenizer"
PROMPT = "In a world where AI controls everything,"
MAX_LENGTH = 50
NUM_THREADS = 4
DEVICE = torch.device("cpu")

def load_compressed_model(model_path, compressed_path, device):
    """Load compressed model with minimal memory footprint"""
    # 1. Load config first
    config = LlamaConfig.from_pretrained(model_path)
    
    # 2. Create empty model
    model = HyperLlamaForCausalLM(config)
    model.to(device)
    model.eval()
    
    # 3. Load compressed state
    compressed_state = torch.load(compressed_path, map_location=device)
    
    # 4. Load genome and hypernet weights
    model.model.genome.data = compressed_state['genome'].float()
    model.load_state_dict(compressed_state['hypernet'], strict=False)
    
    return model

def main():
    # Set CPU threads
    set_cpu_threads(NUM_THREADS)
    
    # Load tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(TOKENIZER_PATH)
    
    # Load compressed model
    print("Loading compressed model...")
    start_load = time.time()
    model = load_compressed_model(MODEL_PATH, COMPRESSED_PATH, DEVICE)
    load_time = time.time() - start_load
    print(f"Model loaded in {load_time:.2f}s")
    
    # Print memory stats
    current_ram, _ = measure_ram()
    print(f"Model RAM usage: {current_ram:.2f}MB")
    
    # Encode prompt
    input_ids = tokenizer.encode(PROMPT, return_tensors="pt").to(DEVICE)
    
    # Warm up
    with torch.no_grad():
        _ = model.generate(input_ids[:, :1], max_length=2)
    
    # Inference with caching
    start_time = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids, 
            max_length=MAX_LENGTH,
            use_cache=True
        )
    total_time = time.time() - start_time
    
    # Decode and print
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("\nGenerated text:")
    print(output_text)
    print(f"\nTotal time: {total_time:.2f}s")
    print(f"Time per token: {total_time / (MAX_LENGTH - input_ids.size(1)) * 1000:.1f}ms")
    
    # Memory usage after inference
    current_ram, peak_ram = measure_ram()
    print(f"RAM usage: Current={current_ram:.2f}MB, Peak={peak_ram:.2f}MB")
    
    # Perplexity measurement (on subset)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ppl = compute_perplexity(model, tokenizer, dataset, DEVICE, max_samples=50)
    print(f"Perplexity: {ppl:.2f}")

if __name__ == "__main__":
    main()