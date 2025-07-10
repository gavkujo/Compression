import torch
import time
from models.hyper_model import HyperLlamaForCausalLM
from transformers import LlamaTokenizer
from scripts.utils import set_cpu_threads, measure_latency, measure_ram, compute_perplexity, load_compressed

# Config
MODEL_PATH = "checkpoints/hyperfold_llama"
TOKENIZER_PATH = "path/to/llama-tokenizer"
CHECKPOINT = "checkpoints/hyperfold_genome.pth"
PROMPT = "In a world where AI controls everything,"
MAX_LENGTH = 50
NUM_THREADS = 4  # Adjust based on your CPU
DEVICE = torch.device("cpu")
COMPRESSED_PATH = "checkpoints/compressed_model.pt"

def main():
    # Set CPU threads
    set_cpu_threads(NUM_THREADS)
    
    # Load tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(TOKENIZER_PATH)
    
    # Load model
    print("Loading model...")
    model = HyperLlamaForCausalLM.from_pretrained(MODEL_PATH)
    model.to(DEVICE)
    model.eval()
    
    # Load genome and hypernets
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
    model.model.genome.data = ckpt['genome'].to(DEVICE)
    model.load_state_dict(ckpt['hypernets'], strict=False)
    
    # Encode prompt
    input_ids = tokenizer.encode(PROMPT, return_tensors="pt").to(DEVICE)
    
    # Inference with caching
    start_time = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids, 
            max_length=MAX_LENGTH,
            use_cache=True  # Enable weight caching
        )
    total_time = time.time() - start_time
    
    # Decode and print
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("\nGenerated text:")
    print(output_text)
    print(f"\nTotal time: {total_time:.2f}s")
    print(f"Time per token: {total_time / (MAX_LENGTH - input_ids.size(1)) * 1000:.1f}ms")
    
    # Memory usage
    current_ram, peak_ram = measure_ram()
    print(f"RAM usage: Current={current_ram:.2f}MB, Peak={peak_ram:.2f}MB")
    
    # Perplexity measurement
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ppl = compute_perplexity(model, tokenizer, dataset, DEVICE, max_samples=100)
    print(f"Perplexity: {ppl:.2f}")
    
    # Save compressed version
    save_compressed(model, COMPRESSED_PATH)

if __name__ == "__main__":
    main()