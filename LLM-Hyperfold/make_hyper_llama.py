import torch
from models.hyper_model import HyperLlamaForCausalLM
from transformers import LlamaConfig
import time
import os

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] >> {msg}", flush=True)

def print_compression_stats(model):
    total_params = sum(p.numel() for p in model.parameters())
    hyper_params = 0
    genome_size = model.model.genome.numel()
    
    # Calculate hypernetwork parameters
    for name, param in model.named_parameters():
        if "hyper" in name:
            hyper_params += param.numel()
    
    print(f"\n⚡ Compression Stats ⚡")
    print(f"Total Params: {total_params/1e6:.2f}M")
    print(f"Hypernetwork Params: {hyper_params/1e6:.2f}M")
    print(f"Genome Size: {genome_size/1e3:.1f}K")
    print(f"Compression Ratio: {total_params/(hyper_params + genome_size):.1f}x")
    
    # Estimate memory usage
    param_bytes = (hyper_params + genome_size) * 4  # FP32
    print(f"Estimated RAM: {param_bytes/1e6:.1f}MB (FP32)")
    
    # Add quantization estimate
    quant_bytes = (hyper_params + genome_size) * 0.5  # 8-bit
    print(f"Estimated RAM (8-bit): {quant_bytes/1e6:.1f}MB")

def build_hyper_llama(
    vocab_size=32000,
    hidden_size=4096,
    intermediate_size=11008,
    num_hidden_layers=32,
    num_attention_heads=32,
    genome_dim=96, # 96 for 6B model
    hyper_hidden=256, # 256 for 6B model
    M=32, # 32 for 6B model
    rank=64 # 64 for 6B model
):
    """Build a custom HyperLlama model"""
    log("Building LlamaConfig...")
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
    )
    
    log("Instantiating HyperLlamaForCausalLM...")
    model = HyperLlamaForCausalLM(
        config,
        genome_dim=genome_dim,
        hyper_hidden=hyper_hidden,
        M=M,
        rank=rank
    )
    
    log("Model instantiated ✅")
    return model, config

if __name__ == "__main__":
    # Build a ~6B parameter model
    log("Starting HyperLLaMA model build")
    start_time = time.time()
    model, config = build_hyper_llama(
        hidden_size=1024, # 4096 for 6B model
        intermediate_size=2048, # 11008 for 6B model
        num_hidden_layers=8, # 32 layers for 6B model
        num_attention_heads=8 # 32 heads for 6B model
    )
    elapsed = time.time() - start_time
    log(f"Model build done in {elapsed:.2f} seconds")
    
    log("Counting total parameters...")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params/1e9:.2f}B")
    log("Counting genome size...")
    print(f"Genome size: {model.model.genome.numel()/1e3:.1f}K")
    
    # Estimate hypernetwork parameters
    log("Counting hypernetwork parameters...")
    hyper_params = 0
    i=1
    for layer in model.model.layers:
        hyper_attn = sum(p.numel() for p in layer.self_attn.parameters())
        hyper_mlp = sum(p.numel() for p in layer.mlp.parameters())
        layer_total = hyper_attn + hyper_mlp
        log(f"Layer {i}: {layer_total/1e6:.2f}M params")
        i += 1
        hyper_params += layer_total
        
    
    print(f"Hypernetwork parameters: {hyper_params/1e6:.2f}M")

    print_compression_stats(model)

    SAVE_DIR = "hyperllama-init"
    log(f"Saving model to {SAVE_DIR}...")

    # Save model weights
    os.makedirs(SAVE_DIR, exist_ok=True)
    torch.save(model.state_dict(), f"{SAVE_DIR}/pytorch_model.bin")

    # Save config as a JSON (Hugging Face style)
    with open(f"{SAVE_DIR}/config.json", "w") as f:
        f.write(config.to_json_string())

    log("Model and config saved successfully!")