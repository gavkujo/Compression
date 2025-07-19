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
    genome_size = 0

    genome_size += model.model.global_genome.numel()
    genome_size += model.model.layer_genome.numel()
    genome_size += model.model.layer_position.weight.numel()
    
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
    vocab_size=1000,  # Compressed vocabulary for edge deployment
    hidden_size=512,  # Compressed hidden size for edge deployment  
    intermediate_size=1024,  # Compressed intermediate size for edge deployment
    num_hidden_layers=8,  # Reduced layers for edge deployment
    num_attention_heads=8,  # Reduced heads for edge deployment
    genome_dim=96, # 96 for edge deployment
    hyper_hidden=128, # 128 for edge deployment (reduced from 256)
    M=16, # 16 for edge deployment (reduced from 32)
    rank=32, # 32 for edge deployment (reduced from 64)
    top_k=4, # 4 for edge deployment
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
    
    # ADD: Save hypernetwork parameters in config for loading
    config.genome_dim = genome_dim
    config.hyper_hidden = hyper_hidden
    config.M = M
    config.rank = rank
    config.top_k = top_k
    
    log("Instantiating HyperLlamaForCausalLM...")
    model = HyperLlamaForCausalLM(
        config,
        genome_dim=genome_dim,
        hyper_hidden=hyper_hidden,
        M=M,
        rank=rank,
        top_k=top_k,
        lora_rank=8,
        lora_alpha=1.0,
        use_lora=True
    )
    
    log("Model instantiated ✅")
    return model, config

if __name__ == "__main__":
    # Build edge deployment model (optimized for <500MB RAM, <10ms per token)
    log("Starting HyperLLaMA edge deployment model build")
    start_time = time.time()
    model, config = build_hyper_llama(
        vocab_size=1000,         # Compressed vocabulary
        hidden_size=512,         # Compressed hidden (down from 4096)
        intermediate_size=1024,  # Compressed intermediate (down from 11008)
        num_hidden_layers=8,     # Fewer layers (down from 32)
        num_attention_heads=8,   # Fewer heads (down from 32)
        genome_dim=96,           # Optimized genome
        hyper_hidden=128,        # Compressed hypernetwork (down from 256)
        M=16,                    # Fewer experts (down from 32)
        rank=32,                 # Lower rank (down from 64)
        top_k=4                  # Efficient routing
    )
    elapsed = time.time() - start_time
    log(f"Model build done in {elapsed:.2f} seconds")
    
    log("Counting total parameters...")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params/1e9:.2f}B")
    log("Counting genome size...")
    # Multi-scale genome stats
    g_global = model.model.global_genome.numel()
    g_layer  = model.model.layer_genome.numel()
    g_pos    = model.model.layer_position.num_embeddings * model.model.layer_position.embedding_dim
    log("Counting genome components...")
    print(f"  • Global genome: {g_global} floats")
    print(f"  • Layer genome:  {g_layer} floats")
    print(f"  • Position vec:  {g_pos} floats")
    
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