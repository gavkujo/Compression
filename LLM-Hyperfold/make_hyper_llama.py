import torch
from models.hyper_model import HyperLlamaForCausalLM
from transformers import LlamaConfig

def build_hyper_llama(
    vocab_size=32000,
    hidden_size=4096,
    intermediate_size=11008,
    num_hidden_layers=32,
    num_attention_heads=32,
    genome_dim=96,
    hyper_hidden=256,
    M=32,
    rank=64
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
    model, config = build_hyper_llama(
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32
    )
    
    log("Counting total parameters...")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params/1e9:.2f}B")
    log("Counting genome size...")
    print(f"Genome size: {model.model.genome.numel()/1e3:.1f}K")
    
    # Estimate hypernetwork parameters
    log("Counting hypernetwork parameters...")
    hyper_params = 0
    for layer in model.model.layers:
        hyper_attn = sum(p.numel() for p in layer.self_attn.parameters())
        hyper_mlp = sum(p.numel() for p in layer.mlp.parameters())
        layer_total = hyper_attn + hyper_mlp
        log(f"Layer {i}: {layer_total/1e6:.2f}M params")
        hyper_params += layer_total
        
    
    print(f"Hypernetwork parameters: {hyper_params/1e6:.2f}M")