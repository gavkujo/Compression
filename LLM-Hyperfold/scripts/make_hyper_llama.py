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
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
    )
    
    model = HyperLlamaForCausalLM(
        config,
        genome_dim=genome_dim,
        hyper_hidden=hyper_hidden,
        M=M,
        rank=rank
    )
    
    return model, config

if __name__ == "__main__":
    # Build a ~6B parameter model
    model, config = build_hyper_llama(
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params/1e9:.2f}B")
    print(f"Genome size: {model.model.genome.numel()/1e3:.1f}K")
    
    # Estimate hypernetwork parameters
    hyper_params = 0
    for layer in model.model.layers:
        hyper_params += sum(p.numel() for p in layer.self_attn.parameters())
        hyper_params += sum(p.numel() for p in layer.mlp.parameters())
    
    print(f"Hypernetwork parameters: {hyper_params/1e6:.2f}M")