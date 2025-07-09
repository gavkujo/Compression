# scripts/make_6b_model.py
import torch
from transformers import GPT2Config, GPT2LMHeadModel

def build_6b_gpt():
    config = GPT2Config(
        vocab_size=50257, n_positions=1024, n_ctx=1024,
        n_embd=4096, n_layer=32, n_head=32, n_inner=16384,
        activation_function="gelu_new"
    )
    return GPT2LMHeadModel(config)

if __name__ == "__main__":
    model = build_6b_gpt()
    print("Params (B):", sum(p.numel() for p in model.parameters())/1e9)
