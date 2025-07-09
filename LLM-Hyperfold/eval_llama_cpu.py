# eval_llama_cpu.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset

from models.hyper_llama import HyperLlamaAttention, HyperLlamaMLP
from scripts.utils import set_cpu_threads, measure_latency, measure_ram, compute_perplexity


# ─── CONFIG ────────────────────────────────────────────────────────────────
PRETRAINED_NAME = "decapoda-research/llama-7b"
CHECKPOINT_PATH = "checkpoints/genome_meta.pth"
PROMPT         = "In a distant future, humanity"
BATCH_SIZE     = 1   # for prompt inference
NUM_THREADS    = None  # None → max available
DEVICE         = torch.device("cpu")


def build_hyperfolded_llama():
    # load base config
    cfg = LlamaConfig.from_pretrained(PRETRAINED_NAME)
    # inject our HyperDecoderLayer
    # assume you’ve patched the code so config knows to use it:
    # e.g. config.decoder_layer_cls = HyperDecoderLayer
    model = torch.hub.load("your-org/LLM-HyperFold", "hyper_llama_model", config=cfg)
    model.to(DEVICE)
    return model


def load_genome_and_hypernet(model, path: str):
    ckpt = torch.load(path, map_location=DEVICE)
    # genome vector
    model.z = nn.Parameter(ckpt['z'].to(DEVICE))
    # load hypernet states
    model.hyper_attn.load_state_dict(ckpt['hyper_attn'])
    model.hyper_mlp.load_state_dict(ckpt['hyper_mlp'])


def main():
    # 1) threading
    set_cpu_threads(NUM_THREADS)

    # 2) tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(PRETRAINED_NAME)

    # 3) build model
    print("🔨 Building hyperfolded LLaMA…")
    model = build_hyperfolded_llama()
    load_genome_and_hypernet(model, CHECKPOINT_PATH)
    model.eval()

    # 4) sample inference
    enc = tokenizer(PROMPT, return_tensors="pt")
    inputs = {"input_ids": enc["input_ids"].to(DEVICE)}

    avg_lat, std_lat = measure_latency(model, inputs, repeat=5)
    cur_ram, peak_ram = measure_ram(model, inputs)

    print(f"⏱️ Prompt latency: {avg_lat:.1f}±{std_lat:.1f} ms")
    print(f"💾 Mem usage: current {cur_ram:.2f} MB; peak {peak_ram:.2f} MB")

    # 5) Perplexity on WikiText2
    print("📊 Computing perplexity on WikiText2…")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ppl = compute_perplexity(model, tokenizer, ds, DEVICE, max_samples=200)
    print(f"🏅 Perplexity: {ppl:.2f}")


if __name__ == "__main__":
    main()
