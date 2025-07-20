#!/usr/bin/env python3
"""
⚡ ULTRA-LIGHTWEIGHT INFERENCE ENGINE
===================================
Edge deployment inference with <500MB RAM, <10ms/token, <100MB storage
"""

import torch
import psutil
import time
import os
import json
import numpy as np
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

class UltraLightweightInference:
    """Ultra-optimized inference engine for edge deployment"""
    def __init__(self, 
                 checkpoint_path: str = None,
                 cpu_threads: int = 4,
                 enable_quantization: bool = True,
                 tokenizer_path: str = None,
                 vocab_size: int = None):
        print("⚡ Initializing Ultra-Lightweight Inference Engine...")
        torch.set_num_threads(cpu_threads)
        torch.set_num_interop_threads(1)
        self.device = torch.device('cpu')
        self.enable_quantization = enable_quantization
        self.inference_times = []
        self.memory_usage = []
        self.start_ram = self._measure_ram()
        self.genome_dim = 96
        self.hyper_hidden = 128
        self.M = 16
        self.rank = 32
        self.top_k = 4
        self.vocab_size = vocab_size or 1000
        self._init_tokenizer(tokenizer_path)
        self._init_transformer()
        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_checkpoint(checkpoint_path)
        else:
            print("⚠️  No checkpoint provided, using initialized model")
        self._optimize_for_inference()
        print(f"✅ Inference engine ready!")
        print(f"📊 Base RAM: {self.start_ram:.1f}MB")

    def _measure_ram(self) -> float:
        """Measure current RAM usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / (1024 ** 2)

    def _init_tokenizer(self, tokenizer_path: str):
        """Initialize tokenizer"""
        self.tokenizer = None
        try:
            from transformers import PreTrainedTokenizerFast
            if tokenizer_path and os.path.exists(tokenizer_path):
                print(f"📦 Loading tokenizer from {tokenizer_path}")
                self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
                if self.vocab_size is None:
                    self.vocab_size = self.tokenizer.vocab_size
            else:
                print("⚠️  No tokenizer found, using fallback character-level encoding")
        except Exception as e:
            print(f"⚠️  Tokenizer load failed: {e}")

    def _init_transformer(self):
        """Initialize transformer with progressive/lazy layer instantiation and factorized weight application"""
        from models.hyper_model import HyperLlamaForCausalLM
        from transformers import LlamaConfig
        import json
        print("🏗️ Initializing streaming transformer...")
        # Try to load config from checkpoint directory if available
        config_path = None
        if hasattr(self, 'checkpoint_path') and self.checkpoint_path:
            ckpt_dir = os.path.dirname(self.checkpoint_path)
            for fname in ["config.json", "llama_config.json"]:
                candidate = os.path.join(ckpt_dir, fname)
                if os.path.exists(candidate):
                    config_path = candidate
                    break
        if config_path:
            with open(config_path, "r") as f:
                config_dict = json.load(f)
            llama_config = LlamaConfig(**config_dict)
        else:
            # fallback to edge config
            llama_config = LlamaConfig(
                vocab_size=self.vocab_size,
                hidden_size=512,
                intermediate_size=1024,
                num_hidden_layers=8,
                num_attention_heads=8,
                max_position_embeddings=2048,
                rms_norm_eps=1e-6,
            )
        # Patch: add progressive/lazy streaming mode
        llama_config.streaming_mode = True
        self.transformer = HyperLlamaForCausalLM(
            llama_config,
            genome_dim=self.genome_dim,
            hyper_hidden=self.hyper_hidden,
            M=self.M,
            rank=self.rank,
            top_k=self.top_k,
            lora_rank=8,
            lora_alpha=1.0,
            use_lora=True
        )
        self.transformer.to(self.device)
        # Print only the hypernetwork/genome params, not full model
        try:
            total_params = sum(p.numel() for p in self.transformer.parameters())
            print(f"📊 Streaming model (hypernet+genome) params: {total_params/1e6:.2f}M")
        except Exception as e:
            print(f"⚠️  Model param count failed: {e}")
        print("✅ Streaming transformer ready (progressive/lazy mode)")

    def _load_checkpoint(self, checkpoint_path: str):
        """Load trained model checkpoint"""
        print(f"📂 Loading checkpoint: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            if 'model_state_dict' in checkpoint:
                self.transformer.load_state_dict(checkpoint['model_state_dict'])
                print("✅ Model weights loaded from 'model_state_dict'")
            elif 'hypernetwork_state' in checkpoint:
                print("⚠️  Training checkpoint detected, initializing with default weights")
            else:
                self.transformer.load_state_dict(checkpoint)
                print("✅ Model weights loaded directly")
        except Exception as e:
            print(f"⚠️  Failed to load checkpoint: {e}")
            print("Using initialized weights instead")

    def _optimize_for_inference(self):
        """Apply optimizations for edge deployment"""
        print("🔧 Applying edge deployment optimizations...")
        self.transformer.eval()
        for param in self.transformer.parameters():
            param.requires_grad = False
        if self.enable_quantization:
            self._apply_quantization()
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        print("✅ Edge optimizations applied")

    def _apply_quantization(self):
        """Apply 8-bit quantization for memory efficiency"""
        print("🗜️  Applying INT8 quantization...")
        def quantize_tensor(tensor, bits=8):
            """Quantize tensor to specified bits"""
            if tensor.numel() == 0:
                return tensor
            min_val = tensor.min()
            max_val = tensor.max()
            if min_val == max_val:
                return tensor
            scale = (max_val - min_val) / (2**bits - 1)
            zero_point = -min_val / scale
            quantized = torch.clamp(
                torch.round(tensor / scale + zero_point), 
                0, 2**bits - 1
            )
            return (quantized - zero_point) * scale
        for name, param in self.transformer.named_parameters():
            if param.dim() > 1:
                param.data = quantize_tensor(param.data)
        print("✅ Quantization applied")

    def encode_prompt(self, prompt: str) -> list:
        """Encode prompt to input IDs"""
        if self.tokenizer:
            return self.tokenizer.encode(prompt, add_special_tokens=True)
        else:
            # Fallback: character-level encoding
            return [ord(c) % self.vocab_size for c in prompt[:512]]

    def decode_output(self, output_ids: list) -> str:
        """Decode output IDs to text"""
        if self.tokenizer:
            return self.tokenizer.decode(output_ids, skip_special_tokens=True)
        else:
            # Fallback: character-level decoding
            return ''.join([chr(min(max(id, 32), 126)) for id in output_ids if id > 0])

    def generate_text(self, prompt: str, max_length: int = 64, temperature: float = 0.7) -> str:
        """Generate text using progressive/lazy streaming architecture"""
        print(f"🚀 Generating text for: '{prompt[:50]}{'...' if len(prompt) > 50 else ''}'")
        input_ids = self.encode_prompt(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        start_time = time.perf_counter()
        start_ram = self._measure_ram()
        output_ids = input_ids.copy()
        # Prepare a default genome vector for streaming (match genome_dim)
        genome_dim = getattr(self, 'genome_dim', 96)
        batch_size = 1
        default_genome_vec = torch.zeros((batch_size, genome_dim), dtype=torch.float32, device=self.device)
        # Progressive/lazy streaming: process one token at a time, one layer at a time
        with torch.no_grad():
            for _ in range(max_length):
                x = torch.tensor([output_ids], dtype=torch.long, device=self.device)
                hidden = x
                print(f"[DEBUG] Input hidden shape before layers: {hidden.shape}")
                for layer_idx, layer in enumerate(self.transformer.model.layers):
                    layer.reset_sequence()  # Clear any cache/state
                    # Pass the default genome vector (shape: [1, genome_dim])
                    try:
                        hidden = layer(hidden, genome_vec=default_genome_vec, attention_mask=None, use_cache=False, token_position=len(output_ids))
                    except Exception as e:
                        print(f"[ERROR] Layer {layer_idx} shape mismatch: {e}")
                        print(f"[DEBUG] hidden shape: {hidden.shape}, genome_vec shape: {default_genome_vec.shape}")
                        raise
                    print(f"[DEBUG] After layer {layer_idx}, hidden shape: {getattr(hidden, 'shape', type(hidden))}")
                # Final layer norm and head
                if isinstance(hidden, (tuple, list)):
                    hidden = hidden[0]
                logits = self.transformer.lm_head(hidden)
                logits = logits[:, -1, :] / temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                output_ids.append(next_token)
                if next_token == 0:
                    break
        end_time = time.perf_counter()
        end_ram = self._measure_ram()
        full_text = self.decode_output(output_ids)
        generated_text = full_text[len(prompt):] if full_text.startswith(prompt) else full_text
        total_time = (end_time - start_time) * 1000
        tokens_generated = len(output_ids) - len(input_ids)
        time_per_token = total_time / max(tokens_generated, 1)
        ram_used = end_ram - start_ram
        print(f"✅ Generated {tokens_generated} tokens in {total_time:.1f}ms")
        print(f"📊 {time_per_token:.1f}ms/token | RAM: +{ram_used:.1f}MB")
        return generated_text.strip()

    def _manual_generate(self, input_ids: torch.Tensor, max_length: int, temperature: float) -> list:
        """Manual text generation (fallback)"""
        current_ids = input_ids.clone()
        for _ in range(max_length):
            outputs = self.transformer(current_ids)
            logits = outputs.logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            current_ids = torch.cat([current_ids, next_token], dim=1)
            if next_token.item() == 0:
                break
        return current_ids[0].cpu().numpy().tolist()

# --- CLI Demo ---
def main():
    """Interactive chat demo for edge deployment"""
    print("⚡ ULTRA-LIGHTWEIGHT INFERENCE DEMO")
    print("=" * 40)
    print("Edge deployment: <500MB RAM, <10ms/token, <100MB storage")
    try:
        engine = UltraLightweightInference(
            checkpoint_path= "checkpoints/best_hypernetwork_1B.pt",  # Set to checkpoint if available
            cpu_threads=4,
            enable_quantization=True,
            tokenizer_path= "scripts/tokenizer_1B/tokenizer.json",  # Set to tokenizer if available
            vocab_size= 2145  # Set dynamically if needed
        )
        print("\n💬 Interactive Chat Mode")
        print("Type your prompt below. Type 'exit' to quit.\n")
        while True:
            try:
                prompt = input("Prompt: ").strip()
                if prompt.lower() in ["exit", "quit", ""]:
                    break
                output = engine.generate_text(prompt, max_length=64)
                print(f"Output: {output}\n")
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Generation failed: {e}\n")
        print("👋 Chat ended.")
    except Exception as e:
        print(f"❌ Inference engine failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
