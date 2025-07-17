#!/usr/bin/env python3
"""
⚡ ULTRA-LIGHTWEIGHT INFERENCE ENGINE
===================================
CPU-only inference with <500MB RAM, <100ms/token, <1MB storage
"""

import torch
import torch.nn.functional as F
import time
import psutil
import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import our models (lightweight imports only)
from build import UniversalHyperNetwork, MOEGenomeManager, MODEL_CONFIGS

class UltraLightweightInference:
    def print_generation_metrics(self, start_time, output_ids, logits=None):
        # Print RAM usage, time taken, and average token probability
        ram = self._measure_ram()
        elapsed = time.perf_counter() - start_time
        avg_prob = None
        if logits is not None:
            # Calculate average probability of generated tokens
            probs = torch.softmax(logits, dim=-1)
            token_probs = [probs[i, token_id].item() for i, token_id in enumerate(output_ids)]
            avg_prob = np.mean(token_probs)
        print(f"RAM used: {ram:.1f}MB | Time taken: {elapsed:.2f}s" + (f" | Avg token prob: {avg_prob:.4f}" if avg_prob is not None else ""))
    def _init_transformer(self):
        # Import and initialize HyperLlamaForCausalLM
        from models.hyper_model import HyperLlamaForCausalLM
        from transformers import LlamaConfig
        config_params = MODEL_CONFIGS[self.target_model_size]
        config = LlamaConfig(
            vocab_size=config_params["vocab_size"],
            hidden_size=config_params["hidden_size"],
            intermediate_size=config_params["intermediate_size"],
            num_hidden_layers=config_params["num_hidden_layers"],
            num_attention_heads=config_params["num_attention_heads"],
            max_position_embeddings=2048,
            rms_norm_eps=1e-6,
        )
        self.transformer = HyperLlamaForCausalLM(config)
        # Load weights if available (optional)
        # You may want to load from checkpoint here

    def generate_text(self, prompt: str, expert_type: str = None, max_length: int = 64) -> str:
        """Generate text response for a given prompt using the compressed model, printing metrics after each output."""
        if not hasattr(self, "transformer"):
            self._init_transformer()
        input_ids = torch.tensor([self.tokenizer.encode(prompt, add_special_tokens=True)], device=self.device)
        # --- Tokenizer vocab size consistency check ---
        model_vocab_size = self.model_config["vocab_size"]
        tokenizer_vocab_size = self.tokenizer.vocab_size
        if tokenizer_vocab_size != model_vocab_size:
            raise ValueError(f"Tokenizer vocab size ({tokenizer_vocab_size}) does not match model config ({model_vocab_size})")
        # 2. Select expert genome
        if expert_type is None:
            expert_type = self.route_expert(prompt)
        genome = self.genome_manager.get_expert_genome(expert_type,
            global_context=torch.randn(self.genome_dim//4, device=self.device),
            position_id=0)
        # 3. Generate weights for each transformer layer
        layer_weights = []
        for layer_idx in range(self.model_config["num_hidden_layers"]):
            weights = self.hypernetwork.generate_weights(
                genome,
                target_shape=(self.model_config["hidden_size"], self.model_config["hidden_size"]),
                target_model=self.target_model_size,
                token_position=layer_idx
            )
            layer_weights.append(weights)
        # 4. Inject weights into transformer
        self.transformer.set_layer_weights(layer_weights)
        # 5. Generate output tokens
        start_time = time.perf_counter()
        with torch.no_grad():
            output = self.transformer.generate(input_ids, max_length=max_length, return_dict_in_generate=True, output_scores=True)
            output_ids = output.sequences[0].cpu().numpy().tolist()
            logits = None
            if hasattr(output, 'scores') and output.scores:
                logits = torch.cat([score for score in output.scores], dim=0)
            self.print_generation_metrics(start_time, output_ids, logits)
        return self.tokenizer.decode(output_ids, skip_special_tokens=True)
        # 2. Select expert genome
        if expert_type is None:
            expert_type = self.route_expert(prompt)
        genome = self.genome_manager.get_expert_genome(expert_type,
            global_context=torch.randn(self.genome_dim//4, device=self.device),
            position_id=0)
        # 3. Generate weights for each transformer layer
        layer_weights = []
        for layer_idx in range(self.model_config["num_hidden_layers"]):
            weights = self.hypernetwork.generate_weights(
                genome,
                target_shape=(self.model_config["hidden_size"], self.model_config["hidden_size"]),
                target_model=self.target_model_size,
                token_position=layer_idx
            )
            layer_weights.append(weights)
        # 4. Inject weights into transformer
        self.transformer.set_layer_weights(layer_weights)
    """Ultra-optimized inference engine for CPU"""
    
    def __init__(self, 
                 checkpoint_path: str,
                 target_model_size: str = "350M",
                 cpu_threads: int = 4,
                 enable_quantization: bool = True,
                 tokenizer_path: str = None,
                 vocab_size: int = 1000):
        print("⚡ Initializing Ultra-Lightweight Inference Engine...")
        self.target_model_size = target_model_size
        self.enable_quantization = enable_quantization
        torch.set_num_threads(cpu_threads)
        torch.set_num_interop_threads(1)
        self.device = torch.device('cpu')
        self.inference_times = []
        self.memory_usage = []
        self.start_ram = self._measure_ram()
        # --- Tokenizer Integration ---
        from transformers import PreTrainedTokenizerFast
        if tokenizer_path:
            self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
        else:
            self.tokenizer = None
        # Load model components
        self._load_checkpoint(checkpoint_path)
        self._optimize_for_inference()
        print(f"✅ Inference engine ready!")
        print(f"📊 Base RAM: {self.start_ram:.1f}MB")
    
    def route_expert(self, prompt: str) -> str:
        """Keyword-based expert router for prompt (same as training)"""
        prompt_lower = prompt.lower()
        math_keywords = ["sum", "add", "subtract", "multiply", "divide", "math", "equation", "number"]
        code_keywords = ["def ", "function", "code", "python", "list", "append", "return", "variable"]
        creative_keywords = ["story", "haiku", "poem", "creative", "imagine", "describe", "forest", "cat"]
        general_keywords = ["capital", "ocean", "cpu", "invented", "general", "what", "who", "when"]
        if any(k in prompt_lower for k in math_keywords):
            return "math"
        elif any(k in prompt_lower for k in code_keywords):
            return "code"
        elif any(k in prompt_lower for k in creative_keywords):
            return "creative"
        else:
            return "general"

    def encode_prompt(self, prompt: str) -> List[int]:
        """Encode prompt to input IDs using tokenizer"""
        return self.tokenizer.encode(prompt, add_special_tokens=True)

    def decode_output(self, output_ids: List[int]) -> str:
        """Decode output IDs to text using tokenizer"""
        return self.tokenizer.decode(output_ids, skip_special_tokens=True)
        
        # Apply optimizations
        self._optimize_for_inference()
        
        print(f"✅ Inference engine ready!")
        print(f"📊 Base RAM: {self.start_ram:.1f}MB")
    
    def _measure_ram(self) -> float:
        """Measure current RAM usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / (1024 ** 2)
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load trained model checkpoint"""
        print(f"📂 Loading checkpoint: {checkpoint_path}")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        config_dict = checkpoint['config']
        
        # Initialize components
        self.genome_dim = config_dict['genome_dim']
        self.hyper_hidden = config_dict['hyper_hidden']
        self.num_experts = config_dict['num_experts']
        self.expert_types = config_dict['expert_types']
        
        # Create models
        self.hypernetwork = UniversalHyperNetwork(
            genome_dim=self.genome_dim,
            hyper_hidden=self.hyper_hidden,
            target_configs=MODEL_CONFIGS,
            enable_streaming=True,
            enable_temporal=True
        )
        
        self.genome_manager = MOEGenomeManager(
            genome_dim=self.genome_dim,
            num_experts=self.num_experts,
            expert_types=self.expert_types
        )
        
        # Load trained weights
        self.hypernetwork.load_state_dict(checkpoint['hypernetwork_state'])
        self.genome_manager.load_state_dict(checkpoint['genome_manager_state'])
        
        print("✅ Checkpoint loaded successfully")
    
    def _optimize_for_inference(self):
        """Apply CPU and memory optimizations"""
        print("🔧 Applying inference optimizations...")
        
        # Set to evaluation mode
        self.hypernetwork.eval()
        self.genome_manager.eval()
        
        # Disable gradients globally
        for param in self.hypernetwork.parameters():
            param.requires_grad = False
        for param in self.genome_manager.parameters():
            param.requires_grad = False
        
        # Apply quantization if enabled
        if self.enable_quantization:
            self._apply_quantization()
        
        # Compile models for faster execution
        '''
        try:
            self.hypernetwork = torch.jit.script(self.hypernetwork)
            print("✅ HyperNetwork compiled with TorchScript")
        except Exception as e:
            print(f"⚠️  TorchScript compilation failed: {e}")
            pass
        '''
        # Pre-allocate frequently used tensors
        self._preallocate_tensors()
        
        print("✅ Optimizations applied")
    
    def _apply_quantization(self):
        """Apply 8-bit quantization for memory efficiency"""
        print("🗜️  Applying quantization...")
        
        def quantize_tensor(tensor, bits=8):
            """Quantize tensor to specified bits"""
            if tensor.numel() == 0:
                return tensor
            
            # Calculate scale and zero point
            min_val = tensor.min()
            max_val = tensor.max()
            
            if min_val == max_val:
                return tensor
            
            scale = (max_val - min_val) / (2**bits - 1)
            zero_point = -min_val / scale
            
            # Quantize
            quantized = torch.clamp(
                torch.round(tensor / scale + zero_point), 
                0, 2**bits - 1
            )
            
            # Dequantize for use
            return (quantized - zero_point) * scale
        
        # Quantize hypernetwork weights
        for name, param in self.hypernetwork.named_parameters():
            if param.dim() > 1:  # Only quantize matrices
                param.data = quantize_tensor(param.data)
        
        # Quantize genome weights (less aggressive to preserve precision)
        for name, param in self.genome_manager.named_parameters():
            if param.dim() > 1:
                param.data = quantize_tensor(param.data, bits=16)  # Higher precision for genomes
        
        print("✅ Quantization applied")
    
    def _preallocate_tensors(self):
        """Pre-allocate commonly used tensors"""
        model_config = MODEL_CONFIGS[self.target_model_size]
        
        # Pre-allocate hidden states
        self.hidden_buffer = torch.zeros(
            1, 512, model_config['hidden_size'], 
            dtype=torch.float32
        )
        
        # Pre-allocate weight matrices
        self.weight_buffer = torch.zeros(
            model_config['hidden_size'], 
            model_config['hidden_size'],
            dtype=torch.float32
        )
        
        print("✅ Tensors pre-allocated")
    
    def get_expert_weights(self, 
                          expert_type: str,
                          target_layer: str = "attention",
                          token_position: int = 0,
                          sequence_length: int = 1) -> torch.Tensor:
        """
        Generate weights for specific expert and layer
        
        Args:
            expert_type: One of ['math', 'code', 'creative', 'general']
            target_layer: Type of layer ('attention', 'mlp', 'embed')
            token_position: Current token position for temporal optimization
            sequence_length: Total sequence length for streaming optimization
        
        Returns:
            Generated weight matrix
        """
        start_time = time.perf_counter()
        
        with torch.no_grad():
            # Get expert genome
            global_context = torch.zeros(self.genome_dim//4, device=self.device)
            genome = self.genome_manager.get_expert_genome(
                expert_type=expert_type,
                global_context=global_context,
                position_id=token_position % 8  # Cycle position IDs
            )
            
            # Add batch dimension
            genome = genome.unsqueeze(0)
            
            # Determine target shape based on layer type and model size
            model_config = MODEL_CONFIGS[self.target_model_size]
            
            if target_layer == "attention":
                target_shape = (model_config['hidden_size'], model_config['hidden_size'])
            elif target_layer == "mlp":
                target_shape = (model_config['hidden_size'] * 4, model_config['hidden_size'])
            elif target_layer == "embed":
                target_shape = (model_config['vocab_size'], model_config['hidden_size'])
            else:
                target_shape = (model_config['hidden_size'], model_config['hidden_size'])
            
            # Generate weights using universal hypernetwork
            weights = self.hypernetwork.generate_weights(
                genomes=genome,
                target_shape=target_shape,
                target_model=self.target_model_size,
                token_position=token_position
            )
            
            # Remove batch dimension if present
            if weights.dim() == 3:
                weights = weights.squeeze(0)
        
        # Track performance
        inference_time = (time.perf_counter() - start_time) * 1000  # ms
        self.inference_times.append(inference_time)
        
        current_ram = self._measure_ram()
        self.memory_usage.append(current_ram)
        
        return weights
    
    def generate_sequence_weights(self, 
                                 expert_type: str,
                                 sequence_length: int,
                                 layer_type: str = "attention") -> List[torch.Tensor]:
        """
        Generate weights for an entire sequence with temporal optimization
        
        Args:
            expert_type: Expert to use
            sequence_length: Length of sequence
            layer_type: Type of layer weights to generate
        
        Returns:
            List of weight matrices for each position
        """
        print(f"🔄 Generating {sequence_length} weight matrices for {expert_type} expert...")
        
        weights_sequence = []
        total_start_time = time.perf_counter()
        
        # Reset temporal cache for new sequence
        if hasattr(self.hypernetwork, 'reset_sequence'):
            self.hypernetwork.reset_sequence()
        
        for pos in range(sequence_length):
            weights = self.get_expert_weights(
                expert_type=expert_type,
                target_layer=layer_type,
                token_position=pos,
                sequence_length=sequence_length
            )
            weights_sequence.append(weights)
        
        total_time = (time.perf_counter() - total_start_time) * 1000
        avg_time_per_token = total_time / sequence_length
        
        print(f"✅ Generated {sequence_length} matrices in {total_time:.1f}ms")
        print(f"📊 Average: {avg_time_per_token:.1f}ms per token")
        
        return weights_sequence
    
    def benchmark_performance(self, 
                             num_tokens: int = 100,
                             expert_type: str = "math") -> Dict:
        """
        Benchmark inference performance
        
        Args:
            num_tokens: Number of tokens to generate
            expert_type: Expert to benchmark
        
        Returns:
            Performance metrics
        """
        print(f"🧪 Benchmarking performance ({num_tokens} tokens, {expert_type} expert)...")
        
        # Clear previous stats
        self.inference_times.clear()
        self.memory_usage.clear()
        
        start_ram = self._measure_ram()
        start_time = time.perf_counter()
        
        # Generate weights for sequence
        weights_sequence = self.generate_sequence_weights(
            expert_type=expert_type,
            sequence_length=num_tokens,
            layer_type="attention"
        )
        
        total_time = time.perf_counter() - start_time
        end_ram = self._measure_ram()
        
        # Calculate metrics
        avg_token_time = np.mean(self.inference_times) if self.inference_times else 0
        max_token_time = np.max(self.inference_times) if self.inference_times else 0
        min_token_time = np.min(self.inference_times) if self.inference_times else 0
        
        peak_ram = np.max(self.memory_usage) if self.memory_usage else end_ram
        ram_increase = end_ram - start_ram
        
        # Model size estimation
        total_params = sum(p.numel() for p in self.hypernetwork.parameters()) + \
                      sum(p.numel() for p in self.genome_manager.parameters())
        storage_mb = total_params * 4 / (1024 ** 2)  # FP32 to MB
        
        if self.enable_quantization:
            storage_mb *= 0.5  # Approximate quantization savings
        
        metrics = {
            'num_tokens': num_tokens,
            'total_time_ms': total_time * 1000,
            'avg_token_time_ms': avg_token_time,
            'max_token_time_ms': max_token_time,
            'min_token_time_ms': min_token_time,
            'tokens_per_second': num_tokens / total_time,
            'start_ram_mb': start_ram,
            'end_ram_mb': end_ram,
            'peak_ram_mb': peak_ram,
            'ram_increase_mb': ram_increase,
            'storage_mb': storage_mb,
            'total_params': total_params,
            'quantized': self.enable_quantization,
            
            # Requirement checks
            'meets_token_time_req': avg_token_time < 100.0,  # <100ms per token
            'meets_ram_req': peak_ram < 500.0,  # <500MB RAM
            'meets_storage_req': storage_mb < 1.0,  # <1MB storage
        }
        
        return metrics
    
    def print_performance_report(self, metrics: Dict):
        """Print detailed performance report"""
        print("\n🏆 PERFORMANCE REPORT")
        print("=" * 40)
        
        # Speed metrics
        print(f"⚡ Speed Metrics:")
        print(f"   Average token time: {metrics['avg_token_time_ms']:.1f}ms")
        print(f"   Max token time: {metrics['max_token_time_ms']:.1f}ms")
        print(f"   Min token time: {metrics['min_token_time_ms']:.1f}ms")
        print(f"   Tokens per second: {metrics['tokens_per_second']:.1f}")
        
        # Memory metrics
        print(f"\n💾 Memory Metrics:")
        print(f"   Start RAM: {metrics['start_ram_mb']:.1f}MB")
        print(f"   End RAM: {metrics['end_ram_mb']:.1f}MB")
        print(f"   Peak RAM: {metrics['peak_ram_mb']:.1f}MB")
        print(f"   RAM increase: {metrics['ram_increase_mb']:.1f}MB")
        
        # Storage metrics
        print(f"\n📦 Storage Metrics:")
        print(f"   Model size: {metrics['storage_mb']:.2f}MB")
        print(f"   Total parameters: {metrics['total_params']:,}")
        print(f"   Quantized: {metrics['quantized']}")
        
        # Requirement checks
        print(f"\n✅ Requirement Checks:")
        print(f"   Token time <100ms: {'✅' if metrics['meets_token_time_req'] else '❌'} "
              f"({metrics['avg_token_time_ms']:.1f}ms)")
        print(f"   RAM <500MB: {'✅' if metrics['meets_ram_req'] else '❌'} "
              f"({metrics['peak_ram_mb']:.1f}MB)")
        print(f"   Storage <1MB: {'✅' if metrics['meets_storage_req'] else '❌'} "
              f"({metrics['storage_mb']:.2f}MB)")
        
        # Overall status
        all_requirements_met = all([
            metrics['meets_token_time_req'],
            metrics['meets_ram_req'],
            metrics['meets_storage_req']
        ])
        
        print(f"\n🎯 Overall Status: {'✅ ALL REQUIREMENTS MET!' if all_requirements_met else '❌ Some requirements not met'}")
    
    def test_all_experts(self, num_tokens: int = 50) -> Dict:
        """Test all expert types"""
        print(f"\n🔬 Testing all {self.num_experts} experts...")
        
        expert_results = {}
        
        for expert_type in self.expert_types:
            print(f"\n🧪 Testing {expert_type} expert...")
            metrics = self.benchmark_performance(num_tokens, expert_type)
            expert_results[expert_type] = metrics
            
            print(f"   Token time: {metrics['avg_token_time_ms']:.1f}ms")
            print(f"   Peak RAM: {metrics['peak_ram_mb']:.1f}MB")
        
        # Calculate overall metrics
        overall_metrics = {
            'avg_token_time_ms': np.mean([m['avg_token_time_ms'] for m in expert_results.values()]),
            'max_peak_ram_mb': np.max([m['peak_ram_mb'] for m in expert_results.values()]),
            'avg_storage_mb': np.mean([m['storage_mb'] for m in expert_results.values()]),
            'all_experts_meet_requirements': all([
                m['meets_token_time_req'] and m['meets_ram_req'] and m['meets_storage_req']
                for m in expert_results.values()
            ])
        }
        
        print(f"\n📊 Overall Results:")
        print(f"   Average token time: {overall_metrics['avg_token_time_ms']:.1f}ms")
        print(f"   Max peak RAM: {overall_metrics['max_peak_ram_mb']:.1f}MB")
        print(f"   Average storage: {overall_metrics['avg_storage_mb']:.2f}MB")
        print(f"   All experts pass: {'✅' if overall_metrics['all_experts_meet_requirements'] else '❌'}")
        
        return {'expert_results': expert_results, 'overall': overall_metrics}
    
    def save_results(self, results: Dict, filename: str = None):
        """Save benchmark results to file"""
        if filename is None:
            filename = f"inference_results_{self.target_model_size}.json"
        
        os.makedirs("results", exist_ok=True)
        filepath = os.path.join("results", filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"💾 Results saved to {filepath}")

def main():
    """Main inference demo"""
    print("⚡ ULTRA-LIGHTWEIGHT INFERENCE DEMO")
    print("=" * 45)
    # Check for trained checkpoint
    checkpoint_path = "checkpoints/best_hypernetwork_1B.pt"
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    try:
        # Initialize inference engine
        engine = UltraLightweightInference(
            checkpoint_path=checkpoint_path,
            target_model_size="1B",
            cpu_threads=4,
            enable_quantization=True,
            tokenizer_path="scripts/tokenizer_1B/tokenizer.json",
            vocab_size=32000
        )
        # Test with real prompts
        prompts = [
            "Write a function to add two numbers.",
            "What is the capital of France?",
            "Write a short story about a cat.",
            "Solve: 12 + 7",
        ]
        for prompt in prompts:
            print(f"\nPrompt: {prompt}")
            output = engine.generate_text(prompt, max_length=64)
            print(f"Output: {output}")
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
