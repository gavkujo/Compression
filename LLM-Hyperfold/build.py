"""
🏗️ DYNAMIC MODEL BUILDER
========================
Build target models (350M-14B) + Universal Hypernetwork + Expert Genomes
"""

import torch
import torch.nn as nn
from transformers import LlamaConfig
import json
import os
from typing import Dict, Any, List

# Model size configurations
MODEL_CONFIGS = {
    "350M": {"hidden_size": 1024, "intermediate_size": 4096, "num_hidden_layers": 24, "num_attention_heads": 16, "vocab_size": 32000},
    "1B": {"hidden_size": 2048, "intermediate_size": 8192, "num_hidden_layers": 24, "num_attention_heads": 16, "vocab_size": 32000}, 
    "3B": {"hidden_size": 3200, "intermediate_size": 12800, "num_hidden_layers": 26, "num_attention_heads": 32, "vocab_size": 32000},
    "6B": {"hidden_size": 4096, "intermediate_size": 16384, "num_hidden_layers": 32, "num_attention_heads": 32, "vocab_size": 32000},
    "14B": {"hidden_size": 5120, "intermediate_size": 20480, "num_hidden_layers": 40, "num_attention_heads": 40, "vocab_size": 32000}
}

class UniversalHyperNetwork(nn.Module):
    """Universal hypernetwork that works for any model size"""
    
    def __init__(self, 
                 genome_dim: int = 96,
                 hyper_hidden: int = 256, 
                 target_configs: Dict = None,
                 enable_streaming: bool = True,
                 enable_temporal: bool = True,
                 max_hidden_size: int = 5120):
        super().__init__()
        
        # Store configuration
        self.genome_dim = genome_dim
        self.hyper_hidden = hyper_hidden
        super().__init__()
        self.genome_dim = genome_dim
        self.hyper_hidden = hyper_hidden
        self.target_configs = target_configs or MODEL_CONFIGS
        self.enable_streaming = enable_streaming
        self.enable_temporal = enable_temporal
        self.max_hidden = max_hidden_size
        
        # **Core hypernetwork components**
        self.genome_proj = nn.Sequential(
            nn.Linear(self.genome_dim, self.hyper_hidden),
            nn.GELU(),
            nn.Linear(self.hyper_hidden, self.hyper_hidden)
        )
        
        # **INNOVATION: Ultra-Aggressive Hierarchical Factorization**
        self.rank = 64
        self.coarse_rank = 8  # Ultra-small coarse structure
        self.fine_rank = 56   # Most detail in fine structure
        self.M = 16  # Basis matrices
        self.top_k = 4
        
        # **Universal basis matrices (work for any target size)**
        self.U_coarse = nn.Parameter(torch.randn(self.M, 64, self.coarse_rank) * 0.01)  # Fixed small size
        self.V_coarse = nn.Parameter(torch.randn(self.M, self.coarse_rank, 64) * 0.01)   # Fixed small size
        self.U_fine = nn.Parameter(torch.randn(self.M, 32, 32) * 0.005)  # Even smaller fine details
        self.V_fine = nn.Parameter(torch.randn(self.M, 32, 32) * 0.005)
        
        # **Gating network**
        self.gating = nn.Linear(self.hyper_hidden, self.M)
        
        # **INNOVATION: 4-Mode Smart Routing**
        self.position_router = nn.Parameter(torch.randn(4) * 0.01)
        self.delta_scale = 0.05
        
        # **State management**
        self.weight_memory = None
        self.current_token_pos = 0
        self.max_hidden = max_hidden_size
        
        # **Core hypernetwork components**
        self.genome_proj = nn.Sequential(
            nn.Linear(self.genome_dim, self.hyper_hidden),
            nn.GELU(),
            nn.Linear(self.hyper_hidden, self.hyper_hidden)
        )
        
        # **INNOVATION: Ultra-Aggressive Hierarchical Factorization**
        self.rank = 64
        self.coarse_rank = 8  # Ultra-small coarse structure
        self.fine_rank = 56   # Most detail in fine structure
        self.M = 16  # Basis matrices
        self.top_k = 4
        
        # **Universal basis matrices (work for any target size)**
        self.U_coarse = nn.Parameter(torch.randn(self.M, 64, self.coarse_rank) * 0.01)  # Fixed small size
        self.V_coarse = nn.Parameter(torch.randn(self.M, self.coarse_rank, 64) * 0.01)   # Fixed small size
        self.U_fine = nn.Parameter(torch.randn(self.M, 32, 32) * 0.005)  # Even smaller fine details
        self.V_fine = nn.Parameter(torch.randn(self.M, 32, 32) * 0.005)
        
        # **Gating network**
        self.gating = nn.Linear(self.hyper_hidden, self.M)
        
        # **INNOVATION: 4-Mode Smart Routing**
        self.position_router = nn.Parameter(torch.randn(4) * 0.01)
        self.delta_scale = 0.05
        
        # **State management**
        self.weight_memory = None
        self.current_token_pos = 0
        
    def generate_weights(self, 
                        genomes: torch.Tensor, 
                        target_shape: tuple, 
                        target_model: str = "350M",
                        token_position: int = 0) -> torch.Tensor:
        """Generate weights for any target shape using the genome"""
        
        # Handle batch dimension
        if genomes.dim() == 2:
            genome = genomes[0]  # Take first in batch
        else:
            genome = genomes
            
        out_dim, in_dim = target_shape
        
        # Project genome to hypernetwork space
        z = self.genome_proj(genome)  # [genome_dim] → [hyper_hidden]
        
        # **Smart routing based on position and memory**
        route_probs = torch.softmax(self.position_router + token_position * 0.01, dim=0)
        
        if token_position == 0 or self.weight_memory is None:
            return self._hierarchical_generation(z, out_dim, in_dim)
        elif route_probs[1] > 0.5 and self.weight_memory is not None:
            return self._temporal_update(z, out_dim, in_dim)
        elif route_probs[2] > 0.3:
            return self._streaming_generation(z, out_dim, in_dim)
        else:
            return self._fast_approximation(z, out_dim, in_dim)
    
    def _hierarchical_generation(self, z: torch.Tensor, out_dim: int, in_dim: int) -> torch.Tensor:
        """Generate weights using hierarchical coarse-to-fine approach"""
        # Gating
        gate_logits = self.gating(z)
        topk_vals, topk_idx = torch.topk(gate_logits, self.top_k, dim=-1)
        coeffs = torch.zeros_like(gate_logits).scatter_(-1, topk_idx, torch.softmax(topk_vals, dim=-1))
        
        # Generate coarse structure
        U_coarse = torch.einsum('m, mor -> or', coeffs, self.U_coarse)  # [64, coarse_rank]
        V_coarse = torch.einsum('m, mri -> ri', coeffs, self.V_coarse)  # [coarse_rank, 64]
        
        # **Dynamic upsampling to target dimensions**
        U_upsampled = torch.nn.functional.interpolate(
            U_coarse.T.unsqueeze(0).unsqueeze(0),  # [1, 1, coarse_rank, 64]
            size=(self.rank, out_dim),
            mode='bilinear', align_corners=False
        ).squeeze(0).squeeze(0).T  # [out_dim, rank]
        
        V_upsampled = torch.nn.functional.interpolate(
            V_coarse.unsqueeze(0).unsqueeze(0),  # [1, 1, coarse_rank, 64]
            size=(self.rank, in_dim),
            mode='bilinear', align_corners=False
        ).squeeze(0).squeeze(0)  # [rank, in_dim]
        
        # Add fine details
        fine_coeffs = coeffs[:min(len(coeffs), 8)]  # Use subset for fine details
        if len(fine_coeffs) > 0:
            U_fine = torch.einsum('m, mij -> ij', fine_coeffs, self.U_fine[:len(fine_coeffs)])
            V_fine = torch.einsum('m, mij -> ij', fine_coeffs, self.V_fine[:len(fine_coeffs)])
            
            # Inject fine details into center
            center_u = (out_dim - 32) // 2
            center_v = (in_dim - 32) // 2
            if center_u >= 0 and center_v >= 0:
                U_upsampled[center_u:center_u+32, :32] += U_fine * 0.1
                V_upsampled[:32, center_v:center_v+32] += V_fine * 0.1
        
        # Generate final weight matrix
        weight = torch.matmul(U_upsampled, V_upsampled)  # [out_dim, in_dim]
        
        # Cache for temporal inheritance
        self.weight_memory = weight.clone()
        
        return weight
    
    def _temporal_update(self, z: torch.Tensor, out_dim: int, in_dim: int) -> torch.Tensor:
        """Ultra-fast temporal weight inheritance"""
        if self.weight_memory is None:
            return self._hierarchical_generation(z, out_dim, in_dim)
        
        # Generate tiny delta
        gate_logits = self.gating(z)
        delta_coeffs = torch.softmax(gate_logits[:4], dim=-1)  # Use only first 4
        
        # Small perturbation
        delta = torch.einsum('m, mij -> ij', delta_coeffs, self.U_fine[:4]) * self.delta_scale
        
        # Apply to center region of cached weight
        center = (min(out_dim, in_dim) - 32) // 2
        if center >= 0:
            self.weight_memory[center:center+32, center:center+32] += delta
        
        return self.weight_memory
    
    def _streaming_generation(self, z: torch.Tensor, out_dim: int, in_dim: int) -> torch.Tensor:
        """Memory-efficient streaming generation"""
        # Simplified streaming - process in chunks
        chunk_size = 64
        chunks = []
        
        for i in range(0, out_dim, chunk_size):
            end_i = min(i + chunk_size, out_dim)
            chunk_out = end_i - i
            
            # Generate chunk using coarse structure
            chunk_weight = self._fast_approximation(z, chunk_out, in_dim)
            chunks.append(chunk_weight)
        
        return torch.cat(chunks, dim=0)
    
    def _fast_approximation(self, z: torch.Tensor, out_dim: int, in_dim: int) -> torch.Tensor:
        """Emergency ultra-fast mode"""
        # Use only 2 basis matrices for maximum speed
        gate_logits = self.gating(z)
        coeffs = torch.softmax(gate_logits[:2], dim=-1)
        
        # Simple coarse generation
        U_fast = torch.einsum('m, mor -> or', coeffs, self.U_coarse[:2])  # [64, coarse_rank]
        V_fast = torch.einsum('m, mri -> ri', coeffs, self.V_coarse[:2])  # [coarse_rank, 64]
        
        # Direct upsampling
        weight = torch.nn.functional.interpolate(
            (U_fast @ V_fast).unsqueeze(0).unsqueeze(0),  # [1, 1, 64, 64]
            size=(out_dim, in_dim),
            mode='bilinear', align_corners=False
        ).squeeze(0).squeeze(0)  # [out_dim, in_dim]
        
        return weight
    
    def reset_sequence(self):
        """Reset for new sequence"""
        self.weight_memory = None
        self.current_token_pos = 0

class MOEGenomeManager(nn.Module):
    """Manages 4 expert genomes"""
    
    def __init__(self, 
                 genome_dim: int = 96,
                 num_experts: int = 4,
                 expert_types: List[str] = None):
        super().__init__()
        
        self.genome_dim = genome_dim
        self.num_experts = num_experts
        self.expert_types = expert_types or ["math", "code", "creative", "general"]
        
        # **Initialize expert genomes as parameters**
        self.expert_genomes = nn.ParameterDict({
            expert_type: nn.Parameter(torch.randn(genome_dim) * 0.1)
            for expert_type in self.expert_types
        })
        
        # Expert routing network (learns to select expert)
        self.expert_router = nn.Linear(genome_dim//4, num_experts)  # Global context → expert probabilities
        
        # Position embedding for temporal optimization
        self.position_embeddings = nn.Embedding(8, genome_dim//8)  # Up to 8 position embeddings
    
    def get_expert_genome(self, 
                         expert_type: str,
                         global_context: torch.Tensor = None,
                         position_id: int = 0) -> torch.Tensor:
        """Get genome for specific expert with context and position"""
        
        if expert_type not in self.expert_genomes:
            raise ValueError(f"Unknown expert type: {expert_type}. Available: {self.expert_types}")
        
        # Base expert genome
        base_genome = self.expert_genomes[expert_type]
        
        # Add position-specific adjustments
        if position_id < 8:
            position_emb = self.position_embeddings(torch.tensor(position_id))
            # Pad position embedding to match genome dimension
            position_full = torch.zeros_like(base_genome)
            position_full[:len(position_emb)] = position_emb
            base_genome = base_genome + 0.1 * position_full
        
        return base_genome
    
    def route_expert(self, context: torch.Tensor) -> tuple:
        """Route to best expert based on context"""
        expert_probs = torch.softmax(self.expert_router(context), dim=-1)
        expert_idx = torch.argmax(expert_probs).item()
        expert_names = list(self.expert_genomes.keys())
        selected_expert = expert_names[expert_idx]
        return selected_expert, self.expert_genomes[selected_expert]

def build_universal_system(target_model_size: str = "350M"):
    """Build the complete universal system"""
    
    if target_model_size not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model size: {target_model_size}")
    
    config_params = MODEL_CONFIGS[target_model_size]
    
    print(f"🏗️ Building Universal HyperFold System for {target_model_size} model...")
    
    # **1. Target model configuration**
    target_config = LlamaConfig(
        vocab_size=32000,
        hidden_size=config_params["hidden"],
        intermediate_size=config_params["intermediate"], 
        num_hidden_layers=config_params["layers"],
        num_attention_heads=config_params["heads"],
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
    )
    
    # **2. Universal hypernetwork (works for any model size)**
    hypernetwork = UniversalHyperNetwork(max_hidden_size=5120)  # Supports up to 14B
    
    # **3. MOE genome manager**
    genome_manager = MOEGenomeManager(genome_dim=32)
    
    # **4. Calculate compression stats**
    target_params = calculate_model_params(config_params)
    hyper_params = sum(p.numel() for p in hypernetwork.parameters())
    genome_params = sum(genome.numel() for genome in genome_manager.expert_genomes.values())
    total_compressed = hyper_params + genome_params
    
    compression_ratio = target_params / total_compressed
    
    print(f"📊 Compression Statistics:")
    print(f"   Target {target_model_size} model: {target_params/1e6:.1f}M parameters")
    print(f"   Universal hypernetwork: {hyper_params/1e6:.2f}M parameters")
    print(f"   Expert genomes (4x): {genome_params} parameters ({genome_params*4} bytes)")
    print(f"   Total compressed: {total_compressed/1e6:.2f}M parameters")
    print(f"   🚀 Compression ratio: {compression_ratio:.1f}x")
    print(f"   💾 Genome storage: {genome_params*4/1024:.2f}KB")
    
    # **5. Verify compression requirements**
    genome_storage_mb = (genome_params * 4) / (1024 * 1024)  # Convert to MB
    if compression_ratio >= 10 and genome_storage_mb < 1.0:
        print(f"✅ Meets requirements: {compression_ratio:.1f}x compression, {genome_storage_mb:.3f}MB genome")
    else:
        print(f"❌ Requirements not met: {compression_ratio:.1f}x compression, {genome_storage_mb:.3f}MB genome")
    
    return {
        "target_config": target_config,
        "hypernetwork": hypernetwork, 
        "genome_manager": genome_manager,
        "model_size": target_model_size,
        "compression_ratio": compression_ratio
    }

def calculate_model_params(config_params: Dict[str, int]) -> int:
    """Calculate parameters for standard LLaMA model"""
    hidden = config_params["hidden"]
    hidden = config_params["hidden_size"]
    intermediate = config_params["intermediate_size"]
    layers = config_params["num_hidden_layers"]
    heads = config_params["num_attention_heads"]
    # Embedding + output
    vocab_params = 32000 * hidden * 2  # embed + lm_head
    # Per layer: attention + MLP + norms
    attn_params = 4 * (hidden * hidden)  # Q, K, V, O projections
    mlp_params = 2 * (hidden * intermediate) + (intermediate * hidden)  # gate, up, down
    norm_params = 2 * hidden  # input + post attention norms
    layer_params = attn_params + mlp_params + norm_params
    total_params = vocab_params + (layers * layer_params)
    return total_params

def save_system(system: Dict[str, Any], save_dir: str = "hyperfold_system"):
    """Save the complete system"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save target config
    with open(f"{save_dir}/target_config.json", "w") as f:
        json.dump(system["target_config"].to_dict(), f)
    
    # Save hypernetwork
    torch.save(system["hypernetwork"].state_dict(), f"{save_dir}/hypernetwork.pt")
    
    # Save expert genomes
    torch.save(system["genome_manager"].expert_genomes, f"{save_dir}/expert_genomes.pt")
    
    # Save metadata
    metadata = {
        "model_size": system["model_size"],
        "genome_dim": getattr(system["hypernetwork"], "genome_dim", None),
        "num_experts": getattr(system["genome_manager"], "num_experts", None),
        "expert_types": getattr(system["genome_manager"], "expert_types", ["math", "code", "creative", "general"]),
    }
    with open(f"{save_dir}/metadata.json", "w") as f:
        json.dump(metadata, f)
    print(f"💾 System saved to {save_dir}/")

if __name__ == "__main__":
    # Test with different model sizes
    for model_size in ["350M", "1B", "3B", "6B", "14B"]:
        print(f"\n{'='*50}")
        system = build_universal_system(model_size)
        
        # Test weight generation
        hypernetwork = system["hypernetwork"]
        genome_manager = system["genome_manager"]
        
        # Test generating weights for attention layer
        math_genome = genome_manager.get_expert_genome("math")
        hidden_size = system["target_config"].hidden_size
        
        # Test Q projection weight generation
        q_weight = hypernetwork.generate_weights(math_genome, (hidden_size, hidden_size))
        print(f"✅ Generated Q weight: {q_weight.shape}")
        
        hypernetwork.reset_sequence()
    
    # Save the 6B system as default
    system_6b = build_universal_system("6B")
    save_system(system_6b)
    print("\n🎉 Universal HyperFold System built successfully!")