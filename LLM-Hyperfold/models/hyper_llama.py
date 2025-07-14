import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP, LlamaRMSNorm
from .basis_hyper import FactorizedBasisHyperLayer

class SharedGenomeProjection(nn.Module):
    """Shared projection for genome vectors"""
    def __init__(self, genome_dim: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(genome_dim, hidden_dim),
            nn.GELU()
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.proj(z)

class HyperLlamaAttention(LlamaAttention):
    """Attention with hyper-generated weights"""
    def __init__(
        self,
        config,
        layer_idx: int,
        genome_proj: nn.Module,
        hyper_hidden: int,
        M: int = 16, # 32 for 6B model
        rank: int = 32, # 64 for 6B model
        genome_dim: int = 96  # Default genome dimension
    ) -> None:
        super().__init__(config, layer_idx)
        E = config.hidden_size
        self.hidden_size = E  # Store for later use
        
        # Explicitly set these attributes
        self.num_heads = config.num_attention_heads
        self.head_dim = E // self.num_heads
        
        # Remove original projections
        del self.q_proj, self.k_proj, self.v_proj, self.o_proj
        
        # Shared genome projection
        self.genome_proj = genome_proj
        
        # Hypernetworks
        self.hyper_q = FactorizedBasisHyperLayer(
            hyper_hidden, hyper_hidden, E, E, M, rank
        )
        self.hyper_k = FactorizedBasisHyperLayer(
            hyper_hidden, hyper_hidden, E, E, M, rank
        )
        self.hyper_v = FactorizedBasisHyperLayer(
            hyper_hidden, hyper_hidden, E, E, M, rank
        )
        self.hyper_o = FactorizedBasisHyperLayer(
            hyper_hidden, hyper_hidden, E, E, M, rank
        )
        
        # Cache for unfolded weights
        self.cache = {}
        self.head_positional = nn.Parameter(
            torch.randn(config.num_attention_heads, genome_dim // 8)
        )

    def forward(self, hidden_states, genome_vec, attention_mask=None, use_cache=False):
        B, T, _ = hidden_states.shape
        head_outputs = []
        
        for head_idx in range(self.num_heads):
            # Perturb genome for this head
            head_genome = genome_vec + self.head_positional[head_idx]
            
            # Generate weights JUST for this head
            head_dim = self.head_dim
            start_idx = head_idx * head_dim
            end_idx = (head_idx + 1) * head_dim
            
            Wq_head = self.hyper_q(head_genome)[start_idx:end_idx]
            Wk_head = self.hyper_k(head_genome)[start_idx:end_idx]
            Wv_head = self.hyper_v(head_genome)[start_idx:end_idx]
            
            # Process single head
            q_head = F.linear(hidden_states, Wq_head)
            k_head = F.linear(hidden_states, Wk_head)
            v_head = F.linear(hidden_states, Wv_head)
            
            # Immediately discard weights
            del Wq_head, Wk_head, Wv_head
            
            # Reshape for attention
            q_head = q_head.view(B, T, 1, self.head_dim).transpose(1, 2)
            k_head = k_head.view(B, T, 1, self.head_dim).transpose(1, 2)
            v_head = v_head.view(B, T, 1, self.head_dim).transpose(1, 2)
            
            # Attention computation
            attn_weights = torch.matmul(q_head, k_head.transpose(2, 3)) / (self.head_dim ** 0.5)
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask[:, :, :, :T]
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_output = torch.matmul(attn_weights, v_head)
            head_outputs.append(attn_output)
            
        # Combine heads
        attn_output = torch.cat(head_outputs, dim=1)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(B, T, self.hidden_size)
        
        # Output projection (still full matrix)
        Wo = self.hyper_o(genome_vec)
        attn_output = F.linear(attn_output, Wo)
        
        return (attn_output, None, None)

class HyperLlamaMLP(LlamaMLP):
    """MLP with hyper-generated weights"""
    def __init__(
        self,
        config,
        genome_proj: nn.Module,
        hyper_hidden: int,
        M: int = 16, # 32 for 6B model
        rank: int = 32 # 64 for 6B model
    ):
        super().__init__(config)
        E = config.hidden_size
        I = config.intermediate_size
        
        # Remove original projections
        del self.gate_proj, self.up_proj, self.down_proj
        
        # Shared genome projection
        self.genome_proj = genome_proj
        
        # Hypernetworks
        self.hyper_gate = FactorizedBasisHyperLayer(
            hyper_hidden, hyper_hidden, I, E, M, rank
        )
        self.hyper_up = FactorizedBasisHyperLayer(
            hyper_hidden, hyper_hidden, I, E, M, rank
        )
        self.hyper_down = FactorizedBasisHyperLayer(
            hyper_hidden, hyper_hidden, E, I, M, rank
        )
        
        # Cache for unfolded weights
        self.cache = {}

    def forward(self, hidden_states: torch.Tensor, genome_vec: torch.Tensor, use_cache=False):
        # Project genome
        z_proj = self.genome_proj(genome_vec)
        cache_key = tuple(z_proj.flatten().tolist()) if use_cache else None
        
        '''
        # Check cache
        if use_cache and cache_key in self.cache:
            W_gate, W_up, W_down, b_gate, b_up, b_down = self.cache[cache_key]
        else:
            W_gate, b_gate = self.hyper_gate(z_proj)
            W_up, b_up = self.hyper_up(z_proj)
            W_down, b_down = self.hyper_down(z_proj)
            
            if use_cache:
                self.cache[cache_key] = (W_gate, W_up, W_down, b_gate, b_up, b_down)
        '''
                
        # Check cache
        if use_cache and cache_key in self.cache:
            W_gate, W_up, W_down = self.cache[cache_key]
        else:
            W_gate = self.hyper_gate(z_proj)
            W_up = self.hyper_up(z_proj)
            W_down = self.hyper_down(z_proj)
            
            if use_cache:
                self.cache[cache_key] = (W_gate, W_up, W_down)
        
        # Forward pass (without bias since LLaMA doesn't use them)
        gate = F.linear(hidden_states, W_gate)
        up = F.linear(hidden_states, W_up)
        gate_act = F.silu(gate)
        fused = gate_act * up
        down = F.linear(fused, W_down)
        
        return down

# Test
if __name__ == "__main__":
    from transformers import LlamaConfig
    
    config = LlamaConfig(
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=4,
        num_hidden_layers=1
    )
    
    genome_proj = SharedGenomeProjection(32, 64)
    attn = HyperLlamaAttention(config, 0, genome_proj, 64)
    mlp = HyperLlamaMLP(config, genome_proj, 64)
    
    x = torch.randn(2, 10, config.hidden_size)
    z = torch.randn(32)
    
    attn_out = attn(x, z)[0]
    mlp_out = mlp(x, z)
    
    assert attn_out.shape == x.shape
    assert mlp_out.shape == x.shape
    print("✅ HyperLlama modules test passed!")