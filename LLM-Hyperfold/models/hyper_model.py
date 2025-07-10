import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaPreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from hyper_llama import HyperLlamaAttention, HyperLlamaMLP, SharedGenomeProjection

class HyperLlamaDecoderLayer(nn.Module):
    """Single decoder layer with hyper-generated weights"""
    def __init__(
        self,
        config: LlamaConfig,
        layer_idx: int,
        genome_proj: nn.Module,
        hyper_hidden: int,
        M: int,
        rank: int
    ):
        super().__init__()
        self.self_attn = HyperLlamaAttention(
            config, layer_idx, genome_proj, hyper_hidden, M, rank
        )
        self.mlp = HyperLlamaMLP(config, genome_proj, hyper_hidden, M, rank)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        genome_vec: torch.Tensor,
        attention_mask=None,
        use_cache=False,
    ):
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _, _ = self.self_attn(
            hidden_states,
            genome_vec=genome_vec,
            attention_mask=attention_mask,
            use_cache=use_cache
        )
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, genome_vec, use_cache=use_cache)
        hidden_states = residual + hidden_states

        return hidden_states

class HyperLlamaModel(LlamaPreTrainedModel):
    """Full LLaMA model with hyper-generated weights"""
    def __init__(self, config, genome_dim=96, hyper_hidden=256, M=32, rank=64):
        super().__init__(config)
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Genome and projection
        self.genome = nn.Parameter(torch.randn(config.num_hidden_layers, genome_dim))
        self.genome_proj = SharedGenomeProjection(genome_dim, hyper_hidden)
        
        # Decoder layers
        self.layers = nn.ModuleList([
            HyperLlamaDecoderLayer(
                config, i, self.genome_proj, hyper_hidden, M, rank
            ) for i in range(config.num_hidden_layers)
        ])
        
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Initialize weights
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        use_cache: bool = False,
    ):
        # Embed inputs
        hidden_states = self.embed_tokens(input_ids)
        
        # Process through layers
        for layer_idx, layer in enumerate(self.layers):
            genome_vec = self.genome[layer_idx]
            hidden_states = layer(
                hidden_states,
                genome_vec=genome_vec,
                attention_mask=attention_mask,
                use_cache=use_cache
            )
        
        # Final normalization
        hidden_states = self.norm(hidden_states)
        return hidden_states

class HyperLlamaForCausalLM(LlamaPreTrainedModel):
    """Causal LM with hyper-generated weights"""
    def __init__(self, config, genome_dim=96, hyper_hidden=256, M=32, rank=64):
        super().__init__(config)
        self.model = HyperLlamaModel(config, genome_dim, hyper_hidden, M, rank)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        labels: torch.LongTensor = None,
        use_cache: bool = False,
    ):
        # Get hidden states
        hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=use_cache
        )
        
        # Get logits
        logits = self.lm_head(hidden_states)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1)
            )
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
        )
    
    def generate(self, input_ids, max_length=50, **kwargs):
        """Simplified generation method"""
        for _ in range(max_length - input_ids.size(1)):
            outputs = self(input_ids=input_ids, use_cache=True)
            next_token = outputs.logits[:, -1, :].argmax(-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
        return input_ids

# Test
if __name__ == "__main__":
    config = LlamaConfig(
        vocab_size=32000,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=8,
        max_position_embeddings=1024,
        rms_norm_eps=1e-6,
    )
    
    model = HyperLlamaForCausalLM(config, genome_dim=64, hyper_hidden=128)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    input_ids = torch.randint(0, 32000, (1, 10))
    outputs = model(input_ids)
    assert outputs.logits.shape == (1, 10, 32000)
    print("✅ HyperLlama model test passed!")