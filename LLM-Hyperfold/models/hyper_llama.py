# models/hyper_llama.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP
from basis_hyper import BasisHyperLayer


class HyperLlamaAttention(LlamaAttention):
    """
    LlamaAttention with Q/K/V/O projections generated on-the-fly from a genome vector.
    Rotary embeddings have been removed for simplicity.
    """
    def __init__(
        self,
        config,
        layer_idx: int,
        genome_dim: int,
        M: int = 64,
        hidden_dim: int = 128
    ) -> None:
        # Initialize the base; we won't use its q_proj etc.
        super().__init__(config, layer_idx)
        # explicitly re‑set these in case the base init didn't (or got wiped)
        self.num_heads = config.num_attention_heads
        self.head_dim  = config.hidden_size // self.num_heads
        E = config.hidden_size

        # delete pretrained linear layers
        del self.q_proj
        del self.k_proj
        del self.v_proj
        del self.o_proj

        # our tiny hypernets
        self.hyper_q = BasisHyperLayer(genome_dim, hidden_dim, E, E, M)
        self.hyper_k = BasisHyperLayer(genome_dim, hidden_dim, E, E, M)
        self.hyper_v = BasisHyperLayer(genome_dim, hidden_dim, E, E, M)
        self.hyper_o = BasisHyperLayer(genome_dim, hidden_dim, E, E, M)

    def forward(
        self,
        hidden_states: Tensor,
        genome_vec: Tensor,
        attention_mask=None,
        **kwargs,
    ):
        """
        Args:
          hidden_states: [B, T, E]
          genome_vec:    [genome_dim] or [B, genome_dim]
        Returns:
          attn_output: [B, T, E]
          None
          (optionally) attn_probs if output_attentions=True
        """
        B, T, E = hidden_states.size()

        # 1) generate weights
        Wq, bq = self.hyper_q(genome_vec)
        Wk, bk = self.hyper_k(genome_vec)
        Wv, bv = self.hyper_v(genome_vec)
        Wo, bo = self.hyper_o(genome_vec)

        # 2) project
        q = hidden_states @ Wq.T + bq
        k = hidden_states @ Wk.T + bk
        v = hidden_states @ Wv.T + bv

        # 3) reshape to (B, H, T, D)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # 4) attention
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if attention_mask is not None:
            scores = scores + attention_mask
        attn_probs  = torch.softmax(scores, dim=-1)
        attn_output = attn_probs @ v  # [B, H, T, D]

        # 5) merge + out proj
        attn_output = attn_output.transpose(1, 2).reshape(B, T, E)
        out = attn_output @ Wo.T + bo

        return (out, None, attn_probs) if kwargs.get("output_attentions", False) else (out, None, None)


class HyperLlamaMLP(LlamaMLP):
    """
    LlamaMLP with its two Linear layers generated on-the-fly from a genome vector.
    """
    def __init__(
        self,
        config,
        genome_dim: int,
        M: int = 64,
        hidden_dim: int = 128
    ) -> None:
        super().__init__(config)
        E     = config.hidden_size
        inner = config.intermediate_size

        # drop pretrained layers
        del self.gate_proj
        del self.up_proj
        del self.down_proj

        # hypernets for up/down
        self.hyper_up   = BasisHyperLayer(genome_dim, hidden_dim, inner, E, M)
        self.hyper_down = BasisHyperLayer(genome_dim, hidden_dim, E, inner, M)

    def forward(self, hidden_states: Tensor, genome_vec: Tensor) -> Tensor:
        # generate
        W_up,   b_up   = self.hyper_up(genome_vec)
        W_down, b_down = self.hyper_down(genome_vec)

        # gate & up
        x_up   = hidden_states @ W_up.T + b_up
        x_gate = hidden_states @ W_up.T + b_up  # reuse same hyper; ok for prototype
        x      = F.silu(x_gate) * x_up

        # down
        x = x @ W_down.T + b_down
        return x


# ——— Self-test ———
if __name__ == "__main__":
    from transformers import LlamaConfig

    # tiny config
    config = LlamaConfig(hidden_size=64, intermediate_size=256, num_attention_heads=2)
    genome_dim, M, hidden_dim = 16, 8, 32
    B, T = 2, 10

    hs = torch.randn(B, T, config.hidden_size)
    z  = torch.randn(genome_dim)

    # test attention
    attn = HyperLlamaAttention(config, layer_idx=0, genome_dim=genome_dim, M=M, hidden_dim=hidden_dim)
    out, _, _ = attn(hs, z)
    assert out.shape == (B, T, config.hidden_size)

    # test mlp
    mlp = HyperLlamaMLP(config, genome_dim, M, hidden_dim)
    out2 = mlp(hs, z)
    assert out2.shape == (B, T, config.hidden_size)

    print("✅ hyper_llama.py smoke‑test passed!")
