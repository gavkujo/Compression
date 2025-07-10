import torch
import torch.nn as nn
import torch.nn.functional as F  # Fixed import
import torch.optim as optim
from tqdm import tqdm
from transformers import LlamaForCausalLM
from scripts.utils import quantize_model
from models.basis_hyper import FactorizedBasisHyperLayer
import os

# Config
PRETRAINED_PATH = "hyperllama-init"  # Local path
GENOME_DIM = 96
HYPER_HIDDEN = 256
M_BASIS = 32
RANK_START = 64
RANK_END = 32
EPOCHS = 10
LR = 1e-3
BATCH_SIZE = 8
SAVE_PATH = "checkpoints/hyperfold_genome.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print(f"Using device: {DEVICE}")
    
    # Load pretrained model
    print("Loading pretrained model...")
    base = LlamaForCausalLM.from_pretrained(PRETRAINED_PATH)
    base.to(DEVICE)
    base.eval()
    
    # Extract weights
    print("Extracting weights...")
    orig_weights = []
    for layer in base.model.layers:
        layer_weights = {}
        
        # Attention weights
        attn = layer.self_attn
        layer_weights['q'] = (attn.q_proj.weight.clone(), attn.q_proj.bias.clone())
        layer_weights['k'] = (attn.k_proj.weight.clone(), attn.k_proj.bias.clone())
        layer_weights['v'] = (attn.v_proj.weight.clone(), attn.v_proj.bias.clone())
        layer_weights['o'] = (attn.o_proj.weight.clone(), attn.o_proj.bias.clone())
        
        # MLP weights
        mlp = layer.mlp
        layer_weights['gate'] = (mlp.gate_proj.weight.clone(), mlp.gate_proj.bias.clone())
        layer_weights['up'] = (mlp.up_proj.weight.clone(), mlp.up_proj.bias.clone())
        layer_weights['down'] = (mlp.down_proj.weight.clone(), mlp.down_proj.bias.clone())
        
        orig_weights.append(layer_weights)
    
    num_layers = len(orig_weights)
    del base  # Free up memory
    print(f"Extracted weights for {num_layers} layers")
    
    # Initialize genome and hypernets
    genome = nn.Parameter(torch.randn(num_layers, GENOME_DIM, device=DEVICE))
    hypernets = nn.ModuleDict()
    
    # Create hypernets for each weight type
    for weight_type in ['q', 'k', 'v', 'o', 'gate', 'up', 'down']:
        W, _ = orig_weights[0][weight_type]
        hypernets[weight_type] = FactorizedBasisHyperLayer(
            genome_dim=GENOME_DIM,
            hidden_dim=HYPER_HIDDEN,
            out_dim=W.size(0),
            in_dim=W.size(1),
            M=M_BASIS,
            rank=RANK_START
        ).to(DEVICE)
    
    # Optimizer
    opt = optim.AdamW([
        {'params': genome, 'lr': LR},
        {'params': hypernets.parameters(), 'lr': LR}
    ])
    
    # Progressive rank schedule
    ranks = torch.linspace(RANK_START, RANK_END, EPOCHS).int().tolist()
    
    # Training loop
    print("Starting training...")
    for epoch in range(EPOCHS):
        total_loss = 0.0
        current_rank = ranks[epoch]
        
        # Update hypernet ranks
        for hypernet in hypernets.values():
            hypernet.set_rank(current_rank)
        
        # Process each layer
        for layer_idx in tqdm(range(num_layers), desc=f"Epoch {epoch+1}/{EPOCHS}"):
            z = genome[layer_idx].unsqueeze(0)  # [1, genome_dim]
            
            # Reconstruct all weights for this layer
            for weight_type, hypernet in hypernets.items():
                W_true, b_true = orig_weights[layer_idx][weight_type]
                W_true = W_true.to(DEVICE)
                b_true = b_true.to(DEVICE) if b_true is not None else None
                
                # Generate weights
                W_gen, b_gen = hypernet(z)
                W_gen = W_gen.squeeze(0)
                b_gen = b_gen.squeeze(0) if b_gen is not None else None
                
                # Quantization-aware training in second half
                if epoch > EPOCHS // 2:
                    W_gen, scale = hypernet.quantize(W_gen, bits=8)
                    W_gen = W_gen.float() / scale
                
                # Calculate loss
                loss = F.mse_loss(W_gen, W_true)
                if b_true is not None and b_gen is not None:
                    loss += F.mse_loss(b_gen, b_true)
                
                total_loss += loss
        
        # Optimize
        opt.zero_grad()
        total_loss.backward()
        opt.step()
        
        print(f"Epoch {epoch+1} Loss: {total_loss.item():.4f}, Rank: {current_rank}")
    
    # Save results
    torch.save({
        'genome': genome.detach().cpu(),
        'hypernets': hypernets.state_dict(),
        'config': {
            'genome_dim': GENOME_DIM,
            'hyper_hidden': HYPER_HIDDEN,
            'M': M_BASIS,
            'rank': RANK_END
        }
    }, SAVE_PATH)
    print(f"Training complete! Saved to {SAVE_PATH}")

if __name__ == "__main__":
    main()