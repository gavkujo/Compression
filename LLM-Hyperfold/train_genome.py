import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from transformers import LlamaForCausalLM
from models.basis_hyper import FactorizedBasisHyperLayer
import os

# Config
PRETRAINED_PATH = "hyperllama-init"
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


def reconstruction_loss(W_gen, W_true):
    """Combined loss for better weight reconstruction"""
    # Cosine similarity loss preserves direction
    cos_loss = 1 - F.cosine_similarity(W_gen.flatten(), W_true.flatten(), dim=0)
    
    # Magnitude-aware MSE
    norm_ratio = torch.norm(W_true) / (torch.norm(W_gen) + 1e-8)
    mse_loss = F.mse_loss(W_gen * norm_ratio, W_true)
    
    return 0.7 * cos_loss + 0.3 * mse_loss

def extract_layer_weights(layer):
    """Extract weights from a layer, handling missing biases"""
    weights = {}
    
    # Attention weights
    attn = layer.self_attn
    weights['q'] = (attn.q_proj.weight.clone(), None)
    weights['k'] = (attn.k_proj.weight.clone(), None)
    weights['v'] = (attn.v_proj.weight.clone(), None)
    weights['o'] = (attn.o_proj.weight.clone(), None)
    
    # MLP weights
    mlp = layer.mlp
    weights['gate'] = (mlp.gate_proj.weight.clone(), None)
    weights['up'] = (mlp.up_proj.weight.clone(), None)
    weights['down'] = (mlp.down_proj.weight.clone(), None)
    
    return weights

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
        orig_weights.append(extract_layer_weights(layer))
    
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
    
    total_hyper = sum(p.numel() for p in hypernets.parameters())
    print(f"Standalone hypernetwork params: {total_hyper/1e6:.2f}M")
    
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
                W_true, _= orig_weights[layer_idx][weight_type]
                W_true = W_true.to(DEVICE)
                
                # Generate weights - LLaMA doesn't use biases
                W_gen = hypernet(z).squeeze(0)
                
                # Quantization-aware training in second half
                if epoch > EPOCHS // 2:
                    W_gen, quant_params = hypernet.quantize(W_gen, bits=8)
                    scale = quant_params[0]  # Extract scale from tuple
                    W_gen = W_gen.float() / scale
        
                
                # Calculate loss - only weights, no biases
                loss = reconstruction_loss(W_gen, W_true)
                total_loss += loss
        
        # Optimize
        opt.zero_grad()
        total_loss.backward()
        opt.step()
        
        print(f"Epoch {epoch+1} Loss: {total_loss.item():.4f}, Rank: {current_rank}")
    
    # Save results
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True) 
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