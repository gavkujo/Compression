#!/usr/bin/env python3
"""
🚀 UNIVERSAL HYPERNETWORK TRAINER
================================
Train the universal hypernetwork + 4 expert genomes on specialized datasets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import time
import os
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math
from tqdm import tqdm
import psutil

# Import our models
from build import UniversalHyperNetwork, MOEGenomeManager, MODEL_CONFIGS
from scripts.utils import measure_ram, set_cpu_threads

@dataclass
class TrainingConfig:
    # Only field definitions here; instantiation happens in main()
    target_model_size: str = "1B"
    genome_dim: int = 96
    hyper_hidden: int = 256
    batch_size: int = 4
    learning_rate: float = 1e-4
    num_epochs: int = 10
    max_sequence_length: int = 256
    max_genome_size_mb: float = 1.0
    max_inference_ram_mb: float = 500.0
    max_token_time_ms: float = 100.0
    target_compression_ratio: float = 10.0
    num_experts: int = 4
    
    # Expert specialization
    expert_types: List[str] = None
    expert_loss_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.expert_types is None:
            self.expert_types = ["math", "code", "creative", "general"]
        if self.expert_loss_weights is None:
            self.expert_loss_weights = {
                "math": 1.0,
                "code": 1.0, 
                "creative": 1.0,
                "general": 1.0
            }

class ExpertDataset(Dataset):
    """Dataset for expert-specific training"""
    
    def __init__(self, csv_path: str, expert_type: str, max_length: int = 512, tokenizer=None):
        self.df = pd.read_csv(csv_path)
        self.expert_type = expert_type
        self.max_length = max_length
        self.tokenizer = tokenizer
        # Filter for specific expert
        if expert_type != "all":
            self.df = self.df[self.df['expert_type'] == expert_type]
        print(f"📊 Loaded {len(self.df)} samples for expert '{expert_type}'")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        prompt = str(row['prompt'])
        output = str(row['output'])
        expert_type = row['expert_type']
        # Input text (prompt + output for LM)
        input_text = f"{prompt} {output}"
        # Subword tokenization
        if self.tokenizer:
            input_ids = self.tokenizer.encode(input_text, add_special_tokens=True, truncation=True, max_length=self.max_length)
        else:
            # Fallback: character-level
            input_ids = [ord(c) % 1000 for c in input_text[:self.max_length]]
        # Pad or truncate
        if len(input_ids) < self.max_length:
            input_ids.extend([0] * (self.max_length - len(input_ids)))
        else:
            input_ids = input_ids[:self.max_length]
        # Labels for LM (shifted by 1)
        labels = input_ids[1:] + [0]
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'expert_type': expert_type,
            'prompt': prompt,
            'output': output
        }

class HyperNetworkTrainer:
    def compute_perplexity(self, data_loader) -> float:
        """Compute perplexity on a given data loader"""
        self.hypernetwork.eval()
        self.genome_manager.eval()
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                batch_size, seq_len = input_ids.shape
                expert_types = [self.route_expert(batch['prompt'][i]) for i in range(len(batch['prompt']))]
                genomes = torch.stack([
                    self.genome_manager.get_expert_genome(exp_type,
                        global_context=torch.randn(self.config.genome_dim//4, device=self.device),
                        position_id=0
                    ) for exp_type in expert_types
                ])
                model_config = MODEL_CONFIGS[self.config.target_model_size]
                target_shape = (model_config['hidden_size'], model_config['hidden_size'])
                hidden_states = torch.randn(batch_size, seq_len, model_config['hidden_size'], device=self.device)
                batch_loss = 0.0
                for pos in range(min(seq_len, 32)):
                    weights = self.hypernetwork.generate_weights(
                        genomes,
                        target_shape=target_shape,
                        target_model=self.config.target_model_size,
                        token_position=pos
                    )
                    if weights.dim() == 3:
                        output = torch.bmm(hidden_states[:, pos:pos+1, :], weights.transpose(-2, -1))
                    else:
                        output = torch.matmul(hidden_states[:, pos:pos+1, :], weights.T)
                    if pos < seq_len - 1:
                        logits = torch.matmul(output, hidden_states[:, pos+1:pos+2, :].transpose(-2, -1)).squeeze(-1)
                        target_logits = torch.zeros_like(logits)
                        loss = F.mse_loss(logits, target_logits, reduction='sum')
                        batch_loss += loss.item()
                total_loss += batch_loss
                total_tokens += batch_size * min(seq_len, 32)
        if total_tokens == 0:
            return float('inf')
        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)
        return perplexity
    def route_expert(self, prompt: str) -> str:
        """Keyword-based expert router for prompt"""
        prompt_lower = prompt.lower()
        # Simple keyword rules
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
        elif any(k in prompt_lower for k in general_keywords):
            return "general"
        else:
            # Default fallback
            return "general"
    """Universal HyperNetwork Trainer with MOE expert specialization"""
    
    def __init__(self, config: TrainingConfig, tokenizer_path: str = None, vocab_size: int = 1000):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔥 Using device: {self.device}")
        # --- Tokenizer Integration ---
        from transformers import PreTrainedTokenizerFast
        if tokenizer_path and os.path.exists(tokenizer_path):
            print(f"📦 Loading tokenizer from {tokenizer_path}")
            self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
        else:
            print("No tokenizer found, exiting...")
            raise FileNotFoundError("Tokenizer file not found. Please provide a valid tokenizer path.")
        
        self.tokenizer.model_max_length = self.config.max_sequence_length
        # Initialize model components before optimizers
        model_config = MODEL_CONFIGS[self.config.target_model_size]
        self.hypernetwork = UniversalHyperNetwork(
            genome_dim=self.config.genome_dim,
            hyper_hidden=self.config.hyper_hidden,
            target_configs=MODEL_CONFIGS,
            enable_streaming=True,
            enable_temporal=True,
            max_hidden_size=model_config['hidden_size']
        )
        self.genome_manager = MOEGenomeManager(
            genome_dim=self.config.genome_dim,
            num_experts=self.config.num_experts,
            expert_types=self.config.expert_types
        )
        self._init_models()
        self._init_datasets()
        self._init_optimizers()
        # Training state
        self.step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        # Performance tracking
        self.training_stats = {
            'losses': [],
            'expert_losses': {expert: [] for expert in config.expert_types},
            'compression_ratios': [],
            'inference_times': [],
            'ram_usage': []
        }
    
    def _init_models(self):
        """Initialize universal hypernetwork and expert genomes"""
        print("🏗️ Initializing Universal HyperNetwork...")
        
        # Move models to device before optimizer creation
        self.hypernetwork.to(self.device)
        self.genome_manager.to(self.device)

        self.hyper_optimizer = torch.optim.AdamW(
            self.hypernetwork.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.95)
        )

        self.genome_optimizer = torch.optim.AdamW(
            self.genome_manager.parameters(),
            lr=self.config.learning_rate * 0.1,  # Slower learning for genomes
            weight_decay=0.001,
            betas=(0.9, 0.95)
        )
        
        # Print model statistics
        hyper_params = sum(p.numel() for p in self.hypernetwork.parameters())
        genome_params = sum(p.numel() for p in self.genome_manager.parameters())
        total_params = hyper_params + genome_params

        print(f"📊 Model Statistics:")
        print(f"   HyperNetwork: {hyper_params/1e6:.2f}M parameters")
        print(f"   Genome Manager: {genome_params/1e3:.1f}K parameters")
        print(f"   Total: {total_params/1e6:.2f}M parameters")

        # Estimate target model size for compression ratio
        model_config = MODEL_CONFIGS[self.config.target_model_size]
        target_params = model_config['hidden_size'] * model_config['vocab_size'] + \
            model_config['num_hidden_layers'] * (model_config['hidden_size'] ** 2) * 12  # Rough estimate
        compression_ratio = target_params / total_params
        print(f"   Target Compression: {compression_ratio:.1f}x")

        # Check if we meet storage requirements
        storage_mb = total_params * 4 / (1024 ** 2)  # FP32 bytes to MB
        print(f"   Storage: {storage_mb:.2f}MB (target: <{self.config.max_genome_size_mb}MB)")

        if storage_mb > self.config.max_genome_size_mb:
            print(f"⚠️  Warning: Storage exceeds target by {storage_mb - self.config.max_genome_size_mb:.2f}MB")
    
    def _init_datasets(self):
        """Initialize datasets for all experts"""
        print("📚 Loading datasets...")
        # Load combined dataset for mixed training
        self.combined_dataset = ExpertDataset(
            "datasets/combined_dataset.csv", 
            "all", 
            self.config.max_sequence_length,
            tokenizer=self.tokenizer
        )
        # Load expert-specific datasets
        self.expert_datasets = {}
        for expert_type in self.config.expert_types:
            self.expert_datasets[expert_type] = ExpertDataset(
                f"datasets/{expert_type}_dataset.csv",
                expert_type,
                self.config.max_sequence_length,
                tokenizer=self.tokenizer
            )
        # Create data loaders
        self.combined_loader = DataLoader(
            self.combined_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        self.expert_loaders = {}
        for expert_type, dataset in self.expert_datasets.items():
            expert_batch_size = max(1, self.config.batch_size // self.config.num_experts)
            self.expert_loaders[expert_type] = DataLoader(
                dataset,
                batch_size=expert_batch_size,  # Ensure batch size is always >= 1
                shuffle=True,
                num_workers=1,
                pin_memory=True if self.device.type == 'cuda' else False
            )
        print(f"✅ Loaded datasets: {len(self.combined_dataset)} total samples")
    
    def _init_optimizers(self):
        """Initialize optimizers with different learning rates"""
        print("🎯 Setting up optimizers...")
        
        # Separate optimizers for hypernetwork and genomes
        # ...existing code...
        
        # Calculate total steps for schedulers
        if hasattr(self, 'combined_loader'):
            total_steps = len(self.combined_loader) * self.config.num_epochs
        else:
            # Fallback: estimate from batch size and dataset size
            total_steps = 1000 * self.config.num_epochs

        self.hyper_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.hyper_optimizer, 
            T_max=total_steps,
            eta_min=1e-6
        )

        self.genome_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.genome_optimizer,
            T_max=total_steps,
            eta_min=1e-7
        )
    
    def forward_pass(self, batch, expert_type: str = None) -> Tuple[torch.Tensor, Dict]:
        """Forward pass through hypernetwork with expert routing"""
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)
        batch_size, seq_len = input_ids.shape
        # Determine expert for each sample using router if not provided
        if expert_type is None:
            expert_types = [self.route_expert(batch['prompt'][i]) for i in range(len(batch['prompt']))]
            genomes = torch.stack([
                self.genome_manager.get_expert_genome(exp_type,
                    global_context=torch.randn(self.config.genome_dim//4, device=self.device),
                    position_id=0
                ) for exp_type in expert_types
            ])
        else:
            genomes = torch.stack([
                self.genome_manager.get_expert_genome(expert_type,
                    global_context=torch.randn(self.config.genome_dim//4, device=self.device),
                    position_id=0
                ) for _ in range(batch_size)
            ])
        model_config = MODEL_CONFIGS[self.config.target_model_size]
        target_shape = (model_config['hidden_size'], model_config['hidden_size'])
        total_loss = 0.0
        metrics = {'per_token_time': [], 'memory_usage': []}
        hidden_states = torch.randn(batch_size, seq_len, model_config['hidden_size'], device=self.device)
        for pos in range(min(seq_len, 32)):
            start_time = time.perf_counter()
            weights = self.hypernetwork.generate_weights(
                genomes,
                target_shape=target_shape,
                target_model=self.config.target_model_size,
                token_position=pos
            )
            if weights.dim() == 3:
                output = torch.bmm(hidden_states[:, pos:pos+1, :], weights.transpose(-2, -1))
            else:
                output = torch.matmul(hidden_states[:, pos:pos+1, :], weights.T)
            if pos < seq_len - 1:
                logits = torch.matmul(output, hidden_states[:, pos+1:pos+2, :].transpose(-2, -1)).squeeze(-1)
                target_logits = torch.zeros_like(logits)
                loss = F.mse_loss(logits, target_logits)
                total_loss += loss
            hidden_states = hidden_states.clone()
            hidden_states[:, pos:pos+1, :] = output
            token_time = (time.perf_counter() - start_time) * 1000
            metrics['per_token_time'].append(token_time)
        current_ram, _ = measure_ram()
        metrics['memory_usage'].append(current_ram)
        return total_loss / min(seq_len, 32), metrics
    
    def train_epoch(self):
        """Train one epoch with expert specialization"""
        self.hypernetwork.train()
        self.genome_manager.train()
        
        epoch_loss = 0.0
        epoch_expert_losses = {expert: 0.0 for expert in self.config.expert_types}
        num_batches = 0
        
        # Progress bar
        pbar = tqdm(self.combined_loader, desc=f"Epoch {self.epoch+1}")
        
        for batch_idx, batch in enumerate(pbar):
            self.step += 1
            
            # Clear gradients
            self.hyper_optimizer.zero_grad()
            self.genome_optimizer.zero_grad()
            
            # Mixed expert training on combined data
            loss, metrics = self.forward_pass(batch)
            
            # Expert-specific training (alternate between experts)
            expert_idx = self.step % self.config.num_experts
            expert_type = self.config.expert_types[expert_idx]
            
            # Get expert-specific batch
            try:
                expert_batch = next(iter(self.expert_loaders[expert_type]))
                expert_loss, expert_metrics = self.forward_pass(expert_batch, expert_type)
                
                # Weighted combination
                total_loss = loss + self.config.expert_loss_weights[expert_type] * expert_loss
                epoch_expert_losses[expert_type] += expert_loss.item()
                
            except StopIteration:
                # If expert loader is exhausted, just use combined loss
                total_loss = loss
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.hypernetwork.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.genome_manager.parameters(), 0.5)
            
            # Optimizer steps
            self.hyper_optimizer.step()
            self.genome_optimizer.step()
            
            # Scheduler steps
            self.hyper_scheduler.step()
            self.genome_scheduler.step()
            
            # Update metrics
            epoch_loss += total_loss.item()
            num_batches += 1
            
            # Log performance metrics
            if metrics['per_token_time']:
                avg_token_time = np.mean(metrics['per_token_time'])
                self.training_stats['inference_times'].append(avg_token_time)
            
            if metrics['memory_usage']:
                avg_memory = np.mean(metrics['memory_usage'])
                self.training_stats['ram_usage'].append(avg_memory)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'token_ms': f"{avg_token_time:.1f}" if metrics['per_token_time'] else "N/A",
                'ram_mb': f"{avg_memory:.0f}" if metrics['memory_usage'] else "N/A"
            })
            
            # Warmup
            warmup_steps = getattr(self.config, 'warmup_steps', 100)
            if self.step <= warmup_steps:
                warmup_factor = self.step / warmup_steps
                for param_group in self.hyper_optimizer.param_groups:
                    param_group['lr'] = self.config.learning_rate * warmup_factor
                for param_group in self.genome_optimizer.param_groups:
                    param_group['lr'] = self.config.learning_rate * 0.1 * warmup_factor
        
        # Calculate epoch averages
        avg_loss = epoch_loss / num_batches
        avg_expert_losses = {k: v / max(num_batches//self.config.num_experts, 1) 
                           for k, v in epoch_expert_losses.items()}
        
        return avg_loss, avg_expert_losses
    
    def evaluate(self) -> Dict:
        """Evaluate model performance and compression metrics"""
        self.hypernetwork.eval()
        self.genome_manager.eval()
        print("🧪 Running evaluation...")
        # Setup for evaluation
        model_config = MODEL_CONFIGS[self.config.target_model_size]
        test_genome = self.genome_manager.get_expert_genome("math")
        target_shape = (model_config['hidden_size'], model_config['hidden_size'])
        start_time = time.perf_counter()
        for _ in range(10):  # Average over 10 runs
            weights = self.hypernetwork.generate_weights(
                test_genome,
                target_shape=target_shape,
                target_model=self.config.target_model_size,
                token_position=0
            )
        inference_time = (time.perf_counter() - start_time) / 10 * 1000  # ms per forward
        # Measure memory
        current_ram, _ = measure_ram()
        # Calculate compression ratio
        total_params = sum(p.numel() for p in self.hypernetwork.parameters()) + \
                      sum(p.numel() for p in self.genome_manager.parameters())
        target_model_params = model_config['hidden_size'] * model_config['vocab_size'] + \
                            model_config['num_hidden_layers'] * (model_config['hidden_size'] ** 2) * 12
        compression_ratio = target_model_params / total_params
        # Storage size
        storage_mb = total_params * 4 / (1024 ** 2)  # FP32 to MB

        metrics = {
            'inference_time_ms': inference_time,
            'memory_usage_mb': current_ram,
            'compression_ratio': compression_ratio,
            'storage_mb': storage_mb,
            'meets_time_requirement': inference_time < self.config.max_token_time_ms,
            'meets_memory_requirement': current_ram < self.config.max_inference_ram_mb,
            'meets_storage_requirement': storage_mb < self.config.max_genome_size_mb,
            'meets_compression_requirement': compression_ratio > self.config.target_compression_ratio
        }
        return metrics
    
    def save_checkpoint(self, path: str, metrics: Dict = None):
        """Save training checkpoint"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            'epoch': self.epoch,
            'step': self.step,
            'config': self.config.__dict__,
            'hypernetwork_state': self.hypernetwork.state_dict(),
            'genome_manager_state': self.genome_manager.state_dict(),
            'hyper_optimizer_state': self.hyper_optimizer.state_dict(),
            'genome_optimizer_state': self.genome_optimizer.state_dict(),
            'training_stats': self.training_stats,
            'metrics': metrics or {}
        }
        
        torch.save(checkpoint, path)
        print(f"💾 Saved checkpoint: {path}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.hypernetwork.load_state_dict(checkpoint['hypernetwork_state'])
        self.genome_manager.load_state_dict(checkpoint['genome_manager_state'])
        self.hyper_optimizer.load_state_dict(checkpoint['hyper_optimizer_state'])
        self.genome_optimizer.load_state_dict(checkpoint['genome_optimizer_state'])
        self.training_stats = checkpoint['training_stats']
        
        print(f"📂 Loaded checkpoint: {path}")
    
    def train(self):
        """Main training loop with perplexity tracking"""
        print("🚀 Starting Universal HyperNetwork Training...")
        print(f"Target Model: {self.config.target_model_size}")
        print(f"Epochs: {self.config.num_epochs}")
        print(f"Batch Size: {self.config.batch_size}")
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            print(f"\n📅 Epoch {epoch + 1}/{self.config.num_epochs}")
            # Train epoch
            avg_loss, expert_losses = self.train_epoch()
            # Log losses
            self.training_stats['losses'].append(avg_loss)
            for expert, loss in expert_losses.items():
                self.training_stats['expert_losses'][expert].append(loss)
            print(f"📈 Epoch {epoch + 1} Results:")
            print(f"   Average Loss: {avg_loss:.4f}")
            for expert, loss in expert_losses.items():
                print(f"   {expert.title()} Loss: {loss:.4f}")
            # Compute perplexity every epoch
            print("🔎 Computing perplexity on validation set...")
            val_loader = self.combined_loader  # For now, use combined loader as validation
            perplexity = self.compute_perplexity(val_loader)
            print(f"   Perplexity: {perplexity:.2f}")
            self.training_stats.setdefault('perplexities', []).append(perplexity)
            # Evaluate every few epochs
            if (epoch + 1) % 2 == 0 or epoch == self.config.num_epochs - 1:
                metrics = self.evaluate()
                metrics['perplexity'] = perplexity
                # Save best model
                if avg_loss < self.best_loss:
                    self.best_loss = avg_loss
                    self.save_checkpoint(
                        f"checkpoints/best_hypernetwork_{self.config.target_model_size}.pt",
                        metrics
                    )
            # Save regular checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(
                    f"checkpoints/hypernetwork_{self.config.target_model_size}_epoch_{epoch+1}.pt"
                )
        print("\n🎉 Training completed!")
        # Final evaluation
        final_metrics = self.evaluate()
        final_metrics['perplexity'] = self.training_stats['perplexities'][-1] if 'perplexities' in self.training_stats else None
        # Save final model
        self.save_checkpoint(
            f"checkpoints/final_hypernetwork_{self.config.target_model_size}.pt",
            final_metrics
        )
        return final_metrics

def main():
    """Main training function"""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Universal HyperNetwork')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs to train')
    parser.add_argument('--quick-test', action='store_true', help='Quick test mode (minimal training)')
    args = parser.parse_args()
    
    print("🎯 UNIVERSAL HYPERNETWORK TRAINING")
    print("=" * 50)
    
    if args.quick_test:
        print("⚡ Quick test mode enabled")
    
    # Set CPU threads for efficiency
    set_cpu_threads(8)
    batch_size = 4 if not args.quick_test else 2
    if batch_size <= 0:
        batch_size = 1
    
    # Training configuration
    config = TrainingConfig(
        target_model_size="1B",  # Start with smaller model
        genome_dim=96,
        hyper_hidden=256,
        batch_size=batch_size,  # Small batch for memory efficiency
        learning_rate=5e-4,
        num_epochs=args.epochs,
        max_sequence_length=256,  # Shorter sequences for training speed
        max_genome_size_mb=1.0,
        max_inference_ram_mb=500.0,
        max_token_time_ms=100.0,
        target_compression_ratio=10.0,
        num_experts=4
    )
    print(f"⚡ Batch size for training: {config.batch_size}")
    
    # Override for quick test
    if args.quick_test:
        config.num_epochs = 1
        config.batch_size = 2
        config.max_sequence_length = 64
        print("🔧 Quick test config applied")
    
    # Initialize trainer
    tokenizer_path = f"scripts/tokenizer_{config.target_model_size}/tokenizer.json"
    trainer = HyperNetworkTrainer(config, tokenizer_path=tokenizer_path)
    
    # Train model
    try:
        final_metrics = trainer.train()
        
        print("\n🏆 FINAL RESULTS:")
        print("=" * 30)
        for key, value in final_metrics.items():
            if isinstance(value, bool):
                status = "✅" if value else "❌"
                print(f"{status} {key}: {value}")
            else:
                print(f"📊 {key}: {value}")
        
        # Save final results
        results_file = f"results/training_results_{config.target_model_size}.json"
        if args.quick_test:
            results_file = f"results/quick_test_results_{config.target_model_size}.json"
            
        with open(results_file, "w") as f:
            json.dump({
                'config': config.__dict__,
                'final_metrics': final_metrics,
                'training_stats': trainer.training_stats
            }, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to {results_file}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted. Saving checkpoint...")
        trainer.save_checkpoint(f"checkpoints/interrupted_hypernetwork_{config.target_model_size}.pt")
    
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Create directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    main()
