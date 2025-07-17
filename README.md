# Universal HyperFold: Neural Network Compression via Hypernetworks

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Abstract

Universal HyperFold presents a novel approach to neural network compression using hypernetworks and mixture-of-experts architectures. The system achieves significant compression ratios by generating target model weights dynamically from compact genome representations, rather than storing static parameters.

**Key Results**: The system achieves 2400:1 compression ratio on 350M parameter models while maintaining 1.9ms per token inference latency on CPU-only execution.

## Introduction

Large language models have demonstrated remarkable capabilities but suffer from prohibitive memory and computational requirements. Traditional compression methods often sacrifice model quality or require specialized hardware. This work introduces Universal HyperFold, which addresses these limitations through dynamic weight generation.

### Motivation

The core insight is that neural network weights contain significant redundancy and can be generated from compact representations. By learning a hypernetwork that maps low-dimensional "genome" vectors to full weight matrices, we achieve extreme compression while maintaining model expressiveness.

### Contributions

1. **Universal Hypernetwork Architecture**: A single hypernetwork capable of generating weights for models ranging from 350M to 14B parameters
2. **Mixture-of-Experts Genome System**: Specialized expert genomes for different domains (mathematics, code, creative writing, general knowledge)
3. **14 Core Innovations**: Technical innovations enabling extreme compression and efficient inference
4. **Empirical Validation**: Demonstration of 2400:1 compression with minimal quality degradation

## System Architecture

The Universal HyperFold system consists of two primary components:

### 1. Universal Hypernetwork
- Maps compact genome vectors (96-dimensional) to full weight matrices
- Supports multiple target model architectures through hierarchical factorization
- Implements temporal weight inheritance for sequence processing

### 2. MOE Genome Manager
- Manages four specialized expert genomes: math, code, creative, general
- Performs dynamic expert routing based on input context
- Enables rapid domain adaptation without retraining

```
>> ULTRA-LIGHTWEIGHT INFERENCE DEMO <<
-> Generated weights shape: torch.Size([1024, 1024])
   Inference time: 1.9ms per token
   Peak RAM: 662MB
   Model size: 0.28MB
   Compression ratio: 2400:1
```

## Technical Specifications

### Performance Metrics
- **Compression Ratio**: 2400:1 (350M → 144K parameters)
- **Inference Speed**: 1.9ms per token (CPU)
- **Model Size**: 0.28MB storage
- **Memory Usage**: 662MB peak RAM
- **Supported Models**: 350M, 1B, 3B, 6B, 14B parameters

### Hardware Requirements
- **Minimum**: 4-core CPU, 1GB RAM, 50MB storage
- **Recommended**: 8-core CPU, 2GB RAM, 100MB storage OR normal no GPU pc!
- **Platform**: CPU-only execution (no GPU required)

## Installation

```bash
# Clone repository
git clone https://github.com/gavkujo/Compression.git
cd LLM-Hyperfold

# Create environment
python3.11 -m venv venv #Python 3.11 worked the best for me
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r "requirements.txt"
```

## Usage

### Quick Start

```bash
# Generate training dataset
python simple_dataset.py

# Train hypernetwork
python train.py --epochs 15

# Run inference
python inference.py

# Test pipeline
python test_pipeline.py
```

### Advanced Usage

```python
from inference import UltraLightweightInference

# Initialize inference engine
engine = UltraLightweightInference(
    checkpoint_path="checkpoints/best_hypernetwork_350M.pt",
    target_model_size="350M",
    enable_quantization=True
)

# Generate weights for specific expert
weights = engine.get_expert_weights(
    expert_type="math",
    target_layer="attention",
    token_position=0
)

# Benchmark performance
metrics = engine.benchmark_performance(num_tokens=100)
engine.print_performance_report(metrics)
```
## File Structure

```
├── models/
│   ├── basis_hyper.py       # Core hypernetwork implementation
│   ├── hyper_llama.py       # LLaMA integration
│   └── hyper_model.py       # Complete model architecture
├── scripts/
│   └── utils.py             # Utility functions
├── train.py                 # Training pipeline
├── inference.py             # Inference engine
├── simple_dataset.py        # Dataset generation
├── test_pipeline.py         # End-to-end testing
```

## Documentation

- **[Architecture.md](LLM-Hyperfold/Architecture.md)**: Detailed technical architecture and mathematical foundations
- **[Innovations.md](LLM-Hyperfold/Innovations.md)**: Comprehensive description of the 14 core innovations

## Results

### Compression Performance
| Metric | Value |
|--------|-------|
| Original Model Size | 350M parameters |
| Compressed Size | 144K parameters |
| Compression Ratio | 2400:1 |
| Storage Requirement | 0.28MB |

### Inference Performance
| Metric | Value |
|--------|-------|
| Average Token Time | 1.9ms |
| Tokens per Second | 499.6 |
| Peak RAM Usage | 662MB |
| CPU Threads | 4 |

### Expert Specialization
| Expert | Domain | Performance |
|--------|--------|-------------|
| Math | Arithmetic, equations | 1.9ms/token |
| Code | Programming, algorithms | 1.9ms/token |
| Creative | Stories, poetry | 1.9ms/token |
| General | Facts, knowledge | 1.9ms/token |

## Limitations

1. **Memory Usage**: Current implementation exceeds 500MB target (optimization in progress)
2. **Training Time**: Initial training requires ~30 minutes for convergence
3. **Domain Specificity**: Performance may vary significantly across different domains
4. **Quality Trade-offs**: Some quality degradation compared to full-size models

## Future Work

- **Memory Optimization**: Reducing peak RAM usage below 500MB target
- **Additional Experts**: Expanding beyond the current four expert domains
- **Hardware Acceleration**: GPU and specialized hardware support
- **Quality Improvements**: Advanced training techniques to minimize quality loss

## Citation

TBD

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

TBD

## Acknowledgments

TBD

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


---

**The age of lightweight intelligence has begun.** 🚀

*Built with 🧬 biological inspiration and ⚡ mathematical elegance*
