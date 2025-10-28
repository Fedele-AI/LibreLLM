# LibreLLM - Mamba-2 Small Language Model

A small language model implementation using Mamba-2 architecture, trained on the TinyStories dataset.

## Overview

This project implements a **pure PyTorch** version of Mamba-2, a state-space model that serves as an efficient alternative to Transformers. The model automatically detects and uses CUDA GPUs when available, or runs on CPU. Uses only open-source components.

### Key Features
- ✅ Pure PyTorch implementation (no custom CUDA kernels needed)
- ✅ **Automatic GPU detection** - uses CUDA when available, CPU otherwise
- ✅ Mamba-2 architecture with state-space models
- ✅ Trained on TinyStories dataset (~2GB)
- ✅ ~10M parameters (small, efficient model)
- ✅ Works on CPU or GPU (macOS/Linux/Windows)

## Model Architecture

- **Architecture**: Mamba-2 (State-Space Model)
- **Parameters**: ~10M (d_model=256, depth=6)
- **Context Length**: 256 tokens
- **Vocabulary**: GPT-2 tokenizer (50,257 tokens)
- **Training Data**: TinyStories dataset (10% subset, ~50K samples)

## Setup

### Installation

```bash
# 1. Create and activate UV virtual environment
uv venv
source .venv/bin/activate.fish  # or .venv/bin/activate for bash

# 2. Install dependencies
uv pip install -r requirements.txt

# 3. Verify installation (and check for GPU)
python check_cuda.py
```

**For CUDA GPU systems**: See [INSTALL_CUDA.md](INSTALL_CUDA.md) for GPU-specific optimizations.

### Dependencies
- torch (automatically installs with CUDA support if available)
- transformers
- datasets
- tqdm
- numpy
- einops

## Project Structure

```
LibreLLM/
├── mamba2_model.py    # Mamba-2 model implementation
├── train.py           # Training script
├── inference.py       # Text generation script
├── requirements.txt   # Python dependencies
└── checkpoints/       # Saved models (created during training)
```

## Usage

### 1. Training

Train the model on TinyStories dataset:

```bash
python train.py
```

Training configuration:
- **Dataset**: TinyStories (10% subset, ~50K samples, ~2GB download)
- **Epochs**: 3
- **Batch Size**: 16 (increase to 32-64 for GPU)
- **Learning Rate**: 3e-4 with warmup and cosine decay
- **Max Steps**: 5,000
- **Estimated Time**: 
  - **CPU**: 1-2 hours
  - **CUDA GPU**: 15-30 minutes ⚡

The training will:
- **Automatically detect and use GPU** if available
- Download TinyStories dataset (small, high-quality stories)
- Tokenize the data
- Train the model with validation
- Save checkpoints every 1000 steps
- Save the best model based on validation loss
- Generate sample text during training

**GPU Optimization**: For CUDA systems, see [INSTALL_CUDA.md](INSTALL_CUDA.md) for recommended config adjustments (larger batch size, model size, etc.).

### 2. Inference

Generate text with the trained model:

```bash
# Single generation with default prompt
python inference.py --checkpoint checkpoints/best_model.pt

# Custom prompt
python inference.py --checkpoint checkpoints/best_model.pt --prompt "The brave knight"

# Longer generation
python inference.py --checkpoint checkpoints/best_model.pt --max_tokens 200

# Interactive mode
python inference.py --checkpoint checkpoints/best_model.pt --interactive
```

Parameters:
- `--checkpoint`: Path to model checkpoint (default: checkpoints/best_model.pt)
- `--prompt`: Text prompt for generation
- `--max_tokens`: Maximum tokens to generate (default: 100)
- `--temperature`: Sampling temperature (default: 0.8)
- `--top_k`: Top-k sampling (default: 40)
- `--interactive`: Run in interactive mode

## Model Details

### Mamba-2 Architecture

Mamba-2 is based on state-space models (SSMs) which provide an efficient alternative to attention mechanisms:

1. **State-Space Models**: Linear-time sequence modeling
2. **Selective Scan**: Context-aware token processing
3. **No Positional Embeddings**: Position is implicit in the state
4. **Efficient**: O(L) complexity vs O(L²) for attention

### Components

- **Embedding Layer**: Token embeddings
- **Mamba Blocks**: 6 layers of state-space processing
  - RMSNorm normalization
  - Input projection
  - 1D convolution
  - Selective state-space computation
  - Output projection
- **LM Head**: Projects to vocabulary for next-token prediction

### Training Details

- **Loss**: Cross-entropy (next-token prediction)
- **Optimizer**: AdamW (betas=0.9,0.95, weight_decay=0.1)
- **Schedule**: Linear warmup (100 steps) + cosine decay
- **Gradient Clipping**: Max norm of 1.0
- **Validation**: Every 500 steps with text generation samples

## Dataset

**TinyStories** (Eldan & Li, 2023):
- High-quality short stories for children
- Simple vocabulary and grammar
- Perfect for small language models
- Dataset size: ~2GB (using 10% subset)
- Clean, human-written text

## Results

After training, you can expect:
- Coherent short stories generation
- Simple narrative structure
- Vocabulary appropriate for children's stories
- Some grammatical consistency

Example output:
```
Prompt: "Once upon a time"
Generated: "Once upon a time there was a little girl named Lily. She loved to 
play outside in the sunshine. One day, she saw a big red ball in the park..."
```

## Advantages of Mamba-2

1. **Efficiency**: Linear time complexity (vs quadratic for Transformers)
2. **Long Context**: Can handle longer sequences efficiently
3. **No Positional Encoding**: Learned implicitly
4. **Simplicity**: Fewer components than Transformers
5. **CPU Friendly**: Works well without GPU acceleration

## Limitations

- Smaller model (~10M params) → limited knowledge
- Trained on simple stories → limited domain
- No instruction tuning → not chat-optimized
- CPU training is slow (GPU recommended for larger experiments)

## Future Improvements

- [ ] Train on larger dataset (WikiText, OpenWebText)
- [ ] Increase model size (d_model=512, depth=12)
- [ ] Implement gradient checkpointing for larger models
- [ ] Add CUDA kernels for faster GPU training
- [ ] Instruction tuning for chat capabilities
- [ ] Quantization for deployment

## References

- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
- [Transformers are SSMs](https://arxiv.org/abs/2405.21060) (Mamba-2 paper)
- [TinyStories Dataset](https://arxiv.org/abs/2305.07759)

## License

MIT License - Feel free to use for research and learning!

## Acknowledgments

- Agora-Lab-AI for the pure PyTorch Mamba-2 reference implementation
- Hugging Face for datasets and tokenizers
- TinyStories dataset creators
