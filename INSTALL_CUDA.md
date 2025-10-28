# Installation Guide - CUDA Systems

## Quick Install (CUDA GPU Systems)

Since you have a CUDA GPU, the installation is even simpler. The pure PyTorch implementation will automatically use your GPU!

### Step 1: Install Dependencies

```bash
# In your UV virtual environment
uv pip install -r requirements.txt
```

This will install:
- PyTorch with CUDA support (automatically detected)
- Transformers
- Datasets
- Other core dependencies

**Note**: We removed `mamba-ssm` and `causal-conv1d` from requirements because:
- They require CUDA compilation with nvcc
- Our pure PyTorch implementation doesn't need them
- This makes installation much simpler and faster!

### Step 2: Verify GPU Detection

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

Expected output:
```
CUDA available: True
GPU: [Your GPU name]
```

### Step 3: Test the Setup

```bash
python test_setup.py
```

You should see:
```
Device: cuda
```

### Step 4: Start Training!

```bash
python train.py
```

## GPU Training Benefits

With CUDA GPU, you'll get:
- ⚡ **10-20x faster training** (15-30 min instead of 1-2 hours)
- 🚀 **Can use larger models** (increase `d_model`, `depth`)
- 📊 **Bigger batches** (increase `batch_size` for faster training)
- 💪 **More data** (use full dataset if you want)

## Optimizing for GPU

Edit `train.py` to take advantage of your GPU:

```python
class Config:
    # Increase model size (GPU can handle it)
    d_model = 512       # Was 256
    depth = 8           # Was 6
    
    # Larger batches for faster training
    batch_size = 32     # Was 16 (adjust based on GPU memory)
    
    # More training steps
    max_steps = 10000   # Was 5000
    
    # Use more data
    dataset_subset = "train[:50%]"  # Was 10%
    max_samples = 250000            # Was 50000
```

## GPU Memory Usage

Approximate GPU memory needed:

| Config | VRAM | Training Time |
|--------|------|---------------|
| Default (d_model=256, batch=16) | ~2-3 GB | 15-20 min |
| Medium (d_model=512, batch=32) | ~6-8 GB | 20-30 min |
| Large (d_model=768, batch=48) | ~12-16 GB | 30-45 min |

## Monitoring GPU Usage

During training, you can monitor GPU usage:

```bash
# In another terminal
watch -n 1 nvidia-smi
```

## Troubleshooting

### "CUDA out of memory"
→ Reduce `batch_size` in `train.py`
→ Or reduce `d_model` / `depth`

### "CUDA available: False"
→ Check PyTorch installation: `python -c "import torch; print(torch.__version__)"`
→ Reinstall with: `uv pip install torch --upgrade`

### Training still seems slow
→ Check GPU is being used: `nvidia-smi` (should show python process)
→ Make sure you see "Device: cuda" at training start

## Multi-GPU Training (Optional)

If you have multiple GPUs, you can enable data parallelism:

Edit `train.py` and add after model creation:
```python
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)
```

## Quick Start Commands

```bash
# 1. Install (one time)
uv pip install -r requirements.txt

# 2. Verify GPU
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"

# 3. Test setup
python test_setup.py

# 4. Train (15-30 min on GPU!)
python train.py

# 5. Generate text
python inference.py --interactive
```

## Expected Performance

With a modern GPU (e.g., RTX 3080, A100, etc.):
- **Setup**: 2 minutes
- **Dataset download**: 5-10 minutes (one time)
- **Training**: 15-30 minutes
- **Total**: ~30-45 minutes from start to trained model!

Much faster than the 1-2 hours on CPU! 🚀

---

**You're ready to train much faster with CUDA!** 🎉
