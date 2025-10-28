# Quick Installation Commands (CUDA System)

For `kjenkins60@atl1-1-02-010-8-1` with CUDA GPU:

```bash
# 1. Install dependencies (fixed - no CUDA compilation needed!)
uv pip install -r requirements.txt

# 2. Verify CUDA GPU is detected
python check_cuda.py

# 3. Test the setup
python test_setup.py

# 4. Start training (will use GPU automatically!)
python train.py
```

## What Changed

✅ **Removed problematic packages** from `requirements.txt`:
   - `mamba-ssm` (required nvcc compilation)
   - `causal-conv1d` (required nvcc compilation)
   - `wandb` (optional, made comment)

✅ **Pure PyTorch implementation** - no CUDA compilation needed!

✅ **Automatic GPU detection** - all scripts detect and use CUDA when available

## Expected Output

After `python check_cuda.py`:
```
LibreLLM - CUDA System Check
============================================================

✓ PyTorch installed: 2.9.0+cu128

CUDA Available: True

🎉 GPU acceleration is available!

GPU Information:
  Device count: 1
  GPU 0: [Your GPU Name]
    Memory: XX.XX GB
    Compute Capability: X.X

CUDA Version: 12.8
...
```

## Training Time

With your CUDA GPU:
- **15-30 minutes** instead of 1-2 hours! 🚀

## Optimization Tips

For even better performance, edit `train.py`:

```python
class Config:
    # Use GPU power!
    batch_size = 32      # Was 16 (double it!)
    d_model = 512        # Was 256 (bigger model)
    depth = 8            # Was 6 (more layers)
    max_steps = 10000    # Was 5000 (train longer)
```

This will give you a much better model in about the same time!
