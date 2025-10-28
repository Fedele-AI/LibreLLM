#!/usr/bin/env python3
"""
Check CUDA availability and system info.
Run this after installation to verify GPU setup.
"""
import sys


def check_cuda():
    """Check CUDA availability and print system info."""
    print("="*60)
    print("LibreLLM - CUDA System Check")
    print("="*60)
    
    # Check PyTorch
    try:
        import torch
        print(f"\n✓ PyTorch installed: {torch.__version__}")
    except ImportError:
        print("\n✗ PyTorch not found!")
        print("  Install with: uv pip install -r requirements.txt")
        sys.exit(1)
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"\nCUDA Available: {cuda_available}")
    
    if cuda_available:
        print("\n🎉 GPU acceleration is available!")
        print(f"\nGPU Information:")
        print(f"  Device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            
            # Get memory info
            props = torch.cuda.get_device_properties(i)
            total_memory = props.total_memory / (1024**3)  # Convert to GB
            print(f"    Memory: {total_memory:.2f} GB")
            print(f"    Compute Capability: {props.major}.{props.minor}")
        
        # CUDA version
        print(f"\nCUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        
        # Test GPU
        print("\n🧪 Testing GPU computation...")
        try:
            x = torch.randn(1000, 1000).cuda()
            y = x @ x
            print("  ✓ GPU computation successful!")
        except Exception as e:
            print(f"  ✗ GPU computation failed: {e}")
            cuda_available = False
    else:
        print("\n⚠️  No CUDA GPU detected")
        print("  The code will run on CPU (slower but still works)")
        print("\nPossible reasons:")
        print("  1. No NVIDIA GPU in system")
        print("  2. CUDA drivers not installed")
        print("  3. PyTorch installed without CUDA support")
        print("\nTo use GPU, ensure:")
        print("  - NVIDIA GPU is present")
        print("  - CUDA toolkit installed")
        print("  - PyTorch with CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu118")
    
    # Check other dependencies
    print("\n" + "="*60)
    print("Checking Dependencies")
    print("="*60)
    
    deps = {
        'transformers': 'Hugging Face Transformers',
        'datasets': 'Hugging Face Datasets',
        'tqdm': 'Progress bars',
        'numpy': 'Numerical computing',
        'einops': 'Tensor operations'
    }
    
    all_ok = True
    for module, desc in deps.items():
        try:
            __import__(module)
            print(f"✓ {desc:30s} ({module})")
        except ImportError:
            print(f"✗ {desc:30s} ({module}) - NOT INSTALLED")
            all_ok = False
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    if cuda_available and all_ok:
        print("\n🎉 Everything is ready!")
        print("   Device: CUDA GPU (Fast training!)")
        print("\n   Expected training time: 15-30 minutes")
        print("\n   Recommended config adjustments for GPU:")
        print("   - batch_size: 32-64 (in train.py)")
        print("   - d_model: 512 (in train.py)")
        print("   - depth: 8-12 (in train.py)")
        print("\n   Start training with:")
        print("   python train.py")
    elif all_ok:
        print("\n✓ Dependencies installed")
        print("   Device: CPU (Slower but works)")
        print("\n   Expected training time: 1-2 hours")
        print("\n   Start training with:")
        print("   python train.py")
    else:
        print("\n⚠️  Missing dependencies")
        print("   Install with:")
        print("   uv pip install -r requirements.txt")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    check_cuda()
