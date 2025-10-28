#!/usr/bin/env python3
"""
Simple training progress monitor.
Shows a summary of training status from checkpoints.
"""
import os
import json
from pathlib import Path
import torch


def monitor_training():
    """Monitor training progress."""
    checkpoint_dir = Path("checkpoints")
    
    print("="*60)
    print("LibreLLM Training Monitor")
    print("="*60)
    
    if not checkpoint_dir.exists():
        print("\n❌ No checkpoints directory found.")
        print("   Training hasn't started yet.")
        print("\n   Start training with: python train.py")
        return
    
    # Check for config
    config_file = checkpoint_dir / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
        print("\n📋 Training Configuration:")
        print(f"   Model: d_model={config.get('d_model', 'N/A')}, depth={config.get('depth', 'N/A')}")
        print(f"   Batch size: {config.get('batch_size', 'N/A')}")
        print(f"   Learning rate: {config.get('learning_rate', 'N/A')}")
        print(f"   Max steps: {config.get('max_steps', 'N/A')}")
    
    # Find checkpoints
    checkpoints = list(checkpoint_dir.glob("checkpoint_*.pt"))
    best_model = checkpoint_dir / "best_model.pt"
    final_model = checkpoint_dir / "final_model.pt"
    
    print(f"\n📁 Checkpoint Files:")
    print(f"   Intermediate: {len(checkpoints)} checkpoints")
    print(f"   Best model: {'✓' if best_model.exists() else '✗'}")
    print(f"   Final model: {'✓' if final_model.exists() else '✗'}")
    
    if checkpoints:
        print(f"\n📊 Training Progress:")
        # Sort by step number
        checkpoints.sort(key=lambda x: int(x.stem.split('_')[1]))
        
        for ckpt in checkpoints:
            step = int(ckpt.stem.split('_')[1])
            size = ckpt.stat().st_size / (1024**2)  # MB
            print(f"   Step {step:5d}: {size:.1f} MB")
    
    if best_model.exists():
        print(f"\n🏆 Best Model:")
        checkpoint = torch.load(best_model, map_location='cpu')
        if 'val_loss' in checkpoint:
            print(f"   Validation loss: {checkpoint['val_loss']:.4f}")
        if 'global_step' in checkpoint:
            print(f"   Training step: {checkpoint['global_step']}")
        if 'epoch' in checkpoint:
            print(f"   Epoch: {checkpoint['epoch'] + 1}")
        
        size = best_model.stat().st_size / (1024**2)
        print(f"   File size: {size:.1f} MB")
    
    if final_model.exists():
        print(f"\n✅ Training Complete!")
        print(f"   Final model saved.")
        size = final_model.stat().st_size / (1024**2)
        print(f"   File size: {size:.1f} MB")
        print(f"\n   Generate text with:")
        print(f"   python inference.py --checkpoint checkpoints/best_model.pt")
    elif best_model.exists():
        print(f"\n⏳ Training in progress...")
        print(f"   You can test the current best model with:")
        print(f"   python inference.py --checkpoint checkpoints/best_model.pt")
    else:
        print(f"\n⏳ Training just started or no models saved yet.")
        print(f"   Check back in a few minutes!")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    monitor_training()
