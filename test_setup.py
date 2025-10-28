#!/usr/bin/env python3
"""
Quick test script to verify the setup and run a minimal training example.
This is useful for testing before running the full training.
"""
import torch
from transformers import GPT2TokenizerFast
from mamba2_model import create_mamba2_model
from torch.utils.data import DataLoader
import numpy as np


def test_model():
    """Test model creation and forward pass."""
    print("="*50)
    print("Testing Mamba-2 Model Setup")
    print("="*50)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n1. Device: {device}")
    
    # Create small model
    print("\n2. Creating model...")
    model = create_mamba2_model(
        d_model=128,
        depth=4,
        vocab_size=50257,
        device=device
    )
    
    # Test forward pass
    print("\n3. Testing forward pass...")
    batch_size = 2
    seq_len = 64
    input_ids = torch.randint(0, 50257, (batch_size, seq_len)).to(device)
    
    with torch.no_grad():
        logits = model(input_ids)
    
    print(f"   Input shape: {input_ids.shape}")
    print(f"   Output shape: {logits.shape}")
    print("   ✓ Forward pass successful!")
    
    # Test training step
    print("\n4. Testing training step...")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    loss, _ = model(input_ids, labels=input_ids)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"   Loss: {loss.item():.4f}")
    print("   ✓ Training step successful!")
    
    # Test generation
    print("\n5. Testing text generation...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model.eval()
    
    prompt = "Hello"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        generated = model.generate(input_ids, max_new_tokens=20, temperature=1.0, top_k=50)
    
    generated_text = tokenizer.decode(generated[0])
    print(f"   Prompt: '{prompt}'")
    print(f"   Generated: '{generated_text}'")
    print("   ✓ Generation successful!")
    
    print("\n" + "="*50)
    print("All tests passed! ✓")
    print("="*50)
    print("\nYou can now run the full training with:")
    print("  python train.py")
    print("\nEstimated dataset download: ~2GB")
    print("Estimated training time on CPU: 1-2 hours")
    print("="*50)


if __name__ == "__main__":
    test_model()
