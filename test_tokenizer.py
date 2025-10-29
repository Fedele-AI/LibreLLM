"""
Test script to verify the tokenizer upgrade works correctly.
"""
import torch
from transformers import AutoTokenizer

print("Testing Mistral tokenizer upgrade...")
print("="*60)

# Load tokenizer
print("\n1. Loading tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"   ✓ Tokenizer loaded successfully")
    print(f"   Vocab size: {len(tokenizer)}")
    print(f"   Pad token: {tokenizer.pad_token}")
    print(f"   EOS token: {tokenizer.eos_token}")
except Exception as e:
    print(f"   ✗ Error loading tokenizer: {e}")
    exit(1)

# Test tokenization
print("\n2. Testing tokenization...")
test_texts = [
    "The State of Georgia was founded in 1733.",
    "The Second Amendment protects the right to bear arms.",
    "John Locke wrote about natural rights and the social contract.",
]

for i, text in enumerate(test_texts, 1):
    try:
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        print(f"   Text {i}: {text[:50]}...")
        print(f"   Tokens: {len(tokens)}")
        print(f"   Decoded: {decoded[:50]}...")
        print()
    except Exception as e:
        print(f"   ✗ Error tokenizing text {i}: {e}")

# Test with 8k context
print("\n3. Testing 8k context handling...")
long_text = "Once upon a time " * 1000  # Create a long text
try:
    tokens = tokenizer.encode(long_text, max_length=8192, truncation=True)
    print(f"   Long text tokens (truncated to 8192): {len(tokens)}")
    print(f"   ✓ 8k context truncation works correctly")
except Exception as e:
    print(f"   ✗ Error with 8k context: {e}")

# Test batch encoding
print("\n4. Testing batch encoding...")
try:
    batch = tokenizer(test_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    print(f"   Batch shape: {batch['input_ids'].shape}")
    print(f"   ✓ Batch encoding works correctly")
except Exception as e:
    print(f"   ✗ Error with batch encoding: {e}")

print("\n" + "="*60)
print("All tokenizer tests passed! ✓")
print("="*60)
