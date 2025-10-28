"""
Inference script for trained Mamba-2 model.
Generate text using the trained model.
"""
import torch
from transformers import GPT2TokenizerFast
from mamba2_model import create_mamba2_model
import json


def load_model(checkpoint_path, device="cpu"):
    """Load a trained Mamba-2 model."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load config
    if 'config' in checkpoint:
        config = checkpoint['config']
        d_model = config.get('d_model', 256)
        depth = config.get('depth', 6)
    else:
        # Default values
        d_model = 256
        depth = 6
    
    # Create model
    model = create_mamba2_model(
        d_model=d_model,
        depth=depth,
        vocab_size=50257,
        device=device
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def generate_text(model, tokenizer, prompt, max_new_tokens=100, temperature=0.8, top_k=40, device="cpu"):
    """Generate text from a prompt."""
    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k
        )
    
    # Decode
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text


def interactive_mode(model, tokenizer, device="cpu"):
    """Interactive text generation."""
    print("\n" + "="*50)
    print("Interactive Mode - Mamba-2 Text Generation")
    print("="*50)
    print("Type your prompt and press Enter to generate text.")
    print("Type 'quit' or 'exit' to stop.\n")
    
    while True:
        prompt = input("Prompt: ").strip()
        
        if prompt.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not prompt:
            continue
        
        print("\nGenerating...\n")
        generated = generate_text(
            model, tokenizer, prompt,
            max_new_tokens=100,
            temperature=0.8,
            top_k=40,
            device=device
        )
        
        print(f"Generated text:\n{generated}\n")
        print("-" * 50 + "\n")


def main():
    """Main inference function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate text with Mamba-2")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Text prompt for generation")
    parser.add_argument("--max_tokens", type=int, default=100,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=40,
                        help="Top-k sampling parameter")
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive mode")
    
    args = parser.parse_args()
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, device=device)
    print("Model loaded successfully!")
    
    if args.interactive:
        interactive_mode(model, tokenizer, device)
    else:
        # Single generation
        prompt = args.prompt or "Once upon a time"
        print(f"\nPrompt: {prompt}\n")
        
        generated = generate_text(
            model, tokenizer, prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device
        )
        
        print(f"Generated text:\n{generated}\n")


if __name__ == "__main__":
    main()
