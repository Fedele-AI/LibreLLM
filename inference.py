"""
Inference script for trained Mamba-2 model.
Generate text using the trained model with Jinja2 template support.
"""
import torch
from transformers import AutoTokenizer
from mamba2_model import create_mamba2_model
import json
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, Template


class TemplateManager:
    """Manage Jinja2 templates for different prompt formats."""
    
    def __init__(self, template_dir="templates"):
        """Initialize template manager.
        
        Args:
            template_dir: Directory containing Jinja2 templates
        """
        self.template_dir = Path(template_dir)
        if self.template_dir.exists():
            self.env = Environment(loader=FileSystemLoader(str(self.template_dir)))
        else:
            print(f"Warning: Template directory '{template_dir}' not found. Using simple templates.")
            self.env = None
        
    def render_chat(self, messages, add_generation_prompt=True):
        """Render chat-style conversation.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            add_generation_prompt: Whether to add assistant prompt at the end
            
        Returns:
            Formatted prompt string
        """
        if self.env:
            try:
                template = self.env.get_template("chat_template.jinja2")
                return template.render(messages=messages, add_generation_prompt=add_generation_prompt)
            except Exception as e:
                print(f"Warning: Could not load chat template: {e}")
        
        # Fallback to simple format
        result = ""
        for msg in messages:
            role = msg['role']
            content = msg['content']
            result += f"<|{role}|>\n{content}\n<|end|>\n"
        if add_generation_prompt:
            result += "<|assistant|>\n"
        return result
    
    def render_instruct(self, instruction, input_text="", response=""):
        """Render instruction-style prompt.
        
        Args:
            instruction: The instruction/task
            input_text: Optional input context
            response: Optional partial response to continue
            
        Returns:
            Formatted prompt string
        """
        if self.env:
            try:
                template = self.env.get_template("instruct_template.jinja2")
                return template.render(instruction=instruction, input=input_text, response=response)
            except Exception as e:
                print(f"Warning: Could not load instruct template: {e}")
        
        # Fallback to simple format
        result = f"### Instruction:\n{instruction}\n\n"
        if input_text:
            result += f"### Input:\n{input_text}\n\n"
        result += f"### Response:\n{response}"
        return result
    
    def render_simple(self, prompt):
        """Render simple prompt.
        
        Args:
            prompt: The prompt text
            
        Returns:
            Formatted prompt string
        """
        if self.env:
            try:
                template = self.env.get_template("simple_template.jinja2")
                return template.render(prompt=prompt)
            except Exception as e:
                print(f"Warning: Could not load simple template: {e}")
        
        return prompt


def load_model(checkpoint_path, device="cpu"):
    """Load a trained Mamba-2 model."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load config
    if 'config' in checkpoint:
        config = checkpoint['config']
        d_model = config.get('d_model', 256)
        depth = config.get('depth', 6)
        vocab_size = config.get('vocab_size', 32000)  # LLaMA vocab size
    else:
        # Default values
        d_model = 256
        depth = 6
        vocab_size = 32000  # LLaMA vocab size
    
    # Create model
    model = create_mamba2_model(
        d_model=d_model,
        depth=depth,
        vocab_size=vocab_size,
        device=device
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def generate_text(model, tokenizer, prompt, max_new_tokens=100, temperature=0.8, top_k=40, device="cpu", max_context=8192):
    """Generate text from a prompt.
    
    Args:
        model: The language model
        tokenizer: Tokenizer for encoding/decoding
        prompt: Text prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        device: Device to run on
        max_context: Maximum context length (8k tokens)
    
    Returns:
        Generated text
    """
    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Truncate if prompt exceeds max context
    if input_ids.size(1) > max_context:
        print(f"Warning: Prompt length ({input_ids.size(1)}) exceeds max context ({max_context}). Truncating...")
        input_ids = input_ids[:, -max_context:]
    
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


def interactive_mode(model, tokenizer, device="cpu", template_manager=None, template_mode="simple"):
    """Interactive text generation with template support.
    
    Args:
        model: The language model
        tokenizer: Tokenizer
        device: Device to run on
        template_manager: Template manager for formatting prompts
        template_mode: Template mode ('simple', 'chat', 'instruct')
    """
    print("\n" + "="*50)
    print("Interactive Mode - Mamba-2 Text Generation")
    print(f"Context Length: 8192 tokens")
    print(f"Template Mode: {template_mode}")
    print("="*50)
    print("Commands:")
    print("  'quit' or 'exit' - Exit the program")
    print("  'mode:simple' - Switch to simple template")
    print("  'mode:chat' - Switch to chat template")
    print("  'mode:instruct' - Switch to instruct template")
    print("  'clear' - Clear chat history (chat mode only)")
    print("\n")
    
    chat_history = []
    current_mode = template_mode
    
    while True:
        if current_mode == "chat":
            prompt = input("User: ").strip()
        elif current_mode == "instruct":
            prompt = input("Instruction: ").strip()
        else:
            prompt = input("Prompt: ").strip()
        
        if prompt.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if prompt.lower() == 'clear':
            chat_history = []
            print("Chat history cleared.\n")
            continue
        
        if prompt.lower().startswith('mode:'):
            new_mode = prompt.lower().split(':')[1]
            if new_mode in ['simple', 'chat', 'instruct']:
                current_mode = new_mode
                print(f"Switched to {current_mode} mode.\n")
                if new_mode != 'chat':
                    chat_history = []
            else:
                print(f"Unknown mode: {new_mode}\n")
            continue
        
        if not prompt:
            continue
        
        # Format prompt based on template mode
        if current_mode == "chat" and template_manager:
            chat_history.append({"role": "user", "content": prompt})
            formatted_prompt = template_manager.render_chat(chat_history, add_generation_prompt=True)
        elif current_mode == "instruct" and template_manager:
            formatted_prompt = template_manager.render_instruct(instruction=prompt)
        elif template_manager:
            formatted_prompt = template_manager.render_simple(prompt)
        else:
            formatted_prompt = prompt
        
        print("\nGenerating...\n")
        generated = generate_text(
            model, tokenizer, formatted_prompt,
            max_new_tokens=200,  # Increased for better responses
            temperature=0.8,
            top_k=40,
            device=device,
            max_context=8192
        )
        
        # Extract assistant response if in chat mode
        if current_mode == "chat":
            # Try to extract just the new response
            if "<|assistant|>" in generated:
                response = generated.split("<|assistant|>")[-1].split("<|end|>")[0].strip()
            else:
                response = generated[len(formatted_prompt):].strip()
            
            chat_history.append({"role": "assistant", "content": response})
            print(f"Assistant: {response}\n")
        else:
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
    parser.add_argument("--max_tokens", type=int, default=200,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=40,
                        help="Top-k sampling parameter")
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive mode")
    parser.add_argument("--template", type=str, default="simple",
                        choices=["simple", "chat", "instruct"],
                        help="Template mode: simple, chat, or instruct")
    parser.add_argument("--template_dir", type=str, default="templates",
                        help="Directory containing Jinja2 templates")
    
    args = parser.parse_args()
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Context length: 8192 tokens")
    
    # Load tokenizer
    print("Loading tokenizer (Mistral-based)...")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, device=device)
    print("Model loaded successfully!")
    
    # Initialize template manager
    print(f"Initializing template manager (mode: {args.template})...")
    template_manager = TemplateManager(template_dir=args.template_dir)
    
    if args.interactive:
        interactive_mode(model, tokenizer, device, template_manager, args.template)
    else:
        # Single generation
        prompt = args.prompt or "Once upon a time"
        
        # Format prompt based on template
        if args.template == "chat":
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = template_manager.render_chat(messages)
        elif args.template == "instruct":
            formatted_prompt = template_manager.render_instruct(instruction=prompt)
        else:
            formatted_prompt = template_manager.render_simple(prompt)
        
        print(f"\nPrompt: {prompt}\n")
        
        generated = generate_text(
            model, tokenizer, formatted_prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device,
            max_context=8192
        )
        
        print(f"Generated text:\n{generated}\n")


if __name__ == "__main__":
    main()
