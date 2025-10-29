#!/usr/bin/env python3
"""
Example usage of the upgraded LibreLLM with Jinja2 templates and 8K context.
"""

from inference import TemplateManager

# Initialize template manager
tm = TemplateManager()

print("=" * 70)
print("LibreLLM Template Examples")
print("=" * 70)

# Example 1: Simple Template
print("\n1. SIMPLE TEMPLATE")
print("-" * 70)
simple_prompt = "Once upon a time in a distant galaxy"
formatted = tm.render_simple(simple_prompt)
print(f"Input: {simple_prompt}")
print(f"Formatted:\n{formatted}")

# Example 2: Chat Template
print("\n\n2. CHAT TEMPLATE")
print("-" * 70)
chat_messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."},
    {"role": "user", "content": "What is it known for?"}
]
formatted = tm.render_chat(chat_messages, add_generation_prompt=True)
print("Messages:")
for msg in chat_messages:
    print(f"  [{msg['role']}]: {msg['content']}")
print(f"\nFormatted:\n{formatted}")

# Example 3: Instruction Template
print("\n\n3. INSTRUCTION TEMPLATE")
print("-" * 70)
instruction = "Explain photosynthesis in simple terms"
input_text = "Explain it to a 10-year-old"
formatted = tm.render_instruct(instruction, input_text)
print(f"Instruction: {instruction}")
print(f"Input: {input_text}")
print(f"\nFormatted:\n{formatted}")

# Example 4: Long Context Demonstration
print("\n\n4. 8K CONTEXT DEMONSTRATION")
print("-" * 70)
print("The model now supports 8192 tokens (approximately 6000-7000 words)")
print("This is ~16x the previous context length of 256 tokens!")
print("\nYou can now:")
print("  - Process entire book chapters")
print("  - Analyze long documents")
print("  - Maintain context across long conversations")
print("  - Generate coherent long-form content")

# Example 5: Dataset Information
print("\n\n5. TRAINING DATASETS")
print("-" * 70)
datasets = [
    ("TinyStories", "15%", "Simple, coherent children's stories"),
    ("OpenWebText", "5%", "High-quality Reddit discussions"),
    ("BookCorpus", "10%", "Narrative structure from books"),
    ("Wikipedia", "5%", "Factual encyclopedia content")
]

print("Training on 4 diverse datasets for better English:")
for name, split, desc in datasets:
    print(f"  • {name:15} ({split:4}): {desc}")

print("\n" + "=" * 70)
print("Ready to generate coherent English text!")
print("=" * 70)
print("\nUsage Examples:")
print("  # Interactive with chat template:")
print("  python inference.py --checkpoint checkpoints/final_model.pt --interactive --template chat")
print("\n  # Single generation:")
print("  python inference.py --checkpoint checkpoints/final_model.pt --prompt 'Your prompt' --max_tokens 500")
