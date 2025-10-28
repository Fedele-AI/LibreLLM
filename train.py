"""
Training script for Mamba-2 small language model.
Uses TinyStories dataset (small, clean, open-source).
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import GPT2TokenizerFast
from tqdm import tqdm
import json
from pathlib import Path

from mamba2_model import create_mamba2_model


class Config:
    """Training configuration."""
    # Model
    d_model = 256
    depth = 6
    max_seq_len = 256
    
    # Training
    batch_size = 16
    learning_rate = 3e-4
    num_epochs = 3
    warmup_steps = 100
    max_steps = 5000  # Limit training steps
    grad_clip = 1.0
    
    # Data
    dataset_name = "roneneldan/TinyStories"
    dataset_subset = "train[:10%]"  # Use only 10% to stay under 10GB
    max_samples = 50000  # Further limit
    
    # System
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = "checkpoints"
    log_interval = 50
    eval_interval = 500
    save_interval = 1000
    
    # Seed
    seed = 42


def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)


class TextDataset:
    """Dataset for language modeling."""
    
    def __init__(self, texts, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        print("Tokenizing dataset...")
        for text in tqdm(texts):
            if len(text.strip()) > 0:
                tokens = tokenizer.encode(text, max_length=max_length, truncation=True)
                if len(tokens) > 10:  # Skip very short sequences
                    self.examples.append(tokens)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        tokens = self.examples[idx]
        # Pad to max_length
        if len(tokens) < self.max_length:
            tokens = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        return torch.tensor(tokens[:self.max_length], dtype=torch.long)


def collate_fn(batch):
    """Collate function for DataLoader."""
    input_ids = torch.stack(batch)
    return input_ids


def get_lr(step, warmup_steps, max_steps, learning_rate):
    """Learning rate schedule with warmup and cosine decay."""
    if step < warmup_steps:
        return learning_rate * step / warmup_steps
    if step > max_steps:
        return 0.0
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + torch.cos(torch.tensor(decay_ratio * 3.14159)))
    return learning_rate * coeff


def train():
    """Main training function."""
    config = Config()
    set_seed(config.seed)
    
    # Create save directory
    Path(config.save_dir).mkdir(exist_ok=True)
    
    # Save config
    config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith('_')}
    with open(f"{config.save_dir}/config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"Training on device: {config.device}")
    print(f"Using dataset: {config.dataset_name}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(config.dataset_name, split=config.dataset_subset)
    
    # Limit samples
    if len(dataset) > config.max_samples:
        dataset = dataset.select(range(config.max_samples))
    
    print(f"Dataset size: {len(dataset)} samples")
    
    # Create datasets
    texts = [item["text"] for item in dataset]
    train_size = int(0.95 * len(texts))
    train_texts = texts[:train_size]
    val_texts = texts[train_size:]
    
    train_dataset = TextDataset(train_texts, tokenizer, max_length=config.max_seq_len)
    val_dataset = TextDataset(val_texts, tokenizer, max_length=config.max_seq_len)
    
    print(f"Train dataset: {len(train_dataset)} examples")
    print(f"Val dataset: {len(val_dataset)} examples")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Use 0 for macOS compatibility
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Create model
    print("\nCreating Mamba-2 model...")
    model = create_mamba2_model(
        d_model=config.d_model,
        depth=config.depth,
        vocab_size=len(tokenizer),
        device=config.device
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )
    
    # Training loop
    print("\nStarting training...")
    model.train()
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(config.num_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{config.num_epochs}")
        print(f"{'='*50}")
        
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        
        for batch_idx, input_ids in enumerate(progress_bar):
            if global_step >= config.max_steps:
                print(f"\nReached max steps ({config.max_steps}), stopping training.")
                break
            
            input_ids = input_ids.to(config.device)
            
            # Forward pass
            loss, _ = model(input_ids, labels=input_ids)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            
            # Update learning rate
            lr = get_lr(global_step, config.warmup_steps, config.max_steps, config.learning_rate)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            # Optimizer step
            optimizer.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{lr:.6f}'
            })
            
            # Logging
            if global_step % config.log_interval == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                print(f"\nStep {global_step}: loss={loss.item():.4f}, avg_loss={avg_loss:.4f}, lr={lr:.6f}")
            
            # Evaluation
            if global_step % config.eval_interval == 0:
                print("\nEvaluating...")
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for val_input_ids in val_loader:
                        val_input_ids = val_input_ids.to(config.device)
                        loss, _ = model(val_input_ids, labels=val_input_ids)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                print(f"Validation loss: {val_loss:.4f}")
                
                # Generate sample
                print("\nGenerating sample text...")
                prompt = "Once upon a time"
                input_ids = tokenizer.encode(prompt, return_tensors="pt").to(config.device)
                generated = model.generate(input_ids, max_new_tokens=50, temperature=0.8, top_k=40)
                generated_text = tokenizer.decode(generated[0])
                print(f"Generated: {generated_text}\n")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({
                        'epoch': epoch,
                        'global_step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss,
                    }, f"{config.save_dir}/best_model.pt")
                    print(f"Saved best model (val_loss: {val_loss:.4f})")
                
                model.train()
            
            # Save checkpoint
            if global_step % config.save_interval == 0:
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, f"{config.save_dir}/checkpoint_{global_step}.pt")
                print(f"\nSaved checkpoint at step {global_step}")
        
        if global_step >= config.max_steps:
            break
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config_dict,
    }, f"{config.save_dir}/final_model.pt")
    
    print("\n" + "="*50)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Models saved in: {config.save_dir}/")
    print("="*50)


if __name__ == "__main__":
    train()
