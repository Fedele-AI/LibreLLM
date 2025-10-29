"""
Training script for Mamba-2 small language model.
Uses multiple datasets for coherent English and specific knowledge domains.
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import json
from pathlib import Path

from mamba2_model import create_mamba2_model


class Config:
    """Training configuration."""
    # Model
    d_model = 256
    depth = 6
    max_seq_len = 8192  # 8k context length
    
    # Training
    batch_size = 4  # Reduced batch size for 8k context
    learning_rate = 3e-4
    num_epochs = 3
    warmup_steps = 100
    max_steps = 10000  # Increased for more training
    grad_clip = 1.0
    
    # Data - Multiple datasets for better English coherence and specific knowledge
    datasets_config = [
        {
            "name": "roneneldan/TinyStories",
            "split": "train[:8%]",
            "text_column": "text",
            "description": "Children's stories with simple, coherent English"
        },
        {
            "name": "Skylion007/openwebtext",
            "split": "train[:2%]",
            "text_column": "text",
            "description": "High-quality web text from Reddit"
        },
        {
            "name": "wikipedia",
            "config": "20220301.en",
            "split": "train[:1%]",
            "text_column": "text",
            "description": "Wikipedia articles for factual English",
            "filter_keywords": ["Georgia", "American Revolution", "colonial", "founding fathers", "Constitution", "Bill of Rights"]
        },
        {
            "name": "sedthh/gutenberg_english",
            "split": "train[:5%]",
            "text_column": "TEXT",
            "description": "Project Gutenberg books - classic literature and historical texts",
            "filter_keywords": ["history", "philosophy", "government", "liberty", "revolution", "politics"]
        },
        {
            "name": "dell-research-harvard/AmericanStories",
            "config": "all_years",
            "split": "train[:0.5%]",
            "text_column": "article",
            "description": "Historical US newspapers - American history, politics, and culture"
        },
        {
            "name": "pile-of-law/pile-of-law",
            "subset": "founding_docs",
            "split": "train",
            "text_column": "text",
            "description": "Letters from U.S. founding fathers and constitutional documents"
        },
        {
            "name": "pile-of-law/pile-of-law",
            "subset": "constitutions",
            "split": "train[:50%]",
            "text_column": "text",
            "description": "World constitutions including US Constitution and state constitutions"
        },
        {
            "name": "custom_historical",
            "split": "all",
            "text_column": "text",
            "description": "Historical texts: Georgia history, Second Amendment, Pre-revolutionary works",
            "custom_texts": [
                # Georgia State History - Expanded
                """The State of Georgia, one of the original Thirteen Colonies, was founded in 1733 by James Oglethorpe. 
                It was named after King George II of Great Britain. Georgia was the last of the British colonies to be established in North America.
                The colony was conceived as a place where debtors and the unemployed could start anew. Oglethorpe and the trustees prohibited slavery 
                and the importation of alcohol. However, these restrictions were lifted in the 1750s. Georgia played a crucial role in the American 
                Revolution. The Province of Georgia was the southernmost of the Thirteen Colonies. Major battles included the Siege of Savannah in 1779. 
                Georgia ratified the United States Constitution on January 2, 1788, becoming the fourth state to join the Union. Atlanta became the 
                state capital in 1868. The state has a rich history including Creek and Cherokee Native American heritage, colonial settlement, 
                the plantation economy, the Civil War, Reconstruction, the Civil Rights Movement, and modern economic development.
                
                The founding of Savannah in 1733 was carefully planned by James Oglethorpe. He designed the city with a unique grid pattern 
                featuring public squares, which still exists today. The settlement was intended to serve as a buffer between South Carolina 
                and Spanish Florida. Early settlers included English debtors, persecuted Protestants from Europe, and Highland Scots. The 
                Salzburgers, German Protestants fleeing religious persecution, established the town of Ebenezer in 1734. Scottish Highlanders 
                founded Darien to defend the southern frontier. These diverse groups contributed significantly to Georgia's cultural heritage.
                
                Relations with Native American tribes were complex. The Creek and Cherokee nations inhabited Georgia before European settlement. 
                Oglethorpe initially maintained peaceful relations, signing treaties with Creek leaders. However, as the colony grew, conflicts 
                increased over land. After American independence, pressure mounted for Native American removal. The Indian Removal Act of 1830 
                led to the forced relocation of Cherokee and other tribes along the Trail of Tears in the 1830s, a tragic chapter in American history.""",
                
                # Second Amendment - Expanded
                """The Second Amendment to the United States Constitution reads: 'A well regulated Militia, being necessary to the security of a free State, 
                the right of the people to keep and bear Arms, shall not be infringed.' This amendment was adopted on December 15, 1791, as part of the 
                Bill of Rights. The amendment protects an individual right to possess firearms. The framers included this amendment to ensure that citizens 
                could defend themselves, their communities, and their nation. Historical context shows that the Founding Fathers believed an armed citizenry 
                was essential to prevent tyranny. The militia concept was central to early American defense strategy. The right to bear arms has been 
                interpreted by the Supreme Court in landmark cases such as District of Columbia v. Heller (2008) and McDonald v. Chicago (2010), 
                which affirmed the individual right to keep and bear arms for lawful purposes such as self-defense.
                
                The origins of the Second Amendment lie in English common law and colonial experience. The English Bill of Rights of 1689 recognized 
                the right of Protestant subjects to have arms for their defense. American colonists relied on militias composed of armed citizens for 
                defense against Native American raids and foreign threats. During the Revolutionary War, citizen militias played a crucial role alongside 
                the Continental Army. The Founders viewed a standing army with suspicion, seeing it as a potential instrument of tyranny. They believed 
                an armed populace could resist governmental oppression and foreign invasion.
                
                James Madison, the primary author of the Bill of Rights, drew inspiration from state constitutions and Anti-Federalist concerns. 
                Virginia's Declaration of Rights proclaimed that 'a well regulated militia, composed of the body of the people, trained to arms, 
                is the proper, natural, and safe defense of a free state.' Pennsylvania's constitution affirmed the people's right to bear arms 
                for defense of themselves and the state. These state provisions influenced the language of the Second Amendment.""",
                
                # John Locke - Expanded
                """John Locke's Two Treatises of Government, published in 1689, is a foundational text of classical liberalism and influenced the 
                American Revolution. Locke argued that government derives its authority from the consent of the governed. He proposed the concept of 
                natural rights: life, liberty, and property. According to Locke, individuals in a state of nature possess these rights, and they create 
                governments to protect them. When a government fails to protect these rights, the people have the right to alter or abolish it. 
                Locke's social contract theory emphasized that political power should not be arbitrary. He rejected the divine right of kings and 
                absolute monarchy. His ideas about limited government, separation of powers, and the right to revolution deeply influenced the 
                American Founding Fathers, particularly Thomas Jefferson in drafting the Declaration of Independence.
                
                In his Second Treatise, Locke described the state of nature as a state of perfect freedom and equality. Unlike Thomas Hobbes, 
                who viewed the state of nature as brutish and violent, Locke saw it as governed by natural law and reason. He argued that 
                natural law teaches that no one ought to harm another in his life, health, liberty, or possessions. However, the state of 
                nature lacked a common judge to resolve disputes, leading individuals to form a social contract and establish civil government.
                
                Locke's labor theory of property held that individuals acquire property rights by mixing their labor with natural resources. 
                When a person works the land or gathers fruit, they make it their own. This theory justified private property and limited 
                government interference in economic affairs. Locke also emphasized that government power must be limited and divided. He 
                distinguished between legislative, executive, and federative powers. The legislature makes laws, the executive enforces them, 
                and the federative power handles foreign affairs. This separation prevents tyranny by dividing authority among different branches.""",
                
                # More Enlightenment Thought
                """The Enlightenment period produced numerous works that shaped revolutionary thought in America. Philosophers emphasized reason, 
                individual liberty, and skepticism of absolute authority. Montesquieu's 'The Spirit of the Laws' (1748) introduced the concept of 
                separation of powers into executive, legislative, and judicial branches. He argued that liberty is best preserved when governmental 
                powers are divided and balanced against each other. This idea profoundly influenced the structure of the United States Constitution.
                
                Jean-Jacques Rousseau's 'The Social Contract' (1762) argued that legitimate political authority derives from a social contract 
                agreed upon by all citizens for their mutual preservation. Rousseau famously proclaimed that 'Man is born free, and everywhere 
                he is in chains.' He distinguished between the general will, which represents the common good, and the will of all, which is 
                merely the sum of private interests. Government should express the general will of the people.
                
                These Enlightenment ideas, combined with English common law traditions and the rights of Englishmen, formed the intellectual 
                foundation for the American Revolution. Colonial thinkers like Samuel Adams, Thomas Paine, and Patrick Henry drew upon these 
                principles to justify independence from British rule. Thomas Paine's 'Common Sense' (1776) applied Enlightenment reasoning to 
                advocate for American independence, arguing that hereditary monarchy violated natural rights and common sense.""",
                
                # Declaration of Independence
                """The Declaration of Independence, adopted on July 4, 1776, is one of the most important documents in American history. 
                Drafted primarily by Thomas Jefferson, it proclaimed the thirteen American colonies' separation from Great Britain. The 
                Declaration articulated fundamental principles of democratic government: that all men are created equal, that they are 
                endowed by their Creator with certain unalienable rights including life, liberty, and the pursuit of happiness, and that 
                governments derive their just powers from the consent of the governed.
                
                The Declaration drew heavily from Enlightenment philosophy, particularly John Locke's ideas. Jefferson transformed Locke's 
                'life, liberty, and property' into 'life, liberty, and the pursuit of happiness.' The document lists grievances against 
                King George III to justify independence. These grievances included imposing taxes without consent, depriving colonists of 
                trial by jury, and maintaining standing armies without colonial approval. The Declaration asserted the right of revolution: 
                when a government becomes destructive of the people's rights, it is their right and duty to alter or abolish it.""",
                
                # Constitutional Convention
                """The Constitutional Convention met in Philadelphia from May to September 1787 to address weaknesses in the Articles of 
                Confederation. Delegates including George Washington, James Madison, Benjamin Franklin, and Alexander Hamilton debated the 
                structure of the new federal government. The Convention produced the United States Constitution, establishing a federal system 
                with three branches of government: legislative, executive, and judicial.
                
                The Great Compromise resolved disputes between large and small states by creating a bicameral legislature. The House of 
                Representatives would have proportional representation based on population, while the Senate would have equal representation 
                with two senators per state. The Three-Fifths Compromise addressed how enslaved persons would be counted for representation 
                and taxation purposes. The Constitution also included checks and balances to prevent any branch from becoming too powerful.
                
                Ratification of the Constitution required approval by nine states. Federalists, including Alexander Hamilton, James Madison, 
                and John Jay, wrote The Federalist Papers to argue for ratification. Anti-Federalists opposed the Constitution, fearing it 
                created too strong a central government and lacked a bill of rights. The promise to add a bill of rights helped secure 
                ratification. James Madison drafted the amendments that became the Bill of Rights, ratified in 1791.""",
            ]
        }
    ]
    max_samples = 200000  # Increased for better coverage
    
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
    
    print(f"Training on device: {config.device}")
    print(f"Loading multiple datasets for better English coherence...")
    
    # Load tokenizer - Using Mistral tokenizer for better performance
    print("Loading tokenizer (Mistral-based for better English)...")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    
    # Save config with vocab_size
    config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith('_')}
    config_dict['vocab_size'] = len(tokenizer)  # Save vocab size for model loading
    with open(f"{config.save_dir}/config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
    
    # Load and combine multiple datasets
    print("\nLoading datasets...")
    all_texts = []
    
    for dataset_config in config.datasets_config:
        try:
            # Handle custom texts
            if dataset_config['name'] == 'custom_historical':
                print(f"\nLoading {dataset_config['name']}...")
                print(f"  Description: {dataset_config['description']}")
                custom_texts = dataset_config.get('custom_texts', [])
                print(f"  Loaded {len(custom_texts)} custom historical texts")
                all_texts.extend(custom_texts)
                continue
            
            print(f"\nLoading {dataset_config['name']}...")
            print(f"  Description: {dataset_config['description']}")
            
            # Handle pile-of-law with subsets
            if 'subset' in dataset_config:
                dataset = load_dataset(
                    dataset_config['name'],
                    dataset_config['subset'],
                    split=dataset_config['split'],
                    trust_remote_code=True
                )
            # Handle different dataset configurations
            elif 'config' in dataset_config:
                dataset = load_dataset(
                    dataset_config['name'],
                    dataset_config['config'],
                    split=dataset_config['split'],
                    trust_remote_code=True
                )
            else:
                dataset = load_dataset(
                    dataset_config['name'],
                    split=dataset_config['split'],
                    trust_remote_code=True
                )
            
            # Extract texts
            text_column = dataset_config['text_column']
            if isinstance(dataset, dict):
                # If dataset is a dict, take the first split
                dataset = list(dataset.values())[0]
            
            texts = []
            for item in dataset:
                if text_column in item and item[text_column]:
                    texts.append(item[text_column])
            
            # Filter by keywords if specified
            if 'filter_keywords' in dataset_config:
                keywords = dataset_config['filter_keywords']
                filtered_texts = []
                for text in texts:
                    text_lower = str(text).lower()
                    if any(keyword.lower() in text_lower for keyword in keywords):
                        filtered_texts.append(text)
                print(f"  Filtered to {len(filtered_texts)} samples matching keywords: {keywords}")
                texts = filtered_texts
            
            print(f"  Loaded {len(texts)} samples")
            all_texts.extend(texts)
            
        except Exception as e:
            print(f"  Warning: Failed to load {dataset_config['name']}: {e}")
            print(f"  Continuing with other datasets...")
            continue
    
    print(f"\nTotal texts collected: {len(all_texts)}")
    
    # Limit total samples
    if len(all_texts) > config.max_samples:
        import random
        random.seed(config.seed)
        all_texts = random.sample(all_texts, config.max_samples)
        print(f"Sampled {config.max_samples} texts for training")
    
    # Create train/val split
    train_size = int(0.95 * len(all_texts))
    train_texts = all_texts[:train_size]
    val_texts = all_texts[train_size:]
    
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
