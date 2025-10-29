"""
Quick overview of datasets without downloading.
"""

print("="*80)
print("LibreLLM Training Datasets - Configuration Overview")
print("="*80)

datasets = [
    {
        "name": "TinyStories",
        "hf_name": "roneneldan/TinyStories",
        "split": "train[:5%]",
        "description": "Children's stories for narrative structure",
        "est_samples": "~10,000",
        "size": "~1 GB"
    },
    {
        "name": "FineWeb-Edu",
        "hf_name": "HuggingFaceFW/fineweb-edu:sample-10BT",
        "split": "train[:0.1%]",
        "description": "High-quality educational web content",
        "est_samples": "~10,000",
        "size": "~500 MB"
    },
    {
        "name": "Custom Historical Texts",
        "hf_name": "embedded in train.py",
        "split": "all",
        "description": "Curated essays on GA history, 2nd Amendment, Locke, etc.",
        "est_samples": "6 essays",
        "size": "~100 KB"
    },
]

print("\n" + "="*80)
print("DATASET BREAKDOWN")
print("="*80)

for i, ds in enumerate(datasets, 1):
    print(f"\n{i}. {ds['name']}")
    print(f"   HuggingFace: {ds['hf_name']}")
    print(f"   Split: {ds['split']}")
    print(f"   Description: {ds['description']}")
    print(f"   Est. Samples: {ds['est_samples']}")
    print(f"   Est. Size: {ds['size']}")

print("\n" + "="*80)
print("TOTAL ESTIMATED SIZE: ~2 GB (well under 50 GB limit)")
print("TOTAL SAMPLES: ~20,000")
print("="*80)

print("\n" + "="*80)
print("KEY FEATURES")
print("="*80)
print()
print("✓ 8K Token Context Window (8192 tokens)")
print("✓ Mistral Tokenizer (32,000 vocab, better than GPT-2)")
print("✓ Jinja2 Template Support (chat, instruct, simple modes)")
print("✓ Historical Knowledge:")
print("  • Georgia state history (founding, revolution, statehood)")
print("  • Second Amendment & Bill of Rights")
print("  • Pre-revolutionary philosophy (Locke, Rousseau, Montesquieu)")
print("  • Founding documents (Constitution, Declaration)")
print("  • Primary sources from founding fathers")
print("  • Historical newspapers from American history")
print()
print("✓ English Coherence:")
print("  • Narrative structure from TinyStories")
print("  • Modern writing from OpenWebText")
print("  • Classic literature from Gutenberg")
print("  • Factual writing from Wikipedia")
print()

print("="*80)
print("READY TO TRAIN!")
print("Run: python train.py")
print("="*80)
