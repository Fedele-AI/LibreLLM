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
        "split": "train[:8%]",
        "description": "Children's stories for narrative structure",
        "est_samples": "~20,000",
        "size": "~2 GB"
    },
    {
        "name": "OpenWebText",
        "hf_name": "Skylion007/openwebtext",
        "split": "train[:2%]",
        "description": "High-quality web text from Reddit",
        "est_samples": "~15,000",
        "size": "~3 GB"
    },
    {
        "name": "Wikipedia (Filtered)",
        "hf_name": "wikipedia/20220301.en",
        "split": "train[:1%]",
        "description": "Wikipedia filtered for historical topics",
        "est_samples": "~5,000",
        "size": "~1 GB"
    },
    {
        "name": "Project Gutenberg (Filtered)",
        "hf_name": "sedthh/gutenberg_english",
        "split": "train[:5%]",
        "description": "Classic books filtered for philosophy/history",
        "est_samples": "~2,500",
        "size": "~500 MB"
    },
    {
        "name": "American Stories",
        "hf_name": "dell-research-harvard/AmericanStories",
        "split": "train[:0.5%]",
        "description": "Historical US newspapers (1700s-1900s)",
        "est_samples": "~100,000",
        "size": "~5 GB"
    },
    {
        "name": "Pile of Law - Founding Docs",
        "hf_name": "pile-of-law/pile-of-law:founding_docs",
        "split": "train (full)",
        "description": "Letters from US founding fathers",
        "est_samples": "~1,000",
        "size": "~50 MB"
    },
    {
        "name": "Pile of Law - Constitutions",
        "hf_name": "pile-of-law/pile-of-law:constitutions",
        "split": "train[:50%]",
        "description": "World constitutions inc. US & state constitutions",
        "est_samples": "~500",
        "size": "~25 MB"
    },
    {
        "name": "Custom Historical Texts",
        "hf_name": "embedded in train.py",
        "split": "all",
        "description": "Curated essays on GA history, 2nd Amendment, Locke",
        "est_samples": "6 essays",
        "size": "~50 KB"
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
print("TOTAL ESTIMATED SIZE: ~12-15 GB (well under 50 GB limit)")
print("TOTAL SAMPLES: ~145,000")
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
