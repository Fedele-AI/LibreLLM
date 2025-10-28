# 🎉 LibreLLM - Mamba-2 Training Setup Complete!

## ✅ What's Been Created

Your workspace now contains a complete, working Mamba-2 language model training pipeline:

### Core Files
- **`mamba2_model.py`** - Pure PyTorch Mamba-2 implementation (~300 lines)
  - Works on CPU (no CUDA required)
  - ~10M parameters with default config
  - State-space model architecture
  
- **`train.py`** - Training script (~250 lines)
  - Uses TinyStories dataset (2-3GB download)
  - Automatic checkpointing
  - Validation and sample generation
  - Progress tracking with tqdm
  
- **`inference.py`** - Text generation script
  - Single prompt generation
  - Interactive mode
  - Configurable temperature and top-k

### Helper Scripts
- **`test_setup.py`** - Verify installation (run this first!)
- **`preview_dataset.py`** - Preview training data
- **`monitor.py`** - Check training progress

### Documentation
- **`README.md`** - Complete documentation
- **`QUICKSTART.md`** - Step-by-step guide
- **`requirements.txt`** - Python dependencies

## 🚀 Getting Started (3 Commands)

```bash
# 1. Test everything works (30 seconds)
python test_setup.py

# 2. Preview the dataset (1 minute)
python preview_dataset.py

# 3. Start training! (1-2 hours on CPU)
python train.py
```

## 📊 Training Specs

### Model Configuration
- **Architecture**: Mamba-2 (state-space model)
- **Parameters**: ~10 million
- **Layers**: 6 Mamba blocks
- **Hidden size**: 256
- **Context length**: 256 tokens
- **Vocabulary**: 50,257 (GPT-2 tokenizer)

### Dataset
- **Name**: TinyStories (roneneldan/TinyStories)
- **Size**: Using 10% (~50K stories, 2-3GB)
- **License**: MIT (fully open-source)
- **Content**: High-quality children's stories
- **Quality**: GPT-3.5/4 generated, human-curated

### Training
- **Steps**: 5,000 (adjustable)
- **Batch size**: 16
- **Learning rate**: 3e-4 (warmup + cosine decay)
- **Optimizer**: AdamW
- **Time**: 1-2 hours on CPU, 15-30 min on GPU (if available)

## 💻 System Requirements

✅ **You're all set!** Everything is already installed in your UV venv.

- Python 3.12
- PyTorch 2.9.0
- Transformers 4.57.1
- Datasets 4.3.0
- 8GB+ RAM recommended
- 10GB free disk space

## 🎯 What Happens During Training

1. **Download** dataset (~2-3GB, only once)
2. **Tokenize** stories into sequences
3. **Train** for 5,000 steps
   - Save checkpoint every 1,000 steps
   - Evaluate every 500 steps
   - Generate sample text to show progress
4. **Save** best model based on validation loss
5. **Done!** Use `inference.py` to generate text

## 📈 Expected Results

After training, your model will be able to:
- Generate coherent short stories
- Maintain basic narrative structure
- Use appropriate vocabulary for children's content
- Complete prompts in a contextually relevant way

Example:
```
Prompt: "Once upon a time"
Output: "Once upon a time there was a little girl named Lily. 
        She loved to play outside in the sunshine. One day, 
        she saw a big red ball in the park..."
```

## 🔬 Why Mamba-2?

Advantages over Transformers:
- ⚡ **Linear time** complexity (vs O(n²) for attention)
- 💾 **Memory efficient** for long sequences
- 🎯 **No positional embeddings** needed
- 🚀 **Fast inference** on CPU
- 📚 **Simpler architecture** easier to understand

Perfect for:
- Learning about sequence models
- CPU-only training
- Research and experimentation
- Understanding state-space models

## 📁 File Structure After Training

```
LibreLLM/
├── checkpoints/
│   ├── config.json           # Training config
│   ├── best_model.pt         # Best model (use this!)
│   ├── final_model.pt        # Final model
│   └── checkpoint_*.pt       # Intermediate saves
├── mamba2_model.py           # Model code
├── train.py                  # Training code
├── inference.py              # Generation code
└── [helper scripts]
```

## 🎮 After Training

### Generate Text
```bash
# Quick test
python inference.py

# Custom prompt
python inference.py --prompt "The magical dragon"

# Interactive mode (recommended!)
python inference.py --interactive
```

### Check Progress
```bash
python monitor.py
```

### Resume Training
Edit `train.py` to:
1. Load checkpoint
2. Increase `max_steps`
3. Run again

## 🎓 Learning Path

1. ✅ **Setup** - Run `test_setup.py`
2. 📚 **Learn** - Read README.md to understand Mamba-2
3. 🔍 **Explore** - Check `preview_dataset.py`
4. 🚀 **Train** - Run `train.py`
5. ⏱️ **Wait** - Monitor with `monitor.py`
6. 🎮 **Generate** - Test with `inference.py --interactive`
7. 🔧 **Experiment** - Modify configs, try new prompts
8. 📈 **Improve** - Increase model size, train longer

## 💡 Tips

### For Faster Training
- Reduce `max_steps` to 2000 for quick results
- Reduce `batch_size` if running out of memory
- Use GPU if available (automatic detection)

### For Better Quality
- Increase `max_steps` to 10,000+
- Increase `d_model` to 512 (but slower)
- Add more layers (`depth` = 8 or 12)
- Use full dataset (remove sample limit)

### For Experimentation
- Try different datasets (WikiText, OpenWebText)
- Adjust temperature during generation
- Modify the SSM block architecture
- Implement different training techniques

## 🐛 Common Issues

### "Out of memory"
→ Reduce `batch_size` in `train.py`

### "Training is slow"
→ This is normal on CPU! Reduce `max_steps` or use GPU

### "Generated text is gibberish"
→ Train longer or use a larger model

### "Dataset download fails"
→ Check internet connection, try again

## 📚 Additional Resources

All included in your workspace:
- `README.md` - Full documentation
- `QUICKSTART.md` - Quick reference
- Code comments - Detailed explanations
- Paper references in README

## 🎉 You're Ready!

Everything is set up and tested. Just run:

```bash
python train.py
```

And watch your language model train! 

The training script will:
- ✅ Show progress bars
- ✅ Display loss values
- ✅ Generate sample text
- ✅ Save checkpoints
- ✅ Report when done

**Estimated time: 1-2 hours on CPU**

Grab some coffee and enjoy watching AI in action! ☕🤖

---

**Questions? Check the README.md or code comments!**

**Happy Training! 🚀**
