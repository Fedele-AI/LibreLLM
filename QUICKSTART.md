# Quick Start Guide - LibreLLM Mamba-2

## 🚀 Quick Start (3 Steps)

### Step 1: Test the Setup (30 seconds)
```bash
python test_setup.py
```
This verifies that everything is installed correctly.

### Step 2: Preview the Dataset (1 minute)
```bash
python preview_dataset.py
```
This shows you sample stories from the training dataset.

### Step 3: Start Training (1-2 hours)
```bash
python train.py
```
This trains the model. Go get coffee! ☕

## 📊 What to Expect

### During Training
- Progress bars showing epochs and batches
- Loss values (should decrease over time)
- Sample text generation every 500 steps
- Automatic checkpoints saved every 1000 steps
- Best model saved based on validation loss

### After Training
You'll have these files in `checkpoints/`:
- `best_model.pt` - Best performing model
- `final_model.pt` - Final model after all training
- `checkpoint_*.pt` - Intermediate checkpoints
- `config.json` - Training configuration

## 🎮 Using Your Trained Model

### Basic Generation
```bash
python inference.py
```

### Custom Prompt
```bash
python inference.py --prompt "The magical forest"
```

### Interactive Mode
```bash
python inference.py --interactive
```
Then type prompts and see the model generate stories!

## 📈 Training Progress

Typical training progression:
- **Steps 0-100**: High loss (~11.0), random text
- **Steps 100-500**: Loss decreases (~8.0), some words make sense
- **Steps 500-1000**: Loss ~6.0, basic grammar emerges
- **Steps 1000-3000**: Loss ~4.5, coherent short phrases
- **Steps 3000-5000**: Loss ~3.5-4.0, simple stories

## 💾 System Requirements

### Minimum
- **RAM**: 8GB (16GB recommended)
- **Disk**: 10GB free space
- **CPU**: Any modern CPU
- **GPU**: Not required (but helps!)
- **Time**: 1-2 hours on CPU, 15-30 min on GPU

### Dataset Download
- Size: ~2-3 GB
- Name: TinyStories
- Stories: ~50,000 (10% of full dataset)
- Quality: High (GPT-3.5/4 generated, curated)

## 🔧 Customization

Edit `train.py` to customize:

```python
class Config:
    # Model size
    d_model = 256      # Increase for bigger model (512, 768)
    depth = 6          # More layers (8, 12)
    
    # Training
    batch_size = 16    # Adjust based on RAM
    learning_rate = 3e-4
    num_epochs = 3
    max_steps = 5000   # More steps = better model
    
    # Data
    max_samples = 50000  # Use more data
```

## 📚 Model Architecture

```
Input Text
    ↓
Token Embeddings (vocab_size → d_model)
    ↓
Mamba Block 1 ─┐
    ↓           │ (6 layers)
Mamba Block 6 ─┘
    ↓
RMS Normalization
    ↓
Language Model Head (d_model → vocab_size)
    ↓
Output Probabilities
```

Each Mamba Block:
1. RMS Normalization
2. Input Projection (2x expansion)
3. 1D Convolution
4. State-Space Model (SSM)
5. Gating mechanism
6. Output Projection

## 🎯 Example Outputs

After training, you might get:

**Prompt**: "Once upon a time"
**Generated**: "Once upon a time there was a little girl named Lily. She loved to play outside. One day she saw a big red ball in the park. She was so happy!"

**Prompt**: "The brave knight"
**Generated**: "The brave knight went to the castle. He wanted to save the princess. He was very brave and strong."

## 🐛 Troubleshooting

### Out of Memory
- Reduce `batch_size` in `train.py`
- Reduce `d_model` or `depth`
- Use fewer samples with `max_samples`

### Slow Training
- Normal on CPU! Be patient
- Reduce `max_steps` for quicker results
- Consider using fewer samples

### Poor Quality Text
- Train longer (more `max_steps`)
- Use more data (increase `max_samples`)
- Increase model size (`d_model`, `depth`)

### Download Issues
- Check internet connection
- Hugging Face datasets might be slow
- Dataset is cached after first download

## 📖 Learning Resources

Want to understand more?

1. **Mamba Architecture**
   - Read the README.md for detailed explanation
   - Check the references for papers

2. **Code Structure**
   - `mamba2_model.py` - Model architecture
   - `train.py` - Training loop
   - `inference.py` - Text generation

3. **Experiment!**
   - Try different prompts
   - Adjust hyperparameters
   - Train on different data

## ✅ Checklist

- [ ] Ran `test_setup.py` successfully
- [ ] Previewed dataset with `preview_dataset.py`
- [ ] Started training with `python train.py`
- [ ] Waited for training to complete
- [ ] Generated text with `inference.py`
- [ ] Tried interactive mode
- [ ] Experimented with different prompts

## 🎉 Next Steps

After basic training:
1. Try increasing model size
2. Train for more steps
3. Use full TinyStories dataset
4. Experiment with other datasets
5. Fine-tune on specific topics
6. Share your results!

---

**Happy Training! 🚀**

For issues or questions, check the README.md or review the code comments.
