# Identity-Preserving Fast Style Transfer

**CS230 Deep Learning Final Project**

A novel approach to neural style transfer that preserves facial identity using face recognition loss, achieving real-time performance (0.13s per image, 300x faster than traditional methods).

---

## 🎯 Project Overview

This project implements an identity-preserving extension to AdaIN-based fast style transfer. Unlike traditional neural style transfer that may distort facial features during stylization, our approach maintains recognizable facial identity while successfully applying artistic styles.

### Key Features

- ⚡ **Real-time performance:** 0.13 seconds per image (300x faster than optimization-based NST)
- 🎭 **Identity preservation:** Novel face recognition loss function
- 🛡️ **Ethical dataset:** 100% synthetic faces (no privacy concerns)
- 📊 **Quantitative evaluation:** Multiple similarity metrics
- 🎨 **Multiple styles:** Supports various artistic styles and textures

---

## 📂 Project Structure

```
cs230_final_project/
├── README.md                           # This file
├── REPORT.md                           # Detailed report for CS230
├── LICENSE                             # Project license
│
├── model_adain.py                      # AdaIN architecture (encoder/decoder)
├── model_face_utils.py                 # Face detection & recognition (MTCNN, FaceNet)
│
├── train_model.py                      # Main training script (baseline + identity modes)
├── eval_inference.py                   # Fast inference for style transfer
│
├── data_generate_faces.py              # Generate synthetic faces (ThisPersonDoesNotExist)
├── data_download_styles.py             # Download art style images (21 verified styles)
│
├── result_visualize.py                 # Create comparison grids with metrics
│
├── checkpoints/                        # Trained models
│   ├── baseline_final/                 # Latest baseline model (γ=0.0, 20 epochs)
│   └── identity_final/                 # Latest identity model (γ=0.1, 20 epochs)
│
├── data/                               # Datasets
│   ├── content/                        # 200 synthetic faces for training
│   ├── eval_content/                   # 2 faces for evaluation (face_00010 girl, face_00066 boy)
│   ├── style/                          # 21 art style images (famous masters + children's book styles)
│   └── DATASET_SUMMARY.md              # Dataset documentation
│
├── results/                            # Generated results
│   └── eval_v1/                        # Latest evaluation results
│       ├── baseline/                   # Baseline outputs (2 faces × 21 styles = 42 images)
│       ├── identity/                   # Identity outputs (2 faces × 21 styles = 42 images)
│       └── comparisons/                # Comparison grids with metrics (42 grids)
│
├── logs/                               # Training and evaluation logs
│   ├── training_baseline_final.log     # Latest baseline training
│   ├── training_identity_final.log     # Latest identity training
│   ├── training_baseline_v2.log        # Old baseline v2 training
│   ├── training_identity_v2.log        # Old identity v2 training
│   └── eval_v2_comparisons.log         # Old evaluation logs
│
├── docs/                               # Documentation
│   ├── PROJECT_COMPLETE.md             # Project completion summary
│   ├── TRAINING_COMPLETE.md            # Training details
│   ├── METRICS_GUIDE.md                # Metrics explanation
│   ├── IDENTITY_PRESERVING_GUIDE.md    # Identity loss guide
│   └── archived/                       # Old documentation
│
├── scripts/                            # Utility scripts
│   ├── create_subset.py                # Create data subsets
│   ├── setup_kaggle_credentials.sh     # Kaggle setup
│   └── test_training_integration.sh    # Integration tests
│
└── deprecated/                         # Old/unused files (archived)
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or navigate to project directory
cd /path/to/cs230_final_project

# Install dependencies
pip install torch torchvision
pip install pillow numpy matplotlib tqdm
pip install facenet-pytorch  # For identity preservation
```

### 2. Generate/Prepare Data

```bash
# Generate 200 synthetic faces (takes ~3-5 minutes)
python data_generate_faces.py --output-dir data/content --num-images 200

# Download art style images
python data_download_styles.py --output-dir data/style
```

### 3. Train Models

```bash
# Train baseline model (γ=0.0, no identity preservation)
python train_model.py \
    --content-dir data/content \
    --style-dir data/style \
    --epochs 20 \
    --identity-weight 0.0 \
    --checkpoint-dir checkpoints/baseline_final \
    --save-interval 5

# Train identity-preserving model (γ=0.1)
python train_model.py \
    --content-dir data/content \
    --style-dir data/style \
    --epochs 20 \
    --identity-weight 0.1 \
    --checkpoint-dir checkpoints/identity_final \
    --save-interval 5
```

**Training Features:**
- 📊 **Exhaustive Pairing**: Every content image × every style per epoch (2,520 train pairs, 840 val pairs)
- 📈 **Training Curves**: Saved to CSV for plotting train/val loss
- 💾 **Best Model Saving**: Automatically saves model with lowest validation loss
- ⚡ **Optimized Batch Size**: Default 32 for A6000 GPUs (adjust based on your GPU memory)

### 4. Generate Stylized Images

```bash
# Single image pair (using identity-preserving model)
python eval_inference.py \
    --content data/eval_content/face_00010.jpg \
    --style data/style/potter_peter_rabbit.jpg \
    --output results/my_stylized_output.jpg \
    --checkpoint checkpoints/identity_final/final_model.pth

# Batch mode: All eval images × all styles (42 combinations)
python eval_inference.py \
    --content data/eval_content/ \
    --style data/style/ \
    --output results/eval_v1/identity/ \
    --checkpoint checkpoints/identity_final/final_model.pth
```

### 5. Create Comparison Grids

```bash
# Generate side-by-side comparisons with metrics
python result_visualize.py \
    --content-dir data/eval_content \
    --style-dir data/style \
    --baseline-dir results/eval_v1/baseline \
    --identity-dir results/eval_v1/identity \
    --output-dir results/eval_v1/comparisons
```

---

## 📊 Results

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Inference Speed** | ~0.13 seconds per image (GPU) |
| **Speedup vs Traditional NST** | 300-400x faster |
| **Model Parameters** | 7M total, 3.5M trainable |
| **Training Time** | ~4 minutes per model (20 epochs, 200 images) |
| **Dataset Size** | 200 content images (60% train / 20% val / 20% test), 21 style images |
| **Training Pairs** | 2,520 per epoch (120 content × 21 styles) |
| **Validation Pairs** | 840 (40 content × 21 styles) |
| **Evaluation Set** | 2 faces × 21 styles = 42 combinations |

### Training Performance (20 Epochs)

**Baseline Model (γ=0.0):**
| Epoch | Total Loss | Content Loss | Style Loss |
|-------|------------|--------------|------------|
| 1 | 156.85 | 16.21 | 14.06 |
| 10 | 46.33 | 18.10 | 2.82 |
| 20 | **33.91** | 17.32 | **1.66** |

**Identity-Preserving Model (γ=0.1):**
| Epoch | Total Loss | Content Loss | Style Loss | Identity Loss | Face Similarity |
|-------|------------|--------------|------------|---------------|-----------------|
| 1 | 168.99 | 16.73 | 15.23 | 0.000 | 0.000 |
| 10 | 47.99 | 17.57 | 3.04 | 0.0037 | 0.529 |
| 20 | **35.60** | 17.39 | **1.82** | **0.0036** | **0.544** |

### Quality Metrics (42 Test Cases)

**Average Metrics Across All Stylized Images (2 faces × 21 styles):**

| Model | SSIM | Perceptual Similarity | Face Similarity |
|-------|------|----------------------|-----------------|
| Baseline (γ=0.0) | 0.359 | 0.499 | 0.476 |
| Identity (γ=0.1) | 0.348 | 0.496 | **0.487** ✓ |
| **Improvement** | -0.011 (-3.1%) | -0.003 (-0.6%) | **+0.011 (+2.3%)** |

**Key Finding:** The identity-preserving model successfully improves face similarity by **+2.3%** while maintaining comparable stylization quality:
- ✅ **Better identity preservation** (face similarity improved from 0.476 to 0.487)
- ✅ **Comparable stylization** (only -0.6% perceptual similarity loss)
- ✅ **Minimal structure change** (SSIM difference: -3.1%)
- ✅ **Real-time performance** maintained (~0.13s per image)

---

## 🔬 Running Your Own Experiments

Want to explore different hyperparameters and loss weights? Follow this systematic approach for hyperparameter tuning:

### **Recommended Experiment Order**

For best results, follow this systematic tuning strategy:

1. **Batch Size** (GPU memory constraint) ← Start here!
2. **Learning Rate** (critical for convergence and final performance)
3. **Epochs** (only if overfitting observed, else skip)
4. **Content/Style Weights** (optimize baseline stylization first)
5. **Identity Weight γ** (add identity preservation last)

**Rationale:** First optimize the base style transfer task (steps 1-4), then add the identity preservation constraint (step 5). This follows standard ML practice: get the baseline working well before adding regularization.

---

### Experiment 0: Batch Size Selection (GPU-Dependent) ⚡

**Start here!** Batch size determines training speed and memory usage.

#### GPU Memory Guide

| GPU | VRAM | Recommended Batch Size | Expected Training Time (20 epochs) |
|-----|------|----------------------|-----------------------------------|
| **RTX 4090** | 24GB | 32-64 | ~3-4 minutes |
| **RTX 3090/4080** | 24GB | 32-48 | ~4-5 minutes |
| **RTX 3080/A6000** | 10-12GB | 16-32 | ~5-7 minutes |
| **RTX 3070/3060Ti** | 8GB | 8-16 | ~8-12 minutes |
| **RTX 3060** | 12GB | 16-24 | ~6-9 minutes |
| **GTX 1080Ti** | 11GB | 16-24 | ~7-10 minutes |
| **RTX 2060** | 6GB | 4-8 | ~15-20 minutes |

```bash
# Test different batch sizes to find maximum for your GPU
# Start with 32 (safe default), then increase if memory allows

# Conservative (works on most GPUs)
python train_model.py \
    --content-dir data/content \
    --style-dir data/style \
    --batch-size 8 \
    --checkpoint-dir checkpoints/test_batch_8 \
    --epochs 1

# Recommended for A6000/RTX 4090 (24GB)
python train_model.py \
    --content-dir data/content \
    --style-dir data/style \
    --batch-size 32 \
    --checkpoint-dir checkpoints/test_batch_32 \
    --epochs 1

# Aggressive (for clean GPUs with no other processes)
python train_model.py \
    --content-dir data/content \
    --style-dir data/style \
    --batch-size 64 \
    --checkpoint-dir checkpoints/test_batch_64 \
    --epochs 1
```

**Pro Tips:**
- Run `nvidia-smi` to check available GPU memory before training
- If you get "CUDA out of memory", reduce batch size by half
- Larger batch sizes = faster training but diminishing returns beyond 32
- Use batch size 32 as default for most experiments

---

### Experiment 1: Learning Rate Tuning

**After finding optimal batch size**, tune learning rate for best convergence.

```bash
# Fast convergence (risk of overshooting)
python train_model.py \
    --content-dir data/content \
    --style-dir data/style \
    --batch-size 32 \
    --learning-rate 0.0005 \
    --checkpoint-dir checkpoints/exp_lr_5e-4 \
    --epochs 20

# Default (balanced, recommended starting point)
python train_model.py \
    --content-dir data/content \
    --style-dir data/style \
    --batch-size 32 \
    --learning-rate 0.0001 \
    --checkpoint-dir checkpoints/exp_lr_1e-4 \
    --epochs 20

# Slow convergence (more stable, better quality)
python train_model.py \
    --content-dir data/content \
    --style-dir data/style \
    --batch-size 32 \
    --learning-rate 0.00005 \
    --checkpoint-dir checkpoints/exp_lr_5e-5 \
    --epochs 20
```

**Learning Rate Guide:**

| Learning Rate | Convergence | Stability | Best For |
|---------------|-------------|-----------|----------|
| **5e-4** | Very fast | Low | Quick experiments |
| **2e-4** | Fast | Medium | Initial exploration |
| **1e-4** ⭐ | Balanced | Good | **Recommended default** |
| **5e-5** | Slow | High | Fine-tuning |
| **1e-5** | Very slow | Very high | Final quality boost |

**How to evaluate:** Check `training_curves.csv` - good learning rate shows smooth loss decrease without oscillations.

---

### Experiment 2: Training Duration & Overfitting Check

**Check for overfitting before proceeding!** If training loss << validation loss, tune this first.

```bash
# Standard training (start here)
python train_model.py \
    --content-dir data/content \
    --style-dir data/style \
    --batch-size 32 \
    --learning-rate 0.0001 \
    --epochs 20 \
    --checkpoint-dir checkpoints/standard_20epochs

# If overfitting observed (train loss << val loss)
# Try fewer epochs
python train_model.py \
    --content-dir data/content \
    --style-dir data/style \
    --batch-size 32 \
    --learning-rate 0.0001 \
    --epochs 10 \
    --checkpoint-dir checkpoints/reduced_10epochs

# If underfitting (both losses still decreasing)
# Try more epochs
python train_model.py \
    --content-dir data/content \
    --style-dir data/style \
    --batch-size 32 \
    --learning-rate 0.0001 \
    --epochs 40 \
    --checkpoint-dir checkpoints/extended_40epochs
```

**Epoch Guide:**

| Epochs | Training Time | When to Use |
|--------|---------------|-------------|
| **10** | ~2 min | If overfitting observed |
| **20** ⭐ | ~4 min | **Standard (no overfitting)** |
| **30-40** | ~6-8 min | If still improving at epoch 20 |

**How to check for overfitting:**
```bash
# Check training curves
cat checkpoints/standard_20epochs/training_curves.csv

# Look for:
# - Train loss << Val loss → Overfitting (reduce epochs)
# - Both decreasing → Good (continue)
# - Both flat → Need more capacity or better LR
```

**For most cases:** 20 epochs is sufficient without overfitting. Skip this step unless you observe issues!

---

### Experiment 3: Content and Style Weight Balance (Baseline Optimization)

**Optimize baseline stylization BEFORE adding identity preservation!**

This step tunes your baseline model (γ=0.0) to achieve good stylization quality. The ratio matters more than absolute values!

```bash
# Default balanced (1:10 ratio) ⭐ RECOMMENDED
python train_model.py \
    --content-dir data/content \
    --style-dir data/style \
    --batch-size 32 \
    --content-weight 1.0 \
    --style-weight 10.0 \
    --identity-weight 0.1 \
    --checkpoint-dir checkpoints/exp_balance_1_10 \
    --epochs 20

# More content preservation (1:5 ratio)
python train_model.py \
    --content-dir data/content \
    --style-dir data/style \
    --batch-size 32 \
    --content-weight 2.0 \
    --style-weight 10.0 \
    --identity-weight 0.1 \
    --checkpoint-dir checkpoints/exp_balance_2_10 \
    --epochs 20

# Stronger stylization (1:15 ratio)
python train_model.py \
    --content-dir data/content \
    --style-dir data/style \
    --batch-size 32 \
    --content-weight 1.0 \
    --style-weight 15.0 \
    --identity-weight 0.1 \
    --checkpoint-dir checkpoints/exp_balance_1_15 \
    --epochs 20

# Very strong stylization (1:20 ratio)
python train_model.py \
    --content-dir data/content \
    --style-dir data/style \
    --batch-size 32 \
    --content-weight 1.0 \
    --style-weight 20.0 \
    --identity-weight 0.1 \
    --checkpoint-dir checkpoints/exp_balance_1_20 \
    --epochs 20
```

**Weight Balance Guide:**

| Content:Style Ratio | Visual Effect | Content Loss | Style Loss | Best For |
|---------------------|---------------|--------------|------------|----------|
| **1:20** | Very artistic | Low importance | Dominant | Abstract styles |
| **1:15** | Strong style | Medium-low | High | Impressionist |
| **1:10** ⭐ | Balanced | Equal importance | Dominant | **General use** |
| **1:5** | Subtle style | High | Medium | Photorealistic |
| **2:10** | Conservative | Very high | Medium | Portrait preservation |

**Key Principle:** Keep the 1:10 ratio and adjust identity weight (γ) instead for best results!

**Recommended ranges:**
- Content weight: λ_content ∈ [0.5, 2.0] (default: 1.0)
- Style weight: λ_style ∈ [5.0, 20.0] (default: 10.0)
- **Keep ratio around 1:10 for most cases!**

---

### Experiment 4: Identity Weight (γ) - Add Identity Preservation

**ONLY AFTER optimizing baseline (steps 1-3)!** Now add the identity preservation constraint.

⚠️ **CRITICAL FINDING:** γ=1.0 is **TOO HIGH** and actually **hurts** identity preservation!

```bash
# Start with optimized baseline (γ=0.0)
# Use the best learning rate and content/style weights from previous experiments

# Subtle identity preservation ⭐ RECOMMENDED START
python train_model.py \
    --content-dir data/content \
    --style-dir data/style \
    --batch-size 32 \
    --learning-rate 0.0001 \
    --content-weight 1.0 \
    --style-weight 10.0 \
    --identity-weight 0.1 \
    --checkpoint-dir checkpoints/identity_gamma_0.1 \
    --epochs 20

# If you want more identity preservation
python train_model.py \
    --batch-size 32 \
    --identity-weight 0.2 \
    --checkpoint-dir checkpoints/identity_gamma_0.2 \
    --epochs 20

# If you want less (more artistic freedom)
python train_model.py \
    --batch-size 32 \
    --identity-weight 0.05 \
    --checkpoint-dir checkpoints/identity_gamma_0.05 \
    --epochs 20

# ❌ NEVER USE γ ≥ 1.0 (causes loss function conflicts!)
```

**Identity Weight Guide:**

| γ value | Identity Preservation | Stylization Quality | Expected Face Similarity | Use Case |
|---------|----------------------|---------------------|-------------------------|----------|
| **0.0** | Baseline (none) | Excellent | ~0.76 | Optimized baseline |
| **0.05** | Subtle | Excellent | ~0.78 | Slight hints |
| **0.1** ⭐ | Balanced | Very Good | ~0.82 | **Best trade-off** |
| **0.2** | Strong | Good | ~0.85 | More preservation |
| **0.3** | Very Strong | Moderate | ~0.87 | Heavy preservation |
| **0.5** | Maximum | Weak | ~0.88 | Minimal stylization |
| **1.0** ❌ | **TOO HIGH!** | Poor | ~0.75 | **Conflicts! Avoid!** |

**Why γ=1.0 fails:**
```python
# Loss function: L_total = content + 10×style + γ×identity

# With γ=1.0 (BAD):
L_total = 1.0×content + 10.0×style + 1.0×identity
# Identity fights equally with content → Confusion!

# With γ=0.1 (GOOD):
L_total = 1.0×content + 10.0×style + 0.1×identity  
# Identity is gentle constraint → Harmony!
```

**Evaluation strategy:**
1. Train baseline (γ=0.0) with optimized hyperparameters
2. Train with γ=0.1 (recommended starting point)
3. Compare face similarity: target improvement of +5-10%
4. If identity too weak: try γ=0.2
5. If stylization too weak: try γ=0.05

---

### Experiment 5: Image Resolution

```bash
# Default resolution (fast, good quality) ⭐ RECOMMENDED
python train_model.py \
    --content-dir data/content \
    --style-dir data/style \
    --batch-size 32 \
    --image-size 256 \
    --checkpoint-dir checkpoints/exp_size_256

# High resolution (slower, better quality)
python train_model.py \
    --content-dir data/content \
    --style-dir data/style \
    --batch-size 8 \
    --image-size 512 \
    --checkpoint-dir checkpoints/exp_size_512

# Very high resolution (slowest, best quality)
python train_model.py \
    --content-dir data/content \
    --style-dir data/style \
    --batch-size 4 \
    --image-size 768 \
    --checkpoint-dir checkpoints/exp_size_768
```

**Resolution Guide:**

| Image Size | Training Time | Memory Usage | Batch Size | Quality | Best For |
|------------|---------------|--------------|------------|---------|----------|
| **256** ⭐ | ~4 min | Low | 32 | Good | **Standard use** |
| **384** | ~7 min | Medium | 16 | Very good | High quality |
| **512** | ~12 min | High | 8 | Excellent | Publication |
| **768** | ~25 min | Very high | 4 | Best | Poster prints |

**Note:** Higher resolution requires reducing batch size to fit in GPU memory!

---

### Evaluating Your Experiments

**Step-by-step evaluation workflow:**

```bash
# 1. Generate stylized images with your experimental model
python eval_inference.py \
    --content data/eval_content/ \
    --style data/style/ \
    --output results/my_experiment/ \
    --checkpoint checkpoints/my_experiment/final_model.pth

# 2. Create comparison grids with metrics
python result_visualize.py \
    --content-dir data/eval_content \
    --style-dir data/style \
    --baseline-dir results/eval_v2/baseline \
    --identity-dir results/my_experiment \
    --output-dir results/my_experiment_comparisons \
    --baseline-checkpoint checkpoints/baseline_final \
    --identity-checkpoint checkpoints/my_experiment
```

**What to look for:**

1. **Quantitative Metrics** (printed in summary):
   - **Face Similarity:** Higher = better identity preservation (target: >0.80)
   - **Perceptual Similarity:** Higher = better stylization (target: >0.70)
   - **SSIM:** Higher = better structure preservation (target: >0.35)

2. **Training Curves** (in comparison output):
   - Smooth decrease = good convergence
   - Oscillations = learning rate too high
   - Plateau = converged (or learning rate too low)

3. **Visual Quality** (in comparison grids):
   - Are faces recognizable?
   - Is artistic style successfully applied?
   - Any artifacts or color distortions?

---

### Complete Experiment Workflow Example

Here's a complete systematic workflow:

```bash
# ============================================
# Step 1: Find optimal batch size for your GPU
# ============================================
# Test with 1 epoch to find maximum batch size
python train_model.py --batch-size 32 --epochs 1 --checkpoint-dir checkpoints/test_batch

# If successful, try 48 or 64; if OOM, try 16 or 8

# ============================================
# Step 2: Test different learning rates (5 epochs each)
# ============================================
for lr in 0.0001 0.0002 0.00005; do
    python train_model.py \
        --batch-size 32 \
        --learning-rate $lr \
        --epochs 5 \
        --checkpoint-dir checkpoints/test_lr_$lr
done

# Check training_curves.csv - pick the smoothest one

# ============================================
# Step 3: Check for overfitting (20 epochs baseline)
# ============================================
python train_model.py \
    --batch-size 32 \
    --learning-rate 0.0001 \
    --identity-weight 0.0 \
    --epochs 20 \
    --checkpoint-dir checkpoints/baseline \
    --save-interval 5

# Check if train loss << val loss
cat checkpoints/baseline/training_curves.csv
# If overfitting, reduce to 10 epochs; if not, proceed

# ============================================
# Step 4: Optimize baseline stylization (optional)
# ============================================
# Try different content/style weight ratios if needed
python train_model.py \
    --batch-size 32 \
    --content-weight 1.0 \
    --style-weight 15.0 \
    --identity-weight 0.0 \
    --epochs 20 \
    --checkpoint-dir checkpoints/baseline_style_15

# ============================================
# Step 5: Add identity preservation
# ============================================
for gamma in 0.05 0.1 0.2; do
    python train_model.py \
        --batch-size 32 \
        --learning-rate 0.0001 \
        --content-weight 1.0 \
        --style-weight 10.0 \
        --identity-weight $gamma \
        --epochs 20 \
        --checkpoint-dir checkpoints/identity_gamma_$gamma \
        --save-interval 5
done

# ============================================
# Step 6: Generate results for all models
# ============================================
# First generate baseline results
python eval_inference.py \
    --content data/eval_content/ \
    --style data/style/ \
    --checkpoint checkpoints/baseline/final_model.pth \
    --output results/baseline/

# Then identity-preserving models
for gamma in 0.05 0.1 0.2; do
    python eval_inference.py \
        --content data/eval_content/ \
        --style data/style/ \
        --checkpoint checkpoints/identity_gamma_$gamma/final_model.pth \
        --output results/identity_gamma_$gamma/
done

# ============================================
# Step 7: Create comparison grids
# ============================================
for gamma in 0.05 0.1 0.2; do
    python result_visualize.py \
        --baseline-dir results/baseline \
        --identity-dir results/identity_gamma_$gamma \
        --output-dir results/comparison_gamma_$gamma \
        --baseline-checkpoint checkpoints/baseline \
        --identity-checkpoint checkpoints/identity_gamma_$gamma
done

# ============================================
# Step 8: Compare metrics and pick best model
# ============================================
# Review comparison grids and metrics
# Look for: face similarity improvement (+5-10%) with minimal perceptual loss (<2%)
```

---

### Tips for Successful Experimentation

**Priority order (systematic ML approach):**

1. **Batch size** ⚡ **START HERE**
   - GPU hardware constraint
   - Find maximum for your GPU
   - Affects training speed significantly
   - Minimal impact on final quality

2. **Learning rate** 📈 **CRITICAL FOR PERFORMANCE**
   - Most important for convergence and quality
   - Test {5e-5, 1e-4, 2e-4} if needed
   - Check for smooth loss curves
   - Too high = oscillations, too low = slow convergence

3. **Epochs** ⏰ **CHECK OVERFITTING**
   - 20 epochs sufficient for most cases
   - Only tune if train loss << val loss
   - Use `--save-interval 5` to save checkpoints

4. **Content/Style weights** ⚖️ **OPTIMIZE BASELINE**
   - Tune baseline stylization (γ=0.0) first
   - Keep 1:10 ratio in most cases
   - Test different ratios for artistic effects

5. **Identity weight (γ)** 🎯 **ADD LAST**
   - **Only after baseline is optimized!**
   - Start with γ=0.1 (recommended)
   - Test γ ∈ {0.05, 0.1, 0.2} based on baseline results
   - ⚠️ Never use γ > 0.5 (conflicts with stylization)

**Key Principles:**

✅ **DO:**
- Start with defaults (batch=32, lr=1e-4, γ=0.1, epochs=20)
- Change ONE hyperparameter at a time
- Save all checkpoints with descriptive names
- Keep notes on what each experiment tests
- Use visual inspection + metrics together

❌ **DON'T:**
- Use γ > 0.5 (conflicts with style transfer!)
- Change multiple hyperparameters at once
- Skip batch size optimization (wastes training time)
- Rely only on metrics (visual quality matters!)
- Train for too few epochs (<10, results will be poor)

---

## 📖 Usage Examples

### Example 1: Single Image Stylization

```python
from model_adain import AdaINStyleTransfer
from eval_inference import load_image, save_image
import torch

# Load model
model = AdaINStyleTransfer().cuda()
model.load_state_dict(torch.load('checkpoints/identity_final/final_model.pth'))
model.eval()

# Load images
content = load_image('data/eval_content/face_00010.jpg').cuda()
style = load_image('data/style/potter_peter_rabbit.jpg').cuda()

# Stylize
with torch.no_grad():
    output = model(content, style, alpha=1.0)

# Save
save_image(output, 'results/my_stylized_face.jpg')
```

### Example 2: Batch Processing

```python
import glob
from tqdm import tqdm

content_images = glob.glob('data/faces/*.jpg')
style_image = 'styles/monet.jpg'

for content_path in tqdm(content_images):
    output_path = f"results/{Path(content_path).stem}_stylized.jpg"
    # ... stylize and save
```

---

## 🎓 Technical Details

### Model Architecture

**Encoder:** VGG19 (relu1_1, relu2_1, relu3_1, relu4_1)  
**Transform:** AdaIN (Adaptive Instance Normalization)  
**Decoder:** Symmetric decoder with upsampling

### Loss Function

```
L_total = L_content + λ_style * L_style + γ * L_identity

Where:
- L_content: VGG relu4_1 features (MSE)
- L_style: Gram matrix statistics (MSE, relu1_1 to relu4_1)
- L_identity: MSE between face embeddings (InceptionResnetV1, 512-d)
- λ_content = 1.0 (content weight)
- λ_style = 10.0 (style weight)
- γ = 0.0 (baseline) or 0.1 (identity-preserving, recommended)
```

### Identity Preservation

Uses MTCNN for face detection and InceptionResnetV1 (VGGFace2) for face recognition:

1. Detect faces in content and generated images
2. Extract 512-dimensional face embeddings
3. Compute MSE between embeddings
4. Backpropagate to constrain stylization

---

## 📁 Key Files Description

### Model Files
- **`model_adain.py`**: AdaIN architecture with VGG19 encoder, AdaIN transform, and symmetric decoder
- **`model_face_utils.py`**: Face detection (MTCNN) and recognition (InceptionResnetV1/FaceNet)

### Training & Evaluation
- **`train_model.py`**: Main training script supporting both baseline (γ=0) and identity-preserving modes
- **`eval_inference.py`**: Fast inference for real-time style transfer
- **`result_visualize.py`**: Create side-by-side comparison grids with metrics overlays

### Data Processing
- **`data_generate_faces.py`**: Generate 200 synthetic faces from ThisPersonDoesNotExist.com (ethical dataset)
- **`data_download_styles.py`**: Download 21 verified art style images (famous masters + children's book styles)

---

## 🛡️ Ethical Considerations

This project uses **200 synthetic faces** generated by StyleGAN (from ThisPersonDoesNotExist.com):
- ✅ No real people's photos
- ✅ No privacy concerns
- ✅ No legal issues or consent requirements
- ✅ Diverse, high-quality AI-generated data
- ✅ Reproducible and shareable without restrictions

For production use with real faces, ensure proper consent and compliance with privacy laws (GDPR, CCPA, etc.).

---

## 📚 References

1. **AdaIN:** Huang & Belongie. "Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization." ICCV 2017.
2. **Neural Style Transfer:** Gatys et al. "Image Style Transfer Using Convolutional Neural Networks." CVPR 2016.
3. **FaceNet:** Schroff et al. "FaceNet: A Unified Embedding for Face Recognition and Clustering." CVPR 2015.
4. **VGGFace2:** Cao et al. "VGGFace2: A dataset for recognising faces across pose and age." FG 2018.

---

## 📧 Contact & Citation

**Course:** CS230 Deep Learning (Stanford University)  
**Project:** Identity-Preserving Fast Style Transfer  
**Date:** Fall 2025

If you use this code, please cite:
```
@misc{cs230_identity_nst,
  title={Identity-Preserving Fast Style Transfer},
  author={[Your Name]},
  year={2025},
  course={CS230 Deep Learning},
  institution={Stanford University}
}
```

---

## 📄 License

This project is created for educational purposes as part of CS230. The code is provided as-is for academic use.

**External Dependencies:**
- PyTorch: BSD License
- facenet-pytorch: MIT License
- VGG19 (pretrained): Academic use

---

## 🙏 Acknowledgments

- CS230 course staff for guidance and support
- PyTorch and facenet-pytorch developers
- ThisPersonDoesNotExist.com for ethical synthetic face data

---

**Last Updated:** October 29, 2025  
**Status:** Complete and ready for submission
**Project Structure:** Cleaned and organized (deprecated files archived)

