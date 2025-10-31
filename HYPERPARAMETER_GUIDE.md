# Hyperparameter Tuning Quick Reference

**Last Updated:** October 31, 2025

This guide provides quick commands and decision trees for conducting hyperparameter tuning experiments.

---

## 🎯 Quick Start: Optimal Default Configuration

```bash
# Best configuration (start here!)
python train_model.py \
    --content-dir data/content \
    --style-dir data/style \
    --batch-size 32 \
    --learning-rate 0.0001 \
    --content-weight 1.0 \
    --style-weight 10.0 \
    --identity-weight 0.1 \
    --epochs 20 \
    --checkpoint-dir checkpoints/optimal_config
```

**Expected results:** Face Similarity ~0.487, Perceptual ~0.496, Training Time ~4 min

---

## 📋 Tuning Priority Order (Systematic ML Approach)

**Follow this order for best results:**

| Priority | Hyperparameter | Impact | Time to Test | Why This Order? |
|----------|----------------|--------|--------------|-----------------|
| **1** ⚡ | **Batch Size** | Training Speed | ~5 min | GPU hardware constraint - determine first! |
| **2** 📈 | **Learning Rate** | Very High | ~15 min (3 values) | **Critical for convergence** |
| **3** ⏰ | **Epochs** | Overfitting | ~10 min | Check generalization; skip if no overfitting |
| **4** ⚖️ | **Content/Style Weights** | Baseline Quality | ~20 min | **Optimize base task first** |
| **5** 🎯 | **Identity Weight (γ)** | Identity vs Style | ~15 min (3 values) | **Add constraint last** |

**Key Principle:** Optimize baseline style transfer (steps 1-4), then add identity preservation (step 5).

---

## ⚡ Batch Size - START HERE!

**GPU hardware constraint - optimize this first to minimize training time!**

### Quick Test Command

```bash
# Test with 1 epoch to find maximum batch size
python train_model.py --batch-size 32 --epochs 1 --checkpoint-dir checkpoints/test_batch

# If successful, try 48 or 64
# If OOM error, try 16 or 8
```

### GPU Guide

| Your GPU | VRAM | Test Batch Sizes | Expected Training Time |
|----------|------|------------------|------------------------|
| RTX 4090 | 24GB | Try: 64 → 48 → 32 | ~3-4 min |
| A6000 | 48GB | Try: 64 → 48 → 32 | ~3-4 min |
| RTX 3080 | 10GB | Try: 32 → 24 → 16 | ~5-7 min |
| RTX 3070 | 8GB | Try: 16 → 12 → 8 | ~8-12 min |

**Impact:** 2x batch size ≈ 1.3x faster training (not 2x due to overhead)

---

## 📈 Learning Rate - CRITICAL FOR PERFORMANCE!

**Most important hyperparameter for convergence and final quality!**

### Quick Test Command

```bash
# Test recommended learning rates (5 epochs each)
for lr in 0.00005 0.0001 0.0002; do
    python train_model.py \
        --batch-size 32 \
        --learning-rate $lr \
        --epochs 5 \
        --checkpoint-dir checkpoints/test_lr_$lr
done

# Check training_curves.csv - pick the smoothest convergence
```

### Decision Table

| Learning Rate | Convergence Speed | Stability | Best For |
|---------------|-------------------|-----------|----------|
| **5e-5** | Slow | Very High | Fine-tuning, avoiding overshoot |
| **1e-4** ⭐ | Balanced | Good | **Default - start here!** |
| **2e-4** | Fast | Medium | Quick experiments |
| **5e-4** | Very fast | Low | Only if 2e-4 is stable |

### How to Evaluate

Check `training_curves.csv`:
- **Good:** Smooth, steady decrease in both train and val loss
- **Too high:** Oscillating or increasing loss
- **Too low:** Very slow decrease (might need more epochs)

---

## ⏰ Epochs & Overfitting Check

**Only tune if overfitting observed!** Most cases: 20 epochs is sufficient.

### Quick Check

```bash
# Train baseline for 20 epochs
python train_model.py \
    --batch-size 32 \
    --learning-rate 0.0001 \
    --identity-weight 0.0 \
    --epochs 20 \
    --checkpoint-dir checkpoints/baseline

# Check for overfitting
cat checkpoints/baseline/training_curves.csv
# Look for: train_loss << val_loss

# If overfitting: reduce to 10 epochs
# If not: proceed with 20 epochs
```

---

## ⚖️ Content/Style Weights - Optimize Baseline First

**Tune baseline (γ=0.0) stylization before adding identity preservation!**

### Quick Test

```bash
# Default (1:10 ratio) ⭐ RECOMMENDED
python train_model.py \
    --batch-size 32 \
    --content-weight 1.0 \
    --style-weight 10.0 \
    --identity-weight 0.0 \
    --epochs 20

# Try stronger stylization if needed (1:15 ratio)
python train_model.py \
    --content-weight 1.0 \
    --style-weight 15.0 \
    --identity-weight 0.0 \
    --epochs 20
```

---

## 🎯 Identity Weight (γ) - Add LAST!

**ONLY AFTER baseline is optimized!** This is the final tuning step.

### Quick Test Command

```bash
# After baseline is good, add identity preservation
for gamma in 0.05 0.1 0.2; do
    python train_model.py \
        --batch-size 32 \
        --learning-rate 0.0001 \
        --content-weight 1.0 \
        --style-weight 10.0 \
        --identity-weight $gamma \
        --epochs 20 \
        --checkpoint-dir checkpoints/identity_gamma_$gamma
done
```

### Decision Table

| Your Goal | Recommended γ | Expected Face Sim | Expected Perceptual |
|-----------|--------------|-------------------|---------------------|
| Optimized baseline | 0.0 | 0.476 | 0.499 |
| Subtle identity hints | 0.05 | ~0.48 | ~0.498 |
| **Balanced (best!)** ⭐ | **0.1** | **0.487** | **0.496** |
| More identity focus | 0.2 | ~0.50 | ~0.49 |
| Strong identity | 0.3 | ~0.51 | ~0.48 |

⚠️ **CRITICAL:** Never use γ ≥ 1.0 (causes loss function conflicts!)

### Why γ=1.0 Fails

```
Loss = 1.0×content + 10.0×style + γ×identity

γ=1.0: Identity fights equally with content → Confusion → Worse results!
γ=0.1: Identity is gentle constraint → Harmony → Best results!
```

---

## ⚡ Batch Size (GPU-Dependent)

**Find maximum batch size first to minimize training time!**

### GPU Guide

| Your GPU | VRAM | Test Batch Sizes | Expected Training Time |
|----------|------|------------------|------------------------|
| RTX 4090 | 24GB | Try: 64 → 48 → 32 | ~3-4 min |
| A6000 | 48GB | Try: 64 → 48 → 32 | ~3-4 min |
| RTX 3080 | 10GB | Try: 32 → 24 → 16 | ~5-7 min |
| RTX 3070 | 8GB | Try: 16 → 12 → 8 | ~8-12 min |

### Quick Test

```bash
# Test maximum batch size (1 epoch = ~12 seconds)
python train_model.py --batch-size 64 --epochs 1 --checkpoint-dir checkpoints/test_batch

# If OOM error, try 48, then 32, then 16...
```

**Impact:** 2x batch size ≈ 1.3x faster training (not 2x due to overhead)

---

## 📈 Learning Rate

**Only tune if default doesn't converge smoothly.**

### Quick Test (5 epochs each)

```bash
for lr in 0.00005 0.0001 0.0002; do
    python train_model.py \
        --batch-size 32 \
        --learning-rate $lr \
        --epochs 5 \
        --checkpoint-dir checkpoints/test_lr_$lr
done
```

### How to Evaluate

```bash
# Check training curves
cat checkpoints/test_lr_*/training_curves.csv

# Good: Smooth decrease in loss
# Bad: Oscillating loss (too high) or flat loss (too low)
```

### Decision Tree

| Observation | Diagnosis | Action |
|-------------|-----------|--------|
| Smooth loss decrease ✅ | Perfect! | Keep current LR |
| Loss oscillates/increases | LR too high | Try 5e-5 |
| Loss decreases very slowly | LR too low | Try 2e-4 |
| Loss plateaus early | May need more epochs | Try 30-40 epochs |

---

## ⚖️ Content/Style Weight Balance

**Only tune if γ=0.1 doesn't give desired effect!**

### Default (Recommended)

```bash
--content-weight 1.0 --style-weight 10.0  # 1:10 ratio
```

### Special Cases

```bash
# More photorealistic (less stylization)
--content-weight 2.0 --style-weight 10.0  # 1:5 ratio

# Very artistic (abstract styles)
--content-weight 1.0 --style-weight 20.0  # 1:20 ratio
```

**Key Principle:** Keep 1:10 ratio, adjust γ for identity instead!

---

## ⏰ Training Duration (Epochs)

| Epochs | Time (batch=32) | Quality | When to Use |
|--------|-----------------|---------|-------------|
| 1-2 | ~30s | Very low | Pipeline testing |
| 5 | ~1 min | Low | Quick experiments |
| 10 | ~2 min | Medium | Hyperparameter search |
| **20** ⭐ | **~4 min** | **Good** | **Standard** |
| 30 | ~6 min | Very good | Final model |
| 40-50 | ~8-10 min | Best | Publication quality |

**Tip:** Use `--save-interval 5` to save checkpoints and resume with `--resume` if needed.

---

## 🚀 Complete Workflow Example

```bash
# ============================================
# Phase 1: Find GPU-specific batch size (~1 min)
# ============================================
python train_model.py --batch-size 32 --epochs 1 --checkpoint-dir checkpoints/test

# ============================================
# Phase 2: Identity weight sweep (~20 min)
# ============================================
for gamma in 0.0 0.05 0.1 0.2 0.3; do
    python train_model.py \
        --batch-size 32 \
        --identity-weight $gamma \
        --epochs 20 \
        --checkpoint-dir checkpoints/gamma_$gamma \
        --save-interval 5
done

# ============================================
# Phase 3: Generate results (~3 min)
# ============================================
for gamma in 0.0 0.05 0.1 0.2 0.3; do
    python eval_inference.py \
        --content data/eval_content/ \
        --style data/style/ \
        --checkpoint checkpoints/gamma_$gamma/final_model.pth \
        --output results/gamma_$gamma/
done

# ============================================
# Phase 4: Create comparisons (~2 min)
# ============================================
for gamma in 0.05 0.1 0.2 0.3; do
    python result_visualize.py \
        --baseline-dir results/gamma_0.0 \
        --identity-dir results/gamma_$gamma \
        --output-dir results/comparison_gamma_$gamma \
        --baseline-checkpoint checkpoints/gamma_0.0 \
        --identity-checkpoint checkpoints/gamma_$gamma
done

# ============================================
# Phase 5: Review metrics
# ============================================
# Check comparison grids in results/comparison_gamma_*/
# Look for: Face Similarity > 0.80, Perceptual > 0.70
```

**Total time:** ~26 minutes for complete sweep!

---

## ✅ Quick Decision Guide

### "I want better identity preservation"

→ Increase γ: Try 0.2 or 0.3 (but not > 0.5!)

### "Identity is too strong, losing artistic style"

→ Decrease γ: Try 0.05 or 0.0

### "Training is too slow"

→ Increase batch size (test with `--epochs 1`)

### "Loss is oscillating"

→ Decrease learning rate to 5e-5

### "Training converges too slowly"

→ Increase learning rate to 2e-4 or train longer (30-40 epochs)

### "Results are photorealistic, not stylized enough"

→ Increase style weight to 15.0 or 20.0

### "Results are too abstract, losing structure"

→ Increase content weight to 2.0

---

## 📊 Expected Metrics by Configuration

| Config | Face Similarity | Perceptual | SSIM | Training Time | Use Case |
|--------|----------------|------------|------|---------------|----------|
| γ=0.0, 20ep | 0.476 | 0.499 | 0.359 | 4 min | Baseline |
| **γ=0.1, 20ep** ⭐ | **0.487** | **0.496** | **0.348** | **4 min** | **Recommended** |
| γ=0.2, 20ep | ~0.50 | ~0.49 | ~0.34 | 4 min | Strong identity |
| γ=0.3, 20ep | ~0.51 | ~0.48 | ~0.33 | 4 min | Very strong |
| γ=0.1, 40ep | ~0.49 | ~0.50 | ~0.35 | 8 min | Best quality |

**Target metrics for good model:**
- Face Similarity: > 0.48 (identity preserved)
- Perceptual Similarity: > 0.49 (good stylization)
- SSIM: > 0.34 (structure maintained)

---

## ⚠️ Common Mistakes to Avoid

| ❌ Don't Do This | ✅ Do This Instead |
|------------------|-------------------|
| Use γ=1.0 or higher | Use γ ∈ [0.05, 0.3] |
| Change multiple hyperparameters at once | Change ONE at a time |
| Skip batch size optimization | Test batch size first! |
| Train for only 5-10 epochs | Use minimum 20 epochs |
| Ignore visual quality | Check both metrics AND images |
| Use γ to control stylization | Use content/style weights instead |

---

## 📝 Experiment Tracking Template

```bash
# Create experiment log
echo "Experiment: [NAME]
Date: $(date)
Goal: [WHAT ARE YOU TESTING?]
Hypothesis: [WHAT DO YOU EXPECT?]

Configuration:
- Batch size: [VALUE]
- Learning rate: [VALUE]
- Identity weight: [VALUE]
- Content weight: [VALUE]
- Style weight: [VALUE]
- Epochs: [VALUE]

Results:
- Face Similarity: [VALUE]
- Perceptual Similarity: [VALUE]
- SSIM: [VALUE]
- Training Time: [VALUE]

Observations:
[WHAT DID YOU NOTICE?]

Next Steps:
[WHAT TO TRY NEXT?]
" > experiments/experiment_$(date +%Y%m%d_%H%M%S).txt
```

---

**For complete details, see `README.md` (Experiment sections) and `REPORT.md` (Appendix C).**

