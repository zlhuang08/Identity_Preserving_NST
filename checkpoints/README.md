# Checkpoints Directory

This directory contains all trained models and hyperparameter tuning results.

---

## Main Models (Final Trained Models)

These are the 5 production models trained on the full dataset (120 train, 40 val, 40 test) for 20 epochs:

### `0_baseline/`
- **Description:** Standard AdaIN without identity preservation
- **Hyperparameters:** γ=0 (no identity loss)
- **Performance:** 54.0% face similarity (baseline)
- **Files:**
  - `final_model.pth` - Final trained model weights
  - `training_curves.csv` - Complete training/val/test metrics per epoch

### `1_identity/`
- **Description:** AdaIN with identity loss
- **Hyperparameters:** γ=1000 (optimal identity weight)
- **Performance:** 78.7% test face similarity (+21.7% vs baseline)
- **Files:**
  - `final_model.pth` - Final trained model weights
  - `training_curves.csv` - Complete training/val/test metrics per epoch

### `2_identity_plus_eye/`
- **Description:** AdaIN with identity loss + enhanced eye-specific loss (LPIPS + Color Preservation)
- **Hyperparameters:** γ=1000 (identity weight), β=1 (eye weight)
- **Components:** Multi-scale VGG, Sobel edge, LPIPS perceptual, color statistics
- **Performance:** 82.1% test face similarity (+28.1% vs baseline)
- **Files:**
  - `final_model.pth` - Final trained model weights
  - `training_curves.csv` - Complete training/val/test metrics per epoch

### `2_face_aware_plus_identity/`
- **Description:** Face-aware AdaIN with identity loss
- **Hyperparameters:** α=0.3 (face preservation), γ=1000 (identity weight)
- **Performance:** 81.0% test face similarity (+24.0% vs baseline)
- **Files:**
  - `final_model.pth` - Final trained model weights
  - `training_curves.csv` - Complete training/val/test metrics per epoch

### `3_all_combined/`
- **Description:** Face-aware AdaIN + identity loss + enhanced eye-specific loss (all three combined)
- **Hyperparameters:** α=0.3 (face preservation), γ=1000, β=1
- **Components:** Face-aware regional control + Identity embedding + Enhanced eye loss (LPIPS+Color)
- **Performance:** 83.3% test face similarity (+29.3% vs baseline) 🏆 **HIGHEST IDENTITY**
- **Files:**
  - `final_model.pth` - Final trained model weights
  - `training_curves.csv` - Complete training/val/test metrics per epoch

---

## Hyperparameter Tuning Results

This directory contains archived results from systematic hyperparameter tuning experiments. Only `training_curves.csv` files are kept (full checkpoints removed to save space).

### `hyperparameter_tuning/learning_rate/`
**Experiment:** Learning rate sweep (5 values)
- `lr_0.00001/` - 1e-5 (too slow)
- `lr_0.00003/` - 3e-5 (slow but stable)
- `lr_0.0001/` - 1e-4 ⭐ **OPTIMAL** (balanced)
- `lr_0.0003/` - 3e-4 (slightly unstable)
- `lr_0.001/` - 1e-3 (diverges)

**Result:** LR=1e-4 achieves best validation loss (31.94) with stable convergence

### `hyperparameter_tuning/content_style_weight/`
**Experiment:** Content:Style weight ratio sweep (5 ratios)
- `weights_1_1/` - 1:1 (weak style)
- `weights_1_5/` - 1:5 (subtle)
- `weights_1_10/` - 1:10 ⭐ **OPTIMAL** (balanced)
- `weights_1_20/` - 1:20 (strong style)
- `weights_1_50/` - 1:50 (maximum style, poor structure)

**Result:** 1:10 ratio sits at "knee" of Pareto curve - best trade-off

### `hyperparameter_tuning/identity_weight/`
**Experiment:** Identity weight (γ) sweep (8 orders of magnitude)
- `gamma_0_00/` - γ=0 (baseline, 54.0% face sim)
- `gamma_0_10/` - γ=0.1 (worse than baseline, "noise region")
- `gamma_1_00/` - γ=1 (still worse, "noise region")
- `gamma_10_00/` - γ=10 (still noise)
- `gamma_100_00/` - γ=100 (starting to work, 73.2%)
- `gamma_1000_00/` - γ=1000 ⭐ **OPTIMAL** (78.5% face sim)
- `gamma_10000_00/` - γ=10,000 (degraded, style collapse)
- `gamma_100000_00/` - γ=100,000 (collapsed)

**Result:** γ=1000 is Pareto optimal - achieves strong face similarity (76.8%, +21% vs baseline) with minimal style degradation (5% increase in style loss). Higher values improve face similarity further but cause style collapse.

### `hyperparameter_tuning/eye_weight/`
**Experiment:** Eye-specific loss weight (β) sweep with enhanced LPIPS+Color eye loss (4 values)
- `beta_0_1/` - β=0.1 (too weak, 80.6% face sim)
- `beta_1/` - β=1 ⭐ **OPTIMAL** (82.4% face sim, best balance)
- `beta_10/` - β=10 (diminishing returns, 83.4% but 2x loss)
- `beta_100/` - β=100 (over-constrained, 80.8%, training failure)

**Result:** β=1 achieves excellent face similarity (82.4%) with stable training. β=10 gains only +1% but nearly doubles validation loss (60.78 vs 33.79). β=100 causes severe over-constraint.

---

## File Structure

```
checkpoints/
├── 0_baseline/
│   ├── final_model.pth              # 14 MB (decoder weights)
│   └── training_curves.csv          # 22 rows × 22 columns
├── 1_identity/
│   ├── final_model.pth
│   └── training_curves.csv
├── 2_identity_plus_eye/
│   ├── final_model.pth
│   └── training_curves.csv
├── 2_face_aware_plus_identity/
│   ├── final_model.pth
│   └── training_curves.csv
├── 3_all_combined/
│   ├── final_model.pth
│   └── training_curves.csv
├── hyperparameter_tuning/
│   ├── learning_rate/
│   │   ├── lr_0.00001/training_curves.csv
│   │   ├── lr_0.00003/training_curves.csv
│   │   ├── lr_0.0001/training_curves.csv
│   │   ├── lr_0.0003/training_curves.csv
│   │   └── lr_0.001/training_curves.csv
│   ├── content_style_weight/
│   │   ├── weights_1_1/training_curves.csv
│   │   ├── weights_1_5/training_curves.csv
│   │   ├── weights_1_10/training_curves.csv
│   │   ├── weights_1_20/training_curves.csv
│   │   └── weights_1_50/training_curves.csv
│   ├── identity_weight/
│   │   ├── gamma_0_00/training_curves.csv
│   │   ├── gamma_0_10/training_curves.csv
│   │   ├── gamma_1_00/training_curves.csv
│   │   ├── gamma_10_00/training_curves.csv
│   │   ├── gamma_100_00/training_curves.csv
│   │   ├── gamma_1000_00/training_curves.csv
│   │   ├── gamma_10000_00/training_curves.csv
│   │   └── gamma_100000_00/training_curves.csv
│   └── eye_weight/
│       ├── beta_0_1/training_curves.csv
│       ├── beta_1/training_curves.csv
│       ├── beta_10/training_curves.csv
│       └── beta_100/training_curves.csv
└── README.md                        # This file
```

**Note:** Hyperparameter tuning directories contain ONLY `training_curves.csv` files. Model checkpoints (`final_model.pth`) have been removed to save disk space (~123 MB freed).

---

## Training Curves CSV Format

Each `training_curves.csv` contains per-epoch metrics with 22 columns:

```csv
epoch,train_loss,train_content,train_style,train_identity,train_eye,
val_loss,val_content,val_style,val_identity,val_eye,
test_loss,test_content,test_style,test_identity,test_eye,
train_similarity,val_similarity,test_similarity,
train_detect_rate,val_detect_rate,test_detect_rate
```

**Key Columns:**
- `*_similarity`: Face similarity (0-1, higher is better)
- `*_detect_rate`: Face detection rate (0-1, higher is better)
- `*_loss`: Respective loss values
- Metrics are reported for train/val/test splits at each epoch

---

## Usage

### Loading a Model

```python
from model_adain import AdaINStyleTransfer
import torch

# Load the best model (all three methods combined)
model = AdaINStyleTransfer().cuda()
checkpoint = torch.load('checkpoints/3_all_combined/final_model.pth')
model.decoder.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Use for inference
with torch.no_grad():
    stylized = model(content_img, style_img)
```

### Loading Training Curves

```python
import pandas as pd

# Load training history
df = pd.read_csv('checkpoints/3_all_combined/training_curves.csv')

# Plot face similarity over epochs
import matplotlib.pyplot as plt
plt.plot(df['epoch'], df['train_similarity'], label='Train')
plt.plot(df['epoch'], df['val_similarity'], label='Val')
plt.plot(df['epoch'], df['test_similarity'], label='Test')
plt.legend()
plt.show()
```

---

## Disk Space

- **Main Models:** ~205 MB (5 models × 41 MB each)
- **Training Curves:** ~0.5 MB (all CSV files)
- **Total:** ~205 MB

**Space Optimization:**
- Removed 6 OLD/test model versions (~205 MB freed)
- Removed hyperparameter tuning model checkpoints (~123 MB freed)
- Total space freed: ~328 MB

---

## Notes

1. **Only final models kept:** All OLD/test versions removed; only 5 production models remain
2. **Hyperparameter tuning:** Model checkpoints removed, CSV files preserved for analysis
3. **Reproducible:** All models trained with seed=42 for deterministic results
4. **Dataset:** 200 synthetic faces (120 train, 40 val, 40 test) × 21 styles
5. **Hardware:** NVIDIA RTX 6000 Ada Generation (48GB VRAM)
6. **Training Time:** ~50-60 minutes per model (20 epochs)
7. **Enhanced Eye Loss:** Models 2 & 3 use LPIPS + Color Preservation for superior eye quality

---

**Last Updated:** December 2, 2025

