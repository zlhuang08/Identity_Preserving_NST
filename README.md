# Identity-Preserving Neural Style Transfer

**Identity-Preserving Neural Style Transfer**

A comprehensive approach to neural style transfer that preserves facial identity through three complementary methods: **Face-Aware AdaIN**, **Identity Loss**, and **Enhanced Eye-Specific Loss**. Achieves **83.1% face similarity** (+26.1% over baseline) while maintaining real-time performance (~0.03-0.13s per image).

---

## 🎯 Overview

This project extends AdaIN-based fast style transfer with three identity-preserving methods:

1. **Identity Loss (γ=1000):** Global constraint using FaceNet embeddings → **78.7% face similarity** (+21.7%)
2. **Enhanced Eye Loss (β=1):** Fine-grained LPIPS + color preservation on eye regions → **82.4% face similarity** (+25.4%)
3. **Face-Aware AdaIN (α=0.3):** Regional adaptive normalization for face vs. background → **81.0% face similarity** (+24.0%)
4. **All Combined:** Best result → **83.1% face similarity** (+26.1%) 🏆

### Key Features

- ⚡ **Real-time:** ~0.03-0.13s per 512×512 image (300× faster than optimization-based methods)
- 🎭 **Superior identity preservation:** Up to +26.1% face similarity improvement
- 🛡️ **Ethical dataset:** 100% synthetic faces from StyleGAN (no privacy concerns)
- 📊 **Rigorous evaluation:** 840 test pairs (40 faces × 21 styles)
- 🔁 **Reproducible:** Fixed random seeds and comprehensive documentation

### Recommended Model

**Identity + Eye Loss (82.4%)** provides the best balance:
- ✅ Better performance than Face-Aware + Identity (82.4% vs 81.0%)
- ✅ Lower computational cost (no face masking required)
- ✅ Unified aesthetic (no visual disconnect)
- ✅ 99% of maximum performance with reduced complexity

---

## 📂 Project Structure

```
Identity_Preserving_NST/
├── README.md                           # Project overview (this file)
├── REPORT.md                           # Comprehensive technical report
├── LICENSE                             # MIT License
│
├── model_adain.py                      # AdaIN architecture with face-aware support
├── model_face_utils.py                 # Face detection, identity loss, eye loss
│
├── train_model.py                      # Main training script
├── eval_inference.py                   # Inference script for style transfer
│
├── data/                               # Datasets
│   ├── content/                        # 200 synthetic faces (training)
│   ├── content_splits/                 # Train/val/test split files
│   ├── eval_content/                   # 2 representative faces for visualization
│   └── style/                          # 21 art style images
│
├── checkpoints/                        # Trained models (~205 MB)
│   ├── 0_baseline/                     # AdaIN only (57.0%)
│   ├── 1_identity/                     # + Identity loss (78.7%)
│   ├── 2_identity_plus_eye/            # + Eye loss (82.4%) ⭐
│   ├── 2_face_aware_plus_identity/     # Face-aware + identity (81.0%)
│   ├── 3_all_combined/                 # All three methods (83.1%)
│   └── hyperparameter_tuning/          # Tuning experiments (CSV only)
│
└── results/                            # Visualizations and analysis
    ├── model_progression/comparisons/  # 42 comparison grids (2×3 layouts)
    └── hyperparameter_tuning/          # 4 tuning plots + ablation studies
```

---

## 🚀 Quick Start

### 1. Environment Setup

**Requirements:**
- Python 3.8+
- PyTorch 1.10+
- CUDA 11.3+ (for GPU acceleration)
- 16GB+ RAM recommended

**Installation:**

```bash
# Clone repository
git clone <repository-url>
cd Identity_Preserving_NST

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install facenet-pytorch lpips pillow matplotlib pandas scipy

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### 2. Inference (Using Pre-trained Models)

**Option A: Best Balanced Model (Recommended)**

```bash
# Identity + Eye Loss (82.4% face similarity)
python eval_inference.py \
    --checkpoint checkpoints/2_identity_plus_eye/final_model.pth \
    --content data/eval_content/face_00016.jpg \
    --style data/style/starry_night.jpg \
    --output results/my_output.jpg
```

**Option B: Maximum Identity Preservation**

```bash
# All Three Combined (83.1% face similarity)
python eval_inference.py \
    --checkpoint checkpoints/3_all_combined/final_model.pth \
    --content data/eval_content/face_00016.jpg \
    --style data/style/starry_night.jpg \
    --output results/my_output.jpg \
    --use-face-aware-adain  # Required for face-aware models
```

**Batch Processing:**

```bash
# Process all test faces with all styles
python eval_inference.py \
    --checkpoint checkpoints/2_identity_plus_eye/final_model.pth \
    --content data/eval_content \
    --style data/style \
    --output results/batch_output
```

### 3. Training from Scratch

**Train Identity + Eye Loss Model (Recommended):**

```bash
python train_model.py \
    --content-dir data/content \
    --style-dir data/style \
    --checkpoint-dir checkpoints/my_model \
    --batch-size 64 \
    --learning-rate 0.0001 \
    --epochs 20 \
    --content-weight 1.0 \
    --style-weight 10.0 \
    --identity-weight 1000.0 \
    --eye-weight 1.0 \
    --image-size 256
```

**Train All Three Combined:**

```bash
python train_model.py \
    --content-dir data/content \
    --style-dir data/style \
    --checkpoint-dir checkpoints/my_all_combined \
    --batch-size 64 \
    --learning-rate 0.0001 \
    --epochs 20 \
    --content-weight 1.0 \
    --style-weight 10.0 \
    --identity-weight 1000.0 \
    --eye-weight 1.0 \
    --use-face-aware-adain \
    --face-preservation-alpha 0.3 \
    --image-size 256
```

**Training time:** ~4-6 hours on NVIDIA A6000 GPU

---

## 📊 Results Summary

### Final Test Metrics (40 faces × 21 styles = 840 pairs)

| Model | Face Similarity | Improvement | Key Innovation |
|-------|----------------|-------------|----------------|
| Baseline (AdaIN only) | 57.0% | — | Fast style transfer |
| Identity Loss (γ=1000) | 78.7% | +21.7% | FaceNet embeddings |
| **Identity + Eye Loss** ⭐ | **82.4%** | **+25.4%** | LPIPS + color preservation |
| Face-Aware + Identity | 81.0% | +24.0% | Regional adaptive normalization |
| All Three Combined | 83.1% | +26.1% 🏆 | Maximum identity preservation |

### Key Innovations

**1. Enhanced Eye-Specific Loss (β=1)**

Our eye loss uniquely combines five components for perceptually natural eyes:

```python
Eye Loss = λ₁·VGG_relu2_1    # Structure (λ=4.0)
         + λ₂·VGG_relu3_1    # Pattern (λ=1.0)  
         + λ₃·Sobel_edges    # Sharpness (λ=2.0)
         + λ₄·LPIPS          # Perceptual beauty (λ=0.8)
         + λ₅·Color_stats    # Natural iris color (λ=0.5)
```

**Benefits:**
- No "mosaic" or "swollen" artifacts (fixed via MTCNN landmark alignment)
- Crisp, clear iris boundaries (Sobel edge loss)
- Natural eye colors preserved (color statistics matching)
- Perceptually realistic (LPIPS loss prioritizes human perception)

**2. Pareto-Optimal Identity Weight (γ=1000)**

Through testing 8 orders of magnitude (γ ∈ [0, 0.1, 1, 10, 100, 1000, 10000, 100000]), we discovered:
- Face similarity **monotonically increases** with γ
- Style quality **monotonically decreases** with γ
- γ=1000 achieves optimal **Pareto trade-off** (78.7% similarity, minimal style degradation)
- Critical finding: Identity loss must be **3-15% of total loss** to be effective

**3. Face-Aware AdaIN (α=0.3)**

Regional control prevents face over-stylization but creates aesthetic trade-offs:
- ✅ Highest single-method improvement (+24.0%)
- ⚠️ Visual disconnect between realistic face and stylized background
- **Recommendation:** Reserve for applications requiring maximum identity (e.g., character consistency in children's books)

---

## 📈 Model Selection Guide

**Choose your model based on your application:**

| Application | Recommended Model | Face Similarity | Rationale |
|-------------|------------------|-----------------|-----------|
| **General portraits** | Identity + Eye | 82.4% | Best balance of quality, speed, and aesthetic unity |
| **Children's books** | All Combined | 83.1% | Character consistency more important than unified style |
| **Fine art galleries** | Baseline | 57.0% | Artistic expression prioritized over identity |
| **Fast prototyping** | Identity only | 78.7% | Lowest computational cost, strong results |
| **Research/maximum quality** | All Combined | 83.1% | State-of-the-art identity preservation |

---

## 🔬 Technical Highlights

### Optimal Hyperparameters (Discovered via Comprehensive Ablation)

- **Learning rate:** 1e-4 (best convergence vs. speed trade-off)
- **Content:Style ratio:** 1:10 (optimal balance after testing 1:1, 1:5, 1:10, 1:20, 1:50)
- **Identity weight (γ):** 1000 (Pareto-optimal after testing 8 orders of magnitude)
- **Eye weight (β):** 1 (optimal from non-monotonic tuning: tested 0.1, 1, 10, 100)
- **Face-aware alpha (α):** 0.3 (30% content preservation in face regions)
- **Batch size:** 64 (maximizes GPU utilization on A6000)

### Dataset

- **Content:** 200 synthetic faces from StyleGAN (ThisPersonDoesNotExist.com)
- **Styles:** 21 diverse art styles (Van Gogh, Monet, Hokusai, children's book illustrations)
- **Split:** 70% train (140) / 15% val (30) / 15% test (40)
- **Ethics:** 100% synthetic data eliminates privacy concerns

### Evaluation Metrics

1. **Face Similarity:** Cosine similarity of FaceNet embeddings (primary metric)
2. **Perceptual Similarity:** LPIPS distance (human perception alignment)
3. **SSIM:** Structural similarity (pixel-level fidelity)
4. **Face Detection Rate:** % of faces successfully detected post-stylization
5. **Training Loss:** Content loss, style loss, identity loss, eye loss

---

## 📖 Documentation

- **`README.md`** (this file): Quick start, key results, and usage examples
- **`REPORT.md`**: Comprehensive technical report with:
  - Detailed methodology and loss formulations
  - Complete ablation studies and hyperparameter tuning
  - Visual comparisons and analysis
  - Limitations and future work
  - Full experimental results

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Original AdaIN Paper:** Huang & Belongie (2017) - "Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization"
- **FaceNet:** Schroff et al. (2015) - Face recognition embeddings for identity loss
- **LPIPS:** Zhang et al. (2018) - Perceptual similarity metric
- **Datasets:**
  - ThisPersonDoesNotExist.com (StyleGAN) for synthetic faces
  - Wikimedia Commons for public domain artwork

---

**For detailed technical documentation, experimental results, and comprehensive ablation studies, see [REPORT.md](REPORT.md).**
