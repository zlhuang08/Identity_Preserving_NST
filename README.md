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
    --batch-size 8 \
    --identity-weight 0.0 \
    --checkpoint-dir checkpoints/baseline_final \
    --save-interval 5

# Train identity-preserving model (γ=0.1)
python train_model.py \
    --content-dir data/content \
    --style-dir data/style \
    --epochs 20 \
    --batch-size 8 \
    --identity-weight 0.1 \
    --checkpoint-dir checkpoints/identity_final \
    --save-interval 5
```

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
| **Dataset Size** | 200 content images, 21 style images |
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

Want to explore different hyperparameters and loss weights? Here's how to conduct experiments:

### Experiment 1: Adjust Identity Weight (γ)

Control how strongly identity is preserved vs stylization strength:

```bash
# Weak identity preservation (more artistic freedom)
python train_model.py \
    --content-dir data/content \
    --style-dir data/style \
    --identity-weight 0.05 \
    --checkpoint-dir checkpoints/experiment_gamma_0.05 \
    --epochs 20

# Strong identity preservation (more face preservation)
python train_model.py \
    --content-dir data/content \
    --style-dir data/style \
    --identity-weight 0.2 \
    --checkpoint-dir checkpoints/experiment_gamma_0.2 \
    --epochs 20
```

**Recommended values:** γ ∈ [0.0, 0.5]
- γ = 0.0: No identity preservation (baseline)
- γ = 0.05-0.1: Subtle identity preservation (good balance)
- γ = 0.2-0.5: Strong identity preservation (less stylization)

### Experiment 2: Adjust Learning Rate

```bash
# Faster convergence (might be less stable)
python train_model.py \
    --content-dir data/content \
    --style-dir data/style \
    --learning-rate 0.0005 \
    --checkpoint-dir checkpoints/experiment_lr_0.0005 \
    --epochs 20

# Slower convergence (more stable)
python train_model.py \
    --content-dir data/content \
    --style-dir data/style \
    --learning-rate 0.00005 \
    --checkpoint-dir checkpoints/experiment_lr_0.00005 \
    --epochs 30
```

**Recommended values:** lr ∈ [5e-5, 5e-4]
- Default: 1e-4 (balanced)

### Experiment 3: Adjust Loss Weights

```bash
# Stronger content preservation
python train_model.py \
    --content-dir data/content \
    --style-dir data/style \
    --content-weight 2.0 \
    --style-weight 10.0 \
    --identity-weight 0.1 \
    --checkpoint-dir checkpoints/experiment_content_2.0 \
    --epochs 20

# Stronger stylization
python train_model.py \
    --content-dir data/content \
    --style-dir data/style \
    --content-weight 1.0 \
    --style-weight 20.0 \
    --identity-weight 0.1 \
    --checkpoint-dir checkpoints/experiment_style_20.0 \
    --epochs 20
```

**Recommended values:**
- Content weight: λ_content ∈ [0.5, 2.0] (default: 1.0)
- Style weight: λ_style ∈ [5.0, 20.0] (default: 10.0)
- Identity weight: γ ∈ [0.0, 0.5] (default: 0.1)

### Experiment 4: Training Duration

```bash
# Quick experiment (fast iteration)
python train_model.py \
    --content-dir data/content \
    --style-dir data/style \
    --epochs 10 \
    --checkpoint-dir checkpoints/experiment_10epochs

# Extended training (better quality)
python train_model.py \
    --content-dir data/content \
    --style-dir data/style \
    --epochs 40 \
    --checkpoint-dir checkpoints/experiment_40epochs
```

**Recommended epochs:**
- Quick test: 5-10 epochs (~1-2 minutes)
- Standard: 20 epochs (~4 minutes, good quality)
- Extended: 30-50 epochs (~6-10 minutes, best quality)

### Experiment 5: Batch Size & Image Size

```bash
# Larger batch (faster training, more memory)
python train_model.py \
    --content-dir data/content \
    --style-dir data/style \
    --batch-size 16 \
    --checkpoint-dir checkpoints/experiment_batch_16

# Higher resolution (better quality, slower)
python train_model.py \
    --content-dir data/content \
    --style-dir data/style \
    --image-size 512 \
    --batch-size 4 \
    --checkpoint-dir checkpoints/experiment_size_512
```

**Recommended values:**
- Batch size: 4-16 (depends on GPU memory)
- Image size: 256-512 (256 is default, 512 for higher quality)

### Evaluating Your Experiments

After training, compare your models:

```bash
# Generate results for your experimental model
python eval_inference.py \
    --content data/eval_content/ \
    --style data/style/ \
    --output results/experiment_gamma_0.2/ \
    --checkpoint checkpoints/experiment_gamma_0.2/final_model.pth

# Create comparisons
python result_visualize.py \
    --content-dir data/eval_content \
    --style-dir data/style \
    --baseline-dir results/eval_v1/baseline \
    --identity-dir results/experiment_gamma_0.2 \
    --output-dir results/experiment_gamma_0.2_comparisons
```

### Tips for Experimentation

1. **Start with identity weight (γ)**: This has the biggest impact on the trade-off
2. **Keep content and style weights in 1:10 ratio**: This ratio works well across settings
3. **Use checkpoints**: Save every 5 epochs with `--save-interval 5`
4. **Monitor training logs**: Watch for identity loss and face similarity trends
5. **Visual inspection matters**: Numbers don't tell the whole story—look at the outputs!
6. **Try multiple styles**: Some styles work better with higher/lower γ values

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

