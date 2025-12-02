# Data Directory

This directory contains all datasets used for training, validation, testing, and evaluation of the Identity-Preserving Fast Style Transfer project.

---

## 📁 Directory Structure

```
data/
├── README.md                    # This file
├── DATASET_SUMMARY.md           # Detailed dataset documentation
├── content/                     # Training content images (200 synthetic faces)
├── eval_content/                # Evaluation content images (2 test faces)
├── style/                       # Style images (21 art styles)
└── content_splits/              # Train/val/test split indices
    ├── train.txt                # 120 images (60%)
    ├── val.txt                  # 40 images (20%)
    └── test.txt                 # 40 images (20%)
```

---

## 📊 Dataset Summary

| Category | Count | Size | Purpose |
|----------|-------|------|---------|
| **Content Images** | 200 | ~100 MB | Training/validation/testing |
| **Eval Content Images** | 2 | ~1 MB | Final evaluation & visualization |
| **Style Images** | 21 | ~30 MB | Style reference for all training |
| **Split Files** | 3 | ~2 KB | Train/val/test indices |
| **Total** | 223 files | ~131 MB | Complete dataset |

---

## 🖼️ Content Images (200 faces)

### **content/**
- **Count:** 200 synthetic face images
- **Format:** JPG, 1024×1024 pixels
- **Source:** ThisPersonDoesNotExist.com (StyleGAN2)
- **Naming:** `face_00001.jpg` to `face_00200.jpg`
- **Ethics:** 100% AI-generated, no real people, no privacy concerns
- **Diversity:** Various ages, genders, ethnicities, expressions
- **Purpose:** Training, validation, and testing the neural style transfer model

### **Split Distribution:**
The 200 content images are split into three sets (using index files in `content_splits/`):

| Split | Count | Percentage | Purpose |
|-------|-------|------------|---------|
| **Train** | 120 images | 60% | Model training |
| **Validation** | 40 images | 20% | Hyperparameter tuning & early stopping |
| **Test** | 40 images | 20% | Final performance evaluation |

**Split Method:** 
- Random split with fixed seed (42) for reproducibility
- Ensures `face_00010.jpg` and `face_00066.jpg` are in test set
- No overlap between splits

**Training Pairs per Epoch:**
- Train: 120 content × 21 styles = **2,520 pairs**
- Val: 40 content × 21 styles = **840 pairs**
- Test: 40 content × 21 styles = **840 pairs**

### **Generation Script:**
```bash
# Regenerate all 200 synthetic faces
python data_generate_faces.py \
    --output-dir data/content \
    --num-images 200

# Automatically creates/updates data/content_splits/ with 60/20/20 split
```

**Note:** Split files are automatically generated/updated after face generation.

---

## 🎨 Style Images (21 styles)

### **style/**
- **Count:** 21 art style images
- **Format:** JPG, various sizes (resized during training)
- **Source:** Public domain artworks from Wikimedia Commons
- **Purpose:** Style reference for neural style transfer

### **Style Categories:**

#### **1. Famous Masters (8 styles)**
Classic artworks from renowned artists:

| Style | Artist | Movement | File |
|-------|--------|----------|------|
| Starry Night | Vincent van Gogh | Post-Impressionism | `starry_night.jpg` |
| The Scream | Edvard Munch | Expressionism | `the_scream.jpg` |
| The Great Wave | Hokusai | Japanese Ukiyo-e | `great_wave.jpg` |
| The Kiss | Gustav Klimt | Art Nouveau | `the_kiss.jpg` |
| Water Lilies | Claude Monet | Impressionism | `water_lilies.jpg` |
| Impression Sunrise | Claude Monet | Impressionism | `impression_sunrise.jpg` |
| Sunday Afternoon | Georges Seurat | Pointillism | `sunday_afternoon.jpg` |
| Café Terrace | Vincent van Gogh | Post-Impressionism | `cafe_terrace.jpg` |

#### **2. Children's Book Illustrations (6 styles)**
Gentle, child-friendly artistic styles:

| Style | Artist | Era | File |
|-------|--------|-----|------|
| Peter Rabbit | Beatrix Potter | 1902 | `potter_peter_rabbit.jpg` |
| Audubon Flamingo | John James Audubon | 1838 | `audubon_flamingo.jpg` |
| Dürer's Hare | Albrecht Dürer | 1502 | `durer_hare.jpg` |
| Kate Greenaway Christmas | Kate Greenaway | 1880s | `greenaway_christmas.jpg` |
| Homer Beach Scene | Winslow Homer | 1870s | `homer_beach.jpg` |
| Homer Boys with Kitten | Winslow Homer | 1873 | `homer_boys_kitten.jpg` |

#### **3. Textures (4 styles)**
Natural and abstract textures:

| Style | Type | File |
|-------|------|------|
| Sandstone | Natural texture | `sandstone.jpg` |
| Stone | Natural texture | `stone_style.jpg` |
| Drop of Water | Macro texture | `drop-of-water.jpg` |
| Monet (general) | Impressionist texture | `claude-monet.jpg` |

#### **4. Additional Styles (3 styles)**
| Style | Type | File |
|-------|------|------|
| Sunflowers | Van Gogh | `sunflowers.jpg` |
| Van Gogh Irises | Van Gogh | `van_gogh_irises.jpg` |
| Homer Girl on Swing | Winslow Homer | `homer_girl_swing.jpg` |

### **Download Script:**
```bash
# Download all 21 verified style images
python data_download_styles.py \
    --output-dir data/style
```

**Note:** All style images are verified public domain works with working URLs.

---

## 👤 Evaluation Content Images (2 faces)

### **eval_content/**
- **Count:** 2 carefully selected faces for final evaluation
- **Format:** JPG, 1024×1024 pixels
- **Purpose:** Generate comparison grids and visualizations for reports

| File | Description | Gender | Included in Test Set |
|------|-------------|--------|----------------------|
| `face_00016.jpg` | Young adult with clear features | Female | ❌ No |
| `face_00066.jpg` | Young adult with natural lighting | Male | ✅ Yes |

**Why these two?**
- **Diverse representation:** Different demographics and features
- **Good facial features:** Clear eyes, well-defined features, ideal for identity preservation
- **Consistent quality:** Both are high-quality synthetic faces
- **Test set coverage:** face_00066 is in test set (ensures consistency with quantitative metrics)
- **Report-ready:** Used for all comparison grids in final report

**Usage:**
```bash
# Generate stylized versions using all 4 models
python eval_inference.py \
    --content data/eval_content/ \
    --style data/style/ \
    --output results/my_output/ \
    --checkpoint checkpoints/3_all_combined/final_model.pth \
    --use-face-aware-adain
```

**Output:** 2 faces × 21 styles × 4 models = **168 stylized images** for comparison

---

## 📑 Content Split Files

### **content_splits/**
Index files that specify which content images belong to train/val/test sets.

| File | Count | Format | Purpose |
|------|-------|--------|---------|
| `train.txt` | 120 lines | One filename per line | Training set indices |
| `val.txt` | 40 lines | One filename per line | Validation set indices |
| `test.txt` | 40 lines | One filename per line | Test set indices |

### **File Format:**
```
face_00001.jpg
face_00003.jpg
face_00005.jpg
...
```

**Split Ratios:** 60% train / 20% val / 20% test (following CS230 guidelines)

**Special Requirement:** `face_00010.jpg` and `face_00066.jpg` are guaranteed to be in `test.txt`

### **How Splits Are Used:**

**Training (`train_model.py`):**
```python
# Automatically loads splits and creates datasets
train_dataset = load_dataset('data/content', 'data/content_splits/train.txt')
val_dataset = load_dataset('data/content', 'data/content_splits/val.txt')

# Training loop uses exhaustive pairing:
# - 120 train images × 21 styles = 2,520 pairs per epoch
# - 40 val images × 21 styles = 840 pairs per validation
```

**Reproducibility:**
- Splits are fixed after initial generation
- Same split used for all experiments
- Ensures fair comparison between models

---

## 🔬 Dataset Statistics

### **Content Images (Synthetic Faces)**
- **Total:** 200 faces
- **Resolution:** 1024×1024 pixels
- **Format:** JPEG (RGB)
- **File size:** ~500 KB per image
- **Generation method:** StyleGAN2 (ThisPersonDoesNotExist.com)
- **Diversity metrics:**
  - Age range: Children to elderly
  - Gender: Mixed (not manually balanced)
  - Ethnicity: Diverse (inherent from StyleGAN training)
  - Expression: Various (smiling, neutral, serious)

### **Style Images**
- **Total:** 21 styles
- **Resolution:** Variable (512×512 to 2048×2048)
- **Format:** JPEG (RGB)
- **Average file size:** ~1.5 MB per image
- **Style diversity:**
  - 8 famous masterpieces (Impressionism, Expressionism, etc.)
  - 6 children's book illustrations
  - 4 natural/abstract textures
  - 3 additional artistic styles

### **Training Combinations**
- **Total unique pairs:** 200 content × 21 styles = **4,200 combinations**
- **Per epoch:**
  - Train: 2,520 pairs
  - Validation: 840 pairs
- **Per training run (15 epochs):**
  - Total training pairs processed: 2,520 × 15 = **37,800 pairs**
  - Total validation pairs processed: 840 × 15 = **12,600 pairs**

---

## 📥 Data Acquisition

### **Content Images (Synthetic Faces)**

**Option 1: Generate New Faces (Recommended)**
```bash
# Generate 200 fresh synthetic faces
python data_generate_faces.py \
    --output-dir data/content \
    --num-images 200

# Takes ~3-5 minutes
# Automatically creates splits in data/content_splits/
```

**Option 2: Use Existing Faces**
- Faces are already generated and included in the repository
- Ready to use for training immediately
- No additional download required

### **Style Images**

**Download All Styles:**
```bash
# Download all 21 verified public domain style images
python data_download_styles.py \
    --output-dir data/style

# Takes ~30 seconds
# All URLs verified and working (as of November 2025)
```

**Manual Download:**
- All style images are from Wikimedia Commons
- Public domain (no copyright restrictions)
- URLs listed in `data_download_styles.py`

---

## 🛡️ Ethical Considerations

### **Synthetic Faces (ThisPersonDoesNotExist.com)**
✅ **Ethical Advantages:**
- No real people's photos → No privacy concerns
- No consent required → No legal issues
- 100% AI-generated → Fully reproducible
- Diverse representation → No bias from manual selection
- Shareable without restrictions → Open for research

⚠️ **Considerations:**
- StyleGAN inherits biases from training data (FFHQ dataset)
- Not suitable for production without additional bias testing
- For educational/research use only

### **Style Images (Public Domain Art)**
✅ **All Legal:**
- All images from Wikimedia Commons
- 100% public domain (copyright expired or CC0)
- No attribution required (though we provide artist names)
- Verified URLs and licenses
- Ethical to use for academic research

---

## 📊 Data Quality

### **Content Images Quality Checks**
✅ **Resolution:** All 1024×1024 pixels (consistent)
✅ **Format:** All JPEG RGB (no transparency issues)
✅ **Face Detection:** All images contain detectable faces (verified by MTCNN)
✅ **Diversity:** Good mix of ages, genders, ethnicities
✅ **Quality:** High-quality StyleGAN2 generation (no artifacts)

### **Style Images Quality Checks**
✅ **Resolution:** All > 512×512 pixels (sufficient for style extraction)
✅ **Format:** All JPEG RGB
✅ **Artistic Quality:** All from renowned artists or high-quality sources
✅ **Diversity:** Wide range of artistic movements and techniques
✅ **Child-Friendly:** Subset of 6 children's book illustrations included

---

## 🔄 Regenerating Data

### **Regenerate Content Images**
```bash
# Clear existing faces
rm data/content/*.jpg

# Generate new faces
python data_generate_faces.py \
    --output-dir data/content \
    --num-images 200

# Splits are automatically regenerated in data/content_splits/
```

### **Re-download Style Images**
```bash
# Clear existing styles
rm data/style/*.jpg

# Download all styles
python data_download_styles.py \
    --output-dir data/style
```

### **Recreate Splits**
```bash
# Splits are automatically created when generating faces
# Or manually regenerate:
python data_generate_faces.py \
    --output-dir data/content \
    --num-images 200  # Re-splits existing 200 images
```

---

## 📈 Data Usage Statistics

### **Training (20 epochs, typical run)**
- **Images loaded:** 2,520 (train) + 840 (val) = 3,360 per epoch
- **Total images processed:** 3,360 × 20 = **67,200 image pairs**
- **Training time:** ~10-15 minutes (batch size 64, A6000 GPU)
- **Data augmentation:** None (relies on diverse style pairing)

### **Evaluation (Final report)**
- **Content images:** 2 (face_00010, face_00066)
- **Style images:** 21 (all styles)
- **Models:** 4 (baseline, identity, face-aware, all-combined)
- **Total generated images:** 2 × 21 × 4 = **168 stylized images**
- **Comparison grids:** 2 × 21 = **42 grids** (2×3 layout)

---

## 🔍 Quick Reference Commands

```bash
# Count files
ls data/content/*.jpg | wc -l        # Should be 200
ls data/eval_content/*.jpg | wc -l   # Should be 2
ls data/style/*.jpg | wc -l          # Should be 21

# Check split counts
wc -l data/content_splits/*.txt
# train.txt: 120 lines
# val.txt: 40 lines
# test.txt: 40 lines

# Verify eval faces are in test set
grep -E "face_00010|face_00066" data/content_splits/test.txt

# Check file sizes
du -sh data/content         # ~100 MB
du -sh data/style           # ~30 MB
du -sh data/eval_content    # ~1 MB
du -sh data/                # ~131 MB total

# List all style filenames
ls data/style/*.jpg | xargs -n 1 basename
```

---

## 📚 Related Documentation

- **Main README:** `../README.md` - Project overview and usage
- **Dataset Details:** `DATASET_SUMMARY.md` - Extended dataset documentation
- **Generation Script:** `../data_generate_faces.py` - Face generation code
- **Download Script:** `../data_download_styles.py` - Style download code
- **Training Guide:** `../README.md` (Experiment section) - How to use this data

---

## 🎯 For New Users

**Quickstart checklist:**

1. ✅ **Verify data exists:**
   ```bash
   ls data/content/*.jpg | wc -l      # Should output: 200
   ls data/style/*.jpg | wc -l        # Should output: 21
   ```

2. ✅ **If data missing, generate/download:**
   ```bash
   python data_generate_faces.py --output-dir data/content --num-images 200
   python data_download_styles.py --output-dir data/style
   ```

3. ✅ **Verify splits exist:**
   ```bash
   ls data/content_splits/*.txt      # Should list: train.txt, val.txt, test.txt
   ```

4. ✅ **Ready to train:**
   ```bash
   python train_model.py \
       --content-dir data/content \
       --style-dir data/style \
       --checkpoint-dir checkpoints/my_model \
       --epochs 20
   ```

---

**Last Updated:** November 29, 2025  
**Total Dataset Size:** ~131 MB  
**Status:** Complete and ready for training/evaluation

