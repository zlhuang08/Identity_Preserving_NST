# Dataset Summary

## 📊 Overview

**Content Images:** 200 synthetic faces (AI-generated, ethical)  
**Style Images:** 13 famous artworks (public domain)  
**Total Possible Combinations:** 2,600 stylized images

---

## 📁 Directory Structure

```
data/
├── content/              # 200 synthetic face images
│   ├── face_00000.jpg
│   ├── face_00001.jpg
│   ├── ...
│   └── face_00199.jpg
│
├── style/                # 13 public domain artworks
│   ├── cafe_terrace.jpg            (Van Gogh)
│   ├── claude-monet.jpg            (Monet - original)
│   ├── drop-of-water.jpg           (Original texture)
│   ├── great_wave.jpg              (Hokusai)
│   ├── impression_sunrise.jpg      (Monet)
│   ├── sandstone.jpg               (Original texture)
│   ├── starry_night.jpg            (Van Gogh)
│   ├── stone_style.jpg             (Original texture)
│   ├── sunday_afternoon.jpg        (Seurat)
│   ├── sunflowers.jpg              (Van Gogh)
│   ├── the_kiss.jpg                (Klimt)
│   ├── the_scream.jpg              (Munch)
│   └── water_lilies.jpg            (Monet)
│
└── eval_content/         # 2 selected evaluation images
    ├── face_00010.jpg    (Girl)
    └── face_00066.jpg    (Boy)
```

---

## 🎨 Style Images Details

### Famous Artworks (9 new)

| Filename | Artist | Style Period | Year | Notes |
|----------|--------|--------------|------|-------|
| **starry_night.jpg** | Vincent van Gogh | Post-Impressionism | 1889 | Iconic swirling night sky |
| **sunflowers.jpg** | Vincent van Gogh | Post-Impressionism | 1888 | Bold colors, thick brushstrokes |
| **cafe_terrace.jpg** | Vincent van Gogh | Post-Impressionism | 1888 | Night scene, warm colors |
| **water_lilies.jpg** | Claude Monet | Impressionism | 1906 | Soft, fluid brushwork |
| **impression_sunrise.jpg** | Claude Monet | Impressionism | 1872 | Origin of "Impressionism" |
| **the_scream.jpg** | Edvard Munch | Expressionism | 1893 | Bold, emotional colors |
| **great_wave.jpg** | Hokusai | Ukiyo-e | 1831 | Japanese woodblock print |
| **the_kiss.jpg** | Gustav Klimt | Art Nouveau | 1908 | Gold leaf, decorative patterns |
| **sunday_afternoon.jpg** | Georges Seurat | Pointillism | 1884 | Dotted technique |

### Original Styles (4 existing)

| Filename | Type | Characteristics |
|----------|------|-----------------|
| **claude-monet.jpg** | Painting | Original sample |
| **drop-of-water.jpg** | Texture | Abstract water patterns |
| **sandstone.jpg** | Texture | Natural stone texture |
| **stone_style.jpg** | Texture | Rough stone surface |

---

## 👤 Content Images Details

### Generation Method
- **Source:** ThisPersonDoesNotExist.com (StyleGAN-based)
- **Count:** 200 unique synthetic faces
- **Demographics:** Mixed (primarily adult faces)
- **Ethics:** 100% AI-generated, no privacy concerns
- **Quality:** High resolution, photorealistic

### File Naming
- Format: `face_XXXXX.jpg` (5-digit zero-padded)
- Range: `face_00000.jpg` to `face_00199.jpg`
- Continuous indexing for easy expansion

### Data Pipeline
To add more faces:
```bash
# Add 100 more (to reach 300 total)
python data_generate_faces.py --num-images 300

# Add 200 more (to reach 400 total)
python data_generate_faces.py --num-images 400
```

---

## 📈 Dataset Statistics

### Current Status

| Metric | Value |
|--------|-------|
| Content images | 200 |
| Style images | 13 (was 4, added 9) |
| Evaluation images | 2 |
| Total combinations | 2,600 |
| Disk usage (content) | ~120 MB |
| Disk usage (style) | ~3.6 MB |

### Compared to Initial Setup

| Dataset | Initial | Current | Growth |
|---------|---------|---------|--------|
| Content | 100 | 200 | **+100%** |
| Style | 4 | 13 | **+225%** |
| Total combinations | 400 | 2,600 | **+550%** |

---

## 🎯 Impact on Training

### Benefits of Larger Dataset

1. **More Content Diversity (200 vs 100)**
   - Better generalization
   - Less overfitting
   - More robust identity preservation

2. **More Style Diversity (13 vs 4)**
   - Wider range of artistic styles
   - Better style transfer capabilities
   - More impressive visual results

3. **More Artistic Styles**
   - Impressionism: Monet (3), Van Gogh (3)
   - Expressionism: Munch (1)
   - Post-Impressionism: Van Gogh (3)
   - Art Nouveau: Klimt (1)
   - Pointillism: Seurat (1)
   - Japanese: Hokusai (1)
   - Textures: Original (4)

### Expected Training Changes

| Aspect | Before | After | Impact |
|--------|--------|-------|--------|
| Batch iterations/epoch | 25 | 50 | 2x longer |
| Training time/epoch | ~7 sec | ~14 sec | 2x longer |
| Total training time (10 epochs) | ~2 min | ~4 min | 2x longer |
| Total training time (20 epochs) | ~4 min | ~8 min | 2x longer |
| Style variety | Limited | Excellent | Much better results |

---

## 🔄 Adding More Style Images

### Option 1: Use the Script (Recommended)

```bash
# List all available artworks
python data_download_styles.py --list

# Download all available artworks
python data_download_styles.py --all

# Download specific artworks
python data_download_styles.py --artworks starry_night great_wave the_scream
```

### Option 2: Manual Download

**Recommended Sources:**
1. **Wikimedia Commons** - https://commons.wikimedia.org/
   - Filter: Public Domain
   - Search: "oil painting" or specific artists

2. **Metropolitan Museum Open Access** - https://www.metmuseum.org/art/collection
   - Filter: Public Domain
   - High-quality scans

3. **Art Institute of Chicago** - https://www.artic.edu/collection
   - Filter: Public Domain
   - Excellent collection

4. **Rijksmuseum** - https://www.rijksmuseum.nl/en/rijksstudio
   - Dutch masters
   - Free to download

### Option 3: Your Own Images

You can also add your own images:
```bash
# Copy any JPG/PNG image to data/style/
cp ~/Downloads/my_painting.jpg data/style/
```

**Good style characteristics:**
- ✓ Strong, distinctive visual patterns
- ✓ Rich colors or interesting textures
- ✓ Medium to high contrast
- ✗ Avoid plain/uniform colors
- ✗ Avoid low resolution (<500px)

---

## 📝 Notes

### Ethical Considerations
- ✅ All content images are AI-generated (no privacy issues)
- ✅ All style images are public domain (no copyright issues)
- ✅ Perfect for academic research (CS230 project)

### Dataset Expansion Strategy
1. ✅ **Content:** Doubled to 200 (completed)
2. ✅ **Style:** Tripled to 13 (completed)
3. 🔄 **Next:** Can add more as needed
4. 🎯 **Goal:** Balance between variety and training time

### Best Practices
- Keep style images < 2MB each (faster loading)
- Use distinctive artistic styles (better results)
- Organize by artist/period if collection grows large
- Document source and copyright status

---

**Last Updated:** October 28, 2024  
**Generated by:** `data_download_styles.py`  
**Total Dataset Size:** ~124 MB


