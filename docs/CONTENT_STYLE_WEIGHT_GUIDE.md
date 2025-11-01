# Content/Style Weight Tuning Guide

## Why is this different from Learning Rate Tuning?

**Learning Rate Tuning:**
- Objective: **Convergence speed and stability**
- Metric: Loss curves (should decrease smoothly)
- Clear "winner": Lowest final loss with stable convergence

**Content/Style Weight Tuning:**
- Objective: **Artistic balance** between content preservation and stylization
- Metric: **Trade-off curve** (can't minimize both simultaneously)
- No single "winner": Depends on your artistic goal and subjective preference

## How to Evaluate Content/Style Weights

### 1. Literature Standard (Safe Choice)

**The 1:10 ratio (λ_content=1.0, λ_style=10.0) is industry standard:**

✅ **Proven in research:**
- AdaIN paper [Huang & Belongie, 2017]
- Fast Style Transfer [Johnson et al., 2016]
- Most style transfer implementations

✅ **Why 1:10 works:**
- Style loss has naturally smaller magnitude than content loss
- The 10x multiplier balances their contributions
- Provides good artistic stylization while preserving content structure

**Recommendation:** Start with 1:10 unless you have specific artistic goals.

### 2. Quantitative Approach: Pareto Trade-off Curve

Unlike learning rate (single optimization objective), content/style weights involve **multi-objective optimization**:

```
Minimize: L_content (preserve structure)
Minimize: L_style (apply artistic style)
Conflict: These objectives compete!
```

**The Pareto Frontier:**
- Plot content loss (x-axis) vs style loss (y-axis) for different weight ratios
- Points on the lower-left frontier are "optimal" (can't improve one without hurting the other)
- Pick a point on this frontier based on your preference

**Example:**
```
Ratio    Content Loss    Style Loss    Interpretation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2:10     17.2           3.8           Content-focused (subtle style)
1:5      17.8           2.9           Balanced (weak style)
1:10 ⭐   18.1           1.6           Standard (good balance)
1:15     18.5           1.2           Style-focused (strong artistic)
1:20     19.2           0.8           Very stylized (may lose content)
```

### 3. Qualitative Approach: Visual Inspection (MOST IMPORTANT!)

For style transfer, **visual quality trumps quantitative metrics**. The tuning script generates side-by-side comparisons:

```
Content Image | Style Image | Ratio 1:5 | Ratio 1:10 | Ratio 1:15 | Ratio 1:20
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  face.jpg    |  monet.jpg  |  subtle   |  balanced  |   strong   |  very strong
```

**What to look for:**
- ✅ **1:10 (Standard)**: Face recognizable, strong artistic style, good balance
- ⚠️ **1:5**: Face very clear but weak stylization (too content-focused)
- ⚠️ **1:20**: Strong style but face structure may be distorted (too style-focused)

### 4. Perceptual Metrics

Additional metrics from evaluation:
- **Perceptual Similarity**: Higher = better content preservation
- **SSIM**: Structural similarity to original content
- **Face Similarity**: Identity preservation (for our use case)

**Expected ranges for different ratios:**

| Ratio | Perceptual Sim | SSIM | Face Sim | Use Case |
|-------|---------------|------|----------|----------|
| 2:10  | 0.52-0.55     | 0.38-0.42 | 0.50-0.54 | Portraits, subtle art |
| 1:10 ⭐| 0.48-0.52     | 0.35-0.38 | 0.47-0.50 | **General use** |
| 1:15  | 0.45-0.48     | 0.32-0.35 | 0.44-0.47 | Artistic photos |
| 1:20  | 0.42-0.45     | 0.29-0.32 | 0.41-0.44 | Abstract art |

## Using the Tuning Script

### Quick Test (Recommended First Step)

Test 3 key ratios with minimal compute:

```bash
python tune_content_style_weights.py --quick-test --epochs 5
```

This tests:
- 1:5 (content-focused)
- 1:10 (standard)
- 1:15 (style-focused)

**Output:**
- `results/tuning/weight_comparison.png` - Pareto curve and total loss comparison
- `results/tuning/weight_visuals/` - Generated images for visual comparison
- `results/tuning/weight_tuning_results.json` - Detailed metrics

### Full Experiment

Test 5 ratios for comprehensive analysis:

```bash
python tune_content_style_weights.py --epochs 10
```

Tests: 2:10, 1:5, 1:10, 1:15, 1:20

### Skip Visual Generation (Faster)

If you only want quantitative analysis:

```bash
python tune_content_style_weights.py --epochs 10 --skip-visuals
```

## Interpretation Guide

### Reading the Pareto Curve

```
Style Loss
    ^
    |
    |  A (1:5)     ← High style loss = weak stylization
    |     
    |       B (1:10) ← Balanced point (recommended)
    |
    |           C (1:20) ← Low style loss = strong stylization
    |________________> Content Loss
    Low              High
```

- **Moving right** (higher content loss): Losing content structure
- **Moving up** (higher style loss): Weaker stylization
- **Sweet spot**: Lower-left region, but exact point depends on preference

### Total Loss is Misleading!

⚠️ **Warning**: Don't just pick the ratio with lowest total loss!

**Why?**
```python
# Example:
Ratio 1:5:  Total = 20.7 (content=17.8, style=2.9)  # Low total, but weak style
Ratio 1:10: Total = 34.1 (content=18.1, style=1.6)  # Higher total, but balanced!
Ratio 1:15: Total = 36.5 (content=18.5, style=1.2)  # Even higher, strong style
```

The total loss combines incomparable quantities (content vs style). Use it for convergence check, not for weight selection.

## Decision Framework

### Step 1: Look at Pareto Curve
- Is 1:10 on the efficient frontier? (It usually is)
- Where does it sit relative to other points?

### Step 2: Check Component Losses
- Content loss should be moderate (~17-19)
- Style loss should be low (~1-3)
- Ratios with very high/low values in either indicate imbalance

### Step 3: Visual Inspection (Most Important!)
- Generate 3-5 images with different ratios
- Show to stakeholders/users if possible
- Pick based on artistic preference

### Step 4: Consider Your Use Case

| Use Case | Recommended Ratio | Why |
|----------|------------------|-----|
| Children's portraits | 1:10 or 2:10 | Need clear face recognition |
| Artistic photography | 1:10 or 1:15 | Balance art and content |
| Abstract art | 1:15 or 1:20 | Strong artistic effect OK |
| Medical/ID photos | 2:10 or higher | Must preserve details |

## Common Pitfalls

❌ **Don't:**
1. Pick ratio based solely on total loss
2. Use extreme ratios (1:30, 5:10) without visual inspection
3. Ignore the visual quality in favor of metrics
4. Change both content AND style weights at once (keep one fixed)

✅ **Do:**
1. Start with 1:10 (industry standard)
2. Generate visual comparisons for 3-5 ratios
3. Consider your specific use case and audience
4. Trust your eyes! Metrics guide, vision decides

## Summary

**For Learning Rate:**
- Objective metric: convergence curves
- Clear winner: LR with lowest val loss
- Quantitative decision

**For Content/Style Weights:**
- Subjective metric: artistic balance
- No clear winner: depends on use case
- **Visual inspection is mandatory**
- Standard 1:10 is safe default

**Our Recommendation:**
1. Use 1:10 unless you have specific artistic goals
2. If experimenting, test 1:5, 1:10, 1:15
3. Generate visual comparisons and inspect
4. Let artistic quality guide your final choice

---

**Bottom Line:** Unlike learning rate where metrics tell the full story, content/style weights require human judgment of artistic quality. The 1:10 ratio is battle-tested and recommended for most use cases.

