#!/usr/bin/env python3
"""
Create final 6-image comparison grids showing progressive improvements:
(1,1) Content | (1,2) Style | (1,3) 0_baseline
(2,1) 1_identity | (2,2) 2_identity_plus_eye | (2,3) 2_face_aware_plus_identity (now labeled as "+ Face-Aware")

With full metrics (SSIM, Perceptual Sim, Face Sim) on each generated image.

IMPORTANT: This script now loads PRE-COMPUTED metrics from JSON files!
Metrics are computed during inference (eval_inference.py) on tensors BEFORE saving,
ensuring consistency with training metrics and avoiding compression artifacts.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import json

# No need for torch, model_face_utils, or torchvision imports anymore!

def load_image_pil(path):
    """Load image as PIL Image"""
    return Image.open(path).convert('RGB')

def load_precomputed_metrics(image_path):
    """
    Load pre-computed metrics from JSON file.
    
    During inference, metrics are computed on the tensor (before saving)
    and stored in a JSON file. This ensures:
    - Consistency with training metrics (no compression artifacts)
    - Faster comparison grid generation (no recomputation needed)
    - Reliability (exact same calculation pipeline)
    
    Args:
        image_path: Path to the generated image (e.g., face_00010_starry_night.png)
    
    Returns:
        tuple: (ssim, perceptual_sim, face_sim) or None if metrics file not found
    """
    # Construct metrics path (same name, but _metrics.json)
    metrics_path = str(image_path).rsplit('.', 1)[0] + '_metrics.json'
    
    try:
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        ssim = metrics.get('ssim', 0.0)
        perceptual = metrics.get('perceptual', 0.0)
        face = metrics.get('face', 0.0) if metrics.get('face') is not None else 0.0
        
        return ssim, perceptual, face
    except FileNotFoundError:
        print(f"⚠️  Warning: Metrics file not found: {metrics_path}")
        print("    Metrics were not computed during inference!")
        return None

def create_progression_comparison(content_name, style_name, 
                                  content_dir, style_dir,
                                  baseline_dir, identity_dir,
                                  identity_eye_dir, all_combined_dir,
                                  output_path):
    """Create 2×3 comparison grid with pre-computed metrics"""
    
    # Load original images
    content_path = content_dir / f"{content_name}.jpg"
    style_path = style_dir / f"{style_name}.jpg"
    
    baseline_path = baseline_dir / f"{content_name}_{style_name}.jpg"
    identity_path = identity_dir / f"{content_name}_{style_name}.jpg"
    identity_eye_path = identity_eye_dir / f"{content_name}_{style_name}.jpg"
    all_combined_path = all_combined_dir / f"{content_name}_{style_name}.jpg"
    
    # Check if all files exist
    for path in [content_path, style_path, baseline_path, identity_path, identity_eye_path, all_combined_path]:
        if not path.exists():
            print(f"⚠️  Missing file: {path}")
            return False
    
    # Load PRE-COMPUTED metrics (computed during inference on tensors!)
    print(f"  Loading pre-computed metrics for {content_name} + {style_name}...")
    
    baseline_metrics = load_precomputed_metrics(baseline_path)
    identity_metrics = load_precomputed_metrics(identity_path)
    identity_eye_metrics = load_precomputed_metrics(identity_eye_path)
    all_combined_metrics = load_precomputed_metrics(all_combined_path)
    
    # Check if all metrics were found
    if None in [baseline_metrics, identity_metrics, identity_eye_metrics, all_combined_metrics]:
        print(f"⚠️  Missing metrics files! Run inference with metrics calculation first.")
        return False
    
    # Load images
    content_img = load_image_pil(content_path)
    style_img = load_image_pil(style_path)
    baseline_img = load_image_pil(baseline_path)
    identity_img = load_image_pil(identity_path)
    identity_eye_img = load_image_pil(identity_eye_path)
    all_combined_img = load_image_pil(all_combined_path)
    
    # Create figure (2 rows × 3 columns)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.subplots_adjust(hspace=0.15, wspace=0.15, top=0.95, bottom=0.05, left=0.05, right=0.95)
    
    # Row 1, Col 1: Content
    axes[0, 0].imshow(content_img)
    axes[0, 0].set_title('Content Image', fontsize=14, fontweight='bold', pad=10)
    axes[0, 0].axis('off')
    
    # Row 1, Col 2: Style
    axes[0, 1].imshow(style_img)
    axes[0, 1].set_title('Style Image', fontsize=14, fontweight='bold', pad=10)
    axes[0, 1].axis('off')
    
    # Row 1, Col 3: Baseline
    axes[0, 2].imshow(baseline_img)
    title_baseline = (f'Baseline\n'
                      f'SSIM: {baseline_metrics[0]:.3f} | '
                      f'Percep: {baseline_metrics[1]:.3f} | '
                      f'Face: {baseline_metrics[2]:.3f}')
    axes[0, 2].set_title(title_baseline, fontsize=12, fontweight='bold', pad=10)
    axes[0, 2].axis('off')
    
    # Row 2, Col 1: Identity
    axes[1, 0].imshow(identity_img)
    title_identity = (f'+ Identity Loss\n'
                      f'SSIM: {identity_metrics[0]:.3f} | '
                      f'Percep: {identity_metrics[1]:.3f} | '
                      f'Face: {identity_metrics[2]:.3f}')
    axes[1, 0].set_title(title_identity, fontsize=12, fontweight='bold', pad=10)
    axes[1, 0].axis('off')
    
    # Row 2, Col 2: Identity + Eye
    axes[1, 1].imshow(identity_eye_img)
    title_identity_eye = (f'+ Eye Loss\n'
                          f'SSIM: {identity_eye_metrics[0]:.3f} | '
                          f'Percep: {identity_eye_metrics[1]:.3f} | '
                          f'Face: {identity_eye_metrics[2]:.3f}')
    axes[1, 1].set_title(title_identity_eye, fontsize=12, fontweight='bold', pad=10)
    axes[1, 1].axis('off')
    
    # Row 2, Col 3: All Combined (but title shows "+ Face-Aware")
    axes[1, 2].imshow(all_combined_img)
    title_all_combined = (f'+ Face-Aware\n'
                          f'SSIM: {all_combined_metrics[0]:.3f} | '
                          f'Percep: {all_combined_metrics[1]:.3f} | '
                          f'Face: {all_combined_metrics[2]:.3f}')
    axes[1, 2].set_title(title_all_combined, fontsize=12, fontweight='bold', pad=10)
    axes[1, 2].axis('off')
    
    # Save figure (no main title)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Saved comparison to: {output_path}")
    return True

def main():
    print("="*80)
    print("Creating Final Progression Comparison Grids")
    print("="*80)
    print()
    
    # Set up paths
    project_root = Path(__file__).parent.parent
    content_dir = project_root / "data" / "eval_content"
    style_dir = project_root / "data" / "style"
    
    result_base = project_root / "results" / "model_progression"
    baseline_dir = result_base / "0_baseline"
    identity_dir = result_base / "1_identity"
    identity_eye_dir = result_base / "2_identity_plus_eye"
    all_combined_dir = result_base / "3_all_combined"
    
    output_dir = result_base / "comparisons"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # No need to initialize models - metrics are pre-computed!
    print("✅ Using pre-computed metrics from inference (fast!)")
    print("   Metrics were computed on tensors before saving")
    print("   This ensures consistency with training metrics")
    print()
    print("📝 New progression layout:")
    print("   Row 1: Content | Style | Baseline")
    print("   Row 2: + Identity Loss | + Eye Loss | + Face-Aware (all three combined)")
    print()
    
    # Get all content and style combinations
    content_images = sorted([f.stem for f in content_dir.glob("*.jpg")])
    style_images = sorted([f.stem for f in style_dir.glob("*.jpg")])
    
    print(f"Found {len(content_images)} content images and {len(style_images)} style images")
    print(f"Total comparisons to create: {len(content_images) * len(style_images)}")
    print()
    
    # Create comparisons
    success_count = 0
    total_count = len(content_images) * len(style_images)
    
    for i, content_name in enumerate(content_images):
        for j, style_name in enumerate(style_images):
            idx = i * len(style_images) + j + 1
            print(f"[{idx}/{total_count}] Creating comparison: {content_name} + {style_name}")
            
            output_path = output_dir / f"progression_{content_name}_{style_name}.png"
            
            success = create_progression_comparison(
                content_name, style_name,
                content_dir, style_dir,
                baseline_dir, identity_dir,
                identity_eye_dir, all_combined_dir,
                output_path
            )
            
            if success:
                success_count += 1
            print()
    
    print("="*80)
    print(f"✅ Completed! Created {success_count}/{total_count} comparison grids")
    print(f"   Output directory: {output_dir}")
    print("="*80)

if __name__ == "__main__":
    main()

