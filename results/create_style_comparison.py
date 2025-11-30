#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create side-by-side weight ratio comparison grids

⚠️ NOTE: This script is ARCHIVED and cannot be run again.
The source images (results/tuning/weight_visuals/ratio_*/) were deleted
to save space after generating the final comparison grids.

The generated comparison grids are preserved in:
    results/hyperparameter_tuning/style_comparisons/

This script is kept for documentation purposes only.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import numpy as np

def create_weight_comparison_grid(content_name, style_name, output_dir):
    """Create a comparison grid for one content-style pair across all weight ratios"""
    
    ratios = ["1.0_1.0", "1.0_5.0", "1.0_10.0", "1.0_20.0", "1.0_50.0"]
    ratio_labels = ["1:1\n(Equal)", "1:5\n(Content)", "1:10\n(Standard)", "1:20\n(Strong)", "1:50\n(Max)"]
    
    base_dir = Path("results/tuning/weight_visuals")
    
    # Load images
    images = []
    for ratio in ratios:
        img_path = base_dir / f"ratio_{ratio}" / f"{content_name}_{style_name}.jpg"
        if img_path.exists():
            images.append(mpimg.imread(img_path))
        else:
            images.append(None)
    
    # Also load content and style images
    content_path = Path(f"data/eval_content/{content_name}.jpg")
    style_path = Path(f"data/style/{style_name}.jpg")
    
    content_img = mpimg.imread(content_path) if content_path.exists() else None
    style_img = mpimg.imread(style_path) if style_path.exists() else None
    
    # Create figure
    fig = plt.figure(figsize=(18, 6))
    
    # Top row: content, style, then 5 ratio results
    for i in range(7):
        ax = plt.subplot(2, 7, i + 1)
        
        if i == 0:
            # Content image
            if content_img is not None:
                ax.imshow(content_img)
                ax.set_title("Content\n(Original)", fontsize=11, fontweight='bold')
        elif i == 1:
            # Style image  
            if style_img is not None:
                ax.imshow(style_img)
                ax.set_title("Style\n(Target)", fontsize=11, fontweight='bold')
        else:
            # Stylized result
            idx = i - 2
            if idx < len(images) and images[idx] is not None:
                ax.imshow(images[idx])
                label = ratio_labels[idx]
                if idx == 2:  # 1:10 - our choice
                    ax.set_title(label + "\n⭐", fontsize=11, fontweight='bold', color='red')
                    for spine in ax.spines.values():
                        spine.set_edgecolor('red')
                        spine.set_linewidth(3)
                else:
                    ax.set_title(label, fontsize=11, fontweight='bold')
        
        ax.axis('off')
    
    # Bottom row: Show metrics for each ratio
    for i in range(5):
        ax = plt.subplot(2, 7, i + 3)  # Start from position 3 in second row
        ax.axis('off')
        
        # Add text with key metrics
        ratio_info = {
            0: "Content: 6.72\nStyle: 4.50\nWeak style",
            1: "Content: 13.22\nStyle: 1.91\nSubtle art",
            2: "Content: 16.68\nStyle: 1.53\nBalanced ⭐",
            3: "Content: 20.00\nStyle: 1.44\nStrong art",
            4: "Content: 21.22\nStyle: 1.41\nMax style"
        }
        
        ax.text(0.5, 0.5, ratio_info[i], 
               ha='center', va='center', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='lightblue' if i == 2 else 'wheat', alpha=0.8))
    
    # Main title
    fig.suptitle(f'Weight Ratio Comparison: {content_name.replace("_", " ").title()} + {style_name.replace("_", " ").title()}',
                fontsize=14, fontweight='bold', y=0.98)
    
    # Save
    output_path = Path(output_dir) / f"comparison_{content_name}_{style_name}.png"
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return output_path


def main():
    """Generate comparison grids for key content-style pairs"""
    
    output_dir = Path("results/hyperparameter_tuning/style_comparisons")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Focus on our two evaluation faces and a few representative styles
    content_images = ["face_00010", "face_00066"]
    style_images = [
        "starry_night",
        "water_lilies", 
        "the_scream",
        "great_wave",
        "peter_rabbit"
    ]
    
    print("="*60)
    print("Creating Weight Ratio Comparison Grids")
    print("="*60)
    
    created = []
    for content in content_images:
        for style in style_images:
            print(f"\nProcessing: {content} + {style}")
            try:
                output_path = create_weight_comparison_grid(content, style, output_dir)
                print(f"  ✓ Saved to {output_path}")
                created.append(output_path)
            except Exception as e:
                print(f"  ✗ Error: {e}")
    
    print("\n" + "="*60)
    print(f"✅ Created {len(created)} comparison grids")
    print(f"📁 Location: {output_dir}")
    print("="*60)
    
    print("\nThese grids show:")
    print("  • How each weight ratio affects the same image")
    print("  • The progression from subtle (1:5) to extreme (1:50)")
    print("  • Why 1:10 (marked with ⭐) provides the best balance")


if __name__ == "__main__":
    main()

