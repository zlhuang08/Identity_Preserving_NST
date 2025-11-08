#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create visual comparison grid for identity weight tuning

Shows: Original | Style | γ=0.0 | γ=0.01 | γ=0.1 | γ=1.0 | γ=10.0
"""

import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import numpy as np

def create_identity_comparison_grid(content_name="face_00010", style_name="starry_night"):
    """Create a comparison grid for one content+style pair across all γ values"""
    
    gamma_values = [0.0, 0.01, 0.1, 1.0, 10.0]
    gamma_labels = ["γ=0.0\n(Baseline)", "γ=0.01", "γ=0.1", "γ=1.0", "γ=10.0"]
    
    # Setup figure
    fig, axes = plt.subplots(1, 7, figsize=(21, 3))
    fig.suptitle(f'Identity Weight Comparison: {content_name} + {style_name} style', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    # Load and display original content
    content_path = f"data/eval_content/{content_name}.jpg"
    try:
        content_img = Image.open(content_path)
        axes[0].imshow(content_img)
        axes[0].set_title("Original\nContent", fontsize=10, fontweight='bold')
        axes[0].axis('off')
    except Exception as e:
        print(f"Error loading content: {e}")
    
    # Load and display style
    style_paths = list(Path("data/style").glob(f"{style_name}*"))
    if style_paths:
        try:
            style_img = Image.open(style_paths[0])
            axes[1].imshow(style_img)
            axes[1].set_title("Style\nReference", fontsize=10, fontweight='bold')
            axes[1].axis('off')
        except Exception as e:
            print(f"Error loading style: {e}")
    
    # Load and display each γ result
    for i, (gamma, label) in enumerate(zip(gamma_values, gamma_labels)):
        gamma_name = f"{gamma:.2f}".replace('.', '_')
        result_path = f"results/tuning/identity_visuals/gamma_{gamma_name}/{content_name}_{style_name}.jpg"
        
        try:
            result_img = Image.open(result_path)
            axes[i+2].imshow(result_img)
            axes[i+2].set_title(label, fontsize=10, fontweight='bold' if gamma == 0.1 else 'normal')
            axes[i+2].axis('off')
            
            # Add border for γ=0.1 (our target)
            if gamma == 0.1:
                for spine in axes[i+2].spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(3)
                    spine.set_visible(True)
        except Exception as e:
            axes[i+2].text(0.5, 0.5, f"Missing\n{gamma}", 
                          ha='center', va='center', transform=axes[i+2].transAxes)
            axes[i+2].axis('off')
    
    plt.tight_layout()
    
    # Save
    output_path = f"results/tuning/identity_visual_comparison_{content_name}_{style_name}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved comparison to: {output_path}")
    plt.close()


def create_multi_style_comparison(content_name="face_00010"):
    """Create comparisons for multiple styles"""
    
    # Popular styles that work well
    styles = [
        "starry_night",
        "water_lilies",
        "sunday_afternoon",
        "the_kiss",
        "great_wave"
    ]
    
    print(f"\nCreating visual comparisons for {content_name}...")
    for style in styles:
        try:
            create_identity_comparison_grid(content_name, style)
        except Exception as e:
            print(f"  ✗ Failed for {style}: {e}")


if __name__ == "__main__":
    print("="*80)
    print("Creating Identity Weight Visual Comparisons")
    print("="*80)
    
    # Create comparisons for both evaluation faces
    create_multi_style_comparison("face_00010")  # Girl
    create_multi_style_comparison("face_00066")  # Boy
    
    print("\n" + "="*80)
    print("✓ Visual comparisons complete!")
    print("="*80)
    print("\nGenerated files in results/tuning/:")
    print("  - identity_visual_comparison_face_00010_*.png")
    print("  - identity_visual_comparison_face_00066_*.png")

