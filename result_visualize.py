#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visual Comparison Grid Generator for Style Transfer Evaluation

OVERVIEW:
This script creates side-by-side comparison grids to visually evaluate style transfer
quality. Each grid shows: original face, style image, baseline output, and identity-
preserving output, along with quantitative similarity metrics.

PURPOSE:
Visual evaluation is crucial for style transfer! Numbers alone don't tell the full story.
This script helps you:
1. Visually compare baseline vs identity-preserving models
2. See quantitative metrics directly on comparison images
3. Identify which styles work best for children's book illustrations
4. Create publication-ready figures for reports/papers

WHAT GETS CREATED?
For each content × style combination, a 2×2 comparison grid:
    ┌─────────────────┬─────────────────┐
    │  Content Image  │   Style Image   │
    │  (Original)     │  (Art Reference)│
    ├─────────────────┼─────────────────┤
    │  Baseline       │  Identity Model │
    │  (γ=0.0)        │  (γ=1.0)        │
    │  Metrics shown  │  Metrics + Δ    │
    └─────────────────┴─────────────────┘

METRICS DISPLAYED:
- Perceptual Similarity: VGG-based semantic similarity (higher = better)
- SSIM: Structural similarity (higher = better)
- Improvement indicators: ↑ for better, ↓ for worse (compared to baseline)
- Color coding: Green = identity model better, Orange = baseline better

USE CASES:
1. **Qualitative Evaluation**: Visually inspect stylization quality
2. **Model Comparison**: See if identity preservation helps or hurts
3. **Style Selection**: Identify which styles work best for your use case
4. **Publication Figures**: Generate comparison grids for papers/presentations
5. **Debugging**: Spot issues like artifacts, color shifts, identity loss

TYPICAL WORKFLOW:
1. Train baseline model (no identity loss)
2. Train identity-preserving model (with identity loss)
3. Generate stylized images using both models (eval_inference.py)
4. Run this script to create visual comparison grids
5. Review grids to evaluate which model works better

TYPICAL USAGE:
    # Create comparison grids for evaluation images
    python result_visualize.py \\
        --content-dir data/eval_content \\
        --style-dir data/style \\
        --baseline-dir results/eval_baseline_v2 \\
        --identity-dir results/eval_identity_v2 \\
        --output-dir results/eval_comparisons_v2

    # This will create:
    # - comparison_face_00010_starry_night.jpg
    # - comparison_face_00010_potter_peter_rabbit.jpg
    # - comparison_face_00066_starry_night.jpg
    # - ... (2 content × 21 styles = 42 comparison grids)

OUTPUT EXAMPLE:
    results/eval_comparisons_v2/
    ├── comparison_face_00010_starry_night.jpg
    ├── comparison_face_00010_potter_peter_rabbit.jpg
    ├── comparison_face_00010_great_wave.jpg
    ├── comparison_face_00066_starry_night.jpg
    └── ... (more comparison grids)

REQUIREMENTS:
- matplotlib (for plotting)
- PIL (for image loading)
- eval_metrics.py (for computing similarity metrics)
- Generated results from both baseline and identity models
"""

import matplotlib.pyplot as plt
from PIL import Image
import os
import json
from pathlib import Path
from eval_metrics import SimilarityComputer
import torch
from tqdm import tqdm


def create_comparison_with_metrics(content_path, style_path, baseline_path, 
                                   identity_path, output_path, style_name,
                                   baseline_metrics, identity_metrics):
    """
    Create a 2×2 comparison grid with quantitative metrics overlay.
    
    This function creates a publication-ready comparison figure showing:
    - Top row: Original content and style reference images
    - Bottom row: Baseline and identity-preserving model outputs
    - Metrics: Perceptual similarity and SSIM displayed on outputs
    - Improvements: Delta values (↑ or ↓) compared to baseline
    - Color coding: Green if identity model is better, orange otherwise
    
    GRID LAYOUT:
        [Content Image]    [Style Image]
        [Baseline + metrics] [Identity + metrics + improvements]
    
    Args:
        content_path: Path to original content image (e.g., face_00010.jpg)
        style_path: Path to style image (e.g., potter_peter_rabbit.jpg)
        baseline_path: Path to baseline model output
        identity_path: Path to identity-preserving model output
        output_path: Path to save the comparison grid (e.g., comparison_face_00010_peter_rabbit.jpg)
        style_name: Display name for the style (shown in title)
        baseline_metrics: Dict with baseline metrics {'ssim': float, 'perceptual': float, 'face': float}
        identity_metrics: Dict with identity metrics {'ssim': float, 'perceptual': float, 'face': float}
    
    Example:
        create_comparison_with_metrics(
            content_path='data/eval_content/face_00010.jpg',
            style_path='data/style/potter_peter_rabbit.jpg',
            baseline_path='results/eval_baseline_v2/face_00010_potter_peter_rabbit.jpg',
            identity_path='results/eval_identity_v2/face_00010_potter_peter_rabbit.jpg',
            output_path='results/comparisons/comparison_face_00010_peter_rabbit.jpg',
            style_name='Peter Rabbit (Beatrix Potter)',
            baseline_metrics={'ssim': 0.854, 'perceptual': 0.923, 'face': 0.745},
            identity_metrics={'ssim': 0.842, 'perceptual': 0.918, 'face': 0.876}
        )
    """
    
    # ========================================
    # Load all images
    # ========================================
    content_img = Image.open(content_path).convert('RGB')
    style_img = Image.open(style_path).convert('RGB')
    baseline_img = Image.open(baseline_path).convert('RGB')
    identity_img = Image.open(identity_path).convert('RGB')
    
    # ========================================
    # Create 2×2 matplotlib figure
    # ========================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    
    # Add main title at the top
    fig.suptitle(f'Style Transfer Comparison: {style_name}', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # ========================================
    # Top-left: Content image (original face)
    # ========================================
    axes[0, 0].imshow(content_img)
    axes[0, 0].set_title('Content Image\n(Original Synthetic Face)', 
                        fontsize=14, fontweight='bold', pad=15)
    axes[0, 0].axis('off')  # Hide axis ticks and labels
    
    # ========================================
    # Top-right: Style image (art reference)
    # ========================================
    axes[0, 1].imshow(style_img)
    axes[0, 1].set_title('Style Image\n(Artistic Reference)', 
                        fontsize=14, fontweight='bold', pad=15)
    axes[0, 1].axis('off')
    
    # ========================================
    # Bottom-left: Baseline model output
    # ========================================
    axes[1, 0].imshow(baseline_img)
    
    # Build title with metrics (if available)
    baseline_title = 'Baseline Model (γ=0.0)\n'  # γ = identity loss weight
    if baseline_metrics['perceptual'] is not None:
        baseline_title += f"Perceptual Sim: {baseline_metrics['perceptual']:.3f}\n"
    if baseline_metrics['ssim'] is not None:
        baseline_title += f"SSIM: {baseline_metrics['ssim']:.3f}"
    
    axes[1, 0].set_title(baseline_title, fontsize=13, fontweight='bold', 
                        color='blue', pad=15)
    axes[1, 0].axis('off')
    
    # ========================================
    # Bottom-right: Identity-preserving model output
    # ========================================
    axes[1, 1].imshow(identity_img)
    
    # Build title with metrics AND improvements
    identity_title = 'Identity-Preserving (γ=1.0)\n'  # γ = identity loss weight
    
    # Show perceptual similarity with improvement indicator
    if identity_metrics['perceptual'] is not None:
        identity_title += f"Perceptual Sim: {identity_metrics['perceptual']:.3f}"
        
        # Calculate and show improvement (↑ if better, ↓ if worse)
        if baseline_metrics['perceptual'] is not None:
            improvement = identity_metrics['perceptual'] - baseline_metrics['perceptual']
            if improvement > 0:
                identity_title += f" (↑{improvement:.3f})"  # Better than baseline
            else:
                identity_title += f" (↓{abs(improvement):.3f})"  # Worse than baseline
        identity_title += "\n"
    
    # Show SSIM with improvement indicator
    if identity_metrics['ssim'] is not None:
        identity_title += f"SSIM: {identity_metrics['ssim']:.3f}"
        
        # Calculate and show improvement
        if baseline_metrics['ssim'] is not None:
            improvement = identity_metrics['ssim'] - baseline_metrics['ssim']
            if improvement > 0:
                identity_title += f" (↑{improvement:.3f})"
            else:
                identity_title += f" (↓{abs(improvement):.3f})"
    
    # ========================================
    # Color coding: Green if better, Orange if worse
    # ========================================
    color = 'green'  # Default: assume identity is better
    if (baseline_metrics['perceptual'] is not None and 
        identity_metrics['perceptual'] is not None):
        if identity_metrics['perceptual'] < baseline_metrics['perceptual']:
            color = 'orange'  # Identity model performed worse
    
    axes[1, 1].set_title(identity_title, fontsize=13, fontweight='bold', 
                        color=color, pad=15)
    axes[1, 1].axis('off')
    
    # ========================================
    # Add subtle border around each subplot
    # ========================================
    for i in range(2):
        for j in range(2):
            rect = plt.Rectangle((0, 0), 1, 1, fill=False, 
                                edgecolor='gray', linewidth=2, 
                                transform=axes[i, j].transAxes)
            axes[i, j].add_patch(rect)
    
    # ========================================
    # Add explanation footer
    # ========================================
    fig.text(0.5, 0.02, 
             'Perceptual Similarity: Higher = closer to original content | '
             'SSIM: Structural similarity (0-1) | '
             '↑/↓: Change from baseline',
             ha='center', fontsize=10, style='italic', color='gray')
    
    # ========================================
    # Save figure
    # ========================================
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Leave space for title and footer
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()  # Free memory
    
    print(f"✓ Created: {Path(output_path).name}")


def process_evaluation_images(content_dir, style_dir, baseline_dir, identity_dir, output_dir):
    """
    Batch process all content×style combinations to create comparison grids.
    
    This is the main workhorse function that:
    1. Finds all content and style images
    2. For each content×style pair:
       a. Loads baseline and identity model outputs
       b. Computes similarity metrics
       c. Creates a comparison grid with metrics overlay
    3. Generates aggregate statistics
    
    WORKFLOW:
        For each content image (e.g., face_00010.jpg, face_00066.jpg):
            For each style image (e.g., starry_night.jpg, peter_rabbit.jpg):
                1. Load baseline output: face_00010_starry_night.jpg
                2. Load identity output: face_00010_starry_night.jpg
                3. Compute metrics (SSIM, perceptual, face similarity)
                4. Create 2×2 comparison grid
                5. Save as: comparison_face_00010_starry_night.jpg
    
    Args:
        content_dir: Directory with content images
                    Example: 'data/eval_content' (contains face_00010.jpg, face_00066.jpg)
        style_dir: Directory with style images  
                  Example: 'data/style' (contains starry_night.jpg, peter_rabbit.jpg, etc.)
        baseline_dir: Directory with baseline model results
                     Example: 'results/eval_baseline_v2'
        identity_dir: Directory with identity-preserving model results
                     Example: 'results/eval_identity_v2'
        output_dir: Directory to save comparison grids
                   Example: 'results/eval_comparisons_v2'
        
    Returns:
        list: Results with metrics for each comparison, format:
              [
                  {
                      'content': 'face_00010',
                      'style': 'starry_night',
                      'baseline': {'ssim': 0.854, 'perceptual': 0.923, ...},
                      'identity': {'ssim': 0.842, 'perceptual': 0.918, ...},
                      'output': 'comparison_face_00010_starry_night.jpg'
                  },
                  ...
              ]
    
    Example:
        results = process_evaluation_images(
            content_dir='data/eval_content',
            style_dir='data/style',
            baseline_dir='results/eval_baseline_v2',
            identity_dir='results/eval_identity_v2',
            output_dir='results/eval_comparisons_v2'
        )
        # Creates 2 content × 21 styles = 42 comparison grids
    """
    
    print("="*60)
    print("Evaluation Comparison Grid Generator")
    print("="*60)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # ========================================
    # Initialize Similarity Computer
    # ========================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    similarity_computer = SimilarityComputer(device=device)
    
    # ========================================
    # Find All Images
    # ========================================
    content_dir = Path(content_dir)
    style_dir = Path(style_dir)
    baseline_dir = Path(baseline_dir)
    identity_dir = Path(identity_dir)
    
    # Find all .jpg and .png images (sorted for consistency)
    content_images = sorted(list(content_dir.glob('*.jpg')) + list(content_dir.glob('*.png')))
    style_images = sorted(list(style_dir.glob('*.jpg')) + list(style_dir.glob('*.png')))
    
    print(f"\nFound {len(content_images)} content images")
    print(f"Found {len(style_images)} style images")
    print(f"Will create {len(content_images) * len(style_images)} comparison grids\n")
    
    all_results = []
    
    # ========================================
    # Process All Content × Style Combinations
    # ========================================
    total_combinations = len(content_images) * len(style_images)
    
    with tqdm(total=total_combinations, desc="Creating comparisons") as pbar:
        for content_path in content_images:
            content_name = content_path.stem  # e.g., 'face_00010'
            
            for style_path in style_images:
                style_name = style_path.stem  # e.g., 'starry_night'
                
                # ========================================
                # Construct paths to baseline and identity outputs
                # ========================================
                # Expected naming convention: {content_name}_{style_name}.jpg
                baseline_name = f"{content_name}_{style_name}.jpg"
                identity_name = f"{content_name}_{style_name}.jpg"
                
                baseline_path = baseline_dir / baseline_name
                identity_path = identity_dir / identity_name
                
                # ========================================
                # Skip if either output is missing
                # ========================================
                if not baseline_path.exists() or not identity_path.exists():
                    print(f"\n⚠️  Skipping {content_name} + {style_name}: Missing results")
                    pbar.update(1)
                    continue
                
                # ========================================
                # Compute Similarity Metrics
                # ========================================
                # Compare original content with baseline output
                baseline_metrics = similarity_computer.compute_all_metrics(
                    str(content_path), str(baseline_path)
                )
                
                # Compare original content with identity-preserving output
                identity_metrics = similarity_computer.compute_all_metrics(
                    str(content_path), str(identity_path)
                )
                
                # ========================================
                # Create Comparison Grid
                # ========================================
                output_filename = f"comparison_{content_name}_{style_name}.jpg"
                output_path = os.path.join(output_dir, output_filename)
                
                create_comparison_with_metrics(
                    content_path=str(content_path),
                    style_path=str(style_path),
                    baseline_path=str(baseline_path),
                    identity_path=str(identity_path),
                    output_path=output_path,
                    style_name=style_name,
                    baseline_metrics=baseline_metrics,
                    identity_metrics=identity_metrics
                )
                
                # ========================================
                # Store Results for Summary
                # ========================================
                result = {
                    'content': content_name,
                    'style': style_name,
                    'baseline': baseline_metrics,
                    'identity': identity_metrics,
                    'output': output_filename
                }
                all_results.append(result)
                
                pbar.update(1)
    
    # ========================================
    # Print Summary Statistics
    # ========================================
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    if all_results:
        # Calculate average metrics across all comparisons
        # (Only count valid values, exclude None)
        
        # SSIM averages
        avg_baseline_ssim = sum(r['baseline']['ssim'] for r in all_results 
                               if r['baseline']['ssim'] is not None) / len(all_results)
        avg_identity_ssim = sum(r['identity']['ssim'] for r in all_results 
                               if r['identity']['ssim'] is not None) / len(all_results)
        
        # Perceptual similarity averages
        avg_baseline_perceptual = sum(r['baseline']['perceptual'] for r in all_results 
                                     if r['baseline']['perceptual'] is not None) / len(all_results)
        avg_identity_perceptual = sum(r['identity']['perceptual'] for r in all_results 
                                     if r['identity']['perceptual'] is not None) / len(all_results)
        
        # Face similarity averages (may have None values if face detection failed)
        baseline_face_valid = [r for r in all_results if r['baseline']['face'] is not None]
        identity_face_valid = [r for r in all_results if r['identity']['face'] is not None]
        
        avg_baseline_face = sum(r['baseline']['face'] for r in baseline_face_valid) / len(baseline_face_valid) if baseline_face_valid else 0.0
        avg_identity_face = sum(r['identity']['face'] for r in identity_face_valid) / len(identity_face_valid) if identity_face_valid else 0.0
        
        # Display results
        print(f"\nTotal comparisons created: {len(all_results)}")
        print(f"\nAverage Metrics Across All Comparisons:")
        print(f"  Baseline Model:")
        print(f"    - SSIM: {avg_baseline_ssim:.3f}")
        print(f"    - Perceptual: {avg_baseline_perceptual:.3f}")
        print(f"    - Face: {avg_baseline_face:.3f}")
        print(f"  Identity-Preserving Model:")
        print(f"    - SSIM: {avg_identity_ssim:.3f}")
        print(f"    - Perceptual: {avg_identity_perceptual:.3f}")
        print(f"    - Face: {avg_identity_face:.3f}")
        
        # Show improvements
        print(f"\n  Improvements (Identity - Baseline):")
        print(f"    - SSIM: {avg_identity_ssim - avg_baseline_ssim:+.3f}")
        print(f"    - Perceptual: {avg_identity_perceptual - avg_baseline_perceptual:+.3f}")
        print(f"    - Face: {avg_identity_face - avg_baseline_face:+.3f}")
        
        print(f"\n✓ All comparison grids saved to: {output_dir}")
    else:
        print("\n⚠️  No comparisons created!")
        print("   Make sure baseline and identity results exist with correct naming:")
        print("   Expected format: {content_name}_{style_name}.jpg")
    
    print("="*60)
    
    return all_results


if __name__ == "__main__":
    """
    Command-line interface for creating comparison grids.
    
    This script is typically run AFTER:
    1. Training both baseline and identity-preserving models
    2. Running eval_inference.py to generate stylized images
    
    It creates visual comparison grids to help you:
    - Evaluate which model works better
    - Identify best styles for your use case
    - Generate publication-ready figures
    - Spot issues like artifacts or identity loss
    
    TYPICAL WORKFLOW:
        # 1. Generate baseline results
        python eval_inference.py \\
            --checkpoint checkpoints/baseline_v2/final_model.pth \\
            --content data/eval_content/ \\
            --style data/style/ \\
            --output results/eval_baseline_v2/
        
        # 2. Generate identity-preserving results
        python eval_inference.py \\
            --checkpoint checkpoints/identity_v2/final_model.pth \\
            --content data/eval_content/ \\
            --style data/style/ \\
            --output results/eval_identity_v2/
        
        # 3. Create comparison grids (this script)
        python result_visualize.py \\
            --content-dir data/eval_content \\
            --style-dir data/style \\
            --baseline-dir results/eval_baseline_v2 \\
            --identity-dir results/eval_identity_v2 \\
            --output-dir results/eval_comparisons_v2
    """
    import argparse
    
    # ========================================
    # Argument Parser
    # ========================================
    parser = argparse.ArgumentParser(
        description="Create visual comparison grids with quantitative metrics for style transfer evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:

  # Basic usage (process evaluation results with default paths)
  python result_visualize.py

  # Custom paths (recommended for versioned results)
  python result_visualize.py \\
      --content-dir data/eval_content \\
      --style-dir data/style \\
      --baseline-dir results/eval_baseline_v2 \\
      --identity-dir results/eval_identity_v2 \\
      --output-dir results/eval_comparisons_v2

  # Process training images instead of evaluation images
  python result_visualize.py \\
      --content-dir data/content \\
      --style-dir data/style \\
      --baseline-dir results/train_baseline \\
      --identity-dir results/train_identity \\
      --output-dir results/train_comparisons

OUTPUT:
  - Creates one comparison grid per content×style combination
  - Each grid is a 2×2 image: [content, style] / [baseline, identity]
  - Metrics displayed: Perceptual similarity, SSIM, improvements
  - Saves to output directory as: comparison_{content}_{style}.jpg

REQUIREMENTS:
  - Baseline and identity results must exist (run eval_inference.py first)
  - Results must follow naming convention: {content_name}_{style_name}.jpg
  - matplotlib and PIL installed
        """
    )
    
    # ========================================
    # Directory Arguments
    # ========================================
    parser.add_argument('--content-dir', type=str, default='data/eval_content',
                        help='Directory with content images (original faces)')
    parser.add_argument('--style-dir', type=str, default='data/style',
                        help='Directory with style images (art references)')
    parser.add_argument('--baseline-dir', type=str, default='results/eval_baseline',
                        help='Directory with baseline model results (γ=0.0)')
    parser.add_argument('--identity-dir', type=str, default='results/eval_identity',
                        help='Directory with identity-preserving model results (γ=1.0)')
    parser.add_argument('--output-dir', type=str, default='results/eval_comparisons',
                        help='Directory to save comparison grids')
    
    args = parser.parse_args()
    
    # ========================================
    # Run Comparison Grid Generator
    # ========================================
    process_evaluation_images(
        content_dir=args.content_dir,
        style_dir=args.style_dir,
        baseline_dir=args.baseline_dir,
        identity_dir=args.identity_dir,
        output_dir=args.output_dir
    )

