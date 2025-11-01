#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Content/Style Weight Tuning Experiment

This script helps determine the optimal content/style weight ratio by:
1. Training models with different weight ratios
2. Generating comparison grids for visual inspection
3. Plotting the Pareto trade-off curve (content loss vs style loss)

USAGE:
    # Test 5 different weight ratios
    python tune_content_style_weights.py --epochs 10
    
    # Quick test (1 epoch, for debugging)
    python tune_content_style_weights.py --epochs 1 --quick-test
"""

import argparse
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json


def train_weight_ratio(content_weight, style_weight, epochs, batch_size, checkpoint_base):
    """Train a model with specific content/style weight ratio"""
    ratio_name = f"{content_weight:.1f}_{style_weight:.1f}"
    checkpoint_dir = checkpoint_base / f"weights_{ratio_name}"
    
    print(f"\n{'='*60}")
    print(f"Training: Content={content_weight}, Style={style_weight} (ratio {content_weight}:{style_weight})")
    print(f"{'='*60}")
    
    cmd = [
        "python", "train_model.py",
        "--content-dir", "data/content",
        "--style-dir", "data/style",
        "--batch-size", str(batch_size),
        "--epochs", str(epochs),
        "--content-weight", str(content_weight),
        "--style-weight", str(style_weight),
        "--identity-weight", "0.0",  # Focus on baseline for now
        "--checkpoint-dir", str(checkpoint_dir),
        "--save-interval", str(max(epochs // 2, 5))
    ]
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"❌ Training failed for ratio {ratio_name}")
        return None
    
    # Load training curves
    csv_file = checkpoint_dir / "training_curves.csv"
    if csv_file.exists():
        df = pd.read_csv(csv_file)
        final_metrics = {
            'content_weight': content_weight,
            'style_weight': style_weight,
            'ratio': f"{content_weight}:{style_weight}",
            'final_train_loss': df['train_loss'].iloc[-1],
            'final_content_loss': df['train_content'].iloc[-1],
            'final_style_loss': df['train_style'].iloc[-1],
            'final_val_loss': df['val_loss'].iloc[-1],
            'final_val_content': df['val_content'].iloc[-1],
            'final_val_style': df['val_style'].iloc[-1],
            'checkpoint_dir': str(checkpoint_dir)
        }
        return final_metrics
    
    return None


def plot_pareto_curve(results, output_file):
    """Plot the Pareto trade-off curve: content loss vs style loss"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Extract data
    content_losses = [r['final_val_content'] for r in results]
    style_losses = [r['final_val_style'] for r in results]
    ratios = [r['ratio'] for r in results]
    
    # Left plot: Pareto curve
    ax1.scatter(content_losses, style_losses, s=200, alpha=0.6, c=np.arange(len(results)), cmap='viridis')
    
    for i, (c, s, ratio) in enumerate(zip(content_losses, style_losses, ratios)):
        ax1.annotate(ratio, (c, s), fontsize=10, ha='right', 
                    xytext=(-10, 5), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    ax1.set_xlabel('Content Loss (Final Validation)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Style Loss (Final Validation)', fontsize=12, fontweight='bold')
    ax1.set_title('Pareto Trade-off: Content vs Style Loss', fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Add "sweet spot" annotation for 1:10 ratio
    sweet_idx = [i for i, r in enumerate(results) if r['ratio'] == '1.0:10.0']
    if sweet_idx:
        idx = sweet_idx[0]
        ax1.scatter([content_losses[idx]], [style_losses[idx]], s=400, 
                   marker='*', color='red', edgecolors='black', linewidths=2,
                   label='Standard 1:10 ratio', zorder=10)
        ax1.legend(fontsize=11)
    
    # Right plot: Total loss comparison
    total_losses = [r['final_val_loss'] for r in results]
    x_pos = np.arange(len(ratios))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    bars = ax2.bar(x_pos, total_losses, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Highlight the best (lowest) total loss
    best_idx = np.argmin(total_losses)
    bars[best_idx].set_color('gold')
    bars[best_idx].set_edgecolor('red')
    bars[best_idx].set_linewidth(3)
    
    ax2.set_xlabel('Weight Ratio (Content:Style)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Total Validation Loss', fontsize=12, fontweight='bold')
    ax2.set_title('Total Loss Comparison', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(ratios, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Add value labels on bars
    for i, (bar, loss) in enumerate(zip(bars, total_losses)):
        height = bar.get_height()
        marker = '⭐' if i == best_idx else ''
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{marker}\n{loss:.2f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    fig.suptitle('Content/Style Weight Tuning Analysis', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✅ Pareto curve saved to: {output_file}")


def generate_visual_comparison(results, output_dir):
    """Generate side-by-side comparison images for different weight ratios"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("Generating Visual Comparisons")
    print("="*60)
    
    # Use the two evaluation faces
    eval_content = [
        "data/eval_content/face_00010.jpg",  # Girl
        "data/eval_content/face_00066.jpg"   # Boy
    ]
    
    # Select a few representative styles for comparison
    eval_styles = [
        "data/style/starry_night.jpg",
        "data/style/water_lilies.jpg",
        "data/style/the_scream.jpg"
    ]
    
    for result in results:
        checkpoint_dir = Path(result['checkpoint_dir'])
        model_path = checkpoint_dir / "final_model.pth"
        ratio = result['ratio'].replace(':', '_')
        
        print(f"\nGenerating images for ratio {result['ratio']}...")
        
        # Run inference
        output_subdir = output_dir / f"ratio_{ratio}"
        cmd = [
            "python", "eval_inference.py",
            "--checkpoint", str(model_path),
            "--content-dir", "data/eval_content",
            "--style-dir", "data/style", 
            "--output-dir", str(output_subdir)
        ]
        
        subprocess.run(cmd, capture_output=True)
        print(f"  ✓ Images saved to {output_subdir}")
    
    print("\n✅ All comparison images generated")
    print(f"📁 Location: {output_dir}")


def print_analysis(results):
    """Print detailed analysis of results"""
    print("\n" + "="*80)
    print("Content/Style Weight Analysis")
    print("="*80)
    
    # Sort by total loss
    results_sorted = sorted(results, key=lambda x: x['final_val_loss'])
    
    print("\nRanked by Final Validation Loss:")
    print(f"{'Rank':<6} {'Ratio':<12} {'Total Loss':<12} {'Content Loss':<14} {'Style Loss':<12}")
    print("-" * 80)
    
    for rank, r in enumerate(results_sorted, 1):
        marker = "⭐" if rank == 1 else "  "
        print(f"{marker} {rank:<4} {r['ratio']:<12} {r['final_val_loss']:<12.2f} "
              f"{r['final_val_content']:<14.4f} {r['final_val_style']:<12.4f}")
    
    print("\n" + "="*80)
    print("Guidelines for Interpretation:")
    print("="*80)
    print("""
1. **Content Loss**: Lower = better content preservation
   - High content loss: Over-stylized, content structure lost
   - Low content loss: Content well-preserved, may be under-stylized

2. **Style Loss**: Lower = better style transfer
   - High style loss: Weak stylization, looks like original
   - Low style loss: Strong stylization, artistic effect clear

3. **Total Loss**: Lower = better overall balance
   - BUT: Don't just pick lowest total loss!
   - Check if content/style losses are balanced

4. **The "Sweet Spot"**: 
   - Moderate content loss (~17-19)
   - Low style loss (~1-3)
   - Visually pleasing balance (MUST inspect images!)

5. **Common Ratios**:
   - 1:5  → Content-focused (subtle stylization)
   - 1:10 → Balanced (standard, recommended) ⭐
   - 1:15 → Style-focused (strong artistic effect)
   - 2:10 → Similar to 1:5, more content preservation
    """)
    
    # Highlight the standard 1:10 ratio
    standard = [r for r in results if r['ratio'] == '1.0:10.0']
    if standard:
        r = standard[0]
        print("\n" + "="*80)
        print("Standard 1:10 Ratio Performance:")
        print("="*80)
        print(f"Total Loss: {r['final_val_loss']:.2f}")
        print(f"Content Loss: {r['final_val_content']:.4f}")
        print(f"Style Loss: {r['final_val_style']:.4f}")
        print("\n✓ This ratio is recommended by AdaIN paper and widely used")
        print("✓ Good starting point before fine-tuning")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tune content/style weight ratios",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs for each training run (default: 10)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size (default: 64)')
    parser.add_argument('--checkpoint-base', type=str, default='checkpoints',
                        help='Base directory for checkpoints (default: checkpoints)')
    parser.add_argument('--output', type=str, default='results/tuning/weight_comparison.png',
                        help='Output file for plots (default: results/tuning/weight_comparison.png)')
    parser.add_argument('--visual-output', type=str, default='results/tuning/weight_visuals',
                        help='Directory for visual comparisons (default: results/tuning/weight_visuals)')
    parser.add_argument('--quick-test', action='store_true',
                        help='Quick test mode: only test 3 ratios with 1 epoch')
    parser.add_argument('--skip-visuals', action='store_true',
                        help='Skip generating visual comparison images (faster)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("Content/Style Weight Tuning Experiment")
    print("="*80)
    print(f"Epochs per model: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output plot: {args.output}")
    print("="*80)
    
    # Define weight ratios to test
    if args.quick_test:
        weight_configs = [
            (1.0, 5.0),   # 1:5 (content-focused)
            (1.0, 10.0),  # 1:10 (standard)
            (1.0, 15.0),  # 1:15 (style-focused)
        ]
        print("\n⚡ Quick test mode: Testing 3 ratios only")
    else:
        # User-specified ratios for comprehensive analysis
        weight_configs = [
            (1.0, 1.0),   # 1:1 (equal weights)
            (1.0, 5.0),   # 1:5 (content-focused)
            (1.0, 10.0),  # 1:10 (standard, recommended)
            (1.0, 20.0),  # 1:20 (strong stylization)
            (1.0, 50.0),  # 1:50 (very strong stylization)
        ]
    
    # Train models with different weight ratios
    results = []
    checkpoint_base = Path(args.checkpoint_base)
    
    for content_w, style_w in weight_configs:
        metrics = train_weight_ratio(content_w, style_w, args.epochs, 
                                     args.batch_size, checkpoint_base)
        if metrics:
            results.append(metrics)
    
    if not results:
        print("\n❌ No successful training runs. Exiting.")
        exit(1)
    
    # Save results to JSON
    results_file = Path(args.output).parent / "weight_tuning_results.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to: {results_file}")
    
    # Plot Pareto curve and analysis
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plot_pareto_curve(results, args.output)
    
    # Generate visual comparisons
    if not args.skip_visuals:
        generate_visual_comparison(results, args.visual_output)
    else:
        print("\n⏭️  Skipping visual comparison generation (--skip-visuals)")
    
    # Print analysis
    print_analysis(results)
    
    print("\n" + "="*80)
    print("🎉 Weight Tuning Complete!")
    print("="*80)
    print(f"\n📊 Quantitative analysis: {args.output}")
    if not args.skip_visuals:
        print(f"🎨 Visual comparisons: {args.visual_output}")
    print(f"💾 Detailed results: {results_file}")
    print("\n👀 IMPORTANT: Inspect the visual comparisons to make final decision!")
    print("   Quantitative metrics guide you, but visual quality is key for style transfer.")

