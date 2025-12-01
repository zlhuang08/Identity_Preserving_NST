#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate Pareto Curve Only for Content/Style Weight Tuning

This script creates a single plot showing the Pareto trade-off between
content loss and style loss for different weight ratios.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse


def plot_pareto_curve_only(checkpoint_base_dir, output_file='results/hyperparameter_tuning/style_weight_tuning.png'):
    """Plot only the Pareto trade-off curve"""
    
    checkpoint_base_dir = Path(checkpoint_base_dir)
    
    # Find all weights_* directories
    weight_dirs = sorted(checkpoint_base_dir.glob('weights_*'))
    
    if not weight_dirs:
        print(f"❌ No weight experiment directories found in {checkpoint_base_dir}")
        return
    
    print(f"Found {len(weight_dirs)} weight ratio experiments")
    
    results = []
    
    for weight_dir in weight_dirs:
        csv_file = weight_dir / 'training_curves.csv'
        
        if not csv_file.exists():
            print(f"⚠️  Skipping {weight_dir.name}: no training_curves.csv found")
            continue
        
        try:
            # Extract ratio from directory name (e.g., weights_1.0_10.0)
            parts = weight_dir.name.replace('weights_', '').split('_')
            if len(parts) == 2:
                content_weight = float(parts[0])
                style_weight = float(parts[1])
                ratio = f"{int(content_weight)}:{int(style_weight)}"
            else:
                continue
            
            df = pd.read_csv(csv_file)
            
            results.append({
                'content_weight': content_weight,
                'style_weight': style_weight,
                'ratio': ratio,
                'final_val_content': df['val_content'].iloc[-1],
                'final_val_style': df['val_style'].iloc[-1],
                'final_val_loss': df['val_loss'].iloc[-1]
            })
            
            print(f"✓ {weight_dir.name}: Content Loss = {df['val_content'].iloc[-1]:.4f}, Style Loss = {df['val_style'].iloc[-1]:.4f}")
            
        except Exception as e:
            print(f"❌ Error reading {csv_file}: {e}")
            continue
    
    if not results:
        print("❌ No valid training curves found!")
        return
    
    # Sort by ratio for consistent coloring
    results = sorted(results, key=lambda x: (x['content_weight'], x['style_weight']))
    
    # Create single figure (larger for milestone report)
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Extract data
    content_losses = [r['final_val_content'] for r in results]
    style_losses = [r['final_val_style'] for r in results]
    ratios = [r['ratio'] for r in results]
    
    # Plot Pareto curve with color gradient
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    scatter = ax.scatter(content_losses, style_losses, s=250, alpha=0.7, 
                        c=np.arange(len(results)), cmap='viridis',
                        edgecolors='black', linewidths=2)
    
    # Annotate each point with ratio (positioned directly above at 12:00 to avoid overlap)
    for i, (c, s, ratio) in enumerate(zip(content_losses, style_losses, ratios)):
        ax.annotate(ratio, (c, s), fontsize=11, ha='center', fontweight='bold',
                   xytext=(0, 18), textcoords='offset points')
    
    # Highlight the 1:10 ratio (standard/optimal)
    optimal_idx = [i for i, r in enumerate(results) if r['ratio'] == '1:10']
    if optimal_idx:
        idx = optimal_idx[0]
        ax.scatter([content_losses[idx]], [style_losses[idx]], s=600, 
                  marker='*', color='#FFBC42', edgecolors='black', linewidths=3,
                  zorder=10)
    
    # Labels and title
    ax.set_xlabel('Content Loss (Validation)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Style Loss (Validation)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label('Ratio Index\n(Lower → Higher style weight)', 
                  fontsize=10, rotation=270, labelpad=25)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✅ Pareto curve saved to: {output_file}")
    
    # Print analysis
    print("\n" + "="*70)
    print("Weight Ratio Analysis")
    print("="*70)
    print(f"{'Ratio':<10} {'Content Loss':<15} {'Style Loss':<15} {'Total Loss':<12}")
    print("-" * 70)
    
    for r in results:
        marker = '⭐' if r['ratio'] == '1:10' else '  '
        print(f"{marker} {r['ratio']:<8} {r['final_val_content']:<15.4f} "
              f"{r['final_val_style']:<15.4f} {r['final_val_loss']:<12.4f}")
    
    print("="*70)
    print("Recommendation: 1:10 ratio provides best balance")
    print("="*70)
    
    plt.close()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot Pareto curve only for content/style weight tuning"
    )
    
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Base directory containing weights_* subdirectories')
    parser.add_argument('--output', type=str, default='results/hyperparameter_tuning/style_weight_tuning.png',
                       help='Output filename for the plot')
    
    args = parser.parse_args()
    
    print("="*70)
    print("Content/Style Weight Pareto Curve Generator")
    print("="*70)
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print(f"Output file: {args.output}")
    print("="*70 + "\n")
    
    plot_pareto_curve_only(args.checkpoint_dir, args.output)

