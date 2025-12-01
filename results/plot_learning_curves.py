#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Learning Rate Comparison Plot Generator (Combined Train & Val)

This script creates a single comparison plot showing both training and validation
curves for different learning rates on the same axes.

USAGE:
    python plot_learning_curves.py
"""

import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import re


def extract_lr_from_dirname(dirname):
    """Extract learning rate value from directory name like 'lr_0.0001' or 'lr_1e-5'"""
    # Try scientific notation first (lr_1e-5)
    match = re.search(r'lr_(\d+\.?\d*)e-(\d+)', dirname)
    if match:
        base = float(match.group(1))
        exp = int(match.group(2))
        return base * (10 ** -exp)
    
    # Try decimal notation (lr_0.0001)
    match = re.search(r'lr_([\d.]+)', dirname)
    if match:
        return float(match.group(1))
    
    return None


def plot_learning_curves(checkpoint_base_dir, output_file='learning_rate_tuning.png'):
    """
    Plot training and validation curves for all learning rate experiments on single plot.
    
    Args:
        checkpoint_base_dir: Directory containing lr_* subdirectories
        output_file: Where to save the comparison plot
    """
    checkpoint_base_dir = Path(checkpoint_base_dir)
    
    # Find all lr_* directories and sort by learning rate value (smallest to largest)
    lr_dirs = [d for d in checkpoint_base_dir.glob('lr_*') if d.is_dir()]
    
    # Sort by extracted learning rate value
    lr_dirs_with_values = []
    for d in lr_dirs:
        lr_value = extract_lr_from_dirname(d.name)
        if lr_value is not None:
            lr_dirs_with_values.append((lr_value, d))
    
    # Sort by LR value (smallest to largest)
    lr_dirs_with_values.sort(key=lambda x: x[0])
    lr_dirs = [d for _, d in lr_dirs_with_values]
    
    if not lr_dirs:
        print(f"❌ No learning rate experiment directories found in {checkpoint_base_dir}")
        print("   Expected directories like: lr_1e-5, lr_0.0001, etc.")
        return
    
    print(f"Found {len(lr_dirs)} learning rate experiments (sorted smallest to largest):")
    
    # Create single figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    
    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(lr_dirs)))
    
    lr_results = []
    temp_results = []  # Store temporarily to find best LR
    
    # First pass: load all data to find optimal LR
    for idx, lr_dir in enumerate(lr_dirs):
        csv_file = lr_dir / 'training_curves.csv'
        
        if not csv_file.exists():
            continue
        
        lr_value = extract_lr_from_dirname(lr_dir.name)
        if lr_value is None:
            continue
            
        try:
            df = pd.read_csv(csv_file)
            temp_results.append({
                'idx': idx,
                'lr_dir': lr_dir,
                'lr_value': lr_value,
                'df': df,
                'final_val_loss': df['val_loss'].iloc[-1]
            })
        except:
            continue
    
    # Find optimal LR (lowest final validation loss)
    if temp_results:
        optimal_result = min(temp_results, key=lambda x: x['final_val_loss'])
        optimal_lr_value = optimal_result['lr_value']
    else:
        optimal_lr_value = None
    
    # Second pass: plot all curves
    for result in temp_results:
        idx = result['idx']
        lr_dir = result['lr_dir']
        lr_value = result['lr_value']
        df = result['df']
        
        # Format learning rate for display
        if lr_value >= 1e-4:
            lr_display = f"{lr_value:.4f}"
        else:
            lr_display = f"{lr_value:.0e}"
        
        is_optimal = (lr_value == optimal_lr_value)
        
        # Plot styling
        train_line_style = '-'
        val_line_style = '--'
        linewidth = 3.5 if is_optimal else 2.5
        alpha = 1.0 if is_optimal else 0.7
        zorder = 10 if is_optimal else 3
        color = '#D81159' if is_optimal else colors[idx]  # Red for optimal, tab10 colors for others
        
        # Plot train loss (solid line)
        ax.plot(df['epoch'], df['train_loss'], 
               linestyle=train_line_style, linewidth=linewidth, 
               color=color, alpha=alpha, zorder=zorder,
               label=f'LR={lr_display} (Train)')
        
        # Plot val loss (dashed line)
        ax.plot(df['epoch'], df['val_loss'], 
               linestyle=val_line_style, linewidth=linewidth, 
               color=color, alpha=alpha, zorder=zorder,
               label=f'LR={lr_display} (Val)')
        
        lr_results.append({
            'lr': lr_value,
            'lr_display': lr_display,
            'final_val_loss': df['val_loss'].iloc[-1],
            'final_train_loss': df['train_loss'].iloc[-1]
        })
        
        print(f"  {idx+1}. LR={lr_display}: Train Loss={df['train_loss'].iloc[-1]:.4f}, Val Loss={df['val_loss'].iloc[-1]:.4f}")
    
    # Styling
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Loss (log scale)', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    
    # Legend with custom description for line styles
    handles, labels = ax.get_legend_handles_labels()
    # Add custom legend entries for line styles
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color='gray', linestyle='-', linewidth=2.5, label='Train'),
        Line2D([0], [0], color='gray', linestyle='--', linewidth=2.5, label='Val')
    ]
    
    # Combine custom lines with actual plot handles/labels
    all_handles = custom_lines + handles
    all_labels = ['Solid: Train', 'Dashed: Val'] + labels
    
    ax.legend(all_handles, all_labels, loc='upper right', fontsize=9, 
             framealpha=0.95, ncol=2)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✅ Combined plot saved to: {output_path}")
    
    # Summary
    print("\n" + "="*70)
    print("Learning Rate Analysis Summary")
    print("="*70)
    print(f"{'Learning Rate':<15} {'Final Train Loss':<18} {'Final Val Loss':<15}")
    print("-"*70)
    for lr in lr_results:
        marker = '⭐' if lr['lr'] == optimal_lr_value else '  '
        print(f"{marker} {lr['lr_display']:<13} {lr['final_train_loss']:<18.4f} {lr['final_val_loss']:<15.4f}")
    print("="*70)
    if optimal_lr_value:
        print(f"Optimal Learning Rate: {[lr['lr_display'] for lr in lr_results if lr['lr'] == optimal_lr_value][0]}")
    print("="*70)
    
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot combined training and validation curves for learning rate experiments"
    )
    
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/hyperparameter_tuning/learning_rate',
                       help='Base directory containing lr_* subdirectories')
    parser.add_argument('--output', type=str, default='results/hyperparameter_tuning/learning_rate_tuning.png',
                       help='Output filename for the combined plot')
    
    args = parser.parse_args()
    
    print("="*70)
    print("Learning Rate Comparison Plot Generator (Combined)")
    print("="*70)
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print(f"Output file: {args.output}")
    print("="*70 + "\n")
    
    plot_learning_curves(args.checkpoint_dir, args.output)
