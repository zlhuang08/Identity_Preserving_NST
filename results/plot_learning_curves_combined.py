#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Learning Rate Comparison Plot Generator (Combined Train & Val)

This script creates a single comparison plot showing both training and validation
curves for different learning rates on the same axes.

USAGE:
    python plot_learning_curves_combined.py
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


def plot_learning_curves(checkpoint_base_dir, output_file='learning_rate_comparison_combined.png'):
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
    
    # Find optimal LR (lowest validation loss)
    if temp_results:
        optimal_lr = min(temp_results, key=lambda x: x['final_val_loss'])['lr_value']
    else:
        optimal_lr = None
    
    # Second pass: plot with optimal highlighted in red
    for result in temp_results:
        idx = result['idx']
        lr_dir = result['lr_dir']
        lr_value = result['lr_value']
        df = result['df']
        
        # Format learning rate nicely
        if lr_value >= 0.001:
            lr_label = f"{lr_value:.4f}"
        elif lr_value >= 0.0001:
            lr_label = f"{lr_value:.5f}"
        else:
            lr_label = f"{lr_value:.0e}"
        
        print(f"✓ {lr_dir.name}: {len(df)} epochs, final train loss = {df['train_loss'].iloc[-1]:.4f}, val loss = {df['val_loss'].iloc[-1]:.4f}")
        
        # Determine color and style
        is_optimal = (lr_value == optimal_lr)
        if is_optimal:
            color = '#D81159'  # Red for optimal
            linewidth = 3.5
            alpha = 1.0
            zorder = 10  # Plot on top
            lr_label_display = f'{lr_label} ⭐ OPTIMAL'
        else:
            color = colors[idx]
            linewidth = 2.5
            alpha = 0.7
            zorder = 5
            lr_label_display = lr_label
        
        # Plot training loss (solid line)
        ax.plot(df['epoch'], df['train_loss'], 
               color=color, linewidth=linewidth, linestyle='-',
               marker='o', markersize=6 if is_optimal else 5, alpha=alpha,
               label=f'LR={lr_label_display} (train)', zorder=zorder)
        
        # Plot validation loss (dashed line, same color)
        ax.plot(df['epoch'], df['val_loss'], 
               color=color, linewidth=linewidth, linestyle='--',
               marker='s', markersize=6 if is_optimal else 5, alpha=alpha,
               label=f'LR={lr_label_display} (val)', zorder=zorder)
        
        # Store results for analysis
        lr_results.append({
            'lr': lr_value,
            'lr_label': lr_label,
            'final_train_loss': df['train_loss'].iloc[-1],
            'final_val_loss': df['val_loss'].iloc[-1],
            'initial_train_loss': df['train_loss'].iloc[0],
            'convergence_rate': df['train_loss'].iloc[0] - df['train_loss'].iloc[-1],
            'epochs': len(df)
        })
    
    if not lr_results:
        print("❌ No valid training curves found!")
        return
    
    # Configure plot
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Loss (log scale)', fontsize=14, fontweight='bold')
    ax.set_title('Learning Rate Comparison: Training vs Validation Loss', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Legend with two columns for better organization
    ax.legend(fontsize=9, loc='upper right', ncol=2, framealpha=0.95,
             columnspacing=1.0, handlelength=2.5)
    
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    ax.set_xlim(left=1)
    ax.set_yscale('log')  # Use log scale for better differentiation
    
    # Add best LR annotation
    best_result = min(lr_results, key=lambda x: x['final_val_loss'])
    summary_text = (
        f"Best Learning Rate: {best_result['lr_label']}\n"
        f"Final Val Loss: {best_result['final_val_loss']:.4f}\n"
        f"Final Train Loss: {best_result['final_train_loss']:.4f}\n"
        f"Convergence: {best_result['convergence_rate']:.2f}\n\n"
        f"Solid lines: Training Loss\n"
        f"Dashed lines: Validation Loss"
    )
    
    ax.text(0.02, 0.98, summary_text, 
           transform=ax.transAxes, fontsize=10,
           verticalalignment='top', horizontalalignment='left',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, 
                    edgecolor='black', linewidth=1.5))
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✅ Combined plot saved to: {output_file}")
    
    # Print analysis
    print("\n" + "="*70)
    print("Learning Rate Analysis")
    print("="*70)
    
    # Sort by final validation loss
    lr_results_sorted = sorted(lr_results, key=lambda x: x['final_val_loss'])
    
    print("\nRanked by Final Validation Loss:")
    print(f"{'Rank':<6} {'LR':<15} {'Train Loss':<12} {'Val Loss':<12} {'Convergence':<12}")
    print("-" * 70)
    
    for rank, result in enumerate(lr_results_sorted, 1):
        marker = "⭐" if rank == 1 else "  "
        print(f"{marker} {rank:<4} {result['lr_label']:<15} {result['final_train_loss']:<12.4f} "
              f"{result['final_val_loss']:<12.4f} {result['convergence_rate']:<12.2f}")
    
    print("\n" + "="*70)
    print(f"Recommendation: Use LR = {lr_results_sorted[0]['lr_label']} for best performance")
    print("="*70)
    
    plt.close()
    return lr_results_sorted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot combined learning rate comparison (train + val on same plot)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Base directory containing lr_* subdirectories (default: checkpoints)')
    parser.add_argument('--output', type=str, default='results/tuning/learning_rate_comparison_combined.png',
                        help='Output filename for the plot')
    
    args = parser.parse_args()
    
    print("="*70)
    print("Learning Rate Comparison Plotter (Combined Train + Val)")
    print("="*70)
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print(f"Output file: {args.output}")
    print("="*70 + "\n")
    
    plot_learning_curves(args.checkpoint_dir, args.output)

