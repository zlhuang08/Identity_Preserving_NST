#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Learning Rate Comparison Plot Generator

This script creates comparison plots of training curves for different learning rates.
Useful for hyperparameter tuning experiments.

USAGE:
    # Plot all learning rate experiments in checkpoints/
    python plot_learning_curves.py
    
    # Custom checkpoint directory
    python plot_learning_curves.py --checkpoint-dir experiments/lr_sweep
    
    # Custom output filename
    python plot_learning_curves.py --output learning_rate_comparison.png
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


def plot_learning_curves(checkpoint_base_dir, output_file='learning_rate_comparison.png'):
    """
    Plot training curves for all learning rate experiments.
    
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
    
    # Create figure with 2 subplots (train loss and val loss)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(lr_dirs)))
    
    lr_results = []
    
    for idx, lr_dir in enumerate(lr_dirs):
        csv_file = lr_dir / 'training_curves.csv'
        
        if not csv_file.exists():
            print(f"⚠️  Skipping {lr_dir.name}: no training_curves.csv found")
            continue
        
        # Extract learning rate from directory name
        lr_value = extract_lr_from_dirname(lr_dir.name)
        if lr_value is None:
            print(f"⚠️  Could not extract LR from {lr_dir.name}, using directory name")
            lr_label = lr_dir.name
        else:
            # Format learning rate nicely
            if lr_value >= 0.001:
                lr_label = f"LR = {lr_value:.4f}"
            elif lr_value >= 0.0001:
                lr_label = f"LR = {lr_value:.5f}"
            else:
                # Use scientific notation for very small values
                lr_label = f"LR = {lr_value:.0e}"
        
        # Load training curves
        try:
            df = pd.read_csv(csv_file)
            print(f"✓ {lr_dir.name}: {len(df)} epochs, final train loss = {df['train_loss'].iloc[-1]:.4f}, val loss = {df['val_loss'].iloc[-1]:.4f}")
            
            # Plot training loss
            ax1.plot(df['epoch'], df['train_loss'], 
                    color=colors[idx], linewidth=2, marker='o', 
                    markersize=5, label=lr_label, alpha=0.8)
            
            # Plot validation loss
            ax2.plot(df['epoch'], df['val_loss'], 
                    color=colors[idx], linewidth=2, marker='s', 
                    markersize=5, label=lr_label, alpha=0.8)
            
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
            
        except Exception as e:
            print(f"❌ Error reading {csv_file}: {e}")
            continue
    
    if not lr_results:
        print("❌ No valid training curves found!")
        return
    
    # Configure left plot (Training Loss)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Training Loss (log scale)', fontsize=12, fontweight='bold')
    ax1.set_title('Training Loss vs Learning Rate', fontsize=14, fontweight='bold', pad=15)
    ax1.legend(fontsize=10, loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(left=1)
    ax1.set_yscale('log')  # Use log scale for better differentiation
    
    # Configure right plot (Validation Loss)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Validation Loss (log scale)', fontsize=12, fontweight='bold')
    ax2.set_title('Validation Loss vs Learning Rate', fontsize=14, fontweight='bold', pad=15)
    ax2.legend(fontsize=10, loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(left=1)
    ax2.set_yscale('log')  # Use log scale for better differentiation
    
    # Main title
    fig.suptitle('Learning Rate Comparison (Baseline Model, γ=0.0)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Add summary text
    best_result = min(lr_results, key=lambda x: x['final_val_loss'])
    summary_text = (
        f"Best LR: {best_result['lr_label']}\n"
        f"Final Val Loss: {best_result['final_val_loss']:.4f}\n"
        f"Convergence: {best_result['convergence_rate']:.2f}"
    )
    
    ax2.text(0.98, 0.02, summary_text, 
            transform=ax2.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✅ Plot saved to: {output_file}")
    
    # Print analysis
    print("\n" + "="*60)
    print("Learning Rate Analysis")
    print("="*60)
    
    # Sort by final validation loss
    lr_results_sorted = sorted(lr_results, key=lambda x: x['final_val_loss'])
    
    print("\nRanked by Final Validation Loss:")
    print(f"{'Rank':<6} {'Learning Rate':<20} {'Train Loss':<12} {'Val Loss':<12} {'Convergence':<12}")
    print("-" * 68)
    
    for rank, result in enumerate(lr_results_sorted, 1):
        marker = "⭐" if rank == 1 else "  "
        print(f"{marker} {rank:<4} {result['lr_label']:<20} {result['final_train_loss']:<12.4f} "
              f"{result['final_val_loss']:<12.4f} {result['convergence_rate']:<12.2f}")
    
    print("\n" + "="*60)
    print(f"Recommendation: Use {lr_results_sorted[0]['lr_label']} for best performance")
    print("="*60)
    
    return lr_results_sorted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot learning rate comparison from training curves",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:

  # Plot all lr_* experiments in checkpoints/
  python plot_learning_curves.py

  # Custom checkpoint directory
  python plot_learning_curves.py --checkpoint-dir experiments/

  # Custom output filename
  python plot_learning_curves.py --output my_lr_comparison.png
        """
    )
    
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Base directory containing lr_* subdirectories (default: checkpoints)')
    parser.add_argument('--output', type=str, default='results/tuning/learning_rate_comparison.png',
                        help='Output filename for the plot (default: results/tuning/learning_rate_comparison.png)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Learning Rate Comparison Plotter")
    print("="*60)
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print(f"Output file: {args.output}")
    print("="*60 + "\n")
    
    plot_learning_curves(args.checkpoint_dir, args.output)

