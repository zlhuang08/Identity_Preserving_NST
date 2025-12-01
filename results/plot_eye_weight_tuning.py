#!/usr/bin/env python3
"""
Visualize eye weight (β) tuning results with dual y-axes:
- Left y-axis: Face Similarity (train and val)
- Right y-axis: Total Loss (train and val)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set up paths
checkpoints_dir = Path(__file__).parent.parent / "checkpoints" / "hyperparameter_tuning" / "eye_weight"
output_dir = Path(__file__).parent / "hyperparameter_tuning"
output_dir.mkdir(exist_ok=True, parents=True)

# Beta values we tested
beta_values = [0.1, 1, 10, 100]
beta_dirs = {
    0.1: "beta_0_1",
    1: "beta_1",
    10: "beta_10",
    100: "beta_100"
}

# Read final epoch metrics for each beta
results = []
for beta in beta_values:
    csv_path = checkpoints_dir / beta_dirs[beta] / "training_curves.csv"
    
    if not csv_path.exists():
        print(f"⚠️  Warning: {csv_path} not found, skipping β={beta}")
        continue
    
    df = pd.read_csv(csv_path)
    final_epoch = df.iloc[-1]
    
    results.append({
        'beta': beta,
        'val_face_sim': final_epoch['val_similarity'] * 100,  # Convert to percentage
        'train_face_sim': final_epoch['train_similarity'] * 100,
        'val_total_loss': final_epoch['val_loss'],
        'train_total_loss': final_epoch['train_loss'],
    })

# Create DataFrame
df_results = pd.DataFrame(results)

print("Eye Weight (β) Tuning Results:")
print("="*80)
print(df_results.to_string(index=False))
print("="*80)

# Create dual y-axis figure
fig, ax1 = plt.subplots(figsize=(12, 7))

# Left y-axis: Face Similarity
color_train = '#2E86AB'  # Blue for train
color_val = '#A23B72'   # Purple for val
ax1.set_xlabel('Eye Loss Weight (β)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Face Similarity (%)', fontsize=14, fontweight='bold', color='black')
ax1.set_xscale('log')

# Solid lines for face similarity
line1 = ax1.plot(df_results['beta'], df_results['train_face_sim'], 'o-', 
                 linewidth=3, markersize=12, label='Train Face Similarity', 
                 color=color_train, markeredgecolor='black', markeredgewidth=1.5)
line2 = ax1.plot(df_results['beta'], df_results['val_face_sim'], 's-', 
                 linewidth=3, markersize=12, label='Val Face Similarity', 
                 color=color_val, markeredgecolor='black', markeredgewidth=1.5)

ax1.tick_params(axis='y', labelsize=12)
ax1.tick_params(axis='x', labelsize=12)
ax1.grid(True, alpha=0.3, linestyle='--', linewidth=1)

# Highlight optimal β=1
optimal_idx = df_results[df_results['beta'] == 1].index[0]
ax1.axvline(x=1, color='#D81159', linestyle='--', linewidth=2.5, alpha=0.7)
ax1.scatter([1], [df_results.loc[optimal_idx, 'val_face_sim']], 
            color='#FFBC42', s=400, marker='*', zorder=5, edgecolors='black', linewidth=2)

# Right y-axis: Total Loss
ax2 = ax1.twinx()
ax2.set_ylabel('Total Loss', fontsize=14, fontweight='bold', color='black')

# Dashed lines for total loss (matching colors with face similarity)
line3 = ax2.plot(df_results['beta'], df_results['train_total_loss'], 'o--', 
                 linewidth=3, markersize=12, label='Train Total Loss', 
                 color=color_train, markeredgecolor='black', markeredgewidth=1.5)
line4 = ax2.plot(df_results['beta'], df_results['val_total_loss'], 's--', 
                 linewidth=3, markersize=12, label='Val Total Loss', 
                 color=color_val, markeredgecolor='black', markeredgewidth=1.5)

ax2.tick_params(axis='y', labelsize=12)

# Set y-axis limits for better visibility
ax1.set_ylim([40, 95])

# Combine legends
lines = line1 + line2 + line3 + line4
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left', fontsize=11, framealpha=0.95, 
           edgecolor='black', fancybox=True, shadow=True)

plt.tight_layout()

# Save figure
output_path = output_dir / "eye_weight_tuning.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✅ Saved dual-axis chart to: {output_path}")

print("\n" + "="*80)
print("RECOMMENDATION: Use β=1 for final model (optimal trade-off)")
print("="*80)
print(f"β=1 achieves:")
print(f"  • Face Similarity: {df_results.loc[optimal_idx, 'val_face_sim']:.1f}%")
print(f"  • Total Loss: {df_results.loc[optimal_idx, 'val_total_loss']:.4f}")
print("="*80)

plt.close()
