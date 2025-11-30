#!/usr/bin/env python3
"""
Visualize eye weight (β) tuning results with dual y-axes:
- Left y-axis: Face Similarity (train and val)
- Right y-axis: Eye Loss (train and val)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set up paths
checkpoints_dir = Path(__file__).parent.parent / "checkpoints"
output_dir = Path(__file__).parent / "tuning"
output_dir.mkdir(exist_ok=True)

# Beta values we tested
beta_values = [0.1, 1, 10, 100]

# Read final epoch metrics for each beta
results = []
for beta in beta_values:
    csv_path = checkpoints_dir / f"face_aware_identity_eye_beta{beta}" / "training_curves.csv"
    if not csv_path.exists():
        # Handle decimal formatting
        csv_path = checkpoints_dir / f"face_aware_identity_eye_beta{str(beta).replace('.', '_')}" / "training_curves.csv"
    
    df = pd.read_csv(csv_path)
    final_epoch = df.iloc[-1]
    
    results.append({
        'beta': beta,
        'val_face_sim': final_epoch['val_similarity'] * 100,  # Convert to percentage
        'train_face_sim': final_epoch['train_similarity'] * 100,
        'val_eye_loss': final_epoch['val_eye'],
        'train_eye_loss': final_epoch['train_eye'],
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
color1 = '#2E86AB'  # Blue
color2 = '#A23B72'  # Purple
ax1.set_xlabel('Eye Loss Weight (β)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Face Similarity (%)', fontsize=14, fontweight='bold', color='black')
ax1.set_xscale('log')

line1 = ax1.plot(df_results['beta'], df_results['train_face_sim'], 'o-', 
                 linewidth=3, markersize=12, label='Train Face Similarity', 
                 color=color1, markeredgecolor='black', markeredgewidth=1.5)
line2 = ax1.plot(df_results['beta'], df_results['val_face_sim'], 's-', 
                 linewidth=3, markersize=12, label='Val Face Similarity', 
                 color=color2, markeredgecolor='black', markeredgewidth=1.5)

ax1.tick_params(axis='y', labelsize=12)
ax1.tick_params(axis='x', labelsize=12)
ax1.grid(True, alpha=0.3, linestyle='--', linewidth=1)

# Highlight optimal β=1
optimal_idx = df_results[df_results['beta'] == 1].index[0]
ax1.axvline(x=1, color='red', linestyle='--', linewidth=2.5, alpha=0.7)
ax1.scatter([1], [df_results.loc[optimal_idx, 'val_face_sim']], 
            color='red', s=300, marker='*', zorder=5, edgecolors='black', linewidth=2)

# Right y-axis: Eye Loss
ax2 = ax1.twinx()
color3 = '#06A77D'  # Green
color4 = '#F77F00'  # Orange
ax2.set_ylabel('Eye Loss (MSE)', fontsize=14, fontweight='bold', color='black')

line3 = ax2.plot(df_results['beta'], df_results['train_eye_loss'], '^-', 
                 linewidth=3, markersize=12, label='Train Eye Loss', 
                 color=color3, markeredgecolor='black', markeredgewidth=1.5)
line4 = ax2.plot(df_results['beta'], df_results['val_eye_loss'], 'v-', 
                 linewidth=3, markersize=12, label='Val Eye Loss', 
                 color=color4, markeredgecolor='black', markeredgewidth=1.5)

ax2.tick_params(axis='y', labelsize=12)

# Set y-axis limits for better visibility
ax1.set_ylim([40, 95])
ax2.set_ylim([3.5, 8.5])

# Combine legends
lines = line1 + line2 + line3 + line4
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left', fontsize=11, framealpha=0.95, 
           edgecolor='black', fancybox=True, shadow=True)

# Title
plt.title('Eye Loss Weight (β) Tuning: Face Similarity vs Eye Loss', 
          fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()

# Save figure
output_path = output_dir / "beta_tuning_dual_axis.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✅ Saved dual-axis chart to: {output_path}")

print("\n" + "="*80)
print("RECOMMENDATION: Use β=1 for final model (3_all_combined)")
print("="*80)

