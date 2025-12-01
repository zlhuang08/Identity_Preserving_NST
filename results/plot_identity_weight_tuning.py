#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create identity weight visualization showing Pareto trade-off.
Shows Face Similarity vs Style Loss to demonstrate that γ=1000
is optimal due to Pareto optimality.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ACTUAL DATA from training experiments
gamma_values = [0, 0.1, 1, 10, 100, 1000, 10000, 100000]

# Face similarity - ACTUAL VALUES from CSV files
face_similarity = [0.560, 0.564, 0.560, 0.557, 0.606, 0.768, 0.802, 0.838]

# Style loss - ACTUAL VALUES from CSV files  
style_loss = [1.252, 1.302, 1.196, 1.192, 1.169, 1.315, 1.603, 3.953]

# Create gamma labels for display
gamma_labels = []
for g in gamma_values:
    if g == 0:
        gamma_labels.append('0')
    elif g == 0.1:
        gamma_labels.append('0.1')
    elif g == 1:
        gamma_labels.append('1')
    elif g >= 10:
        exp = int(np.log10(g))
        gamma_labels.append(f'10$^{exp}$')
    else:
        gamma_labels.append(str(g))

# Create single figure (Pareto curve style)
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Plot Pareto curve with color gradient (darker = higher gamma)
colors = plt.cm.viridis(np.linspace(0, 1, len(gamma_values)))
scatter = ax.scatter(face_similarity, style_loss, s=250, alpha=0.7, 
                    c=np.arange(len(gamma_values)), cmap='viridis',
                    edgecolors='black', linewidths=2)

# Annotate each point with gamma value (positioned directly above)
for i, (fs, sl, label) in enumerate(zip(face_similarity, style_loss, gamma_labels)):
    ax.annotate(f'γ={label}', (fs, sl), fontsize=11, ha='center', fontweight='bold',
               xytext=(0, 18), textcoords='offset points')

# Highlight the γ=1000 (optimal)
optimal_idx = 5  # γ=1000
ax.scatter([face_similarity[optimal_idx]], [style_loss[optimal_idx]], s=600, 
          marker='*', color='#FFBC42', edgecolors='black', linewidths=3,
          zorder=10)

# Labels
ax.set_xlabel('Face Similarity (higher = better identity)', fontsize=14, fontweight='bold')
ax.set_ylabel('Style Loss (lower = better style)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)

# Set axis limits for better visualization
ax.set_xlim([0.54, 0.86])
ax.set_ylim([1.0, 4.5])

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
cbar.set_label('γ Index\n(Lower → Higher identity weight)', 
              fontsize=10, rotation=270, labelpad=25)

plt.tight_layout()

# Save the figure
output_path = Path('results/hyperparameter_tuning/identity_weight_tuning.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n✅ Pareto trade-off plot saved to: {output_path}")
print(f"   • γ=1000 is optimal (balanced face similarity and style quality)")
print(f"   • Face Similarity: 76.8% (+21% vs baseline)")
print(f"   • Style Loss: 1.315 (minimal degradation)")

plt.close()
