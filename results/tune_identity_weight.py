#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Identity Weight (γ) Tuning Experiment

This script helps determine the optimal identity weight by:
1. Training models with different γ values
2. Generating comparison images
3. Plotting the 3-way trade-off curve (identity vs content vs style)

USAGE:
    python tune_identity_weight.py --epochs 20
"""

import argparse
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json


def train_identity_weight(gamma, epochs, batch_size, checkpoint_base, seed=42):
    """Train a model with specific identity weight"""
    
    gamma_name = f"{gamma:.2f}".replace('.', '_')
    checkpoint_dir = checkpoint_base / f"gamma_{gamma_name}"
    
    print(f"\n{'='*60}")
    print(f"Training: Identity Weight γ = {gamma}")
    print(f"{'='*60}")
    
    cmd = [
        "python", "train_model.py",
        "--content-dir", "data/content",
        "--style-dir", "data/style",
        "--batch-size", str(batch_size),
        "--epochs", str(epochs),
        "--content-weight", "1.0",
        "--style-weight", "10.0",  # Use optimal from previous experiment
        "--identity-weight", str(gamma),
        "--checkpoint-dir", str(checkpoint_dir),
        "--save-interval", str(max(epochs // 2, 5)),
        "--seed", str(seed)  # CRITICAL: Use consistent seed for reproducibility
    ]
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"❌ Training failed for γ={gamma}")
        return None
    
    # Load training curves
    csv_file = checkpoint_dir / "training_curves.csv"
    if csv_file.exists():
        df = pd.read_csv(csv_file)
        
        # Check if identity columns exist (only for γ > 0)
        has_identity = 'train_identity' in df.columns
        
        metrics = {
            'gamma': gamma,
            'gamma_label': f"γ={gamma}",
            'final_train_loss': df['train_loss'].iloc[-1],
            'final_content_loss': df['train_content'].iloc[-1],
            'final_style_loss': df['train_style'].iloc[-1],
            'final_val_loss': df['val_loss'].iloc[-1],
            'final_val_content': df['val_content'].iloc[-1],
            'final_val_style': df['val_style'].iloc[-1],
            'checkpoint_dir': str(checkpoint_dir)
        }
        
        if has_identity and gamma > 0:
            metrics['final_identity_loss'] = df['train_identity'].iloc[-1]
            metrics['final_val_identity'] = df['val_identity'].iloc[-1]
        else:
            metrics['final_identity_loss'] = None
            metrics['final_val_identity'] = None
        
        return metrics
    
    return None


def plot_identity_trade_off(results, metrics_df, output_file):
    """Plot the 3-way trade-off: face similarity vs perceptual quality vs style"""
    
    fig = plt.figure(figsize=(18, 6))
    
    # Left: Face Similarity vs Perceptual Similarity (THE KEY PLOT)
    ax1 = plt.subplot(1, 3, 1)
    
    gammas = metrics_df['gamma'].values
    face_sim = metrics_df['face_similarity'].values
    perceptual_sim = metrics_df['perceptual_similarity'].values
    
    # Plot the curve
    scatter = ax1.scatter(perceptual_sim, face_sim, s=200, c=gammas, 
                         cmap='viridis', alpha=0.8, edgecolors='black', linewidths=2)
    
    # Annotate each point
    for i, (ps, fs, g) in enumerate(zip(perceptual_sim, face_sim, gammas)):
        ax1.annotate(f'γ={g}', (ps, fs), fontsize=9, ha='right',
                    xytext=(-10, 5), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # Highlight γ=0.1
    idx_01 = np.where(gammas == 0.1)[0]
    if len(idx_01) > 0:
        idx = idx_01[0]
        ax1.scatter([perceptual_sim[idx]], [face_sim[idx]], s=400,
                   marker='*', color='red', edgecolors='black', linewidths=2,
                   label='Recommended γ=0.1', zorder=10)
    
    ax1.set_xlabel('Perceptual Similarity (Content Quality)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Face Similarity (Identity Preservation)', fontsize=12, fontweight='bold')
    ax1.set_title('Identity-Content Trade-Off', fontsize=14, fontweight='bold', pad=15)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Identity Weight (γ)', fontsize=10)
    
    # Middle: All three metrics as line plots
    ax2 = plt.subplot(1, 3, 2)
    
    ax2.plot(gammas, face_sim, 'o-', linewidth=2, markersize=8, label='Face Similarity ↑', color='green')
    ax2.plot(gammas, perceptual_sim, 's-', linewidth=2, markersize=8, label='Perceptual Similarity ↑', color='blue')
    
    # Normalize style loss to [0, 1] for comparison
    style_losses = metrics_df['style_loss'].values
    style_norm = 1 - (style_losses - style_losses.min()) / (style_losses.max() - style_losses.min())
    ax2.plot(gammas, style_norm, '^-', linewidth=2, markersize=8, label='Style Quality (normalized) ↑', color='orange')
    
    ax2.set_xlabel('Identity Weight (γ)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Metric Value', fontsize=12, fontweight='bold')
    ax2.set_title('All Metrics vs γ', fontsize=14, fontweight='bold', pad=15)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(-0.02, max(gammas) + 0.02)
    
    # Highlight γ=0.1
    if len(idx_01) > 0:
        ax2.axvline(x=0.1, color='red', linestyle='--', linewidth=2, alpha=0.7, label='γ=0.1')
    
    # Right: Cost-Benefit Analysis
    ax3 = plt.subplot(1, 3, 3)
    
    # Calculate marginal gains
    if len(gammas) > 1:
        face_gains = np.diff(face_sim) / np.diff(gammas)
        perceptual_costs = -np.diff(perceptual_sim) / np.diff(gammas)
        gamma_mid = (gammas[:-1] + gammas[1:]) / 2
        
        width = (gammas[1] - gammas[0]) * 0.8
        x = np.arange(len(gamma_mid))
        
        bars1 = ax3.bar(x - width/2, face_gains, width, label='Face Similarity Gain', 
                       color='green', alpha=0.7, edgecolor='black')
        bars2 = ax3.bar(x + width/2, perceptual_costs, width, label='Perceptual Cost',
                       color='red', alpha=0.7, edgecolor='black')
        
        ax3.set_xlabel('γ Transition', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Marginal Change per γ', fontsize=12, fontweight='bold')
        ax3.set_title('Cost-Benefit Analysis', fontsize=14, fontweight='bold', pad=15)
        ax3.set_xticks(x)
        ax3.set_xticklabels([f'{gammas[i]:.2f}→{gammas[i+1]:.2f}' for i in range(len(gamma_mid))],
                           rotation=45, ha='right')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    fig.suptitle('Identity Weight (γ) Tuning Analysis', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✅ Trade-off curve saved to: {output_file}")


def compute_inference_metrics(checkpoint_dir, output_dir):
    """Run inference and compute face/perceptual similarity metrics"""
    
    import torch
    from model_face_utils import FaceDetector, FaceRecognizer
    from torchvision import transforms
    from PIL import Image
    import torch.nn.functional as F
    
    # Initialize face models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    face_detector = FaceDetector(device=device)
    face_recognizer = FaceRecognizer(device=device)
    
    # Load evaluation images
    eval_dir = Path(output_dir)
    content_dir = Path("data/eval_content")
    
    face_similarities = []
    perceptual_similarities = []
    
    for content_img_path in content_dir.glob("*.jpg"):
        content_name = content_img_path.stem
        
        # Find corresponding stylized images
        stylized_imgs = list(eval_dir.glob(f"{content_name}_*.jpg"))
        
        for stylized_path in stylized_imgs:
            try:
                # Load images
                content_img = Image.open(content_img_path).convert('RGB')
                stylized_img = Image.open(stylized_path).convert('RGB')
                
                transform = transforms.Compose([
                    transforms.Resize((512, 512)),
                    transforms.ToTensor()
                ])
                
                content_tensor = transform(content_img).unsqueeze(0).to(device)
                stylized_tensor = transform(stylized_img).unsqueeze(0).to(device)
                
                # Compute face similarity
                try:
                    # Extract faces (returns tuple: faces, indices, boxes)
                    content_faces, _, _ = face_detector.extract_faces(content_tensor)
                    stylized_faces, _, _ = face_detector.extract_faces(stylized_tensor)
                    
                    if content_faces is not None and stylized_faces is not None:
                        if len(content_faces) > 0 and len(stylized_faces) > 0:
                            # Get embeddings
                            content_emb = face_recognizer.extract_embeddings(content_faces)
                            stylized_emb = face_recognizer.extract_embeddings(stylized_faces)
                            
                            # Compute similarity (unsqueeze to add batch dimension)
                            face_sim = face_recognizer.compute_similarity(
                                content_emb[0].unsqueeze(0), 
                                stylized_emb[0].unsqueeze(0)
                            ).item()
                            face_similarities.append(face_sim)
                except Exception as e:
                    # Face detection can fail for stylized images, that's okay
                    pass
                
                # Compute perceptual similarity (cosine similarity in pixel space)
                perceptual_sim = F.cosine_similarity(
                    content_tensor.flatten(1),
                    stylized_tensor.flatten(1)
                ).item()
                perceptual_similarities.append(perceptual_sim)
                
            except Exception as e:
                print(f"Error processing {stylized_path.name}: {e}")
                continue
    
    return {
        'face_similarity': np.mean(face_similarities) if face_similarities else 0.0,
        'perceptual_similarity': np.mean(perceptual_similarities) if perceptual_similarities else 0.0,
        'num_samples': len(face_similarities)
    }


def print_analysis(results, metrics_df):
    """Print detailed analysis"""
    
    print("\n" + "="*80)
    print("Identity Weight (γ) Analysis")
    print("="*80)
    
    print("\nComplete Results:")
    print(f"{'γ':<8} {'Face Sim':<12} {'Percept Sim':<14} {'Style Loss':<12} {'Val Loss':<12}")
    print("-" * 70)
    
    for _, row in metrics_df.iterrows():
        marker = "⭐" if row['gamma'] == 0.1 else "  "
        print(f"{marker} {row['gamma']:<6.2f} {row['face_similarity']:<12.4f} "
              f"{row['perceptual_similarity']:<14.4f} {row['style_loss']:<12.4f} "
              f"{row['val_loss']:<12.2f}")
    
    print("\n" + "="*80)
    print("Key Findings")
    print("="*80)
    
    # Find optimal γ based on face similarity improvement vs perceptual cost
    if len(metrics_df) > 1:
        baseline = metrics_df[metrics_df['gamma'] == 0.0].iloc[0]
        
        print(f"\nBaseline (γ=0.0):")
        print(f"  Face Similarity: {baseline['face_similarity']:.4f}")
        print(f"  Perceptual Similarity: {baseline['perceptual_similarity']:.4f}")
        
        print(f"\nImprovements over Baseline:")
        for _, row in metrics_df[metrics_df['gamma'] > 0].iterrows():
            face_gain = row['face_similarity'] - baseline['face_similarity']
            perceptual_cost = baseline['perceptual_similarity'] - row['perceptual_similarity']
            ratio = face_gain / perceptual_cost if perceptual_cost > 0 else float('inf')
            
            marker = "⭐" if row['gamma'] == 0.1 else "  "
            print(f"{marker} γ={row['gamma']:.2f}: Face +{face_gain:+.4f}, Perceptual {perceptual_cost:+.4f}, Ratio: {ratio:.2f}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tune identity weight γ",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs for each training run (default: 20)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size (default: 64)')
    parser.add_argument('--checkpoint-base', type=str, default='checkpoints',
                        help='Base directory for checkpoints (default: checkpoints)')
    parser.add_argument('--output', type=str, default='results/tuning/identity_comparison.png',
                        help='Output file for plots')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("Identity Weight (γ) Tuning Experiment")
    print("="*80)
    print(f"Epochs per model: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output plot: {args.output}")
    print("="*80)
    
    # Define gamma values to test (logarithmic spacing based on weighted contribution analysis)
    # Extended range: Testing if γ>1000 continues to improve or plateaus/degrades
    gamma_values = [10000.0, 100000.0]
    
    print(f"\nTesting {len(gamma_values)} identity weights: {gamma_values}")
    print("This will take approximately {:.0f} minutes".format(len(gamma_values) * args.epochs * 2))
    print("\nNote: Using logarithmic spacing to quickly find optimal range")
    
    # Train models
    results = []
    checkpoint_base = Path(args.checkpoint_base)
    
    for gamma in gamma_values:
        metrics = train_identity_weight(gamma, args.epochs, args.batch_size, checkpoint_base, args.seed)
        if metrics:
            results.append(metrics)
    
    if not results:
        print("\n❌ No successful training runs. Exiting.")
        exit(1)
    
    # Save training results
    results_file = Path(args.output).parent / "identity_tuning_results.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate inference and compute metrics
    print("\n" + "="*80)
    print("Generating Inference Images and Computing Metrics")
    print("="*80)
    
    metrics_list = []
    for result in results:
        gamma = result['gamma']
        gamma_name = f"{gamma:.2f}".replace('.', '_')
        checkpoint_dir = Path(result['checkpoint_dir'])
        output_dir = Path(f"results/tuning/identity_visuals/gamma_{gamma_name}")
        
        print(f"\nProcessing γ={gamma}...")
        
        # Run inference
        cmd = [
            "python", "eval_inference.py",
            "--checkpoint", str(checkpoint_dir / "final_model.pth"),
            "--content", "data/eval_content",
            "--style", "data/style",
            "--output", str(output_dir)
        ]
        subprocess.run(cmd, capture_output=True)
        
        # Compute metrics
        inference_metrics = compute_inference_metrics(checkpoint_dir, output_dir)
        
        metrics_list.append({
            'gamma': gamma,
            'face_similarity': inference_metrics['face_similarity'],
            'perceptual_similarity': inference_metrics['perceptual_similarity'],
            'style_loss': result['final_val_style'],
            'val_loss': result['final_val_loss']
        })
        
        print(f"  Face Similarity: {inference_metrics['face_similarity']:.4f}")
        print(f"  Perceptual Similarity: {inference_metrics['perceptual_similarity']:.4f}")
    
    metrics_df = pd.DataFrame(metrics_list)
    
    # Plot trade-off curves
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plot_identity_trade_off(results, metrics_df, args.output)
    
    # Print analysis
    print_analysis(results, metrics_df)
    
    print("\n" + "="*80)
    print("🎉 Identity Weight Tuning Complete!")
    print("="*80)
    print(f"\n📊 Quantitative analysis: {args.output}")
    print(f"💾 Detailed results: {results_file}")

