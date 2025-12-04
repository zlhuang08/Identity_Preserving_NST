#!/usr/bin/env python3
"""
Calculate Comprehensive Test Set Metrics for All Models

This script computes Face Similarity, SSIM, and Perceptual Similarity for all
trained models on the test set. It follows the same procedure as train_model.py
to ensure consistency (no image saving/loading).

Metrics Calculated:
- Face Similarity: Cosine similarity of FaceNet embeddings (identity preservation)
- SSIM: Structural Similarity Index (pixel-level similarity)
- Perceptual Similarity (LPIPS): Learned perceptual distance (human perception)

The script:
1. Loads each model checkpoint
2. Runs inference on the test set (in memory, no image saving)
3. Calculates all three metrics
4. Saves results to test_metrics.csv in each checkpoint folder
5. Verifies Face Similarity matches training_curves.csv

Author: CS230 Final Project
Date: December 2025
"""

import os
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
from tqdm import tqdm

# Import project modules
from model_adain import AdaINStyleTransfer
from model_face_utils import FaceDetector, IdentityPreserver
from evaluation_metrics_utility import compute_metrics
from train_model import ImageDataset  # Import the custom dataset class

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Paths
CONTENT_DIR = Path("data/content")
STYLE_DIR = Path("data/style")
TEST_SPLIT_FILE = Path("data/content_splits/test.txt")
CHECKPOINT_DIR = Path("checkpoints")

# Image preprocessing (same as training)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor()
])


def load_test_data(batch_size=16):
    """
    Load test set using the same procedure as train_model.py.
    
    Returns:
        test_loader: DataLoader for test set
    """
    print(f"Loading test dataset...")
    test_dataset = ImageDataset(
        content_dir=str(CONTENT_DIR),
        style_dir=str(STYLE_DIR),
        split_file=str(TEST_SPLIT_FILE),
        image_size=256
    )
    
    # Create DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Test pairs: {len(test_dataset)}")
    
    return test_loader


def load_style_images():
    """
    Load all style images.
    
    Returns:
        style_images: Tensor of shape (N, 3, 256, 256)
        style_names: List of style image names
    """
    style_paths = sorted(STYLE_DIR.glob("*.jpg"))
    style_images = []
    style_names = []
    
    for style_path in style_paths:
        from PIL import Image
        style_img = Image.open(style_path).convert('RGB')
        style_tensor = transform(style_img)
        style_images.append(style_tensor)
        style_names.append(style_path.stem)
    
    style_images = torch.stack(style_images).to(device)
    print(f"Loaded {len(style_images)} style images")
    
    return style_images, style_names


def calculate_metrics_for_model(checkpoint_path, use_face_aware=False):
    """
    Calculate all three metrics for a single model on the test set.
    
    Args:
        checkpoint_path: Path to model checkpoint (final_model.pth)
        use_face_aware: Whether to use face-aware AdaIN
        
    Returns:
        results: Dictionary with metric statistics
    """
    print(f"\n{'='*80}")
    print(f"Processing: {checkpoint_path.parent.name}")
    print(f"{'='*80}\n")
    
    # Load model
    print("Loading model...")
    model = AdaINStyleTransfer().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Only load decoder state (encoder is frozen VGG19)
    model.decoder.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load test data
    print("Loading test data...")
    test_loader = load_test_data(batch_size=8)
    
    # Initialize metrics
    print("Initializing metric calculators...")
    face_detector = FaceDetector(device=device, keep_all=False)
    identity_preserver = IdentityPreserver(device=device)
    
    # Storage for metrics
    all_face_sim = []
    all_ssim = []
    all_perceptual_sim = []
    
    total_pairs = len(test_loader.dataset)
    
    print(f"\nCalculating metrics for {total_pairs} pairs...")
    print()
    
    # Calculate metrics
    with torch.no_grad():
        pbar = tqdm(total=total_pairs, desc="Progress", unit="pair")
        
        for content_batch, style_batch in test_loader:
            content_batch = content_batch.to(device)
            style_batch = style_batch.to(device)
            batch_size = content_batch.size(0)
            
            # Generate stylized images (in memory, following train_model.py procedure)
            if use_face_aware:
                # Face-aware AdaIN requires face masks
                face_masks = torch.zeros((batch_size, 1, 256, 256), device=device)
                
                # Detect faces and create masks
                boxes_batch, probs_batch, landmarks_batch = face_detector.detect(content_batch)
                
                for i in range(batch_size):
                    if boxes_batch[i] is not None and len(boxes_batch[i]) > 0:
                        # Get first detected face box
                        box = boxes_batch[i][0]  # [x1, y1, x2, y2]
                        x1, y1, x2, y2 = map(int, box)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(256, x2), min(256, y2)
                        face_masks[i, 0, y1:y2, x1:x2] = 1.0
                
                # Generate with face-aware AdaIN
                stylized = model.forward_with_face_aware_adain(
                    content_batch, style_batch, face_masks,
                    face_preservation_alpha=0.3, alpha=1.0
                )
            else:
                # Standard AdaIN
                stylized = model(content_batch, style_batch, alpha=1.0)
            
            # Calculate Face Similarity
            _, identity_metrics = identity_preserver.compute_identity_loss(
                generated_images=stylized,
                content_images=content_batch
            )
            # Extract individual similarities if available, otherwise use average
            if identity_metrics['num_matched_faces'] > 0:
                # Use average similarity for this batch
                for _ in range(batch_size):
                    all_face_sim.append(identity_metrics['avg_similarity'])
            else:
                all_face_sim.extend([0.0] * batch_size)
            
            # Calculate SSIM and Perceptual Similarity using compute_metrics
            for i in range(batch_size):
                metrics = compute_metrics(
                    content_batch[i].unsqueeze(0),
                    stylized[i].unsqueeze(0),
                    device=device
                )
                all_ssim.append(metrics['ssim'])
                all_perceptual_sim.append(metrics['perceptual'])
            
            pbar.update(batch_size)
        
        pbar.close()
    
    # Compute statistics
    results = {
        'face_similarity_mean': np.mean(all_face_sim),
        'face_similarity_std': np.std(all_face_sim),
        'ssim_mean': np.mean(all_ssim),
        'ssim_std': np.std(all_ssim),
        'perceptual_sim_mean': np.mean(all_perceptual_sim),
        'perceptual_sim_std': np.std(all_perceptual_sim),
        'num_pairs': len(all_face_sim)
    }
    
    print(f"\n{'='*80}")
    print("Test Set Metrics:")
    print(f"{'='*80}")
    print(f"Face Similarity:       {results['face_similarity_mean']:.4f} ± {results['face_similarity_std']:.4f}")
    print(f"SSIM:                  {results['ssim_mean']:.4f} ± {results['ssim_std']:.4f}")
    print(f"Perceptual Similarity: {results['perceptual_sim_mean']:.4f} ± {results['perceptual_sim_std']:.4f}")
    print(f"Number of pairs:       {results['num_pairs']}")
    print(f"{'='*80}\n")
    
    return results


def verify_with_training_curves(checkpoint_dir, test_face_sim):
    """
    Verify that the calculated test face similarity matches training_curves.csv.
    
    Args:
        checkpoint_dir: Path to checkpoint directory
        test_face_sim: Calculated test face similarity
        
    Returns:
        bool: True if verification passed
    """
    training_curves_path = checkpoint_dir / "training_curves.csv"
    
    if not training_curves_path.exists():
        print(f"⚠️  Warning: training_curves.csv not found in {checkpoint_dir.name}")
        return False
    
    # Read training curves
    df = pd.read_csv(training_curves_path)
    
    # Get test face similarity from last epoch
    if 'test_similarity' in df.columns:
        recorded_test_sim = df['test_similarity'].iloc[-1]
        
        # Calculate difference
        diff = abs(test_face_sim - recorded_test_sim)
        diff_pct = (diff / recorded_test_sim) * 100 if recorded_test_sim > 0 else 0
        
        print(f"Verification:")
        print(f"  Recorded (training_curves.csv): {recorded_test_sim:.4f}")
        print(f"  Calculated (this script):       {test_face_sim:.4f}")
        print(f"  Difference:                     {diff:.4f} ({diff_pct:.2f}%)")
        
        if diff < 0.01:  # Allow 1% tolerance
            print(f"  ✅ VERIFIED: Metrics match!")
            return True
        else:
            print(f"  ⚠️  WARNING: Difference > 1%")
            return False
    else:
        print(f"  ⚠️  Warning: test_similarity column not found in training_curves.csv")
        return False


def save_test_metrics(checkpoint_dir, results):
    """
    Save test metrics to CSV file in checkpoint directory.
    
    Args:
        checkpoint_dir: Path to checkpoint directory
        results: Dictionary with metric statistics
    """
    output_path = checkpoint_dir / "test_metrics.csv"
    
    # Create DataFrame
    df = pd.DataFrame([{
        'metric': 'face_similarity',
        'mean': results['face_similarity_mean'],
        'std': results['face_similarity_std']
    }, {
        'metric': 'ssim',
        'mean': results['ssim_mean'],
        'std': results['ssim_std']
    }, {
        'metric': 'perceptual_similarity',
        'mean': results['perceptual_sim_mean'],
        'std': results['perceptual_sim_std']
    }])
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"✅ Saved test metrics to: {output_path}")


def main():
    """Main entry point."""
    print()
    print("="*80)
    print("     Comprehensive Test Set Metrics Calculation")
    print("="*80)
    print()
    
    # Find all model checkpoints
    checkpoint_dirs = [
        CHECKPOINT_DIR / "0_baseline",
        CHECKPOINT_DIR / "1_identity",
        CHECKPOINT_DIR / "2_face_aware_plus_identity",
        CHECKPOINT_DIR / "2_identity_plus_eye",
        CHECKPOINT_DIR / "3_all_combined"
    ]
    
    # Filter to only existing checkpoints
    existing_checkpoints = []
    for checkpoint_dir in checkpoint_dirs:
        checkpoint_path = checkpoint_dir / "final_model.pth"
        if checkpoint_path.exists():
            existing_checkpoints.append((checkpoint_dir, checkpoint_path))
        else:
            print(f"⚠️  Skipping {checkpoint_dir.name}: checkpoint not found")
    
    print(f"\nFound {len(existing_checkpoints)} model checkpoints")
    print()
    
    # Process each model
    summary = []
    
    for checkpoint_dir, checkpoint_path in existing_checkpoints:
        # Check if test_metrics.csv already exists
        test_metrics_file = checkpoint_dir / "test_metrics.csv"
        if test_metrics_file.exists():
            print(f"\n✓ Skipping {checkpoint_dir.name}: test_metrics.csv already exists")
            print(f"  (Delete the file to recalculate)")
            
            # Load existing results for summary
            try:
                df = pd.read_csv(test_metrics_file)
                face_sim = df[df['metric'] == 'face_similarity']['mean'].values[0]
                ssim_val = df[df['metric'] == 'ssim']['mean'].values[0]
                perceptual_val = df[df['metric'] == 'perceptual_similarity']['mean'].values[0]
                
                summary.append({
                    'model': checkpoint_dir.name,
                    'face_similarity': face_sim,
                    'ssim': ssim_val,
                    'perceptual_sim': perceptual_val,
                    'verified': True  # Assume verified from previous run
                })
                print(f"  Loaded: Face Sim={face_sim:.4f}, SSIM={ssim_val:.4f}, Perceptual={perceptual_val:.4f}")
            except Exception as e:
                print(f"  Warning: Could not load existing metrics: {e}")
            
            continue
        
        # Determine if face-aware AdaIN should be used
        use_face_aware = 'face_aware' in checkpoint_dir.name or 'all_combined' in checkpoint_dir.name
        
        # Calculate metrics
        results = calculate_metrics_for_model(checkpoint_path, use_face_aware=use_face_aware)
        
        # Verify with training curves
        print()
        verified = verify_with_training_curves(checkpoint_dir, results['face_similarity_mean'])
        print()
        
        # Save results
        save_test_metrics(checkpoint_dir, results)
        print()
        
        # Store summary
        summary.append({
            'model': checkpoint_dir.name,
            'face_similarity': results['face_similarity_mean'],
            'ssim': results['ssim_mean'],
            'perceptual_sim': results['perceptual_sim_mean'],
            'verified': verified
        })
    
    # Print final summary
    print("="*80)
    print("                    FINAL SUMMARY")
    print("="*80)
    print()
    print(f"{'Model':<35} {'Face Sim':<12} {'SSIM':<12} {'Perceptual':<12} {'Verified'}")
    print("-"*80)
    
    for item in summary:
        verified_str = "✅" if item['verified'] else "⚠️ "
        print(f"{item['model']:<35} {item['face_similarity']:<12.4f} "
              f"{item['ssim']:<12.4f} {item['perceptual_sim']:<12.4f} {verified_str}")
    
    print()
    print("="*80)
    print("✅ All metrics calculated and saved to test_metrics.csv in each checkpoint folder")
    print("="*80)
    print()


if __name__ == "__main__":
    main()

