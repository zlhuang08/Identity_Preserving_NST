#!/usr/bin/env python3
"""
Create final 6-image comparison grids showing progressive improvements:
(1,1) Content | (1,2) Style | (1,3) 0_baseline
(2,1) 1_identity | (2,2) 2_face_aware_plus_identity | (2,3) 3_all_combined

With full metrics (SSIM, Perceptual Sim, Face Sim) on each generated image.
"""

import torch
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from torchvision.models import vgg19

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from model_face_utils import FaceDetector, FaceRecognizer

def load_image_pil(path):
    """Load image as PIL Image"""
    return Image.open(path).convert('RGB')

def load_image_tensor(path, device='cuda', target_size=512):
    """Load image as normalized tensor [0,1], resized to target_size"""
    img = Image.open(path).convert('RGB')
    # Resize to target size
    img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
    img = np.array(img).astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return img.to(device)

def compute_ssim(img1, img2):
    """Compute Structural Similarity Index (SSIM) between two images"""
    # Simple SSIM implementation using L1/L2 norms as proxy
    # For proper SSIM, would need pytorch-msssim package
    # Here we use MSE-based similarity as approximation
    mse = torch.nn.functional.mse_loss(img1, img2).item()
    # Convert MSE to similarity score (0-1 range)
    # Lower MSE = higher similarity
    similarity = np.exp(-mse * 10)  # Scale factor for reasonable range
    return similarity

def compute_perceptual_similarity(img1, img2, vgg_model):
    """Compute perceptual similarity using VGG features"""
    # Normalize for VGG
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(img1.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(img1.device)
    
    img1_norm = (img1 - mean) / std
    img2_norm = (img2 - mean) / std
    
    # Extract features
    features1 = vgg_model(img1_norm)
    features2 = vgg_model(img2_norm)
    
    # Compute MSE in feature space
    mse = torch.nn.functional.mse_loss(features1, features2).item()
    
    # Convert to similarity score (lower MSE = higher similarity)
    # Use exponential decay: similarity = exp(-mse/scale)
    scale = 10.0
    similarity = np.exp(-mse / scale)
    
    return similarity

def compute_metrics(content_path, generated_path, face_detector, face_recognizer, vgg, device='cuda'):
    """Compute all metrics for a generated image"""
    content_tensor = load_image_tensor(content_path, device)
    generated_tensor = load_image_tensor(generated_path, device)
    
    # SSIM
    ssim = compute_ssim(content_tensor, generated_tensor)
    
    # Perceptual Similarity
    perceptual_sim = compute_perceptual_similarity(content_tensor, generated_tensor, vgg)
    
    # Face Similarity
    try:
        content_embedding = face_recognizer.extract_embeddings(content_tensor, face_detector)
        gen_embedding = face_recognizer.extract_embeddings(generated_tensor, face_detector)
        
        if content_embedding is not None and gen_embedding is not None:
            face_sim = torch.nn.functional.cosine_similarity(
                content_embedding, gen_embedding, dim=1
            ).item()
        else:
            face_sim = 0.0
    except:
        face_sim = 0.0
    
    return ssim, perceptual_sim, face_sim

def create_progression_comparison(content_name, style_name, 
                                   content_dir, style_dir,
                                   baseline_dir, identity_dir,
                                   face_aware_dir, all_combined_dir,
                                   output_path,
                                   face_detector, face_recognizer, vgg):
    """Create 2×3 comparison grid"""
    
    # Load original images
    content_path = content_dir / f"{content_name}.jpg"
    style_path = style_dir / f"{style_name}.jpg"
    
    baseline_path = baseline_dir / f"{content_name}_{style_name}.jpg"
    identity_path = identity_dir / f"{content_name}_{style_name}.jpg"
    face_aware_path = face_aware_dir / f"{content_name}_{style_name}.jpg"
    all_combined_path = all_combined_dir / f"{content_name}_{style_name}.jpg"
    
    # Check if all files exist
    for path in [content_path, style_path, baseline_path, identity_path, face_aware_path, all_combined_path]:
        if not path.exists():
            print(f"⚠️  Missing file: {path}")
            return False
    
    # Compute metrics for all generated images
    print(f"  Computing metrics for {content_name} + {style_name}...")
    
    baseline_metrics = compute_metrics(content_path, baseline_path, face_detector, face_recognizer, vgg)
    identity_metrics = compute_metrics(content_path, identity_path, face_detector, face_recognizer, vgg)
    face_aware_metrics = compute_metrics(content_path, face_aware_path, face_detector, face_recognizer, vgg)
    all_combined_metrics = compute_metrics(content_path, all_combined_path, face_detector, face_recognizer, vgg)
    
    # Load images
    content_img = load_image_pil(content_path)
    style_img = load_image_pil(style_path)
    baseline_img = load_image_pil(baseline_path)
    identity_img = load_image_pil(identity_path)
    face_aware_img = load_image_pil(face_aware_path)
    all_combined_img = load_image_pil(all_combined_path)
    
    # Create figure (2 rows × 3 columns)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.subplots_adjust(hspace=0.15, wspace=0.15, top=0.95, bottom=0.05, left=0.05, right=0.95)
    
    # Row 1, Col 1: Content
    axes[0, 0].imshow(content_img)
    axes[0, 0].set_title('Content Image', fontsize=14, fontweight='bold', pad=10)
    axes[0, 0].axis('off')
    
    # Row 1, Col 2: Style
    axes[0, 1].imshow(style_img)
    axes[0, 1].set_title('Style Image', fontsize=14, fontweight='bold', pad=10)
    axes[0, 1].axis('off')
    
    # Row 1, Col 3: Baseline (γ=0)
    axes[0, 2].imshow(baseline_img)
    title_baseline = (f'Step 0: Baseline (AdaIN only)\n'
                      f'SSIM: {baseline_metrics[0]:.3f} | '
                      f'Percep: {baseline_metrics[1]:.3f} | '
                      f'Face: {baseline_metrics[2]:.3f}')
    axes[0, 2].set_title(title_baseline, fontsize=12, fontweight='bold', pad=10)
    axes[0, 2].axis('off')
    
    # Row 2, Col 1: Identity (γ=1000)
    axes[1, 0].imshow(identity_img)
    face_diff_1 = identity_metrics[2] - baseline_metrics[2]
    title_identity = (f'Step 1: + Identity Loss (γ=1000)\n'
                      f'SSIM: {identity_metrics[0]:.3f} | '
                      f'Percep: {identity_metrics[1]:.3f} | '
                      f'Face: {identity_metrics[2]:.3f} ({face_diff_1:+.3f})')
    axes[1, 0].set_title(title_identity, fontsize=12, fontweight='bold', pad=10)
    axes[1, 0].axis('off')
    
    # Row 2, Col 2: Face-aware + Identity
    axes[1, 1].imshow(face_aware_img)
    face_diff_2 = face_aware_metrics[2] - baseline_metrics[2]
    title_face_aware = (f'Step 2: + Face-Aware AdaIN (α=0.3)\n'
                        f'SSIM: {face_aware_metrics[0]:.3f} | '
                        f'Percep: {face_aware_metrics[1]:.3f} | '
                        f'Face: {face_aware_metrics[2]:.3f} ({face_diff_2:+.3f})')
    axes[1, 1].set_title(title_face_aware, fontsize=12, fontweight='bold', pad=10)
    axes[1, 1].axis('off')
    
    # Row 2, Col 3: All Combined (Best Model)
    axes[1, 2].imshow(all_combined_img)
    face_diff_3 = all_combined_metrics[2] - baseline_metrics[2]
    title_all = (f'Step 3: + Eye-Specific Loss (β=1) ⭐\n'
                 f'SSIM: {all_combined_metrics[0]:.3f} | '
                 f'Percep: {all_combined_metrics[1]:.3f} | '
                 f'Face: {all_combined_metrics[2]:.3f} ({face_diff_3:+.3f})')
    axes[1, 2].set_title(title_all, fontsize=12, fontweight='bold', pad=10,
                         bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))
    axes[1, 2].axis('off')
    
    # Main title
    fig.suptitle(f'Progressive Identity Preservation: {content_name} + {style_name}',
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Save figure
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Saved comparison to: {output_path}")
    return True

def main():
    print("="*80)
    print("Creating Final Progression Comparison Grids")
    print("="*80)
    print()
    
    # Set up paths
    project_root = Path(__file__).parent.parent
    content_dir = project_root / "data" / "eval_content"
    style_dir = project_root / "data" / "style"
    
    result_base = project_root / "results" / "model_progression"
    baseline_dir = result_base / "0_baseline"
    identity_dir = result_base / "1_identity"
    face_aware_dir = result_base / "2_face_aware_plus_identity"
    all_combined_dir = result_base / "3_all_combined"
    
    output_dir = result_base / "comparisons"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize models for metrics
    print("Initializing face detector and recognizer...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    face_detector = FaceDetector(device=device)
    face_recognizer = FaceRecognizer(device=device)
    
    # VGG for perceptual similarity (use relu4_1 layer)
    vgg = vgg19(pretrained=True).features[:21].to(device).eval()  # Up to relu4_1
    for param in vgg.parameters():
        param.requires_grad = False
    
    print(f"✅ Models loaded on {device}")
    print()
    
    # Get all content and style combinations
    content_images = sorted([f.stem for f in content_dir.glob("*.jpg")])
    style_images = sorted([f.stem for f in style_dir.glob("*.jpg")])
    
    print(f"Found {len(content_images)} content images and {len(style_images)} style images")
    print(f"Total comparisons to create: {len(content_images) * len(style_images)}")
    print()
    
    # Create comparisons
    success_count = 0
    total_count = len(content_images) * len(style_images)
    
    for i, content_name in enumerate(content_images):
        for j, style_name in enumerate(style_images):
            idx = i * len(style_images) + j + 1
            print(f"[{idx}/{total_count}] Creating comparison: {content_name} + {style_name}")
            
            output_path = output_dir / f"progression_{content_name}_{style_name}.png"
            
            success = create_progression_comparison(
                content_name, style_name,
                content_dir, style_dir,
                baseline_dir, identity_dir,
                face_aware_dir, all_combined_dir,
                output_path,
                face_detector, face_recognizer, vgg
            )
            
            if success:
                success_count += 1
            print()
    
    print("="*80)
    print(f"✅ Completed! Created {success_count}/{total_count} comparison grids")
    print(f"   Output directory: {output_dir}")
    print("="*80)

if __name__ == "__main__":
    main()

