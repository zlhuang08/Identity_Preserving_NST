#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create Final Comparison: Baseline (γ=0) vs Optimal Identity Weight (γ=1000)

This script creates side-by-side comparison grids showing:
- Content image (original)
- Style image (target)
- Baseline result (γ=0, no identity preservation)
- Identity result (γ=1000, optimal identity preservation)

With full metrics displayed on each generated image.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
from pathlib import Path
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision import models

# Import face utilities for identity metrics
try:
    from model_face_utils import FaceDetector, FaceRecognizer
    FACE_UTILS_AVAILABLE = True
except ImportError:
    FACE_UTILS_AVAILABLE = False
    print("Warning: Face utilities not available. Face similarity will not be computed.")


class SimilarityComputer:
    """Compute various similarity metrics between images"""
    
    def __init__(self):
        """Initialize similarity computation tools"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # VGG16 for perceptual similarity
        vgg = models.vgg16(pretrained=True).features[:16].to(self.device).eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        
        # Face detection and recognition
        if FACE_UTILS_AVAILABLE:
            self.face_detector = FaceDetector()
            self.face_recognizer = FaceRecognizer()
        else:
            self.face_detector = None
            self.face_recognizer = None
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def compute_ssim(self, img1, img2):
        """Compute SSIM between two images"""
        from skimage.metrics import structural_similarity as ssim
        
        # Convert to numpy if needed
        if isinstance(img1, Image.Image):
            img1 = np.array(img1)
        if isinstance(img2, Image.Image):
            img2 = np.array(img2)
        
        # Resize img2 to match img1 dimensions if needed
        if img1.shape != img2.shape:
            img2_pil = Image.fromarray(img2)
            img2_pil = img2_pil.resize((img1.shape[1], img1.shape[0]), Image.LANCZOS)
            img2 = np.array(img2_pil)
        
        # Compute SSIM
        score = ssim(img1, img2, multichannel=True, channel_axis=2, data_range=255)
        return score
    
    def compute_perceptual_similarity(self, img1, img2):
        """Compute perceptual similarity using VGG features"""
        # Resize img2 to match img1 dimensions if needed
        if isinstance(img1, Image.Image):
            img1_pil = img1
        else:
            img1_pil = Image.fromarray(img1)
        
        if isinstance(img2, Image.Image):
            img2_pil = img2
        else:
            img2_pil = Image.fromarray(img2)
        
        # Resize to common size
        if img1_pil.size != img2_pil.size:
            img2_pil = img2_pil.resize(img1_pil.size, Image.LANCZOS)
        
        # Preprocess images
        tensor1 = self.transform(img1_pil).unsqueeze(0).to(self.device)
        tensor2 = self.transform(img2_pil).unsqueeze(0).to(self.device)
        
        # Extract VGG features
        with torch.no_grad():
            features1 = self.vgg(tensor1)
            features2 = self.vgg(tensor2)
        
        # Compute cosine similarity
        features1 = features1.flatten()
        features2 = features2.flatten()
        similarity = F.cosine_similarity(features1.unsqueeze(0), features2.unsqueeze(0))
        
        return similarity.item()
    
    def compute_face_similarity(self, img1, img2):
        """Compute face similarity using face recognition"""
        if not FACE_UTILS_AVAILABLE or self.face_detector is None:
            return None
        
        try:
            # Convert to numpy if needed
            if isinstance(img1, Image.Image):
                img1_np = np.array(img1)
            else:
                img1_np = img1
            
            if isinstance(img2, Image.Image):
                img2_np = np.array(img2)
            else:
                img2_np = img2
            
            # Resize img2 to match img1 dimensions if needed
            if img1_np.shape != img2_np.shape:
                img2_pil = Image.fromarray(img2_np)
                img2_pil = img2_pil.resize((img1_np.shape[1], img1_np.shape[0]), Image.LANCZOS)
                img2_np = np.array(img2_pil)
            
            # Detect faces
            faces1 = self.face_detector.detect_faces(img1_np)
            faces2 = self.face_detector.detect_faces(img2_np)
            
            if len(faces1) == 0 or len(faces2) == 0:
                return None
            
            # Extract embeddings
            emb1 = self.face_recognizer.extract_embeddings(torch.from_numpy(img1_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0, allow_grad=False)
            emb2 = self.face_recognizer.extract_embeddings(torch.from_numpy(img2_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0, allow_grad=False)
            
            # Compute similarity
            similarity = self.face_recognizer.compute_similarity(emb1, emb2)
            
            return similarity.item()
        except Exception as e:
            print(f"    Warning: Face similarity computation failed: {e}")
            return None
    
    def compute_all_metrics(self, content_img, generated_img):
        """Compute all similarity metrics"""
        metrics = {}
        
        # SSIM
        metrics['ssim'] = self.compute_ssim(content_img, generated_img)
        
        # Perceptual similarity
        metrics['perceptual'] = self.compute_perceptual_similarity(content_img, generated_img)
        
        # Face similarity
        face_sim = self.compute_face_similarity(content_img, generated_img)
        metrics['face'] = face_sim if face_sim is not None else 0.0
        
        return metrics


def create_final_comparison(content_name, style_name, output_dir, similarity_computer):
    """Create final comparison: baseline vs optimal identity weight"""
    
    # Paths
    content_path = Path(f"data/eval_content/{content_name}.jpg")
    style_path = Path(f"data/style/{style_name}.jpg")
    baseline_path = Path(f"results/tuning/identity_visuals/gamma_0_00/{content_name}_{style_name}.jpg")
    identity_path = Path(f"results/tuning/identity_visuals/gamma_1000_00/{content_name}_{style_name}.jpg")
    
    # Check if all files exist
    if not all([content_path.exists(), style_path.exists(), baseline_path.exists(), identity_path.exists()]):
        print(f"  ✗ Missing files for {content_name} + {style_name}")
        return None
    
    # Load images
    content_img = Image.open(content_path).convert('RGB')
    style_img = Image.open(style_path).convert('RGB')
    baseline_img = Image.open(baseline_path).convert('RGB')
    identity_img = Image.open(identity_path).convert('RGB')
    
    # Compute metrics
    print(f"  Computing metrics...")
    baseline_metrics = similarity_computer.compute_all_metrics(content_img, baseline_img)
    identity_metrics = similarity_computer.compute_all_metrics(content_img, identity_img)
    
    # Create figure (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Top-left: Content
    axes[0, 0].imshow(content_img)
    axes[0, 0].set_title('Content\n(Original Face)', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Top-right: Style
    axes[0, 1].imshow(style_img)
    axes[0, 1].set_title('Style\n(Target Art)', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Bottom-left: Baseline (γ=0)
    axes[1, 0].imshow(baseline_img)
    axes[1, 0].set_title('Baseline (γ=0)\nNo Identity Preservation', fontsize=12, fontweight='bold', color='navy')
    axes[1, 0].axis('off')
    
    # Add metrics to baseline
    metrics_text_baseline = (
        f"SSIM: {baseline_metrics['ssim']:.3f}\n"
        f"Perceptual: {baseline_metrics['perceptual']:.3f}\n"
        f"Face Sim: {baseline_metrics['face']:.3f}"
    )
    axes[1, 0].text(0.02, 0.02, metrics_text_baseline,
                   transform=axes[1, 0].transAxes,
                   fontsize=10, verticalalignment='bottom',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='navy', linewidth=2))
    
    # Bottom-right: Identity (γ=1000)
    axes[1, 1].imshow(identity_img)
    axes[1, 1].set_title('Identity-Preserving (γ=1000)\nOptimal Identity Weight ⭐', 
                        fontsize=12, fontweight='bold', color='darkgreen')
    axes[1, 1].axis('off')
    
    # Add metrics to identity (with improvements)
    ssim_change = identity_metrics['ssim'] - baseline_metrics['ssim']
    perceptual_change = identity_metrics['perceptual'] - baseline_metrics['perceptual']
    face_change = identity_metrics['face'] - baseline_metrics['face']
    
    ssim_arrow = "↑" if ssim_change > 0 else "↓"
    perceptual_arrow = "↑" if perceptual_change > 0 else "↓"
    face_arrow = "↑" if face_change > 0 else "↓"
    
    metrics_text_identity = (
        f"SSIM: {identity_metrics['ssim']:.3f} ({ssim_change:+.3f} {ssim_arrow})\n"
        f"Perceptual: {identity_metrics['perceptual']:.3f} ({perceptual_change:+.3f} {perceptual_arrow})\n"
        f"Face Sim: {identity_metrics['face']:.3f} ({face_change:+.3f} {face_arrow})"
    )
    
    bbox_color = 'lightgreen' if face_change > 0 else 'lightyellow'
    axes[1, 1].text(0.02, 0.02, metrics_text_identity,
                   transform=axes[1, 1].transAxes,
                   fontsize=10, verticalalignment='bottom',
                   bbox=dict(boxstyle='round', facecolor=bbox_color, alpha=0.9, 
                            edgecolor='darkgreen', linewidth=2))
    
    # Main title
    fig.suptitle(f'Final Comparison: {content_name.replace("_", " ").title()} + {style_name.replace("_", " ").title()}',
                fontsize=14, fontweight='bold', y=0.98)
    
    # Save
    output_path = Path(output_dir) / f"final_comparison_{content_name}_{style_name}.png"
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return output_path, baseline_metrics, identity_metrics


def main():
    """Generate final comparison grids"""
    
    output_dir = Path("results/tuning/final_comparisons")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize similarity computer
    print("Initializing similarity computer...")
    similarity_computer = SimilarityComputer()
    
    # Content and style combinations
    content_images = ["face_00010", "face_00066"]
    style_images = [
        "starry_night",
        "water_lilies", 
        "the_scream",
        "great_wave",
        "potter_peter_rabbit"
    ]
    
    print("="*80)
    print("Creating Final Comparison Grids: Baseline (γ=0) vs Optimal (γ=1000)")
    print("="*80)
    print()
    
    results = []
    created = []
    
    for content in content_images:
        for style in style_images:
            print(f"\nProcessing: {content} + {style}")
            try:
                output_path, baseline_metrics, identity_metrics = create_final_comparison(
                    content, style, output_dir, similarity_computer
                )
                
                if output_path:
                    print(f"  ✓ Saved to {output_path}")
                    created.append(output_path)
                    
                    # Store results
                    results.append({
                        'content': content,
                        'style': style,
                        'baseline_ssim': baseline_metrics['ssim'],
                        'baseline_perceptual': baseline_metrics['perceptual'],
                        'baseline_face': baseline_metrics['face'],
                        'identity_ssim': identity_metrics['ssim'],
                        'identity_perceptual': identity_metrics['perceptual'],
                        'identity_face': identity_metrics['face'],
                        'face_improvement': identity_metrics['face'] - baseline_metrics['face']
                    })
            except Exception as e:
                print(f"  ✗ Error: {e}")
                import traceback
                traceback.print_exc()
    
    print("\n" + "="*80)
    print(f"✅ Created {len(created)} final comparison grids")
    print(f"📁 Location: {output_dir}")
    print("="*80)
    
    # Print summary statistics
    if results:
        import pandas as pd
        df = pd.DataFrame(results)
        
        print("\n📊 SUMMARY STATISTICS:")
        print("-"*80)
        print(f"Average Face Similarity Improvement: {df['face_improvement'].mean():+.4f}")
        print(f"Best Face Similarity Improvement: {df['face_improvement'].max():+.4f}")
        print(f"Face Similarity improved in {(df['face_improvement'] > 0).sum()}/{len(df)} cases")
        print()
        print(f"Baseline Average Face Sim: {df['baseline_face'].mean():.4f}")
        print(f"Identity Average Face Sim: {df['identity_face'].mean():.4f}")
        print("-"*80)
    
    print("\nThese grids show:")
    print("  • Clean 2×2 layout: Content | Style | Baseline | Identity")
    print("  • Full metrics with improvements (↑/↓ indicators)")
    print("  • Visual evidence of γ=1000 superiority")
    print("  • Ready for inclusion in report/presentation")
    print("="*80)


if __name__ == "__main__":
    main()

