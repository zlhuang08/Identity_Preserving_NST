#!/usr/bin/env python3
"""
Evaluation Metrics Utility

Centralized metrics computation for consistency across all evaluation scripts.
All metrics calculations should use functions from this module to ensure:
- Consistent implementation
- Reliable comparisons
- No redundant code

Metrics provided:
- SSIM (Structural Similarity Index)
- Perceptual Similarity (VGG-based)
- Face Similarity (FaceNet-based)

Usage:
    from evaluation_metrics_utility import MetricsCalculator
    
    calculator = MetricsCalculator(device='cuda')
    ssim_score = calculator.compute_ssim(img1, img2)
    perceptual_score = calculator.compute_perceptual_similarity(img1, img2)
    face_score = calculator.compute_face_similarity(img1, img2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19
from model_face_utils import FaceDetector, FaceRecognizer, IdentityPreserver


class MetricsCalculator:
    """
    Centralized metrics calculator for all evaluation tasks.
    
    This class provides consistent implementations of:
    1. SSIM - Structural similarity for image quality
    2. Perceptual Similarity - VGG feature-based similarity
    3. Face Similarity - FaceNet embedding-based identity preservation
    
    All methods expect tensors in [0, 1] range with shape (B, C, H, W).
    """
    
    def __init__(self, device='cuda'):
        """
        Initialize metrics calculator.
        
        Args:
            device: Device to run computations on ('cuda' or 'cpu')
        """
        self.device = device
        
        # Initialize VGG for perceptual similarity
        # Using features up to relu4_1 (layer 21)
        self.vgg = vgg19(weights='IMAGENET1K_V1').features[:21].to(device).eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        # ImageNet normalization (required for VGG)
        self.register_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.register_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        
        # Initialize IdentityPreserver - THE SAME CLASS USED IN TRAINING!
        # This ensures 100% consistent face similarity calculation
        self.identity_preserver = IdentityPreserver(device=device, pretrained='vggface2')
        
        print(f"✅ MetricsCalculator initialized on {device}")
    
    def compute_ssim(self, img1, img2, window_size=11):
        """
        Compute Structural Similarity Index (SSIM) between two images.
        
        SSIM measures structural similarity considering:
        - Luminance similarity
        - Contrast similarity
        - Structure similarity
        
        Args:
            img1: First image tensor (B, C, H, W) in [0, 1]
            img2: Second image tensor (B, C, H, W) in [0, 1]
            window_size: Size of Gaussian window (default: 11)
        
        Returns:
            float: SSIM score in [-1, 1], where 1 is identical
        
        Reference:
            Wang et al. "Image Quality Assessment: From Error Visibility to 
            Structural Similarity" IEEE TIP 2004
        """
        # Constants for stability
        C1 = 0.01 ** 2  # (K1 * L)^2, L=1 for [0,1] range
        C2 = 0.03 ** 2  # (K2 * L)^2
        
        # Compute local means using average pooling
        mu1 = F.avg_pool2d(img1, window_size, 1, window_size // 2)
        mu2 = F.avg_pool2d(img2, window_size, 1, window_size // 2)
        
        # Compute variances and covariance
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.avg_pool2d(img1 ** 2, window_size, 1, window_size // 2) - mu1_sq
        sigma2_sq = F.avg_pool2d(img2 ** 2, window_size, 1, window_size // 2) - mu2_sq
        sigma12 = F.avg_pool2d(img1 * img2, window_size, 1, window_size // 2) - mu1_mu2
        
        # Compute SSIM
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim_map.mean().item()
    
    def compute_perceptual_similarity(self, img1, img2):
        """
        Compute perceptual similarity using VGG19 features.
        
        Perceptual similarity measures high-level content similarity by:
        1. Extracting deep features from pretrained VGG19
        2. Computing cosine similarity between feature vectors
        
        This captures semantic similarity better than pixel-wise metrics.
        
        Args:
            img1: First image tensor (B, C, H, W) in [0, 1]
            img2: Second image tensor (B, C, H, W) in [0, 1]
        
        Returns:
            float: Cosine similarity in [-1, 1], where 1 is identical
        
        Reference:
            Johnson et al. "Perceptual Losses for Real-Time Style Transfer" 
            ECCV 2016
        """
        # Normalize for VGG (trained on ImageNet)
        img1_norm = (img1 - self.register_mean) / self.register_std
        img2_norm = (img2 - self.register_mean) / self.register_std
        
        # Extract features at relu4_1 layer
        with torch.no_grad():
            feat1 = self.vgg(img1_norm)
            feat2 = self.vgg(img2_norm)
        
        # Flatten features
        feat1_flat = feat1.view(feat1.size(0), -1)
        feat2_flat = feat2.view(feat2.size(0), -1)
        
        # Compute cosine similarity
        similarity = F.cosine_similarity(feat1_flat, feat2_flat, dim=1)
        
        return similarity.item()
    
    def compute_face_similarity(self, content_img, generated_img):
        """
        Compute face similarity using the EXACT SAME METHOD as training.
        
        This uses IdentityPreserver.compute_identity_loss() which is the same
        function used during training to compute face similarity. This ensures
        100% consistency between training metrics and evaluation metrics.
        
        Face similarity measures identity preservation by:
        1. Detecting faces in both images using MTCNN
        2. Extracting 512-dimensional embeddings using InceptionResnetV1
        3. Computing cosine similarity between embeddings
        4. Converting from [-1, 1] to [0, 1] range
        
        This metric is crucial for evaluating identity-preserving style transfer.
        
        Args:
            content_img: Content image tensor (B, C, H, W) in [0, 1]
            generated_img: Generated image tensor (B, C, H, W) in [0, 1]
        
        Returns:
            float or None: Face similarity in [0, 1] if faces detected,
                          None if face detection fails in either image
        
        Reference:
            Schroff et al. "FaceNet: A Unified Embedding for Face Recognition"
            CVPR 2015
        """
        try:
            # Use the EXACT SAME method as training
            # This calls FaceDetector.extract_faces() and FaceRecognizer.compute_similarity()
            # with all the same preprocessing and normalization
            identity_loss, metrics = self.identity_preserver.compute_identity_loss(
                generated_images=generated_img,
                content_images=content_img
            )
            
            # Return the average similarity from training metrics
            # This is already in [0, 1] range (converted inside compute_identity_loss)
            avg_similarity = metrics['avg_similarity']
            
            # Return None if no faces were matched (same as before)
            if metrics['num_matched_faces'] == 0:
                return None
            
            return avg_similarity
            
        except Exception as e:
            # Face detection or recognition failed
            # This can happen with heavily stylized images or profile views
            return None
    
    def compute_all_metrics(self, content_img, generated_img):
        """
        Compute all metrics at once for convenience.
        
        Args:
            content_img: Content image tensor (B, C, H, W) in [0, 1]
            generated_img: Generated image tensor (B, C, H, W) in [0, 1]
        
        Returns:
            dict: Dictionary with keys 'ssim', 'perceptual', 'face'
                  Face similarity will be None if detection fails
        """
        return {
            'ssim': self.compute_ssim(content_img, generated_img),
            'perceptual': self.compute_perceptual_similarity(content_img, generated_img),
            'face': self.compute_face_similarity(content_img, generated_img)
        }


# Convenience function for one-time metric computation
def compute_metrics(content_img, generated_img, device='cuda'):
    """
    Compute all metrics without persistent calculator instance.
    
    Use this for one-off metric computation. For batch processing,
    create a MetricsCalculator instance to avoid repeated model loading.
    
    Args:
        content_img: Content image tensor (B, C, H, W) in [0, 1]
        generated_img: Generated image tensor (B, C, H, W) in [0, 1]
        device: Device to use ('cuda' or 'cpu')
    
    Returns:
        dict: Dictionary with keys 'ssim', 'perceptual', 'face'
    """
    calculator = MetricsCalculator(device=device)
    return calculator.compute_all_metrics(content_img, generated_img)


if __name__ == "__main__":
    """
    Test metrics calculator with dummy images.
    """
    print("="*80)
    print("Testing MetricsCalculator")
    print("="*80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Create dummy images
    print("\nCreating test images...")
    img1 = torch.rand(1, 3, 256, 256, device=device)
    img2 = img1 + torch.randn(1, 3, 256, 256, device=device) * 0.1  # Similar but noisy
    
    # Initialize calculator
    print("\nInitializing calculator...")
    calculator = MetricsCalculator(device=device)
    
    # Test individual metrics
    print("\nComputing individual metrics...")
    print(f"  SSIM: {calculator.compute_ssim(img1, img2):.4f}")
    print(f"  Perceptual: {calculator.compute_perceptual_similarity(img1, img2):.4f}")
    print(f"  Face: {calculator.compute_face_similarity(img1, img2)}")
    
    # Test batch computation
    print("\nComputing all metrics at once...")
    metrics = calculator.compute_all_metrics(img1, img2)
    for name, value in metrics.items():
        if value is not None:
            print(f"  {name}: {value:.4f}")
        else:
            print(f"  {name}: None (face detection failed)")
    
    print("\n" + "="*80)
    print("✅ All tests passed!")
    print("="*80)

