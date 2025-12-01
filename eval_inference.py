#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Inference Script for AdaIN Style Transfer - Fast, Feed-Forward Style Transfer

OVERVIEW:
This script performs FAST style transfer using a trained AdaIN model.
Unlike optimization-based methods that require 100s of iterations per image,
this runs in a SINGLE forward pass - taking just ~0.1 seconds per image!

SPEED COMPARISON:
- Optimization-based NST: ~60 seconds per 512×512 image (Gatys et al.)
- AdaIN (this script): ~0.1 seconds per 512×512 image (300x faster!)

HOW IT WORKS:
1. Load trained decoder weights from checkpoint
2. Load content image (e.g., face photo) and style image (e.g., Peter Rabbit)
3. Single forward pass: Encode → AdaIN → Decode
4. Save stylized output

MODES OF OPERATION:
1. Single Image Pair: Stylize one content image with one style
2. Batch Mode: Stylize all content images with all styles (N×M combinations)

RECOMMENDED TEST IMAGES (for children's book illustrations):
- Content: face_00010.jpg (girl) and face_00066.jpg (boy) in data/eval_content/
- Styles: Any from data/style/ (peter_rabbit, starry_night, etc.)

TYPICAL USAGE:
    # Single image: Girl face + Peter Rabbit style
    python eval_inference.py \\
        --checkpoint checkpoints/best_model.pth \\
        --content data/eval_content/face_00010.jpg \\
        --style data/style/potter_peter_rabbit.jpg \\
        --output results/girl_peter_rabbit.jpg
    
    # Single image: Boy face + Kate Greenaway style
    python eval_inference.py \\
        --checkpoint checkpoints/best_model.pth \\
        --content data/eval_content/face_00066.jpg \\
        --style data/style/greenaway_christmas.jpg \\
        --output results/boy_greenaway.jpg
    
    # Batch mode: All eval images × all styles
    python eval_inference.py \\
        --checkpoint checkpoints/best_model.pth \\
        --content data/eval_content/ \\
        --style data/style/ \\
        --output results/
    
    # Control style strength (subtle effect)
    python eval_inference.py \\
        --checkpoint checkpoints/best_model.pth \\
        --content data/eval_content/face_00010.jpg \\
        --style data/style/potter_peter_rabbit.jpg \\
        --output results/girl_peter_rabbit_subtle.jpg \\
        --alpha 0.5
    
    # High resolution output
    python eval_inference.py \\
        --checkpoint checkpoints/best_model.pth \\
        --content data/eval_content/face_00010.jpg \\
        --style data/style/potter_peter_rabbit.jpg \\
        --output results/girl_peter_rabbit_hires.jpg \\
        --image-size 1024

ALPHA PARAMETER (style strength):
- alpha=0.0: No style transfer (output = content)
- alpha=0.5: Subtle style (50% stylized, 50% original)
- alpha=1.0: Full style (100% stylized, recommended default)

QUALITY vs SPEED:
- image_size=256: Very fast, lower quality
- image_size=512: Balanced (recommended)
- image_size=1024: Slower, higher quality
- image_size=0: Original size (can be very slow for large images)
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import argparse
import os
from pathlib import Path
import time
import json

from model_adain import AdaINStyleTransfer
from model_face_utils import FaceDetector
from evaluation_metrics_utility import MetricsCalculator


def load_image(image_path, image_size=512):
    """
    Load and preprocess an image for style transfer.
    
    This function loads an image from disk and converts it to a PyTorch tensor
    in the format expected by the AdaIN model.
    
    PREPROCESSING STEPS:
    1. Load image with PIL
    2. Convert to RGB (handles grayscale, RGBA, etc.)
    3. Resize to target size (if specified)
    4. Convert to tensor in [0, 1] range
    5. Add batch dimension
    
    Args:
        image_path: Path to image file
                   Example: 'data/eval_content/face_00010.jpg'
        image_size: Size to resize to (default: 512)
                   None = keep original size (can be slow for large images!)
                   Smaller = faster but lower quality
                   Larger = slower but higher quality
    
    Returns:
        image_tensor: (1, 3, H, W) tensor in [0, 1] range
                     Batch size = 1 (single image)
                     3 channels (RGB)
                     H, W = image_size or original dimensions
    
    Example:
        # Load and resize to 512×512
        img = load_image('face_00010.jpg', image_size=512)
        print(img.shape)  # torch.Size([1, 3, 512, 512])
        
        # Load at original size
        img = load_image('face_00010.jpg', image_size=None)
        print(img.shape)  # torch.Size([1, 3, 1024, 768]) or whatever original size is
    """
    # ========================================
    # Step 1: Load image from disk
    # ========================================
    image = Image.open(image_path).convert('RGB')
    # convert('RGB') ensures 3 channels, handles grayscale/RGBA/etc.
    
    # ========================================
    # Step 2: Create preprocessing transform
    # ========================================
    if image_size is not None:
        # Resize + convert to tensor
        transform = transforms.Compose([
            transforms.Resize(image_size),  # Resize to target size
            transforms.ToTensor()           # Convert to [0,1] tensor, (C,H,W) format
        ])
    else:
        # Just convert to tensor (keep original size)
        transform = transforms.ToTensor()
    
    # ========================================
    # Step 3: Apply transform and add batch dimension
    # ========================================
    image_tensor = transform(image)        # (3, H, W)
    image_tensor = image_tensor.unsqueeze(0)  # (1, 3, H, W) - add batch dim
    
    return image_tensor


def save_image(tensor, output_path):
    """
    Save a PyTorch tensor as an image file.
    
    This function converts a tensor back to a PIL Image and saves it to disk.
    
    POSTPROCESSING STEPS:
    1. Remove batch dimension if present
    2. Clamp values to [0, 1] range (handles any overflow/underflow)
    3. Convert tensor to PIL Image
    4. Save to disk
    
    Args:
        tensor: (1, 3, H, W) or (3, H, W) tensor in [0, 1] range
               Can be output from model or any RGB tensor
        output_path: Path to save image
                    Example: 'results/girl_peter_rabbit.jpg'
                    Supported formats: .jpg, .png, .bmp, etc.
    
    Example:
        # After style transfer
        stylized = model(content, style)  # (1, 3, 512, 512)
        save_image(stylized, 'results/output.jpg')
        # Image saved as JPEG
    """
    # ========================================
    # Step 1: Remove batch dimension if present
    # ========================================
    if tensor.dim() == 4:  # (1, 3, H, W) → (3, H, W)
        tensor = tensor.squeeze(0)
    
    # ========================================
    # Step 2: Clamp to valid range [0, 1]
    # ========================================
    # Neural networks can sometimes produce values slightly outside [0, 1]
    # Clamping ensures valid pixel values
    tensor = torch.clamp(tensor, 0, 1)
    
    # ========================================
    # Step 3: Convert to PIL Image and save
    # ========================================
    to_pil = transforms.ToPILImage()  # Converts (C,H,W) tensor to PIL Image
    image = to_pil(tensor.cpu())      # Move to CPU if on GPU, then convert
    image.save(output_path)           # Save to disk
    
    print(f"Saved stylized image to: {output_path}")


def stylize_single_pair(model, content_path, style_path, output_path, 
                        image_size=512, alpha=1.0, device='cuda', 
                        use_face_aware_adain=False, face_detector=None,
                        face_preservation_alpha=0.3, face_mask_margin=1.3,
                        metrics_calculator=None):
    """
    Stylize a single content-style image pair - THE CORE FUNCTION!
    
    This is the main inference function that takes a content image (e.g., a face)
    and a style image (e.g., Peter Rabbit), and produces a stylized output.
    
    THE PROCESS:
    1. Load and preprocess content and style images
    2. Move tensors to GPU (if available)
    3. Single forward pass through model: Encode → AdaIN → Decode
    4. Compute metrics IMMEDIATELY (on tensor, before saving to avoid compression artifacts)
    5. Save stylized output and metrics JSON
    
    TYPICAL TIMING:
    - 512×512 images on GPU: ~0.1 seconds
    - 512×512 images on CPU: ~2-3 seconds
    - 1024×1024 images on GPU: ~0.3 seconds
    
    Args:
        model: AdaINStyleTransfer model (trained and loaded)
        content_path: Path to content image
                     Example: 'data/eval_content/face_00010.jpg' (girl)
                              'data/eval_content/face_00066.jpg' (boy)
        style_path: Path to style image
                   Example: 'data/style/potter_peter_rabbit.jpg'
                            'data/style/greenaway_christmas.jpg'
        output_path: Path to save stylized result
                    Example: 'results/girl_peter_rabbit.jpg'
        image_size: Size to resize images to (default: 512)
                   Larger = higher quality but slower
        alpha: Style strength (default: 1.0)
              0.0 = no style (output = content)
              0.5 = subtle style (half-way blend)
              1.0 = full style (maximum artistic effect)
        device: Device to run on ('cuda' or 'cpu')
        metrics_calculator: MetricsCalculator instance (optional)
                           If provided, metrics will be computed and saved
    
    Returns:
        stylized: (1, 3, H, W) tensor of stylized image
    
    Example:
        # Stylize girl face with Peter Rabbit style
        model = AdaINStyleTransfer(device='cuda')
        # ... load checkpoint ...
        
        result = stylize_single_pair(
            model=model,
            content_path='data/eval_content/face_00010.jpg',
            style_path='data/style/potter_peter_rabbit.jpg',
            output_path='results/girl_peter_rabbit.jpg',
            image_size=512,
            alpha=1.0,
            device='cuda'
        )
    """
    # Set model to evaluation mode (disables dropout, batch norm tracking)
    model.eval()
    
    # ========================================
    # Step 1: Load and preprocess images
    # ========================================
    print(f"Loading content image: {content_path}")
    content = load_image(content_path, image_size).to(device)
    
    print(f"Loading style image: {style_path}")
    style = load_image(style_path, image_size).to(device)
    
    # Print shapes for debugging
    print(f"Content shape: {content.shape}")  # (1, 3, 512, 512)
    print(f"Style shape: {style.shape}")      # (1, 3, 512, 512)
    
    # ========================================
    # Step 2: Perform style transfer (FAST!)
    # ========================================
    print(f"Performing style transfer (alpha={alpha})...")
    start_time = time.time()
    
    # Single forward pass with no gradient computation
    with torch.no_grad():
        # Two modes: standard AdaIN or face-aware AdaIN
        if use_face_aware_adain and face_detector is not None:
            # Face-aware AdaIN: lighter stylization in face regions
            print("   Using face-aware AdaIN (regional adaptive normalization)")
            
            # Generate face mask
            face_masks = face_detector.generate_face_masks(
                content, 
                margin_factor=face_mask_margin
            )
            
            # Perform face-aware style transfer
            stylized = model.forward_with_face_aware_adain(
                content, style, face_masks,
                face_preservation_alpha=face_preservation_alpha,
                alpha=alpha
            )
        else:
            # Standard AdaIN
            print("   Using standard AdaIN")
            stylized = model(content, style, alpha=alpha)
    
    elapsed_time = time.time() - start_time
    print(f"✓ Style transfer completed in {elapsed_time:.3f} seconds")
    
    # ========================================
    # Step 3: Compute metrics (BEFORE saving to avoid compression artifacts)
    # ========================================
    if metrics_calculator is not None:
        print("Computing metrics by generating at 256×256 (matching training resolution)...")
        metrics_start = time.time()
        
        # IMPORTANT: Generate at 256×256 DIRECTLY, don't downsample!
        # Downsampling destroys facial features. We need to run the model again at 256×256.
        content_256 = load_image(content_path, image_size=256).to(device)
        style_256 = load_image(style_path, image_size=256).to(device)
        
        # Generate stylized image at 256×256 (for metrics only, not saved)
        with torch.no_grad():
            if use_face_aware_adain and face_detector is not None:
                face_masks_256 = face_detector.generate_face_masks(
                    content_256, 
                    margin_factor=face_mask_margin
                )
                stylized_256 = model.forward_with_face_aware_adain(
                    content_256, style_256, face_masks_256,
                    face_preservation_alpha=face_preservation_alpha,
                    alpha=alpha
                )
            else:
                stylized_256 = model(content_256, style_256, alpha=alpha)
        
        # Compute metrics on 256×256 tensors
        metrics = metrics_calculator.compute_all_metrics(content_256, stylized_256)
        
        metrics_elapsed = time.time() - metrics_start
        print(f"✓ Metrics computed in {metrics_elapsed:.3f} seconds")
        
        # Format face similarity for display
        if metrics['face'] is not None:
            face_display = f"{metrics['face']:.4f}"
        else:
            face_display = "N/A"
        
        print(f"   SSIM: {metrics['ssim']:.4f}, Perceptual: {metrics['perceptual']:.4f}, Face: {face_display}")
        
        # Save metrics to JSON (same name as output image)
        metrics_path = str(output_path).rsplit('.', 1)[0] + '_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"✓ Saved metrics to: {metrics_path}")
    
    # ========================================
    # Step 4: Save result (1024×1024 for visualization)
    # ========================================
    save_image(stylized, output_path)
    
    return stylized


def stylize_directory(model, content_dir, style_dir, output_dir,
                      image_size=512, alpha=1.0, device='cuda',
                      use_face_aware_adain=False, face_detector=None,
                      face_preservation_alpha=0.3, face_mask_margin=1.3,
                      metrics_calculator=None):
    """
    Batch processing: Stylize all content images with all style images.
    
    This function generates all possible combinations of content and style images!
    Perfect for creating a comprehensive gallery of results.
    
    EXAMPLE SCENARIO:
    - Content dir has: face_00010.jpg (girl), face_00066.jpg (boy), face_00100.jpg
    - Style dir has: peter_rabbit.jpg, starry_night.jpg
    - Output: 6 stylized images (3 content × 2 styles)
        • face_00010_peter_rabbit.jpg
        • face_00010_starry_night.jpg
        • face_00066_peter_rabbit.jpg
        • face_00066_starry_night.jpg
        • face_00100_peter_rabbit.jpg
        • face_00100_starry_night.jpg
    
    SMART RESUME:
    - Automatically skips already-existing outputs
    - Useful for resuming interrupted batch processing
    - Or adding new images to existing results
    
    NAMING CONVENTION:
    Output files are named: {content_name}_{style_name}.jpg
    Example: face_00010_potter_peter_rabbit.jpg
    
    Args:
        model: AdaINStyleTransfer model (trained and loaded)
        content_dir: Directory containing content images
                    Example: 'data/eval_content/' with face_*.jpg files
        style_dir: Directory containing style images
                  Example: 'data/style/' with style artwork files
        output_dir: Directory to save all stylized outputs
                   Will be created if it doesn't exist
                   Example: 'results/batch_processing/'
        image_size: Size to resize images to (default: 512)
        alpha: Style strength (default: 1.0)
        device: Device to run on ('cuda' or 'cpu')
        metrics_calculator: MetricsCalculator instance (optional)
                           If provided, metrics will be computed and saved
    
    Example:
        # Generate all combinations (eval images × all styles)
        stylize_directory(
            model=model,
            content_dir='data/eval_content',
            style_dir='data/style',
            output_dir='results/all_combinations',
            image_size=512,
            alpha=1.0,
            device='cuda'
        )
        
        # This might generate 2 eval images × 21 styles = 42 images!
    """
    # Set model to evaluation mode
    model.eval()
    
    # ========================================
    # Create output directory
    # ========================================
    os.makedirs(output_dir, exist_ok=True)
    
    # ========================================
    # Find all image files in directories
    # ========================================
    content_dir = Path(content_dir)
    style_dir = Path(style_dir)
    
    # Supported image extensions
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG'}
    
    # Collect all content images
    content_images = []
    for ext in extensions:
        content_images.extend(content_dir.glob(f'*{ext}'))
    
    # Collect all style images
    style_images = []
    for ext in extensions:
        style_images.extend(style_dir.glob(f'*{ext}'))
    
    # Report what we found
    print(f"Found {len(content_images)} content images")
    print(f"Found {len(style_images)} style images")
    print(f"Will generate {len(content_images) * len(style_images)} stylized images")
    print(f"Output directory: {output_dir}")
    print()
    
    # ========================================
    # Generate all content × style combinations
    # ========================================
    total_combinations = len(content_images) * len(style_images)
    processed = 0
    skipped = 0
    errors = 0
    
    for content_path in content_images:
        for style_path in style_images:
            # ========================================
            # Generate output filename
            # ========================================
            content_name = content_path.stem  # e.g., "face_00010"
            style_name = style_path.stem      # e.g., "potter_peter_rabbit"
            
            output_name = f"{content_name}_{style_name}.jpg"
            output_path = os.path.join(output_dir, output_name)
            
            # ========================================
            # Skip if already exists (smart resume!)
            # ========================================
            if os.path.exists(output_path):
                print(f"✓ Skipping (already exists): {output_name}")
                skipped += 1
                continue
            
            # ========================================
            # Stylize this content-style pair
            # ========================================
            print(f"\nProcessing [{processed + skipped + errors + 1}/{total_combinations}]: {content_name} + {style_name}")
            
            try:
                stylize_single_pair(
                    model=model,
                    content_path=str(content_path),
                    style_path=str(style_path),
                    output_path=output_path,
                    image_size=image_size,
                    alpha=alpha,
                    device=device,
                    use_face_aware_adain=use_face_aware_adain,
                    face_detector=face_detector,
                    face_preservation_alpha=face_preservation_alpha,
                    face_mask_margin=face_mask_margin,
                    metrics_calculator=metrics_calculator
                )
                processed += 1
            except Exception as e:
                print(f"❌ Error processing {content_name} + {style_name}: {e}")
                errors += 1
    
    # ========================================
    # Print final summary
    # ========================================
    print("\n" + "="*50)
    print("Batch processing complete!")
    print("="*50)
    print(f"Total combinations: {total_combinations}")
    print(f"Processed: {processed}")
    print(f"Skipped (already existed): {skipped}")
    print(f"Errors: {errors}")
    print(f"Output directory: {output_dir}")
    print("="*50)


def main():
    """
    Main inference function - command-line interface for style transfer.
    
    This function orchestrates the entire inference pipeline:
    1. Parse command-line arguments
    2. Load trained model from checkpoint
    3. Perform style transfer (single or batch mode)
    4. Save results
    
    TWO MODES:
    1. Single mode: Both --content and --style are files
    2. Batch mode: Both --content and --style are directories
    
    RECOMMENDED EXAMPLES (for children's book illustrations):
    
    Example 1: Girl with Peter Rabbit style
        python eval_inference.py \\
            --checkpoint checkpoints/best_model.pth \\
            --content data/eval_content/face_00010.jpg \\
            --style data/style/potter_peter_rabbit.jpg \\
            --output results/girl_peter_rabbit.jpg
    
    Example 2: Boy with Kate Greenaway style
        python eval_inference.py \\
            --checkpoint checkpoints/best_model.pth \\
            --content data/eval_content/face_00066.jpg \\
            --style data/style/greenaway_christmas.jpg \\
            --output results/boy_greenaway.jpg
    """
    # ============================================================================
    # ARGUMENT PARSING
    # ============================================================================
    parser = argparse.ArgumentParser(
        description="AdaIN Style Transfer Inference - Fast Feed-Forward Style Transfer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # ========================================
    # Input/output arguments (all required)
    # ========================================
    parser.add_argument('--content', type=str, required=True,
                        help='Path to content image or directory '
                             '(e.g., data/eval_content/face_00010.jpg or data/eval_content/)')
    parser.add_argument('--style', type=str, required=True,
                        help='Path to style image or directory '
                             '(e.g., data/style/potter_peter_rabbit.jpg or data/style/)')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output image or directory '
                             '(e.g., results/output.jpg or results/)')
    
    # ========================================
    # Model arguments
    # ========================================
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint '
                             '(e.g., checkpoints/best_model.pth)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on (cuda or cpu)')
    
    # ========================================
    # Processing arguments
    # ========================================
    parser.add_argument('--image-size', type=int, default=512,
                        help='Size to resize images to (512 is recommended, use 0 for original size)')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Style strength: 0.0=no style, 0.5=subtle, 1.0=full (default)')
    
    # ========================================
    # Face-aware AdaIN arguments
    # ========================================
    parser.add_argument('--use-face-aware-adain', action='store_true',
                        help='Enable face-aware AdaIN (lighter stylization in face regions). '
                             'Recommended for better identity preservation.')
    parser.add_argument('--face-preservation-alpha', type=float, default=0.3,
                        help='Stylization strength in face regions (0.0-1.0). '
                             'Only used if --use-face-aware-adain is enabled. Default: 0.3')
    parser.add_argument('--face-mask-margin', type=float, default=1.3,
                        help='Expand face bounding box by this factor (>= 1.0). '
                             'Only used if --use-face-aware-adain is enabled. Default: 1.3')
    
    args = parser.parse_args()
    
    # ============================================================================
    # SETUP
    # ============================================================================
    
    # ========================================
    # Setup device
    # ========================================
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # ========================================
    # Load model from checkpoint
    # ========================================
    print(f"\nLoading model from: {args.checkpoint}")
    
    # Create model architecture
    model = AdaINStyleTransfer(device=device)
    
    # Load trained decoder weights
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.decoder.load_state_dict(checkpoint['model_state_dict'])
    
    # Set to evaluation mode
    model.eval()
    
    print("✓ Model loaded successfully!")
    if 'epoch' in checkpoint:
        print(f"   Trained for {checkpoint['epoch'] + 1} epochs")
    if 'loss' in checkpoint:
        print(f"   Final loss: {checkpoint['loss']:.4f}")
    
    # ========================================
    # Initialize face detector (if face-aware AdaIN enabled)
    # ========================================
    face_detector = None
    if args.use_face_aware_adain:
        print(f"\nInitializing face detector for face-aware AdaIN...")
        try:
            face_detector = FaceDetector(device=device, keep_all=False, min_face_size=20)
            print("✓ Face detector initialized successfully")
            print(f"   Face preservation alpha: {args.face_preservation_alpha}")
            print(f"   Face mask margin: {args.face_mask_margin}x")
        except ImportError as e:
            print(f"⚠️  Warning: Could not initialize face detector: {e}")
            print("   Install facenet-pytorch: pip install facenet-pytorch")
            print("   Disabling face-aware AdaIN...")
            args.use_face_aware_adain = False
    
    # ========================================
    # Initialize metrics calculator
    # ========================================
    print(f"\nInitializing metrics calculator...")
    metrics_calculator = MetricsCalculator(device=device)
    print("✓ Metrics calculator initialized successfully")
    print("   Metrics will be computed on tensors (before saving)")
    print("   This ensures consistency with training metrics!")
    
    # ========================================
    # Handle image size
    # ========================================
    # 0 means use original image size (no resizing)
    image_size = args.image_size if args.image_size > 0 else None
    if image_size is None:
        print("   Using original image size (may be slow for large images)")
    else:
        print(f"   Image size: {image_size}×{image_size}")
    
    # ============================================================================
    # DETERMINE MODE AND RUN INFERENCE
    # ============================================================================
    
    # ========================================
    # Check if inputs are files or directories
    # ========================================
    content_is_file = os.path.isfile(args.content)
    style_is_file = os.path.isfile(args.style)
    
    print("\n" + "="*50)
    
    if content_is_file and style_is_file:
        # ========================================
        # MODE 1: Single image pair
        # ========================================
        print("Mode: Single image pair")
        print("="*50 + "\n")
        
        stylize_single_pair(
            model=model,
            content_path=args.content,
            style_path=args.style,
            output_path=args.output,
            image_size=image_size,
            alpha=args.alpha,
            device=device,
            use_face_aware_adain=args.use_face_aware_adain,
            face_detector=face_detector,
            face_preservation_alpha=args.face_preservation_alpha,
            face_mask_margin=args.face_mask_margin,
            metrics_calculator=metrics_calculator
        )
    
    elif os.path.isdir(args.content) and os.path.isdir(args.style):
        # ========================================
        # MODE 2: Batch processing (directory)
        # ========================================
        print("Mode: Directory (batch processing)")
        print("="*50 + "\n")
        
        stylize_directory(
            model=model,
            content_dir=args.content,
            style_dir=args.style,
            output_dir=args.output,
            image_size=image_size,
            alpha=args.alpha,
            device=device,
            use_face_aware_adain=args.use_face_aware_adain,
            face_detector=face_detector,
            face_preservation_alpha=args.face_preservation_alpha,
            face_mask_margin=args.face_mask_margin,
            metrics_calculator=metrics_calculator
        )
    
    else:
        # ========================================
        # Error: Mixed file/directory inputs
        # ========================================
        raise ValueError(
            "Both content and style must be either files or directories!\n"
            "Single mode: --content file.jpg --style file.jpg --output file.jpg\n"
            "Batch mode:  --content dir/ --style dir/ --output dir/"
        )
    
    # ============================================================================
    # COMPLETE!
    # ============================================================================
    print("\n" + "="*50)
    print("✨ Style transfer completed!")
    print("="*50)
    print(f"\nResults saved to: {args.output}")
    print("\nRecommended next steps:")
    print("  - View results in your image viewer")
    print("  - Run eval_metrics.py to compute quality metrics")
    print("  - Run result_visualize.py to create comparison grids")


if __name__ == "__main__":
    main()

