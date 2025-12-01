#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training Script for AdaIN-based Style Transfer Network

OVERVIEW:
This script trains the AdaIN decoder to perform arbitrary style transfer.
The goal is to teach the decoder to reconstruct beautiful stylized images
from AdaIN-transformed features.

WHAT GETS TRAINED?
- Decoder: YES! (~3.5M parameters) - Learns to reconstruct images
- VGG Encoder: NO! (~20M parameters) - Frozen, pretrained on ImageNet
- AdaIN Layer: NO! No parameters - Pure statistical operation

TRAINING PROCESS:
1. Load random content-style pairs (e.g., face photo + Peter Rabbit)
2. Extract features using frozen VGG encoder
3. Apply AdaIN to transfer style statistics
4. Decode features back to image using trainable decoder
5. Compute losses (content + style + identity)
6. Backpropagate through decoder only
7. Update decoder weights with Adam optimizer

LOSS COMPONENTS:
- Content Loss (weight=1.0): Preserve content structure
  Ensures output still looks like the original content (face shape preserved)
  
- Style Loss (weight=10.0): Match style appearance
  Ensures output has artistic style (watercolor look, colors, textures)
  
- Identity Loss (weight=0.1, optional): Preserve facial identity
  Ensures stylized faces still look like the same person
  Requires facenet-pytorch for face detection and recognition

WHY IT WORKS:
The decoder learns to invert the feature space transformation:
  Original Image → VGG → Features → AdaIN → Stylized Features → Decoder → Stylized Image
                  (frozen)              (no params)           (TRAINABLE!)

After training, the decoder can reconstruct stylized images that:
✓ Preserve content structure (recognizable faces, objects)
✓ Match style appearance (artistic look, colors, textures)
✓ Look visually pleasing (no artifacts, natural-looking)

TYPICAL USAGE:
    # Train baseline model (exhaustive pairing, no identity preservation)
    python train_model.py \\
        --content-dir data/content \\
        --style-dir data/style \\
        --checkpoint-dir checkpoints/baseline \\
        --epochs 20 \\
        --identity-weight 0.0
    
    # Train identity-preserving model (CS230 60/20/20 splits)
    python train_model.py \\
        --content-dir data/content \\
        --style-dir data/style \\
        --checkpoint-dir checkpoints/identity \\
        --epochs 20 \\
        --identity-weight 0.1
    
    # Fast training with larger batch (if GPUs are clean)
    python train_model.py \\
        --content-dir data/content \\
        --style-dir data/style \\
        --checkpoint-dir checkpoints/identity_fast \\
        --epochs 20 \\
        --batch-size 64 \\
        --identity-weight 0.1
    
    # Custom split files (default: data/content_splits/train.txt and val.txt)
    python train_model.py \\
        --content-dir data/content \\
        --style-dir data/style \\
        --split-file data/content_splits/train.txt \\
        --val-split-file data/content_splits/val.txt \\
        --epochs 10

EXPECTED RESULTS:
- After 10 epochs: Basic stylization working, some artifacts
- After 20 epochs: Good quality, natural-looking stylization
- After 30+ epochs: Excellent quality, production-ready

HARDWARE REQUIREMENTS:
- GPU: Recommended (CUDA-capable, 4GB+ VRAM)
- CPU: Works but 10-50x slower
- RAM: 8GB+ recommended
- Storage: ~1GB for checkpoints
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import random

# Import our custom modules
from model_adain import AdaINStyleTransfer, calc_content_loss, calc_style_loss
from model_face_utils import IdentityPreserver, FaceDetector


class ImageDataset(Dataset):
    """
    Dataset for loading content and style image pairs for AdaIN training.
    
    KEY DESIGN CHOICE - EXHAUSTIVE PAIRING:
    We now iterate through ALL unique content×style combinations per epoch!
    This provides reproducible training curves essential for DL experiments.
    
    Example with 3 content images and 2 styles (6 total pairs per epoch):
        - Batch 1: face_001.jpg + starry_night.jpg, face_001.jpg + peter_rabbit.jpg
        - Batch 2: face_002.jpg + starry_night.jpg, face_002.jpg + peter_rabbit.jpg
        - Batch 3: face_003.jpg + starry_night.jpg, face_003.jpg + peter_rabbit.jpg
    
    Exhaustive pairing benefits:
    ✓ Reproducible training curves (essential for milestone reports)
    ✓ Consistent epoch definition (all data seen once per epoch)
    ✓ Better for small datasets (maximize data usage)
    ✓ Easier to judge convergence and overfitting
    
    With splits (train/val/test):
    - Train: 120 content × 21 styles = 2,520 pairs per epoch
    - Val: 40 content × 21 styles = 840 pairs for validation
    - Test: 40 content × 21 styles = 840 pairs for testing
    
    DATASET REQUIREMENTS:
    - Content images: Natural photos (faces, objects, scenes)
      For our project: Synthetic children's faces from StyleGAN
    
    - Style images: Artistic images with distinct visual styles
      For our project: Famous paintings + children's book illustrations
    
    TRANSFORMS APPLIED:
    1. Resize to square (e.g., 256×256) - ensures consistent size
    2. ToTensor - converts PIL Image to PyTorch tensor [0,1]
    """
    def __init__(self, content_dir, style_dir, split_file=None, transform=None, image_size=256):
        """
        Initialize dataset by loading image paths.
        
        Args:
            content_dir: Directory containing content images
                        Example: 'data/content/' with face_00000.jpg, face_00001.jpg, ...
            style_dir: Directory containing style images
                      Example: 'data/style/' with starry_night.jpg, peter_rabbit.jpg, ...
            split_file: Optional path to .txt file listing which content images to use
                       Example: 'data/content_splits/train.txt' with one filename per line
                       If None, uses all images in content_dir
            transform: Optional custom transform (if None, uses default)
            image_size: Size to resize images to (default: 256×256)
                       Larger = better quality but slower training and more VRAM
                       Smaller = faster training but lower quality
        
        Example:
            # Create dataset for training with split file
            dataset = ImageDataset(
                content_dir='data/content',
                style_dir='data/style',
                split_file='data/content_splits/train.txt',
                image_size=256
            )
            
            # Create dataloader (no shuffle needed - pairs are deterministic)
            loader = DataLoader(dataset, batch_size=8, shuffle=False)
        """
        self.content_dir = Path(content_dir)
        self.style_dir = Path(style_dir)
        self.split_file = split_file
        
        # ========================================
        # Load content images (from split file or directory)
        # ========================================
        if split_file:
            # Load from split file
            with open(split_file, 'r') as f:
                filenames = [line.strip() for line in f if line.strip()]
            self.content_images = [self.content_dir / filename for filename in filenames]
            print(f"Loaded {len(self.content_images)} content images from split file")
        else:
            # Load all images from directory
            self.content_images = self._get_image_files(self.content_dir)
            print(f"Found {len(self.content_images)} content images")
        
        # ========================================
        # Load all style images
        # ========================================
        self.style_images = self._get_image_files(self.style_dir)
        print(f"Found {len(self.style_images)} style images")
        
        # ========================================
        # Create exhaustive pairing
        # ========================================
        # Generate ALL content×style combinations
        self.pairs = []
        for content_img in self.content_images:
            for style_img in self.style_images:
                self.pairs.append((content_img, style_img))
        
        print(f"Total pairs: {len(self.pairs)} ({len(self.content_images)} content × {len(self.style_images)} styles)")
        
        # ========================================
        # Validate dataset
        # ========================================
        if len(self.content_images) == 0:
            raise ValueError(f"No images found for content")
        if len(self.style_images) == 0:
            raise ValueError(f"No images found in style directory: {style_dir}")
        
        # ========================================
        # Setup image transforms
        # ========================================
        if transform is None:
            # Default transform pipeline for training
            self.transform = transforms.Compose([
                # Step 1: Resize to target size
                transforms.Resize((image_size, image_size)),
                
                # Step 2: Convert PIL Image to PyTorch tensor in [0, 1] range
                # This also reorders from (H, W, C) to (C, H, W)
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform
    
    def _get_image_files(self, directory):
        """
        Recursively find all image files in directory.
        
        Supports common image formats: .jpg, .jpeg, .png, .bmp
        Case-insensitive matching (both .jpg and .JPG work)
        
        Args:
            directory: Path object pointing to image directory
        
        Returns:
            sorted_files: Sorted list of Path objects for all images
        
        Example directory structure:
            data/content/
                face_00000.jpg
                face_00001.jpg
                ...
            data/style/
                starry_night.jpg
                peter_rabbit.jpg
                ...
        """
        # Common image file extensions
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG'}
        image_files = []
        
        # Recursively search for images with each extension
        # **/* means search in all subdirectories
        for ext in extensions:
            image_files.extend(directory.glob(f'**/*{ext}'))
        
        # Sort for reproducibility (same order every run)
        return sorted(image_files)
    
    def __len__(self):
        """
        Return the number of training samples.
        
        With exhaustive pairing, this is content_count × style_count.
        Example: 120 content × 21 styles = 2,520 pairs per epoch
        """
        return len(self.pairs)
    
    def __getitem__(self, idx):
        """
        Get a content-style pair for training - THE KEY METHOD!
        
        IMPORTANT: Pairs are now deterministic (exhaustive pairing)!
        Same idx always returns the same content-style combination.
        
        Process:
        1. Get content-style pair from pre-generated pairs list (deterministic)
        2. Load both images from disk
        3. Apply transforms to both
        4. Return as tensor pair
        
        Args:
            idx: Index of pair to load (0 to len(pairs)-1)
        
        Returns:
            content_tensor: (3, H, W) tensor of content image in [0, 1]
            style_tensor: (3, H, W) tensor of style image in [0, 1]
        
        Example:
            dataset = ImageDataset('data/content', 'data/style', 
                                  split_file='data/content_splits/train.txt')
            content, style = dataset[0]  # First pair, always the same
            
            print(content.shape)  # torch.Size([3, 256, 256])
            print(style.shape)    # torch.Size([3, 256, 256])
            print(content.min(), content.max())  # 0.0, 1.0
        """
        # ========================================
        # Get deterministic content-style pair
        # ========================================
        content_path, style_path = self.pairs[idx]
        
        # Load both images
        content_img = Image.open(content_path).convert('RGB')
        style_img = Image.open(style_path).convert('RGB')
        # convert('RGB') ensures 3 channels (handles grayscale, RGBA, etc.)
        
        # ========================================
        # Apply transforms (resize, crop, to tensor)
        # ========================================
        content_tensor = self.transform(content_img)  # (3, 256, 256) in [0, 1]
        style_tensor = self.transform(style_img)      # (3, 256, 256) in [0, 1]
        
        return content_tensor, style_tensor


def train_epoch(model, dataloader, optimizer, device, content_weight=1.0, style_weight=10.0, 
                identity_weight=0.0, identity_preserver=None, eye_weight=0.0, eye_loss_module=None,
                face_aware_adain=False, face_detector=None, face_preservation_alpha=0.3, face_mask_margin=1.3):
    """
    Train the model for one epoch - THE MAIN TRAINING LOOP!
    
    This function performs one complete pass through the training dataset,
    updating the decoder weights to minimize the combined loss.
    
    THE TRAINING PROCESS (for each batch):
    1. Load batch of content-style pairs
    2. Forward pass: content + style → stylized image
    3. Compute losses (content, style, identity)
    4. Backward pass: compute gradients
    5. Update decoder weights with optimizer
    6. Track and display progress
    
    LOSS WEIGHTING:
    Total Loss = content_weight × Content Loss
                + style_weight × Style Loss
                + identity_weight × Identity Loss
                + eye_weight × Eye Loss
    
    Typical weights:
    - content_weight = 1.0 (baseline)
    - style_weight = 10.0 (style is more important for artistic look)
    - identity_weight = 0.0 (baseline) or 1000.0 (with identity preservation)
    - eye_weight = 0.0 (baseline) or 100.0 (with eye-specific preservation)
    
    Args:
        model: AdaINStyleTransfer model (encoder + decoder + adain)
        dataloader: DataLoader yielding (content, style) batches
                   Example batch: (8, 3, 256, 256) images
        optimizer: Adam optimizer for decoder parameters
        device: 'cuda' or 'cpu'
        content_weight: Weight for content loss (default: 1.0)
                       Higher = stronger content preservation
        style_weight: Weight for style loss (default: 10.0)
                     Higher = stronger style matching
        identity_weight: Weight for identity loss (default: 0.0)
                        0.0 = baseline (no identity preservation)
                        0.1 = with identity preservation for faces
        identity_preserver: IdentityPreserver instance (required if identity_weight > 0)
                           None if not using identity preservation
    
    Returns:
        avg_loss: Average total loss for the epoch
        avg_content_loss: Average content loss
        avg_style_loss: Average style loss
        avg_identity_loss: Average identity loss (0 if not used)
        avg_similarity_epoch: Average face similarity (0-1, only if identity used)
    
    Example:
        # Train one epoch
        train_loss, c_loss, s_loss, i_loss, sim = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device='cuda',
            content_weight=1.0,
            style_weight=10.0,
            identity_weight=0.1,  # Enable identity preservation
            identity_preserver=preserver
        )
        
        print(f"Training Loss: {train_loss:.4f}")
        print(f"Content: {c_loss:.4f}, Style: {s_loss:.4f}, Identity: {i_loss:.4f}")
        print(f"Face Similarity: {sim:.2f}")
    """
    # ========================================
    # Set model modes
    # ========================================
    model.train()          # Set overall model to train mode
    model.decoder.train()  # Decoder in train mode (updates weights)
    model.encoder.eval()   # Encoder stays frozen (no weight updates!)
    
    # ========================================
    # Initialize loss accumulators
    # ========================================
    total_loss = 0.0            # Sum of all batch losses
    total_content_loss = 0.0    # Sum of content losses
    total_style_loss = 0.0      # Sum of style losses
    total_identity_loss = 0.0   # Sum of identity losses
    total_eye_loss = 0.0        # Sum of eye-specific losses
    total_identity_samples = 0  # Count of detected face pairs
    total_eye_samples = 0       # Count of detected eye pairs
    total_similarity = 0.0      # Sum of similarity scores (weighted by num faces)
    
    # Progress bar for visual feedback during training
    pbar = tqdm(dataloader, desc="Training")
    
    # ========================================
    # Training loop - iterate through batches
    # ========================================
    for batch_idx, (content, style) in enumerate(pbar):
        # Move data to GPU/CPU
        content = content.to(device)  # (B, 3, 256, 256)
        style = style.to(device)      # (B, 3, 256, 256)
        
        # ========================================
        # Forward pass - generate stylized image
        # ========================================
        # Two modes:
        # 1. Standard AdaIN (baseline)
        # 2. Face-aware AdaIN (regional adaptive normalization)
        
        if face_aware_adain and face_detector is not None:
            # Face-aware AdaIN: lighter stylization in face regions
            # Generate face masks for this batch
            with torch.no_grad():  # No gradients needed for mask generation
                face_masks = face_detector.generate_face_masks(
                    content, 
                    margin_factor=face_mask_margin
                )
            
            # Forward pass with face-aware AdaIN
            stylized, gen_features, content_features, style_features = \
                model.forward_with_features_face_aware(
                    content, style, face_masks, 
                    face_preservation_alpha=face_preservation_alpha
                )
        else:
            # Standard AdaIN (baseline)
            stylized, gen_features, content_features, style_features = \
                model.forward_with_features(content, style)
        
        # stylized: (B, 3, 256, 256) - the output stylized image
        # gen_features: dict - VGG features of stylized image at 4 layers
        # content_features: dict - VGG features of content image at 4 layers
        # style_features: dict - VGG features of style image at 4 layers
        
        # ========================================
        # Calculate content and style losses
        # ========================================
        # Content loss: MSE between stylized and content features at relu4_1
        # Ensures stylized image has same content structure as original
        content_loss = calc_content_loss(gen_features, content_features)
        
        # Style loss: MSE between mean/std of stylized and style features at 4 layers
        # Ensures stylized image has same style appearance as style image
        style_loss = calc_style_loss(gen_features, style_features)
        
        # ========================================
        # Calculate identity loss (if enabled) and similarity metrics (always)
        # ========================================
        # IMPORTANT: We ALWAYS compute face similarity for monitoring,
        # but only use it as a loss term when identity_weight > 0
        identity_loss = torch.tensor(0.0, device=device)  # Default: no identity loss
        num_faces = 0
        avg_similarity = 0.0
        
        if identity_preserver is not None:
            # Always compute similarity (even for baseline model)
            # This allows us to compare all models on the same metric
            id_loss, metrics = identity_preserver.compute_identity_loss(stylized, content)
            num_faces = metrics['num_matched_faces']      # How many face pairs detected
            avg_similarity = metrics['avg_similarity']    # Cosine similarity (0-1)
            
            # Track statistics for epoch-level metrics (always track for all models)
            total_identity_samples += num_faces
            if num_faces > 0:
                total_similarity += avg_similarity * num_faces
            
            # Only use as loss term if identity_weight > 0
            if identity_weight > 0:
                identity_loss = id_loss  # Use the computed loss for backprop
        
        # ========================================
        # Eye-Specific Loss (YOUR NOVEL CONTRIBUTION!)
        # ========================================
        eye_loss = torch.tensor(0.0, device=device, requires_grad=True)
        num_eyes = 0
        
        if eye_weight > 0 and eye_loss_module is not None:
            eye_loss, eye_metrics = eye_loss_module.compute_eye_loss(stylized, content)
            num_eyes = eye_metrics['num_matched_eyes']
            total_eye_samples += num_eyes
        
        # ========================================
        # Compute total weighted loss
        # ========================================
        # Typical values:
        # - content_weight = 1.0
        # - style_weight = 10.0 (style is weighted 10x more!)
        # - identity_weight = 0.0 (baseline) or 1000.0 (with identity)
        # - eye_weight = 0.0 (baseline) or 100.0 (with eye-specific)
        loss = content_weight * content_loss + style_weight * style_loss + identity_weight * identity_loss + eye_weight * eye_loss
        
        # ========================================
        # Backward pass and optimization
        # ========================================
        optimizer.zero_grad()  # Clear previous gradients
        loss.backward()        # Compute gradients (only for decoder!)
        optimizer.step()       # Update decoder weights
        
        # ========================================
        # Accumulate losses for epoch statistics
        # ========================================
        total_loss += loss.item()
        total_content_loss += content_loss.item()
        total_style_loss += style_loss.item()
        total_identity_loss += identity_loss.item()
        total_eye_loss += eye_loss.item()
        
        # ========================================
        # Update progress bar with current batch stats
        # ========================================
        postfix = {
            'loss': loss.item(),
            'content': content_loss.item(),
            'style': style_loss.item()
        }
        # Add identity metrics if using identity preservation
        if identity_weight > 0:
            postfix['identity'] = identity_loss.item()
            postfix['faces'] = num_faces
            if num_faces > 0:
                postfix['sim'] = avg_similarity  # Show similarity if faces detected
        # Add eye metrics if using eye-specific loss
        if eye_weight > 0:
            postfix['eye'] = eye_loss.item()
            postfix['eyes'] = num_eyes
        
        pbar.set_postfix(postfix)
    
    # ========================================
    # Calculate epoch-level averages
    # ========================================
    avg_loss = total_loss / len(dataloader)
    avg_content_loss = total_content_loss / len(dataloader)
    avg_style_loss = total_style_loss / len(dataloader)
    avg_identity_loss = total_identity_loss / len(dataloader)
    avg_eye_loss = total_eye_loss / len(dataloader)
    
    # Calculate average face similarity (weighted by number of faces)
    # Only meaningful if identity preservation is enabled and faces were detected
    avg_similarity_epoch = total_similarity / total_identity_samples if total_identity_samples > 0 else 0.0
    
    # Calculate face detection rate (percentage of images where faces were detected)
    # This is important to track - if detection rate drops, similarity becomes less reliable
    total_images = len(dataloader) * dataloader.batch_size
    detect_rate = total_identity_samples / total_images if total_images > 0 else 0.0
    
    return avg_loss, avg_content_loss, avg_style_loss, avg_identity_loss, avg_eye_loss, avg_similarity_epoch, detect_rate


def validate(model, dataloader, device, content_weight=1.0, style_weight=10.0,
             identity_weight=0.0, identity_preserver=None, eye_weight=0.0, eye_loss_module=None,
             face_aware_adain=False, face_detector=None, face_preservation_alpha=0.3, face_mask_margin=1.3):
    """
    Validate the model on a held-out validation set.
    
    This is similar to train_epoch but WITHOUT weight updates.
    It evaluates the model's performance on unseen data to:
    - Monitor for overfitting
    - Choose the best checkpoint
    - Track generalization performance
    
    Key differences from training:
    - model.eval() mode (no dropout, batch norm in eval mode)
    - torch.no_grad() context (no gradient computation = faster + less memory)
    - No optimizer.step() (weights don't change)
    - No progress bar updates per batch (quieter output)
    
    Args:
        (same as train_epoch)
    
    Returns:
        (same as train_epoch)
    """
    # Set model to evaluation mode
    model.eval()
    
    # Initialize loss accumulators (same as training)
    total_loss = 0.0
    total_content_loss = 0.0
    total_style_loss = 0.0
    total_identity_loss = 0.0
    total_eye_loss = 0.0
    total_identity_samples = 0
    total_eye_samples = 0
    total_similarity = 0.0
    
    # ========================================
    # Validation loop (no gradient computation!)
    # ========================================
    with torch.no_grad():  # Disable gradient tracking = faster + less memory
        for content, style in tqdm(dataloader, desc="Validating"):
            content = content.to(device)
            style = style.to(device)
            
            # Forward pass (same as training)
            if face_aware_adain and face_detector is not None:
                # Face-aware AdaIN
                face_masks = face_detector.generate_face_masks(
                    content, 
                    margin_factor=face_mask_margin
                )
                stylized, gen_features, content_features, style_features = \
                    model.forward_with_features_face_aware(
                        content, style, face_masks,
                        face_preservation_alpha=face_preservation_alpha
                    )
            else:
                # Standard AdaIN
                stylized, gen_features, content_features, style_features = \
                    model.forward_with_features(content, style)
            
            # Calculate losses (same as training)
            content_loss = calc_content_loss(gen_features, content_features)
            style_loss = calc_style_loss(gen_features, style_features)
            
            # Calculate identity loss if enabled (same as training)
            # IMPORTANT: Always compute similarity for monitoring
            identity_loss = torch.tensor(0.0, device=device)
            num_faces = 0
            avg_similarity = 0.0
            
            if identity_preserver is not None:
                # Always compute similarity (even for baseline model)
                id_loss, metrics = identity_preserver.compute_identity_loss(stylized, content)
                num_faces = metrics['num_matched_faces']
                avg_similarity = metrics['avg_similarity']
                
                total_identity_samples += num_faces
                if num_faces > 0:
                    total_similarity += avg_similarity * num_faces
                
                # Only use as loss term if identity_weight > 0
                if identity_weight > 0:
                    identity_loss = id_loss
            
            # Calculate eye loss if enabled
            eye_loss = torch.tensor(0.0, device=device)
            num_eyes = 0
            
            if eye_weight > 0 and eye_loss_module is not None:
                eye_loss, eye_metrics = eye_loss_module.compute_eye_loss(stylized, content)
                num_eyes = eye_metrics['num_matched_eyes']
                total_eye_samples += num_eyes
            
            # Total weighted loss
            loss = content_weight * content_loss + style_weight * style_loss + identity_weight * identity_loss + eye_weight * eye_loss
            
            # Accumulate losses
            total_loss += loss.item()
            total_content_loss += content_loss.item()
            total_style_loss += style_loss.item()
            total_identity_loss += identity_loss.item()
            total_eye_loss += eye_loss.item()
    
    # Calculate averages
    avg_loss = total_loss / len(dataloader)
    avg_content_loss = total_content_loss / len(dataloader)
    avg_style_loss = total_style_loss / len(dataloader)
    avg_identity_loss = total_identity_loss / len(dataloader)
    avg_eye_loss = total_eye_loss / len(dataloader)
    avg_similarity_epoch = total_similarity / total_identity_samples if total_identity_samples > 0 else 0.0
    
    # Calculate face detection rate
    total_images = len(dataloader) * dataloader.batch_size
    detect_rate = total_identity_samples / total_images if total_images > 0 else 0.0
    
    return avg_loss, avg_content_loss, avg_style_loss, avg_identity_loss, avg_eye_loss, avg_similarity_epoch, detect_rate


def worker_init_fn(worker_id):
    """
    Initialize random seeds for DataLoader workers.
    
    This ensures reproducibility when using multiple workers (num_workers > 0).
    Each worker gets a different but deterministic seed based on:
    - Base seed from args.seed
    - Worker ID (0, 1, 2, ...)
    
    Without this, each worker would use a random (non-reproducible) seed!
    
    Args:
        worker_id: Integer ID of the current worker (0-indexed)
    
    Example:
        If args.seed=42 and num_workers=4:
        - Worker 0: seed = 42 + 0 = 42
        - Worker 1: seed = 42 + 1 = 43
        - Worker 2: seed = 42 + 2 = 44
        - Worker 3: seed = 42 + 3 = 45
    """
    # Get the base seed from torch's generator state
    # This is set by torch.manual_seed(args.seed) earlier
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_path):
    """
    Save model checkpoint to disk for resuming training or inference.
    
    WHAT GETS SAVED:
    - Decoder weights (the only trainable part)
    - Optimizer state (for resuming training with same momentum/learning rate)
    - Current epoch number
    - Current loss value
    
    WHAT DOESN'T GET SAVED:
    - VGG encoder weights (pretrained, loaded from torchvision)
    - AdaIN layer (no parameters to save)
    
    Args:
        model: AdaINStyleTransfer model
        optimizer: Adam optimizer
        epoch: Current epoch number (0-indexed)
        loss: Current loss value
        checkpoint_path: Where to save (e.g., 'checkpoints/epoch_10.pth')
    
    Example:
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=10,
            loss=0.5,
            checkpoint_path='checkpoints/checkpoint_epoch_10.pth'
        )
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.decoder.state_dict(),  # Only decoder weights!
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def main():
    """
    Main training function - orchestrates the entire training pipeline.
    
    This function:
    1. Parses command-line arguments
    2. Sets up datasets and dataloaders
    3. Creates model and optimizer
    4. Optionally initializes identity preserver
    5. Runs training loop for specified epochs
    6. Saves checkpoints periodically
    
    Example command:
        python train_model.py \\
            --content-dir data/content \\
            --style-dir data/style \\
            --epochs 20 \\
            --identity-weight 0.1
    """
    # ============================================================================
    # ARGUMENT PARSING
    # ============================================================================
    parser = argparse.ArgumentParser(
        description="Train AdaIN Style Transfer Network",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # ========================================
    # Dataset arguments
    # ========================================
    parser.add_argument('--content-dir', type=str, required=True,
                        help='Directory containing content images (e.g., data/content)')
    parser.add_argument('--style-dir', type=str, required=True,
                        help='Directory containing style images (e.g., data/style)')
    parser.add_argument('--val-content-dir', type=str, default=None,
                        help='Directory containing validation content images (optional)')
    parser.add_argument('--val-style-dir', type=str, default=None,
                        help='Directory containing validation style images (optional)')
    
    # Split file arguments (for train/val/test splits)
    parser.add_argument('--split-file', type=str, default='data/content_splits/train.txt',
                        help='Path to training split file listing content images to use')
    parser.add_argument('--val-split-file', type=str, default='data/content_splits/val.txt',
                        help='Path to validation split file listing content images to use')
    
    # ========================================
    # Training hyperparameters
    # ========================================
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs (20 is typical)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (32 safe for dual A6000, up to 64 if GPUs are clean, 8 for smaller GPUs)')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning rate for Adam optimizer')
    parser.add_argument('--content-weight', type=float, default=1.0,
                        help='Weight for content loss (typically 1.0)')
    parser.add_argument('--style-weight', type=float, default=10.0,
                        help='Weight for style loss (typically 10.0)')
    parser.add_argument('--identity-weight', type=float, default=0.0,
                        help='Weight for identity loss: 0.0=baseline, 1000.0=optimal (from comprehensive tuning)')
    parser.add_argument('--eye-weight', type=float, default=0.0,
                        help='Weight for eye-specific loss: 0.0=disabled, 100.0=recommended (YOUR novel contribution!)')
    
    # ========================================
    # Face-aware AdaIN (Regional Adaptive Normalization)
    # ========================================
    parser.add_argument('--use-face-aware-adain', action='store_true',
                        help='Enable face-aware AdaIN (lighter stylization in face regions). '
                             'Inspired by Ulyanov et al. for better identity preservation. '
                             'Default: False (use standard AdaIN)')
    parser.add_argument('--face-preservation-alpha', type=float, default=0.3,
                        help='Stylization strength in face regions (0.0-1.0). '
                             '0.0 = full stylization (same as baseline), '
                             '0.3 = 30%% stylization, 70%% content (RECOMMENDED), '
                             '0.5 = half stylization, '
                             '1.0 = no stylization (pure content). '
                             'Only used if --use-face-aware-adain is enabled. Default: 0.3')
    parser.add_argument('--face-mask-margin', type=float, default=1.3,
                        help='Expand face bounding box by this factor (>= 1.0). '
                             '1.0 = exact box, 1.3 = expand by 30%% (default, includes hair/ears), '
                             '1.5 = expand by 50%% (more context). '
                             'Only used if --use-face-aware-adain is enabled. Default: 1.3')
    
    # ========================================
    # Model configuration
    # ========================================
    parser.add_argument('--image-size', type=int, default=256,
                        help='Size to resize images to (256 is standard)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to train on (cuda or cpu)')
    
    # ========================================
    # Checkpoint management
    # ========================================
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--save-interval', type=int, default=1,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    
    # ========================================
    # Other settings
    # ========================================
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers (0 = main process)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # ============================================================================
    # SETUP
    # ============================================================================
    
    # ========================================
    # Set random seeds for reproducibility
    # ========================================
    # CRITICAL: Set ALL random seeds for reproducible results
    torch.manual_seed(args.seed)           # PyTorch CPU random seed
    np.random.seed(args.seed)              # NumPy random seed
    random.seed(args.seed)                 # Python random seed
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)      # CUDA random seed (current GPU)
        torch.cuda.manual_seed_all(args.seed)  # CUDA random seed (all GPUs)
        
        # CRITICAL: Enable deterministic CUDA operations
        # This ensures identical results across runs with same seed
        # WARNING: May slightly reduce performance (~5-10%)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        print(f"✓ Reproducibility mode enabled (seed={args.seed})")
        print(f"  Note: Deterministic CUDA may reduce performance by ~5-10%")
    
    # ========================================
    # Setup device
    # ========================================
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    
    # ========================================
    # Create checkpoint directory
    # ========================================
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # ============================================================================
    # DATASET LOADING
    # ============================================================================
    print("\n" + "="*50)
    print("Loading datasets...")
    print("="*50)
    
    # ========================================
    # Create training dataset and dataloader
    # ========================================
    train_dataset = ImageDataset(
        content_dir=args.content_dir,
        style_dir=args.style_dir,
        split_file=args.split_file,  # Use train split
        image_size=args.image_size
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,              # Shuffle for better training
        num_workers=args.num_workers,  # Parallel data loading
        pin_memory=True,           # Faster transfer to GPU
        worker_init_fn=worker_init_fn  # Seed workers for reproducibility
    )
    
    # ========================================
    # Validation dataset and dataloader (using val split file)
    # ========================================
    print(f"\nLoading validation dataset...")
    val_dataset = ImageDataset(
        content_dir=args.content_dir,  # Same directory, different split
        style_dir=args.style_dir,
        split_file=args.val_split_file,  # Use val split
        image_size=args.image_size
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,             # Don't shuffle validation (deterministic for curves)
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn  # Seed workers for reproducibility
    )
    print(f"Validation pairs: {len(val_dataset)}")
    
    # ========================================
    # Create test dataset and loader
    # ========================================
    print("\nLoading test dataset...")
    test_dataset = ImageDataset(
        content_dir=args.content_dir,  # Same directory, different split
        style_dir=args.style_dir,
        split_file='data/content_splits/test.txt',  # Use test split
        image_size=args.image_size
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,             # Don't shuffle test (deterministic for curves)
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn  # Seed workers for reproducibility
    )
    print(f"Test pairs: {len(test_dataset)}")
    
    # ============================================================================
    # MODEL CREATION
    # ============================================================================
    print("\n" + "="*50)
    print("Creating model...")
    print("="*50)
    
    # ========================================
    # Create AdaIN Style Transfer model
    # ========================================
    model = AdaINStyleTransfer(device=device)
    print(f"✓ Model created and moved to {device}")
    
    # ========================================
    # Count parameters
    # ========================================
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.decoder.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters (decoder): {trainable_params:,}")
    print(f"Frozen parameters (encoder): {total_params - trainable_params:,}")
    
    # ========================================
    # Create optimizer (only for decoder!)
    # ========================================
    # Adam optimizer with default betas=(0.9, 0.999)
    # Only decoder parameters are passed, so encoder stays frozen
    optimizer = optim.Adam(model.decoder.parameters(), lr=args.learning_rate)
    print(f"✓ Adam optimizer created (lr={args.learning_rate})")
    
    # ============================================================================
    # IDENTITY PRESERVER (always initialize for similarity monitoring)
    # ============================================================================
    # Always initialize identity preserver so we can track face similarity
    # for ALL models (including baseline). This allows fair comparison.
    print(f"\nInitializing identity preserver...")
    if args.identity_weight > 0:
        print(f"  Mode: WITH identity loss (weight={args.identity_weight})")
    else:
        print(f"  Mode: Monitoring only (no identity loss in training)")
    
    identity_preserver = None
    try:
        identity_preserver = IdentityPreserver(device=device)
        print("✓ Identity preserver initialized successfully")
        print("   Face detector: MTCNN")
        print("   Face recognizer: InceptionResnetV1 (VGGFace2)")
        print("   Usage: Face similarity tracked for all models")
    except ImportError as e:
        print(f"⚠️  Warning: Could not initialize identity preserver: {e}")
        print("   Install facenet-pytorch: pip install facenet-pytorch")
        print("   Continuing without identity tracking...")
        identity_preserver = None
    
    # ============================================================================
    # EYE-SPECIFIC LOSS (Novel Contribution!)
    # ============================================================================
    eye_loss_module = None
    if args.eye_weight > 0:
        print(f"\nInitializing eye-specific loss (weight={args.eye_weight})...")
        try:
            from model_face_utils import EyeSpecificLoss
            eye_loss_module = EyeSpecificLoss(device=device)
            print("✓ Eye-specific loss initialized successfully")
            print("   Eye detector: MTCNN (eye landmarks)")
            print("   Feature extractor: VGG19 (relu2_1)")
            print("   🎯 YOUR NOVEL CONTRIBUTION!")
        except ImportError as e:
            print(f"⚠️  Warning: Could not initialize eye-specific loss: {e}")
            print("   Install facenet-pytorch: pip install facenet-pytorch")
            print("   Continuing without eye loss...")
            args.eye_weight = 0
    
    # ============================================================================
    # FACE DETECTOR (for face-aware AdaIN)
    # ============================================================================
    face_detector = None
    if args.use_face_aware_adain:
        print(f"\nInitializing face detector for face-aware AdaIN...")
        print(f"   Face preservation alpha: {args.face_preservation_alpha} (0=full style, 1=full content)")
        print(f"   Face mask margin: {args.face_mask_margin}x (expand bounding box)")
        try:
            face_detector = FaceDetector(device=device, keep_all=False, min_face_size=20)
            print("✓ Face detector initialized successfully")
            print("   Method: MTCNN (Multi-task Cascaded CNN)")
            print("   Usage: Generates masks for regional adaptive normalization")
            print("   Effect: Lighter stylization in face regions, full style in background")
        except ImportError as e:
            print(f"⚠️  Warning: Could not initialize face detector: {e}")
            print("   Install facenet-pytorch: pip install facenet-pytorch")
            print("   Disabling face-aware AdaIN...")
            args.use_face_aware_adain = False
    
    # ============================================================================
    # RESUME FROM CHECKPOINT (optional)
    # ============================================================================
    start_epoch = 0
    
    if args.resume:
        print(f"\nLoading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.decoder.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"✓ Resumed from epoch {start_epoch}")
        print(f"   Previous loss: {checkpoint['loss']:.4f}")
    
    # ============================================================================
    # TRAINING LOOP
    # ============================================================================
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50)
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Image size: {args.image_size}×{args.image_size}")
    print(f"\nLoss weights:")
    print(f"  Content: {args.content_weight}")
    print(f"  Style: {args.style_weight}")
    if args.identity_weight > 0:
        print(f"  Identity: {args.identity_weight}")
    
    if args.use_face_aware_adain:
        print(f"\nFace-aware AdaIN: ENABLED")
        print(f"  Face preservation alpha: {args.face_preservation_alpha}")
        print(f"  Face mask margin: {args.face_mask_margin}x")
    else:
        print(f"\nFace-aware AdaIN: DISABLED (standard AdaIN)")
    print("="*50 + "\n")
    
    # Track best validation loss for model selection
    best_val_loss = float('inf')
    
    # ========================================
    # Setup CSV logging for training curves
    # ========================================
    # Use same header format for ALL models for easy comparison
    # Columns that don't apply to a model will be filled with 0.0
    # Format: epoch, train_metrics, val_metrics, test_metrics, similarities
    csv_path = os.path.join(args.checkpoint_dir, 'training_curves.csv')
    csv_file = open(csv_path, 'w')
    csv_header = 'epoch,'
    csv_header += 'train_loss,train_content,train_style,train_identity,train_eye,'
    csv_header += 'val_loss,val_content,val_style,val_identity,val_eye,'
    csv_header += 'test_loss,test_content,test_style,test_identity,test_eye,'
    csv_header += 'train_similarity,val_similarity,test_similarity,'
    csv_header += 'train_detect_rate,val_detect_rate,test_detect_rate'
    csv_file.write(csv_header + '\n')
    csv_file.flush()  # Ensure header is written immediately
    print(f"✓ Training curves will be saved to: {csv_path}\n")
    
    # ========================================
    # Main training loop
    # ========================================
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 50)
        
        # ========================================
        # Training phase
        # ========================================
        train_loss, train_content_loss, train_style_loss, train_identity_loss, train_eye_loss, train_similarity, train_detect_rate = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            content_weight=args.content_weight,
            style_weight=args.style_weight,
            identity_weight=args.identity_weight,
            identity_preserver=identity_preserver,
            eye_weight=args.eye_weight,
            eye_loss_module=eye_loss_module,
            face_aware_adain=args.use_face_aware_adain,
            face_detector=face_detector,
            face_preservation_alpha=args.face_preservation_alpha,
            face_mask_margin=args.face_mask_margin
        )
        
        # Print training statistics
        train_stats = f"\nTrain Loss: {train_loss:.4f} (Content: {train_content_loss:.4f}, Style: {train_style_loss:.4f}"
        if args.identity_weight > 0:
            train_stats += f", Identity: {train_identity_loss:.4f}, Similarity: {train_similarity:.4f}"
        if args.eye_weight > 0:
            train_stats += f", Eye: {train_eye_loss:.4f}"
        train_stats += ")"
        print(train_stats)
        
        # ========================================
        # Validation phase (always run for proper training curves)
        # ========================================
        val_loss, val_content_loss, val_style_loss, val_identity_loss, val_eye_loss, val_similarity, val_detect_rate = validate(
            model=model,
            dataloader=val_loader,
            device=device,
            content_weight=args.content_weight,
            style_weight=args.style_weight,
            identity_weight=args.identity_weight,
            identity_preserver=identity_preserver,
            eye_weight=args.eye_weight,
            eye_loss_module=eye_loss_module,
            face_aware_adain=args.use_face_aware_adain,
            face_detector=face_detector,
            face_preservation_alpha=args.face_preservation_alpha,
            face_mask_margin=args.face_mask_margin
        )
        
        # Print validation statistics
        val_stats = f"Val Loss: {val_loss:.4f} (Content: {val_content_loss:.4f}, Style: {val_style_loss:.4f}"
        if args.identity_weight > 0:
            val_stats += f", Identity: {val_identity_loss:.4f}, Similarity: {val_similarity:.4f}"
        if args.eye_weight > 0:
            val_stats += f", Eye: {val_eye_loss:.4f}"
        val_stats += ")"
        print(val_stats)
        
        # ========================================
        # Test phase (evaluate on held-out test set)
        # ========================================
        test_loss, test_content_loss, test_style_loss, test_identity_loss, test_eye_loss, test_similarity, test_detect_rate = validate(
            model=model,
            dataloader=test_loader,
            device=device,
            content_weight=args.content_weight,
            style_weight=args.style_weight,
            identity_weight=args.identity_weight,
            identity_preserver=identity_preserver,
            eye_weight=args.eye_weight,
            eye_loss_module=eye_loss_module,
            face_aware_adain=args.use_face_aware_adain,
            face_detector=face_detector,
            face_preservation_alpha=args.face_preservation_alpha,
            face_mask_margin=args.face_mask_margin
        )
        
        # Print test statistics
        test_stats = f"Test Loss: {test_loss:.4f} (Content: {test_content_loss:.4f}, Style: {test_style_loss:.4f}"
        if args.identity_weight > 0:
            test_stats += f", Identity: {test_identity_loss:.4f}, Similarity: {test_similarity:.4f}"
        if args.eye_weight > 0:
            test_stats += f", Eye: {test_eye_loss:.4f}"
        test_stats += ")"
        print(test_stats)
        
        # ========================================
        # Log to CSV for training curves
        # ========================================
        # Format: epoch, train_metrics, val_metrics, test_metrics, similarities
        # Always write all columns (use 0.0 for unused metrics)
        
        csv_line = f"{epoch+1},"
        
        # Train metrics
        csv_line += f"{train_loss:.6f},{train_content_loss:.6f},{train_style_loss:.6f},"
        csv_line += f"{train_identity_loss:.6f}," if args.identity_weight > 0 else "0.0,"
        csv_line += f"{train_eye_loss:.6f}," if args.eye_weight > 0 else "0.0,"
        
        # Val metrics
        csv_line += f"{val_loss:.6f},{val_content_loss:.6f},{val_style_loss:.6f},"
        csv_line += f"{val_identity_loss:.6f}," if args.identity_weight > 0 else "0.0,"
        csv_line += f"{val_eye_loss:.6f}," if args.eye_weight > 0 else "0.0,"
        
        # Test metrics
        csv_line += f"{test_loss:.6f},{test_content_loss:.6f},{test_style_loss:.6f},"
        csv_line += f"{test_identity_loss:.6f}," if args.identity_weight > 0 else "0.0,"
        csv_line += f"{test_eye_loss:.6f}," if args.eye_weight > 0 else "0.0,"
        
        # Similarities (always write, computed for all models)
        csv_line += f"{train_similarity:.6f},{val_similarity:.6f},{test_similarity:.6f},"
        
        # Detection rates (always write, computed for all models)
        csv_line += f"{train_detect_rate:.6f},{val_detect_rate:.6f},{test_detect_rate:.6f}"
        
        csv_file.write(csv_line + '\n')
        csv_file.flush()  # Ensure data is written immediately
        
        # ========================================
        # Update best model if validation improved
        # ========================================
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"  → New best validation loss: {best_val_loss:.4f}")
    
    # ========================================
    # Save final model only
    # ========================================
    final_checkpoint_path = os.path.join(args.checkpoint_dir, 'final_model.pth')
    save_checkpoint(model, optimizer, args.epochs - 1, train_loss, final_checkpoint_path)
    print(f"\n✓ Final model saved to: {final_checkpoint_path}")
    
    # ========================================
    # Close CSV file
    # ========================================
    csv_file.close()
    print(f"\n✓ Training curves saved to: {csv_path}")
    
    # ============================================================================
    # TRAINING COMPLETE!
    # ============================================================================
    print("\n" + "="*50)
    print("🎉 Training completed!")
    print("="*50)
    print(f"\nFinal model saved to: {args.checkpoint_dir}/final_model.pth")
    print(f"Training curves saved to: {csv_path}")
    print("\nCSV contains train/val/test metrics for all epochs:")
    print("  - Losses: train_loss, val_loss, test_loss")
    print("  - Content: train_content, val_content, test_content")
    print("  - Style: train_style, val_style, test_style")
    print("  - Identity: train_identity, val_identity, test_identity")
    print("  - Similarity: train_similarity, val_similarity, test_similarity")
    print("  - Eye: train_eye, val_eye, test_eye")
    print("\nNext steps:")
    print("  1. Analyze training curves to check for overfitting")
    print("  2. Compare val vs test similarity at each epoch")
    print("  3. Run final evaluation on test set")
    print()


if __name__ == "__main__":
    main()

