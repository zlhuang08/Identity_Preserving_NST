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
    # Basic training (baseline, no identity preservation)
    python train_model.py \\
        --content-dir data/content \\
        --style-dir data/style \\
        --epochs 20 \\
        --batch-size 8
    
    # Advanced training (with identity preservation for faces)
    python train_model.py \\
        --content-dir data/content \\
        --style-dir data/style \\
        --epochs 20 \\
        --batch-size 8 \\
        --identity-weight 0.1
    
    # Resume from checkpoint
    python train_model.py \\
        --content-dir data/content \\
        --style-dir data/style \\
        --resume checkpoints/checkpoint_epoch_10.pth

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
from model_face_utils import IdentityPreserver


class ImageDataset(Dataset):
    """
    Dataset for loading content and style image pairs for AdaIN training.
    
    KEY DESIGN CHOICE - RANDOM PAIRING:
    During training, we randomly pair content images with style images!
    This teaches the decoder to handle arbitrary style transfer.
    
    Example:
        - Epoch 1, Batch 1: face_001.jpg + starry_night.jpg
        - Epoch 1, Batch 2: face_002.jpg + peter_rabbit.jpg
        - Epoch 2, Batch 1: face_001.jpg + the_scream.jpg  (different style!)
    
    This random pairing is crucial because:
    ✓ Prevents overfitting to specific content-style combinations
    ✓ Enables arbitrary style transfer (any content + any style)
    ✓ Maximizes dataset diversity (N×M combinations from N content + M styles)
    
    DATASET REQUIREMENTS:
    - Content images: Natural photos (faces, objects, scenes)
      Examples: CelebA faces, COCO objects, MS-COCO scenes
      For our project: Synthetic children's faces from ThisPersonDoesNotExist
    
    - Style images: Artistic images with distinct visual styles
      Examples: Famous paintings, watercolors, sketches
      For our project: Peter Rabbit, Kate Greenaway, etc.
    
    TRANSFORMS APPLIED:
    1. Resize to square (e.g., 256×256) - ensures consistent size
    2. RandomCrop (during training) - data augmentation
    3. ToTensor - converts PIL Image to PyTorch tensor [0,1]
    """
    def __init__(self, content_dir, style_dir, transform=None, image_size=256):
        """
        Initialize dataset by loading image paths.
        
        Args:
            content_dir: Directory containing content images
                        Example: 'data/content/' with face_00000.jpg, face_00001.jpg, ...
            style_dir: Directory containing style images
                      Example: 'data/style/' with starry_night.jpg, peter_rabbit.jpg, ...
            transform: Optional custom transform (if None, uses default)
            image_size: Size to resize images to (default: 256×256)
                       Larger = better quality but slower training and more VRAM
                       Smaller = faster training but lower quality
        
        Example:
            # Create dataset for training
            dataset = ImageDataset(
                content_dir='data/content',
                style_dir='data/style',
                image_size=256
            )
            
            # Create dataloader
            loader = DataLoader(dataset, batch_size=8, shuffle=True)
        """
        self.content_dir = Path(content_dir)
        self.style_dir = Path(style_dir)
        
        # ========================================
        # Load all image file paths
        # ========================================
        self.content_images = self._get_image_files(self.content_dir)
        self.style_images = self._get_image_files(self.style_dir)
        
        # Report dataset sizes
        print(f"Found {len(self.content_images)} content images")
        print(f"Found {len(self.style_images)} style images")
        
        # ========================================
        # Validate dataset
        # ========================================
        if len(self.content_images) == 0:
            raise ValueError(f"No images found in content directory: {content_dir}")
        if len(self.style_images) == 0:
            raise ValueError(f"No images found in style directory: {style_dir}")
        
        # ========================================
        # Setup image transforms
        # ========================================
        if transform is None:
            # Default transform pipeline for training
            self.transform = transforms.Compose([
                # Step 1: Resize to target size (maintaining aspect ratio is not critical)
                transforms.Resize((image_size, image_size)),
                
                # Step 2: Random crop for data augmentation (helps prevent overfitting)
                # Note: Since we already resized to exact size, this is a no-op
                # but kept for compatibility if you want larger resize + crop
                transforms.RandomCrop(image_size),
                
                # Step 3: Convert PIL Image to PyTorch tensor in [0, 1] range
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
        
        Since we randomly pair content with styles, the effective dataset
        size is len(content_images), with each content image paired with
        a different random style each epoch.
        """
        return len(self.content_images)
    
    def __getitem__(self, idx):
        """
        Get a content-style pair for training - THE KEY METHOD!
        
        IMPORTANT: Style images are chosen RANDOMLY!
        This is not a bug - it's intentional for arbitrary style transfer.
        
        Process:
        1. Load content image at index idx (sequential access)
        2. Load RANDOM style image (random access)
        3. Apply transforms to both
        4. Return as tensor pair
        
        Args:
            idx: Index of content image to load (0 to len(content_images)-1)
        
        Returns:
            content_tensor: (3, H, W) tensor of content image in [0, 1]
            style_tensor: (3, H, W) tensor of style image in [0, 1]
        
        Example:
            dataset = ImageDataset('data/content', 'data/style')
            content, style = dataset[0]
            
            print(content.shape)  # torch.Size([3, 256, 256])
            print(style.shape)    # torch.Size([3, 256, 256])
            print(content.min(), content.max())  # 0.0, 1.0
        """
        # ========================================
        # Load content image (sequential)
        # ========================================
        content_path = self.content_images[idx]
        content_img = Image.open(content_path).convert('RGB')
        # convert('RGB') ensures 3 channels (handles grayscale, RGBA, etc.)
        
        # ========================================
        # Load random style image
        # ========================================
        # Pick a random style image (different each time for same idx!)
        style_idx = random.randint(0, len(self.style_images) - 1)
        style_path = self.style_images[style_idx]
        style_img = Image.open(style_path).convert('RGB')
        
        # ========================================
        # Apply transforms (resize, crop, to tensor)
        # ========================================
        content_tensor = self.transform(content_img)  # (3, 256, 256) in [0, 1]
        style_tensor = self.transform(style_img)      # (3, 256, 256) in [0, 1]
        
        return content_tensor, style_tensor


def train_epoch(model, dataloader, optimizer, device, content_weight=1.0, style_weight=10.0, 
                identity_weight=0.0, identity_preserver=None):
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
    
    Typical weights:
    - content_weight = 1.0 (baseline)
    - style_weight = 10.0 (style is more important for artistic look)
    - identity_weight = 0.0 (baseline) or 0.1 (with identity preservation)
    
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
    total_identity_samples = 0  # Count of detected face pairs
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
        # This extracts features, applies AdaIN, decodes, AND gets intermediate features for loss
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
        # Calculate identity loss (if enabled)
        # ========================================
        identity_loss = torch.tensor(0.0, device=device)  # Default: no identity loss
        num_faces = 0
        avg_similarity = 0.0
        
        if identity_weight > 0 and identity_preserver is not None:
            # Detect faces in both stylized and content images
            # Extract face embeddings and compute MSE loss
            # This encourages stylized faces to preserve identity
            identity_loss, metrics = identity_preserver.compute_identity_loss(stylized, content)
            num_faces = metrics['num_matched_faces']      # How many face pairs detected
            avg_similarity = metrics['avg_similarity']    # Cosine similarity (0-1)
            
            # Track statistics for epoch-level metrics
            total_identity_samples += num_faces
            if num_faces > 0:
                total_similarity += avg_similarity * num_faces
        
        # ========================================
        # Compute total weighted loss
        # ========================================
        # Typical values:
        # - content_weight = 1.0
        # - style_weight = 10.0 (style is weighted 10x more!)
        # - identity_weight = 0.0 (baseline) or 0.1 (with identity)
        loss = content_weight * content_loss + style_weight * style_loss + identity_weight * identity_loss
        
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
        
        pbar.set_postfix(postfix)
    
    # ========================================
    # Calculate epoch-level averages
    # ========================================
    avg_loss = total_loss / len(dataloader)
    avg_content_loss = total_content_loss / len(dataloader)
    avg_style_loss = total_style_loss / len(dataloader)
    avg_identity_loss = total_identity_loss / len(dataloader)
    
    # Calculate average face similarity (weighted by number of faces)
    # Only meaningful if identity preservation is enabled and faces were detected
    avg_similarity_epoch = total_similarity / total_identity_samples if total_identity_samples > 0 else 0.0
    
    return avg_loss, avg_content_loss, avg_style_loss, avg_identity_loss, avg_similarity_epoch


def validate(model, dataloader, device, content_weight=1.0, style_weight=10.0,
             identity_weight=0.0, identity_preserver=None):
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
    total_identity_samples = 0
    total_similarity = 0.0
    
    # ========================================
    # Validation loop (no gradient computation!)
    # ========================================
    with torch.no_grad():  # Disable gradient tracking = faster + less memory
        for content, style in tqdm(dataloader, desc="Validating"):
            content = content.to(device)
            style = style.to(device)
            
            # Forward pass (same as training)
            stylized, gen_features, content_features, style_features = \
                model.forward_with_features(content, style)
            
            # Calculate losses (same as training)
            content_loss = calc_content_loss(gen_features, content_features)
            style_loss = calc_style_loss(gen_features, style_features)
            
            # Calculate identity loss if enabled (same as training)
            identity_loss = torch.tensor(0.0, device=device)
            num_faces = 0
            avg_similarity = 0.0
            
            if identity_weight > 0 and identity_preserver is not None:
                identity_loss, metrics = identity_preserver.compute_identity_loss(stylized, content)
                num_faces = metrics['num_matched_faces']
                avg_similarity = metrics['avg_similarity']
                
                total_identity_samples += num_faces
                if num_faces > 0:
                    total_similarity += avg_similarity * num_faces
            
            # Total weighted loss
            loss = content_weight * content_loss + style_weight * style_loss + identity_weight * identity_loss
            
            # Accumulate losses
            total_loss += loss.item()
            total_content_loss += content_loss.item()
            total_style_loss += style_loss.item()
            total_identity_loss += identity_loss.item()
    
    # Calculate averages
    avg_loss = total_loss / len(dataloader)
    avg_content_loss = total_content_loss / len(dataloader)
    avg_style_loss = total_style_loss / len(dataloader)
    avg_identity_loss = total_identity_loss / len(dataloader)
    avg_similarity_epoch = total_similarity / total_identity_samples if total_identity_samples > 0 else 0.0
    
    return avg_loss, avg_content_loss, avg_style_loss, avg_identity_loss, avg_similarity_epoch


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
            --batch-size 8 \\
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
    
    # ========================================
    # Training hyperparameters
    # ========================================
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs (20 is typical)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size (8 works for 8GB GPU, use 4 for 4GB GPU)')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning rate for Adam optimizer')
    parser.add_argument('--content-weight', type=float, default=1.0,
                        help='Weight for content loss (typically 1.0)')
    parser.add_argument('--style-weight', type=float, default=10.0,
                        help='Weight for style loss (typically 10.0)')
    parser.add_argument('--identity-weight', type=float, default=0.0,
                        help='Weight for identity loss: 0.0=baseline, 0.1=with identity')
    
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
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
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
        image_size=args.image_size
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,              # Shuffle for better training
        num_workers=args.num_workers,  # Parallel data loading
        pin_memory=True            # Faster transfer to GPU
    )
    
    # ========================================
    # Optional validation dataset and dataloader
    # ========================================
    val_loader = None
    if args.val_content_dir and args.val_style_dir:
        print(f"\nLoading validation dataset...")
        val_dataset = ImageDataset(
            content_dir=args.val_content_dir,
            style_dir=args.val_style_dir,
            image_size=args.image_size
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,             # Don't shuffle validation
            num_workers=args.num_workers,
            pin_memory=True
        )
    
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
    print("✓ Model created")
    
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
    # IDENTITY PRESERVER (optional)
    # ============================================================================
    identity_preserver = None
    if args.identity_weight > 0:
        print(f"\nInitializing identity preserver (weight={args.identity_weight})...")
        try:
            identity_preserver = IdentityPreserver(device=device)
            print("✓ Identity preserver initialized successfully")
            print("   Face detector: MTCNN")
            print("   Face recognizer: InceptionResnetV1 (VGGFace2)")
        except ImportError as e:
            print(f"⚠️  Warning: Could not initialize identity preserver: {e}")
            print("   Install facenet-pytorch: pip install facenet-pytorch")
            print("   Continuing without identity loss...")
            args.identity_weight = 0
    
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
    print("="*50 + "\n")
    
    # Track best validation loss for model selection
    best_val_loss = float('inf')
    
    # ========================================
    # Main training loop
    # ========================================
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 50)
        
        # ========================================
        # Training phase
        # ========================================
        train_loss, train_content_loss, train_style_loss, train_identity_loss, train_similarity = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            content_weight=args.content_weight,
            style_weight=args.style_weight,
            identity_weight=args.identity_weight,
            identity_preserver=identity_preserver
        )
        
        # Print training statistics
        train_stats = f"\nTrain Loss: {train_loss:.4f} (Content: {train_content_loss:.4f}, Style: {train_style_loss:.4f}"
        if args.identity_weight > 0:
            train_stats += f", Identity: {train_identity_loss:.4f}, Similarity: {train_similarity:.4f}"
        train_stats += ")"
        print(train_stats)
        
        # ========================================
        # Validation phase (if validation set provided)
        # ========================================
        if val_loader:
            val_loss, val_content_loss, val_style_loss, val_identity_loss, val_similarity = validate(
                model=model,
                dataloader=val_loader,
                device=device,
                content_weight=args.content_weight,
                style_weight=args.style_weight,
                identity_weight=args.identity_weight,
                identity_preserver=identity_preserver
            )
            
            # Print validation statistics
            val_stats = f"Val Loss: {val_loss:.4f} (Content: {val_content_loss:.4f}, Style: {val_style_loss:.4f}"
            if args.identity_weight > 0:
                val_stats += f", Identity: {val_identity_loss:.4f}, Similarity: {val_similarity:.4f}"
            val_stats += ")"
            print(val_stats)
            
            # ========================================
            # Save best model based on validation loss
            # ========================================
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
                save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)
                print(f"✓ New best model saved (val_loss: {val_loss:.4f})")
        
        # ========================================
        # Save periodic checkpoint
        # ========================================
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            save_checkpoint(model, optimizer, epoch, train_loss, checkpoint_path)
    
    # ========================================
    # Save final model
    # ========================================
    final_checkpoint_path = os.path.join(args.checkpoint_dir, 'final_model.pth')
    save_checkpoint(model, optimizer, args.epochs - 1, train_loss, final_checkpoint_path)
    
    # ============================================================================
    # TRAINING COMPLETE!
    # ============================================================================
    print("\n" + "="*50)
    print("🎉 Training completed!")
    print("="*50)
    print(f"\nCheckpoints saved to: {args.checkpoint_dir}/")
    print(f"  - best_model.pth (lowest validation loss)")
    print(f"  - final_model.pth (last epoch)")
    print(f"  - checkpoint_epoch_*.pth (periodic saves)")
    print("\nNext steps:")
    print("  1. Run inference: python eval_inference.py --checkpoint <path>")
    print("  2. Evaluate metrics: python eval_metrics.py --checkpoint <path>")
    print("  3. Visualize results: python result_visualize.py")
    print()


if __name__ == "__main__":
    main()

