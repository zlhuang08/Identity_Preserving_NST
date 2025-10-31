#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate synthetic faces using ThisPersonDoesNotExist.com

This script downloads AI-generated faces from ThisPersonDoesNotExist.com (powered by StyleGAN).
It's designed for ethical research - 100% synthetic faces with no privacy concerns.

FEATURES:
- Smart resume: Automatically detects existing images and continues from where you left off
- Rate limiting: Respects the server with configurable delays between requests
- Error handling: Gracefully handles network issues and server errors
- Progress tracking: Shows real-time progress with tqdm progress bar
- Cache busting: Uses timestamps to ensure unique faces (no duplicates)
- Data splitting: Automatically create train/val/test splits (60/20/20) with guaranteed test images

USAGE EXAMPLES:
    # Basic: Generate 200 faces + auto-create splits (simplest!)
    python data_generate_faces.py
    
    # Custom number of images (splits created automatically)
    python data_generate_faces.py --num-images 500
    
    # Custom output directory
    python data_generate_faces.py --output-dir data/my_faces --num-images 200
    
    # Slower rate limiting (be extra nice to server)
    python data_generate_faces.py --delay 2.0
    
    # Ask before resuming (interactive mode)
    python data_generate_faces.py --no-auto-continue
    
    # Add more faces later (splits updated automatically, test/val preserved!)
    python data_generate_faces.py --num-images 300  # adds 100 more to train split

NAMING CONVENTION:
    Images are saved as: face_00000.jpg, face_00001.jpg, face_00002.jpg, ...
    Zero-padding ensures proper sorting and easy indexing.

DATA SPLITS:
    Automatically creates three .txt files in data/content_splits/:
    - train.txt: 60% of images (e.g., 120 out of 200)
    - val.txt: 20% of images (e.g., 40 out of 200)
    - test.txt: 20% of images (e.g., 40 out of 200), includes face_00010.jpg and face_00066.jpg
    
    These files list which images belong to each split WITHOUT moving any files.
    This follows CS230 course standards (60/20/20 split ratio)!
"""

import os
import time
import urllib.request
from pathlib import Path
from tqdm import tqdm
import argparse
import random


def generate_synthetic_faces(output_dir="data/content", num_images=1000, delay=1.0, auto_continue=True):
    """
    Generate synthetic faces from ThisPersonDoesNotExist.com
    
    Args:
        output_dir: Output directory (e.g., 'data/content')
        num_images: TOTAL target number of images (not additional) - if you already have 50 images
                   and set num_images=100, it will download 50 more
        delay: Delay between requests in seconds (rate limiting to be nice to server)
        auto_continue: If True, automatically resume from where you left off;
                      If False, ask for user confirmation before resuming
    
    Returns:
        Path object pointing to the output directory
    """
    # Print header banner
    print("="*50)
    print("Generating Synthetic Faces")
    print("="*50)
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)  # exist_ok=True means no error if already exists
    
    # Display configuration info
    print(f"\nOutput directory: {output_dir}")
    print(f"Target total images: {num_images}")
    print(f"Delay between requests: {delay}s")
    print("\nNote: This is 100% ethical - all faces are AI-generated")
    print("Source: ThisPersonDoesNotExist.com (StyleGAN)")
    
    # ========================================
    # STEP 1: Check for existing images
    # ========================================
    # Look for any existing face_*.jpg files in the output directory
    existing = list(output_path.glob("face_*.jpg"))
    
    if existing:
        # Extract numeric indices from filenames to determine where we left off
        # Example: face_00123.jpg -> extract the number 123
        indices = []
        for f in existing:
            try:
                # f.stem gets filename without extension: "face_00123.jpg" -> "face_00123"
                # split('_')[1] gets the part after underscore: "face_00123" -> "00123"
                # int() converts to integer: "00123" -> 123
                idx = int(f.stem.split('_')[1])
                indices.append(idx)
            except (IndexError, ValueError):
                # Skip files that don't match the expected pattern
                # IndexError: if filename doesn't have '_'
                # ValueError: if the part after '_' isn't a number
                continue
        
        if indices:
            # Find the highest index number to resume from next one
            # Example: if we have face_00000.jpg through face_00099.jpg,
            # max(indices) = 99, so start_idx = 100
            start_idx = max(indices) + 1
            existing_count = len(existing)
            
            print(f"\n✓ Found {existing_count} existing images (up to face_{max(indices):05d}.jpg)")
            
            # Decide whether to resume or start over
            if auto_continue:
                # Automatically resume (non-interactive mode)
                print(f"✓ Auto-continuing from index {start_idx}")
            else:
                # Ask user for confirmation (interactive mode)
                response = input(f"Continue from index {start_idx}? (y/n): ")
                if response.lower() != 'y':
                    start_idx = 0
                    print("⚠️  Starting from 0 (will overwrite existing files!)")
        else:
            # Edge case: files exist but none match face_*.jpg pattern
            start_idx = 0
    else:
        # No existing images found at all
        start_idx = 0
        print("\nNo existing images found. Starting from 0.")
    
    # ========================================
    # STEP 2: Check if we're already done
    # ========================================
    # If we already have enough images, exit early
    # Example: target is 100 but we already have 150 images
    if num_images <= start_idx:
        print(f"\n✓ Already have {start_idx} images (target: {num_images})")
        print("Nothing to do!")
        return output_path
    
    # Calculate how many NEW images we need to download
    # Example: target=200, start_idx=150 -> need 50 more images
    new_images_count = num_images - start_idx
    print(f"\nGenerating {new_images_count} new images (from {start_idx} to {num_images-1})...")
    print("This may take a while (be patient, it's worth it!)")
    
    # ========================================
    # STEP 3: Download images from the web
    # ========================================
    url = "https://thispersondoesnotexist.com/"
    
    # Track success/failure rates
    success_count = 0
    fail_count = 0
    
    # Main download loop with progress bar
    for i in tqdm(range(start_idx, num_images)):
        try:
            # Construct output filename with zero-padding
            # Example: i=5 -> "face_00005.jpg"
            # The :05d format means: decimal integer, padded to 5 digits with zeros
            output_file = output_path / f"face_{i:05d}.jpg"
            
            # Add timestamp parameter to avoid browser/CDN caching
            # Without this, we might get the same face multiple times!
            # Each request gets a unique timestamp -> unique face
            timestamped_url = f"{url}?t={time.time()}"
            
            # Download and save the image
            urllib.request.urlretrieve(timestamped_url, output_file)
            
            success_count += 1
            
            # RATE LIMITING: Be respectful to the server
            # Don't hammer the server with rapid requests
            time.sleep(delay)
            
        except Exception as e:
            # Handle any download errors (network issues, server errors, etc.)
            print(f"\nError downloading image {i}: {e}")
            fail_count += 1
            
            # Wait longer after an error before retrying
            time.sleep(delay * 2)
            
            # Safety mechanism: stop if too many consecutive failures
            # This prevents infinite loops if the website is down
            if fail_count > 10:
                print("\nToo many failures. Stopping.")
                print("The website might be down or rate-limiting us.")
                break
    
    # ========================================
    # STEP 4: Print summary statistics
    # ========================================
    print(f"\n✓ Generation complete!")
    print(f"  Successfully generated: {success_count}")
    print(f"  Failed: {fail_count}")
    # Count total images in directory (including pre-existing ones)
    print(f"  Total images: {len(list(output_path.glob('face_*.jpg')))}")
    
    return output_path


def update_data_splits(output_dir="data/content", split_dir="data/content_splits",
                       train_ratio=0.6, val_ratio=0.2, test_ratio=0.2,
                       required_test_images=None, seed=42):
    """
    Create or update train/val/test split files for the generated faces.
    
    This function intelligently handles splits:
    - If splits don't exist: Create them from scratch
    - If splits exist: Preserve existing splits, add new images to train split
    
    This ensures reproducibility - your test/val sets remain constant even when
    you add more training data later!
    
    Args:
        output_dir: Directory containing the face images (e.g., 'data/content')
        split_dir: Directory where split files will be saved (e.g., 'data/content_splits')
        train_ratio: Fraction of data for training (default: 0.6 = 60%)
        val_ratio: Fraction of data for validation (default: 0.2 = 20%)
        test_ratio: Fraction of data for testing (default: 0.2 = 20%)
        required_test_images: List of images that MUST be in test set 
                             (e.g., ['face_00010.jpg', 'face_00066.jpg'])
        seed: Random seed for reproducibility (default: 42)
    
    Returns:
        Dictionary with keys 'train', 'val', 'test' containing lists of image filenames
    
    Example:
        # After generating 200 faces, split them 60/20/20 with specific test images
        splits = update_data_splits(
            output_dir='data/content',
            required_test_images=['face_00010.jpg', 'face_00066.jpg']
        )
    """
    print("\n" + "="*50)
    print("Updating Train/Val/Test Splits")
    print("="*50)
    
    # ========================================
    # STEP 1: Get all face images
    # ========================================
    output_path = Path(output_dir)
    all_images = sorted(output_path.glob("face_*.jpg"))
    
    if not all_images:
        print(f"❌ No face images found in {output_dir}")
        return None
    
    # Convert Path objects to just filenames (e.g., "face_00123.jpg")
    all_filenames = [img.name for img in all_images]
    total_count = len(all_filenames)
    
    print(f"\nFound {total_count} images in {output_dir}")
    
    # ========================================
    # STEP 2: Check if splits already exist
    # ========================================
    split_path = Path(split_dir)
    train_file = split_path / "train.txt"
    val_file = split_path / "val.txt"
    test_file = split_path / "test.txt"
    
    splits_exist = train_file.exists() and val_file.exists() and test_file.exists()
    
    if splits_exist:
        # Load existing splits
        print("\n✓ Found existing splits - preserving them!")
        
        with open(train_file, 'r') as f:
            train_images = [line.strip() for line in f if line.strip()]
        with open(val_file, 'r') as f:
            val_images = [line.strip() for line in f if line.strip()]
        with open(test_file, 'r') as f:
            test_images = [line.strip() for line in f if line.strip()]
        
        # Find new images (not in any split yet)
        existing_in_splits = set(train_images + val_images + test_images)
        new_images = [img for img in all_filenames if img not in existing_in_splits]
        
        if new_images:
            print(f"  Existing: {len(existing_in_splits)} images already split")
            print(f"  New: {len(new_images)} images will be added to train split")
            # Add all new images to training set
            train_images.extend(sorted(new_images))
        else:
            print(f"  All {total_count} images already in splits - nothing to update")
            return {
                'train': train_images,
                'val': val_images,
                'test': test_images
            }
    else:
        # Create splits from scratch
        print(f"\n✓ No existing splits found - creating new splits")
        print(f"Split ratios: {train_ratio:.0%} train / {val_ratio:.0%} val / {test_ratio:.0%} test (CS230 standard)")
        
        # ========================================
        # STEP 3: Ensure required images are in test set
        # ========================================
        if required_test_images is None:
            required_test_images = []
        
        # Check that all required test images actually exist
        test_images = []
        for img in required_test_images:
            if img in all_filenames:
                test_images.append(img)
                print(f"✓ Guaranteed in test set: {img}")
            else:
                print(f"⚠️  Warning: Required test image not found: {img}")
        
        # Remove required test images from the pool of remaining images
        remaining = [img for img in all_filenames if img not in test_images]
        
        # ========================================
        # STEP 4: Shuffle and split remaining images
        # ========================================
        # Set random seed for reproducibility
        # Same seed = same split every time you run this
        random.seed(seed)
        random.shuffle(remaining)
        
        # Calculate how many images go to each split
        # Example: 200 images, 2 already in test -> 198 remaining
        # Need 40 test total -> add 38 more
        # Need 40 val -> take 40
        # Rest go to train -> 198 - 38 - 40 = 120
        test_target = int(total_count * test_ratio)  # e.g., 200 * 0.2 = 40
        val_target = int(total_count * val_ratio)    # e.g., 200 * 0.2 = 40
        
        # How many more test images do we need?
        test_needed = max(0, test_target - len(test_images))
        
        # Add more images to test set
        test_images.extend(remaining[:test_needed])
        remaining = remaining[test_needed:]
        
        # Take images for validation set
        val_images = remaining[:val_target]
        remaining = remaining[val_target:]
        
        # All remaining images go to training set
        train_images = remaining
    
    # ========================================
    # STEP 5: Verify split sizes
    # ========================================
    print(f"\nFinal split sizes:")
    print(f"  Train: {len(train_images)} ({len(train_images)/total_count:.1%})")
    print(f"  Val:   {len(val_images)} ({len(val_images)/total_count:.1%})")
    print(f"  Test:  {len(test_images)} ({len(test_images)/total_count:.1%})")
    print(f"  Total: {len(train_images) + len(val_images) + len(test_images)} (should be {total_count})")
    
    # Sanity check: make sure we didn't lose or duplicate any images
    assert len(train_images) + len(val_images) + len(test_images) == total_count
    assert len(set(train_images) & set(val_images)) == 0  # No overlap between train and val
    assert len(set(train_images) & set(test_images)) == 0  # No overlap between train and test
    assert len(set(val_images) & set(test_images)) == 0    # No overlap between val and test
    
    # ========================================
    # STEP 6: Create split directory and save files
    # ========================================
    split_path = Path(split_dir)
    split_path.mkdir(parents=True, exist_ok=True)
    
    # Save train split
    train_file = split_path / "train.txt"
    with open(train_file, 'w') as f:
        f.write('\n'.join(sorted(train_images)) + '\n')
    print(f"\n✓ Saved: {train_file}")
    
    # Save validation split
    val_file = split_path / "val.txt"
    with open(val_file, 'w') as f:
        f.write('\n'.join(sorted(val_images)) + '\n')
    print(f"✓ Saved: {val_file}")
    
    # Save test split
    test_file = split_path / "test.txt"
    with open(test_file, 'w') as f:
        f.write('\n'.join(sorted(test_images)) + '\n')
    print(f"✓ Saved: {test_file}")
    
    # ========================================
    # STEP 7: Print summary
    # ========================================
    action = "updated" if splits_exist else "created"
    print(f"\n{'='*50}")
    print(f"✓ Splits {action} successfully!")
    print(f"{'='*50}")
    print(f"Split files saved to: {split_dir}")
    
    return {
        'train': train_images,
        'val': val_images,
        'test': test_images
    }


# ========================================
# Command-Line Interface (CLI)
# ========================================
# This section only runs when the script is executed directly (not imported)
if __name__ == "__main__":
    # Set up argument parser for command-line options
    parser = argparse.ArgumentParser(
        description="Generate synthetic faces from ThisPersonDoesNotExist.com"
    )
    
    # Define command-line arguments
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='data/content',
        help='Output directory (default: data/content)'
    )
    parser.add_argument(
        '--num-images', 
        type=int, 
        default=200,
        help='TOTAL target number of images (default: 200)'
    )
    parser.add_argument(
        '--delay', 
        type=float, 
        default=1.0,
        help='Delay between requests in seconds (default: 1.0)'
    )
    parser.add_argument(
        '--no-auto-continue', 
        action='store_true',
        help='Ask for confirmation before continuing from existing images'
    )
    
    # Parse command-line arguments
    args = parser.parse_args()
    
    # ========================================
    # STEP 1: Generate synthetic faces
    # ========================================
    # Call the main function with parsed arguments
    # Note: auto_continue is the inverse of no_auto_continue
    # --no-auto-continue flag sets no_auto_continue=True, so auto_continue=False
    generate_synthetic_faces(
        args.output_dir, 
        args.num_images, 
        args.delay,
        auto_continue=not args.no_auto_continue
    )
    
    # ========================================
    # STEP 2: Automatically create/update data splits
    # ========================================
    # This is ALWAYS done (no flag needed) - it's a fundamental step!
    # If splits exist, they're preserved and only new images are added.
    # If splits don't exist, they're created with face_00010 and face_00066 in test set.
    update_data_splits(
        output_dir=args.output_dir,
        split_dir='data/content_splits',
        required_test_images=['face_00010.jpg', 'face_00066.jpg']
    )
    
    # ========================================
    # Print helpful next steps for the user
    # ========================================
    print("\n" + "="*50)
    print("✓ All Done! Dataset and Splits Ready")
    print("="*50)
    print("\nNext steps:")
    print("1. Verify dataset:")
    print(f"   ls -l {args.output_dir}/ | wc -l")
    print("   cat data/content_splits/train.txt | wc -l")
    print("   cat data/content_splits/val.txt | wc -l")
    print("   cat data/content_splits/test.txt | wc -l")
    
    print("\n2. Train with your splits:")
    print(f"   python train_model.py \\")
    print(f"       --content-dir {args.output_dir} \\")
    print(f"       --split-file data/content_splits/train.txt \\")
    print(f"       --val-split-file data/content_splits/val.txt \\")
    print(f"       --style-dir data/style \\")
    print(f"       --epochs 20")

