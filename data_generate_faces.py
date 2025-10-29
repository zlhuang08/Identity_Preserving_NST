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

USAGE EXAMPLES:
    # Basic: Generate 200 faces (default)
    python data_generate_faces.py
    
    # Custom number of images
    python data_generate_faces.py --num-images 500
    
    # Custom output directory
    python data_generate_faces.py --output-dir data/content --num-images 200
    
    # Slower rate limiting (be extra nice to server)
    python data_generate_faces.py --delay 2.0
    
    # Ask before resuming (interactive mode)
    python data_generate_faces.py --no-auto-continue

NAMING CONVENTION:
    Images are saved as: face_00000.jpg, face_00001.jpg, face_00002.jpg, ...
    Zero-padding ensures proper sorting and easy indexing.
"""

import os
import time
import urllib.request
from pathlib import Path
from tqdm import tqdm
import argparse


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
        default='data/synthetic_children',
        help='Output directory (default: data/synthetic_children)'
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
    # Print helpful next steps for the user
    # ========================================
    print("\nNext steps:")
    print("1. Verify dataset:")
    print(f"   ls -l {args.output_dir}/ | wc -l")
    print("\n2. Train with synthetic faces:")
    print(f"   python train_adain.py \\")
    print(f"       --content-dir {args.output_dir} \\")
    print(f"       --style-dir data/wikiart \\")
    print(f"       --epochs 20")

