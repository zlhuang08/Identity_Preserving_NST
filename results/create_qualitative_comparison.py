#!/usr/bin/env python3
"""
Create Qualitative Comparison Grid for CS230 Final Project

This script generates a comprehensive 4×6 grid comparing four models across four
representative content-style combinations. The grid is designed for presentation
and publication, showing the progressive improvement from baseline to all methods combined.

Grid Layout:
- Rows (4): Different content-style combinations
  - Row 1: face_00016 + durer_hare (with subtitles)
  - Row 2: face_00016 + starry_night
  - Row 3: face_00066 + durer_hare
  - Row 4: face_00066 + the_scream
  
- Columns (6): Progressive model improvements
  - Col 1: Content image
  - Col 2: Style image
  - Col 3: Baseline (AdaIN only)
  - Col 4: + Identity Loss
  - Col 5: + Eye Loss
  - Col 6: + Face-Aware

Output:
- High-resolution 4×6 grid (2400×1600 pixels at 300 DPI for print quality)
- Only first row has subtitles to keep the visualization clean
- Saved as: results/qualitative_comparison_grid.png

Author: CS230 Final Project
Date: December 2025
"""

import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Configuration
CONTENT_DIR = Path("../data/eval_content")
STYLE_DIR = Path("../data/style")
RESULTS_DIR = Path(".")
OUTPUT_PATH = Path("qualitative_comparison_grid.png")

# Grid configuration
IMG_SIZE = 400  # Size of each cell in the grid
PADDING = 20    # Padding between images
TITLE_HEIGHT = 50  # Height for subtitle text (only for row 1)
GRID_ROWS = 4
GRID_COLS = 6

# Model directories
MODEL_DIRS = {
    'baseline': 'model_progression/0_baseline',
    'identity': 'model_progression/1_identity',
    'eye': 'model_progression/2_identity_plus_eye',
    'face_aware': 'model_progression/3_all_combined'
}

# Combinations to display (row_index: (content, style))
COMBINATIONS = [
    ('face_00016.jpg', 'durer_hare.jpg'),      # Row 1 (with subtitles)
    ('face_00066.jpg', 'durer_hare.jpg'),      # Row 2
    ('face_00016.jpg', 'starry_night.jpg'),    # Row 3
    ('face_00066.jpg', 'the_scream.jpg')       # Row 4
]

# Column subtitles (only shown in row 1)
SUBTITLES = [
    "Content",
    "Style",
    "Baseline",
    "+ Identity Loss",
    "+ Eye Loss",
    "+ Face-Aware"
]


def load_and_resize_image(image_path, size=IMG_SIZE):
    """
    Load an image and resize it to a square while maintaining aspect ratio.
    
    Args:
        image_path: Path to the image file
        size: Target size for the square image
        
    Returns:
        PIL Image object resized to (size, size)
    """
    try:
        img = Image.open(image_path).convert('RGB')
        
        # Resize to square, maintaining aspect ratio and centering
        img.thumbnail((size, size), Image.Resampling.LANCZOS)
        
        # Create a white background square
        background = Image.new('RGB', (size, size), (255, 255, 255))
        
        # Calculate position to paste the image (center it)
        offset = ((size - img.width) // 2, (size - img.height) // 2)
        background.paste(img, offset)
        
        return background
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        # Return a white placeholder
        return Image.new('RGB', (size, size), (255, 255, 255))


def get_stylized_image_path(model_dir, content_name, style_name):
    """
    Construct the path to a stylized image.
    
    Args:
        model_dir: Directory containing the model outputs
        content_name: Name of content image (e.g., 'face_00016.jpg')
        style_name: Name of style image (e.g., 'durer_hare.jpg')
        
    Returns:
        Path to the stylized image
    """
    content_stem = Path(content_name).stem  # e.g., 'face_00016'
    style_stem = Path(style_name).stem      # e.g., 'durer_hare'
    
    # Pattern: {content}_{style}.jpg
    filename = f"{content_stem}_{style_stem}.jpg"
    return Path(model_dir) / filename


def add_subtitle(draw, text, x, y, width, font_size=24):
    """
    Add centered subtitle text above an image.
    
    Args:
        draw: PIL ImageDraw object
        text: Text to display
        x: Left x-coordinate of the image
        y: Top y-coordinate where text should start (above the image)
        width: Width of the image (for centering)
        font_size: Font size for the text
    """
    try:
        # Try to load a nice font
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        # Fallback to default font
        font = ImageFont.load_default()
    
    # Get text bounding box for centering
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Center the text horizontally
    text_x = x + (width - text_width) // 2
    text_y = y + 5  # Small offset from top
    
    # Draw text with a slight shadow for readability
    shadow_offset = 2
    draw.text((text_x + shadow_offset, text_y + shadow_offset), text, fill=(200, 200, 200), font=font)
    draw.text((text_x, text_y), text, fill=(0, 0, 0), font=font)


def create_qualitative_grid():
    """
    Create the 4×6 qualitative comparison grid.
    
    The grid shows 4 content-style combinations across 6 progressive model stages.
    Only the first row includes subtitles to keep the visualization clean.
    """
    print("=" * 80)
    print("           Creating Qualitative Comparison Grid (4×6)")
    print("=" * 80)
    print()
    
    # Calculate canvas size
    # First row has TITLE_HEIGHT above it, then all rows have same IMG_SIZE
    canvas_width = GRID_COLS * (IMG_SIZE + PADDING) + PADDING
    canvas_height = PADDING + TITLE_HEIGHT + GRID_ROWS * (IMG_SIZE + PADDING) + PADDING
    
    print(f"Canvas size: {canvas_width} × {canvas_height} pixels")
    print()
    
    # Create white canvas
    canvas = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    
    # Process each row
    for row_idx, (content_name, style_name) in enumerate(COMBINATIONS):
        print(f"Row {row_idx + 1}/{GRID_ROWS}: {content_name} + {style_name}")
        
        # Calculate y position for this row
        if row_idx == 0:
            # First row: padding + title height for subtitles
            y_pos = PADDING + TITLE_HEIGHT
        else:
            # Subsequent rows: first row position + row height
            y_pos = PADDING + TITLE_HEIGHT + row_idx * (IMG_SIZE + PADDING)
        
        # Column 1: Content image
        content_path = CONTENT_DIR / content_name
        content_img = load_and_resize_image(content_path, IMG_SIZE)
        x_pos = PADDING
        if row_idx == 0:
            # Add subtitle ABOVE the image
            add_subtitle(draw, SUBTITLES[0], x_pos, PADDING, IMG_SIZE)
        canvas.paste(content_img, (x_pos, y_pos))
        print(f"  ✓ Content image")
        
        # Column 2: Style image
        style_path = STYLE_DIR / style_name
        style_img = load_and_resize_image(style_path, IMG_SIZE)
        x_pos = PADDING + (IMG_SIZE + PADDING)
        if row_idx == 0:
            add_subtitle(draw, SUBTITLES[1], x_pos, PADDING, IMG_SIZE)
        canvas.paste(style_img, (x_pos, y_pos))
        print(f"  ✓ Style image")
        
        # Column 3: Baseline
        baseline_path = get_stylized_image_path(MODEL_DIRS['baseline'], content_name, style_name)
        baseline_img = load_and_resize_image(baseline_path, IMG_SIZE)
        x_pos = PADDING + 2 * (IMG_SIZE + PADDING)
        if row_idx == 0:
            add_subtitle(draw, SUBTITLES[2], x_pos, PADDING, IMG_SIZE)
        canvas.paste(baseline_img, (x_pos, y_pos))
        print(f"  ✓ Baseline")
        
        # Column 4: + Identity Loss
        identity_path = get_stylized_image_path(MODEL_DIRS['identity'], content_name, style_name)
        identity_img = load_and_resize_image(identity_path, IMG_SIZE)
        x_pos = PADDING + 3 * (IMG_SIZE + PADDING)
        if row_idx == 0:
            add_subtitle(draw, SUBTITLES[3], x_pos, PADDING, IMG_SIZE)
        canvas.paste(identity_img, (x_pos, y_pos))
        print(f"  ✓ + Identity Loss")
        
        # Column 5: + Eye Loss
        eye_path = get_stylized_image_path(MODEL_DIRS['eye'], content_name, style_name)
        eye_img = load_and_resize_image(eye_path, IMG_SIZE)
        x_pos = PADDING + 4 * (IMG_SIZE + PADDING)
        if row_idx == 0:
            add_subtitle(draw, SUBTITLES[4], x_pos, PADDING, IMG_SIZE)
        canvas.paste(eye_img, (x_pos, y_pos))
        print(f"  ✓ + Eye Loss")
        
        # Column 6: + Face-Aware
        face_aware_path = get_stylized_image_path(MODEL_DIRS['face_aware'], content_name, style_name)
        face_aware_img = load_and_resize_image(face_aware_path, IMG_SIZE)
        x_pos = PADDING + 5 * (IMG_SIZE + PADDING)
        if row_idx == 0:
            add_subtitle(draw, SUBTITLES[5], x_pos, PADDING, IMG_SIZE)
        canvas.paste(face_aware_img, (x_pos, y_pos))
        print(f"  ✓ + Face-Aware")
        print()
    
    # Save the grid
    output_path = RESULTS_DIR / OUTPUT_PATH
    canvas.save(output_path, quality=95, dpi=(300, 300))
    
    print("=" * 80)
    print(f"✅ Qualitative comparison grid saved: {output_path}")
    print(f"   Size: {canvas_width} × {canvas_height} pixels")
    print(f"   Resolution: 300 DPI (print quality)")
    print("=" * 80)
    print()
    print("Grid layout:")
    print("  • 4 rows: Different content-style combinations")
    print("  • 6 columns: Progressive model improvements")
    print("  • Only Row 1 has subtitles for clean presentation")
    print()


def main():
    """Main entry point for the script."""
    # Change to the results directory
    os.chdir(Path(__file__).parent)
    
    print()
    print("CS230 Final Project - Qualitative Comparison Grid Generator")
    print()
    
    # Verify that model output directories exist
    print("Verifying model output directories...")
    all_exist = True
    for name, path in MODEL_DIRS.items():
        full_path = Path(path)
        if not full_path.exists():
            print(f"  ✗ {name}: {full_path} (NOT FOUND)")
            all_exist = False
        else:
            num_files = len(list(full_path.glob("*.jpg")))
            print(f"  ✓ {name}: {full_path} ({num_files} images)")
    
    if not all_exist:
        print()
        print("ERROR: Some model output directories are missing!")
        print("Please run inference first using eval_inference.py")
        return 1
    
    print()
    
    # Create the grid
    create_qualitative_grid()
    
    print("Done!")
    return 0


if __name__ == "__main__":
    exit(main())

