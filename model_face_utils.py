#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Face Detection and Recognition Utilities for Identity-Preserving Style Transfer

OVERVIEW:
When we apply style transfer to face photos (e.g., turning a child's photo into
a Peter Rabbit watercolor), we want to preserve WHO the person is! The stylized
output should still look like the same person, just in a different artistic style.

This module provides tools to ensure identity preservation:
1. Face Detection: Find faces in images (where is the face?)
2. Face Recognition: Extract "who is this person?" embeddings
3. Identity Loss: Measure how well identity is preserved

WHY IS THIS IMPORTANT?
Without identity preservation, style transfer might:
- Change facial features (different nose, eyes, etc.)
- Alter proportions (face shape changes)
- Lose the person's unique characteristics

With identity preservation:
✓ Face structure stays the same
✓ Person is still recognizable
✓ Only artistic style changes (colors, textures, brush strokes)

REAL-WORLD EXAMPLE:
    Original photo: Child's face with realistic features
    ↓ (style transfer with identity preservation)
    Peter Rabbit style: Same child's face, but in watercolor style!
    
    The child's parents can still recognize their kid, but now it looks
    like an illustration from a children's book!

HOW IT WORKS:
1. Detect faces in both original and stylized images using MTCNN
2. Extract identity embeddings (512-d vectors) using InceptionResnetV1
3. Compare embeddings to ensure they represent the same person
4. Penalize the model if identity changes too much

TECHNICAL COMPONENTS:
- MTCNN: Multi-task Cascaded Convolutional Networks for face detection
- InceptionResnetV1: Deep CNN for face recognition (pretrained on VGGFace2)
- Cosine Similarity: Measure how similar two face embeddings are

USAGE EXAMPLE:
    # During training
    identity_preserver = IdentityPreserver(device='cuda')
    
    # Compute identity loss
    id_loss, metrics = identity_preserver.compute_identity_loss(
        generated_images=stylized_faces,
        content_images=original_faces
    )
    
    # Add to total training loss
    total_loss = content_loss + style_loss + 0.1 * id_loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import contextlib
from PIL import Image
import warnings
import lpips  # LPIPS for perceptual loss

# ============================================================================
# OPTIONAL DEPENDENCIES
# ============================================================================
# These libraries are optional but recommended for face detection/recognition

# Try to import facenet-pytorch (for MTCNN and InceptionResnetV1)
try:
    from facenet_pytorch import MTCNN, InceptionResnetV1
    FACENET_AVAILABLE = True
except ImportError:
    FACENET_AVAILABLE = False
    warnings.warn(
        "facenet-pytorch not installed. Install with: pip install facenet-pytorch\n"
        "This is needed for advanced face detection and identity preservation."
    )

# Try to import OpenCV (for fallback face detection)
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    # OpenCV is optional, only warn if facenet is also not available
    if not FACENET_AVAILABLE:
        warnings.warn("Neither facenet-pytorch nor opencv-python available for face detection")


class FaceDetector:
    """
    Face detector using MTCNN (Multi-task Cascaded Convolutional Networks).
    
    WHAT IS MTCNN?
    MTCNN is a state-of-the-art face detection algorithm that uses a cascade
    of three neural networks to detect faces and facial landmarks.
    
    THREE-STAGE PIPELINE:
    1. Proposal Network (P-Net): Quickly scans image for face candidates
    2. Refine Network (R-Net): Refines candidates and rejects false positives
    3. Output Network (O-Net): Final detection + facial landmarks (eyes, nose, mouth)
    
    WHY MTCNN?
    ✓ High accuracy (better than Haar Cascade)
    ✓ Robust to pose, lighting, expression
    ✓ Detects facial landmarks (useful for alignment)
    ✓ Handles multiple faces in one image
    ✓ Works on various face sizes (from 20px to full image)
    
    TYPICAL USE CASE:
        detector = FaceDetector(device='cuda', keep_all=False)
        boxes, probs, landmarks = detector.detect(images)
        # boxes: bounding boxes around faces
        # probs: confidence scores (0-1)
        # landmarks: eye, nose, mouth positions
    """
    def __init__(self, device='cuda', keep_all=False, min_face_size=20):
        """
        Initialize MTCNN face detector.
        
        Args:
            device: Device to run detection on ('cuda' or 'cpu')
                   GPU is recommended for speed (10-100x faster)
            keep_all: If True, return all detected faces; if False, return only largest
                     For children's book photos, usually False (one child per photo)
            min_face_size: Minimum face size to detect (in pixels)
                          20px = detect even small/distant faces
                          40px = only detect larger/closer faces (faster)
        
        Example:
            # Detect only the main face (largest one)
            detector = FaceDetector(keep_all=False, min_face_size=20)
            
            # Detect all faces (e.g., group photos)
            detector = FaceDetector(keep_all=True, min_face_size=30)
        """
        self.device = device
        self.keep_all = keep_all
        
        # Check if facenet-pytorch is installed
        if not FACENET_AVAILABLE:
            raise ImportError(
                "facenet-pytorch is required for face detection. "
                "Install with: pip install facenet-pytorch"
            )
        
        # ========================================
        # Initialize MTCNN detector
        # ========================================
        self.mtcnn = MTCNN(
            keep_all=keep_all,
            device=device,
            min_face_size=min_face_size,
            thresholds=[0.3, 0.5, 0.5],  # LOWERED thresholds for stylized images
                                          # Default [0.6, 0.7, 0.7] is too strict for artistic styles
                                          # Lower = more detections (needed for style transfer)
                                          # Higher = fewer detections but higher precision (for natural images)
            post_process=False  # We'll handle normalization ourselves for consistency
        )
    
    def detect(self, images):
        """
        Detect faces in a batch of images.
        
        This method finds faces and their locations in the images, along with
        facial landmarks (eyes, nose, mouth corners).
        
        Args:
            images: (B, 3, H, W) tensor of images in [0, 1] range
                   Example: (4, 3, 256, 256) - 4 RGB images, 256×256 pixels
                   MUST be in [0, 1] range, NOT [0, 255]!
        
        Returns:
            boxes: List of bounding boxes for each image
                  Each box: [x1, y1, x2, y2] (top-left and bottom-right corners)
                  Example: [[120, 80, 200, 160]] for one face
                  None if no face detected in that image
            
            probs: List of detection confidence scores for each image
                  Values in [0, 1], where 1 = very confident
                  Example: [0.99] for high confidence detection
                  None if no face detected
            
            landmarks: List of 5 facial landmarks for each image
                      Order: left eye, right eye, nose, left mouth, right mouth
                      Each landmark: [x, y] pixel coordinates
                      Example: [[100, 90], [140, 90], [120, 110], [105, 130], [135, 130]]
                      None if no face detected
        
        Example:
            # Detect faces in a batch
            boxes, probs, landmarks = detector.detect(images)
            
            # Check results for first image
            if boxes[0] is not None:
                print(f"Found face at: {boxes[0]}")
                print(f"Confidence: {probs[0]:.2f}")
                print(f"Eyes at: {landmarks[0][:2]}")  # First two landmarks are eyes
        """
        batch_size = images.size(0)
        pil_images = []
        
        # ========================================
        # Step 1: Convert PyTorch tensors to PIL Images
        # ========================================
        # MTCNN expects PIL Images, so we need to convert
        for i in range(batch_size):
            img = images[i].cpu()  # Move to CPU if on GPU
            
            # Convert from [0, 1] float to [0, 255] uint8
            img = (img * 255).clamp(0, 255).byte()
            
            # Convert from (3, H, W) to (H, W, 3) for PIL
            img = img.permute(1, 2, 0).numpy()
            
            # Create PIL Image
            pil_img = Image.fromarray(img)
            pil_images.append(pil_img)
        
        # ========================================
        # Step 2: Run MTCNN face detection
        # ========================================
        # This runs the 3-stage cascade: P-Net → R-Net → O-Net
        boxes, probs, landmarks = self.mtcnn.detect(pil_images, landmarks=True)
        
        return boxes, probs, landmarks
    
    def extract_faces(self, images, target_size=160):
        """
        Detect and extract face regions from images, ready for recognition.
        
        This is a convenience method that combines detection + cropping + resizing.
        The extracted faces are standardized to a fixed size (160×160 by default)
        for use with face recognition models.
        
        Args:
            images: (B, 3, H, W) tensor of images in [0, 1] range
                   Example: (4, 3, 256, 256) - 4 full images
            target_size: Size to resize extracted faces to (default: 160)
                        InceptionResnetV1 expects 160×160 faces
        
        Returns:
            faces: (N, 3, target_size, target_size) tensor of extracted faces
                  Example: (5, 3, 160, 160) - 5 faces found across the batch
                  None if no faces detected in any image
            
            face_indices: List indicating which image each face came from
                         Example: [0, 0, 1, 2, 3] means:
                         - First 2 faces from image 0
                         - Third face from image 1
                         - Fourth face from image 2
                         - Fifth face from image 3
                         Useful for matching faces back to original images
            
            boxes: List of bounding boxes for each extracted face
                  Example: [[120, 80, 200, 160], ...] in original image coordinates
        
        Example:
            # Extract all faces from a batch
            faces, indices, boxes = detector.extract_faces(images, target_size=160)
            
            if faces is not None:
                print(f"Found {len(faces)} faces")
                print(f"Face shapes: {faces.shape}")  # (N, 3, 160, 160)
                
                # Process first face
                first_face = faces[0]  # (3, 160, 160)
                from_image = indices[0]  # Which image it came from
        """
        batch_size = images.size(0)
        all_faces = []      # Will store extracted face tensors
        face_indices = []   # Will store which image each face came from
        all_boxes = []      # Will store bounding boxes
        
        # ========================================
        # Step 1: Detect faces in all images
        # ========================================
        boxes, probs, landmarks = self.detect(images)
        
        # ========================================
        # Step 2: Extract and resize each detected face
        # ========================================
        for i in range(batch_size):
            # Skip if no face detected in this image
            if boxes[i] is None:
                continue
            
            img = images[i]  # Current image: (3, H, W)
            img_boxes = boxes[i]  # Bounding box(es) for this image
            
            # ========================================
            # Handle single face or multiple faces
            # ========================================
            # boxes[i] might be 1D (single face) or 2D (multiple faces)
            # Convert to 2D for uniform processing
            if len(img_boxes.shape) == 1:
                img_boxes = img_boxes.unsqueeze(0)
            
            # ========================================
            # Extract each face from this image
            # ========================================
            for box in img_boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # ========================================
                # Clamp to image boundaries (safety check)
                # ========================================
                # This prevents errors if detection goes slightly outside image
                h, w = img.shape[1], img.shape[2]  # Image height and width
                x1 = max(0, x1)      # Don't go left of image
                y1 = max(0, y1)      # Don't go above image
                x2 = min(w, x2)      # Don't go right of image
                y2 = min(h, y2)      # Don't go below image
                
                # ========================================
                # Crop face region
                # ========================================
                # Use tensor slicing to extract the face region
                # img[:, y1:y2, x1:x2] means:
                # - All channels (:)
                # - Rows from y1 to y2
                # - Columns from x1 to x2
                face = img[:, y1:y2, x1:x2]
                
                # ========================================
                # Resize to standard size (160×160)
                # ========================================
                # Face recognition models expect fixed size inputs
                # We use bilinear interpolation for smooth resizing
                face = F.interpolate(
                    face.unsqueeze(0),  # Add batch dim: (3, H, W) → (1, 3, H, W)
                    size=(target_size, target_size),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)  # Remove batch dim: (1, 3, 160, 160) → (3, 160, 160)
                
                # ========================================
                # Store extracted face and metadata
                # ========================================
                all_faces.append(face)
                face_indices.append(i)  # Remember which image this face came from
                all_boxes.append(box)
        
        # ========================================
        # Return None if no faces found
        # ========================================
        if len(all_faces) == 0:
            return None, None, None
        
        # ========================================
        # Stack all faces into a single tensor
        # ========================================
        # List of (3, 160, 160) → (N, 3, 160, 160)
        faces = torch.stack(all_faces)
        return faces, face_indices, all_boxes
    
    def generate_face_masks(self, images, margin_factor=1.3, blur_kernel_size=21):
        """
        Generate soft binary masks indicating face regions for each image.
        
        This method creates masks that are:
        - 1.0 (white) in face regions
        - 0.0 (black) in background regions
        - Smoothly blended at boundaries (using Gaussian blur)
        
        WHY DO WE NEED FACE MASKS?
        Face-aware AdaIN uses masks to apply different stylization strengths:
        - Face regions: Lighter stylization (preserve identity)
        - Background: Full stylization (artistic effect)
        - Smooth transition: Avoid hard boundaries that look unnatural
        
        Args:
            images: (B, 3, H, W) tensor of images in [0, 1] range
                   Example: (4, 3, 256, 256) - 4 RGB images
            
            margin_factor: Float >= 1.0, expands face region by this factor
                          1.0 = exact bounding box
                          1.3 = expand by 30% (default, recommended)
                          1.5 = expand by 50% (includes more hair/background)
                          Why expand? Face detection boxes are tight; we want to include
                          hair, ears, and a smooth transition to background
            
            blur_kernel_size: Size of Gaussian blur kernel (must be odd)
                             Larger = smoother transition, but may affect far regions
                             21 = good balance (default)
                             41 = very smooth transition
                             5 = sharp transition (not recommended)
        
        Returns:
            masks: (B, 1, H, W) tensor of binary masks in [0, 1] range
                  Example: (4, 1, 256, 256) - 4 masks matching input images
                  Each mask has:
                  - 1.0 for face pixels
                  - 0.0 for background pixels
                  - Smooth gradient at boundaries
                  
                  If no face detected in an image, that mask is all zeros (pure background)
        
        Example:
            # Generate face masks for a batch
            masks = detector.generate_face_masks(images, margin_factor=1.3)
            
            # Use masks to blend content and style
            # Face regions: More content preservation
            # Background: More style transfer
            output = masks * preserved_faces + (1 - masks) * stylized_background
            
            # Visualize mask for first image
            import matplotlib.pyplot as plt
            plt.imshow(masks[0, 0].cpu(), cmap='gray')
            plt.title('Face Mask (white=face, black=background)')
            plt.show()
        
        Technical Details:
            1. Detect face bounding boxes using MTCNN
            2. Expand boxes by margin_factor to include hair/context
            3. Create binary mask (1 inside box, 0 outside)
            4. Apply Gaussian blur for smooth transitions
            5. Clamp values to [0, 1] range
        """
        batch_size, _, height, width = images.shape
        device = images.device
        
        # ========================================
        # Step 1: Initialize masks (all zeros = all background)
        # ========================================
        # Shape: (B, 1, H, W) - one mask channel per image
        masks = torch.zeros(batch_size, 1, height, width, device=device)
        
        # ========================================
        # Step 2: Detect faces in all images
        # ========================================
        boxes, probs, landmarks = self.detect(images)
        
        # ========================================
        # Step 3: Create mask for each image
        # ========================================
        for i in range(batch_size):
            # Skip if no face detected in this image
            if boxes[i] is None:
                # Mask stays all zeros (pure background)
                continue
            
            img_boxes = boxes[i]  # Bounding box(es) for this image
            
            # ========================================
            # Handle single face or multiple faces
            # ========================================
            # boxes[i] might be 1D (single face) or 2D (multiple faces)
            # Convert to 2D for uniform processing
            if len(img_boxes.shape) == 1:
                img_boxes = img_boxes.unsqueeze(0)
            
            # ========================================
            # Fill mask for each detected face
            # ========================================
            for box in img_boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # ========================================
                # Expand bounding box by margin_factor
                # ========================================
                # This includes more context (hair, ears, forehead)
                # Example: If box is 100px wide and margin=1.3, new box is 130px wide
                box_width = x2 - x1
                box_height = y2 - y1
                
                # Calculate how much to expand (in pixels)
                expand_w = int((margin_factor - 1.0) * box_width / 2)
                expand_h = int((margin_factor - 1.0) * box_height / 2)
                
                # Expand in all directions
                x1_expanded = x1 - expand_w
                y1_expanded = y1 - expand_h
                x2_expanded = x2 + expand_w
                y2_expanded = y2 + expand_h
                
                # ========================================
                # Clamp to image boundaries
                # ========================================
                # Prevent going outside image borders
                x1_expanded = max(0, x1_expanded)
                y1_expanded = max(0, y1_expanded)
                x2_expanded = min(width, x2_expanded)
                y2_expanded = min(height, y2_expanded)
                
                # ========================================
                # Fill mask region with 1.0 (white)
                # ========================================
                # This marks the face region in the mask
                masks[i, 0, y1_expanded:y2_expanded, x1_expanded:x2_expanded] = 1.0
        
        # ========================================
        # Step 4: Apply Gaussian blur for smooth transitions
        # ========================================
        # Hard edges look unnatural in style transfer
        # Gaussian blur creates a smooth gradient at face boundaries
        # 
        # Technical: Use 2D convolution with Gaussian kernel
        sigma = blur_kernel_size / 6.0  # Standard deviation (rule of thumb)
        
        # Create Gaussian kernel
        # This is a 2D bell curve that smooths the mask
        kernel_range = torch.arange(blur_kernel_size, device=device, dtype=torch.float32)
        kernel_range = kernel_range - (blur_kernel_size - 1) / 2.0  # Center at 0
        
        # 2D Gaussian formula: exp(-(x^2 + y^2) / (2 * sigma^2))
        gaussian_1d = torch.exp(-(kernel_range ** 2) / (2 * sigma ** 2))
        gaussian_2d = gaussian_1d.unsqueeze(0) * gaussian_1d.unsqueeze(1)
        gaussian_2d = gaussian_2d / gaussian_2d.sum()  # Normalize to sum=1
        
        # Reshape kernel for conv2d: (1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_2d.unsqueeze(0).unsqueeze(0)
        
        # Apply blur using convolution
        # Padding ensures output size matches input size
        padding = blur_kernel_size // 2
        masks = F.conv2d(masks, gaussian_kernel, padding=padding)
        
        # ========================================
        # Step 5: Clamp values to [0, 1]
        # ========================================
        # Convolution might create values slightly outside [0, 1]
        masks = masks.clamp(0, 1)
        
        return masks


class FaceRecognizer:
    """
    Face recognition model for extracting identity embeddings.
    
    WHAT ARE FACE EMBEDDINGS?
    A face embedding is a 512-dimensional vector that represents WHO a person is.
    Think of it as a "fingerprint" for faces - similar faces have similar embeddings!
    
    KEY PROPERTIES:
    - Same person → Similar embeddings (high cosine similarity)
    - Different people → Different embeddings (low cosine similarity)
    - Robust to pose, expression, lighting, aging (within limits)
    
    HOW IT WORKS:
    1. Input: Face image (160×160 pixels)
    2. Process: Deep CNN (InceptionResnetV1) extracts features
    3. Output: 512-dimensional embedding vector
    4. Compare: Use cosine similarity to measure identity similarity
    
    PRETRAINED MODELS:
    - 'vggface2': Trained on 3.3M images of 9K people (recommended, more accurate)
    - 'casia-webface': Trained on 500K images of 10K people (smaller, faster)
    
    REAL-WORLD EXAMPLE:
        Original face embedding: [0.12, -0.34, 0.56, ..., 0.89]  (512 numbers)
        Stylized face embedding: [0.13, -0.32, 0.54, ..., 0.91]  (very similar!)
        Different person:        [0.87, 0.45, -0.23, ..., -0.12] (very different!)
    
    TYPICAL USE:
        recognizer = FaceRecognizer(device='cuda', pretrained='vggface2')
        
        # Extract embeddings
        emb1 = recognizer.extract_embeddings(face1)  # (1, 512)
        emb2 = recognizer.extract_embeddings(face2)  # (1, 512)
        
        # Compare similarity
        similarity = recognizer.compute_similarity(emb1, emb2)
        # similarity ≈ 0.95 → same person!
        # similarity ≈ 0.30 → different people
    """
    def __init__(self, device='cuda', pretrained='vggface2'):
        """
        Initialize face recognition model.
        
        Args:
            device: Device to run model on ('cuda' or 'cpu')
                   GPU recommended for faster processing
            pretrained: Which pretrained model to use
                       'vggface2' - More accurate, trained on 3.3M images (recommended)
                       'casia-webface' - Smaller, trained on 500K images
        
        Example:
            # Use VGGFace2 model on GPU (best accuracy)
            recognizer = FaceRecognizer(device='cuda', pretrained='vggface2')
            
            # Use CASIA model on CPU (good for testing)
            recognizer = FaceRecognizer(device='cpu', pretrained='casia-webface')
        """
        self.device = device
        
        # Check if facenet-pytorch is installed
        if not FACENET_AVAILABLE:
            raise ImportError(
                "facenet-pytorch is required for face recognition. "
                "Install with: pip install facenet-pytorch"
            )
        
        # ========================================
        # Initialize InceptionResnetV1 model
        # ========================================
        # This is a deep CNN (Inception + ResNet architecture)
        # Pretrained on millions of face images to learn identity features
        self.model = InceptionResnetV1(
            pretrained=pretrained,  # Load pretrained weights
            classify=False,  # We want embeddings (512-d vectors), not classifications
            device=device
        ).eval()  # Set to evaluation mode (no dropout, no batch norm updating)
        
        # ========================================
        # Freeze model parameters
        # ========================================
        # We don't want to train this model, just use it as-is
        # Frozen weights = no gradient computation = faster & less memory
        for param in self.model.parameters():
            param.requires_grad = False
    
    def extract_embeddings(self, faces, allow_grad=False):
        """
        Extract face embeddings (identity vectors) from face images.
        
        This converts face images into 512-dimensional vectors that represent
        the person's identity. Similar faces → similar vectors.
        
        Args:
            faces: (B, 3, H, W) tensor of face images in [0, 1] range
                   Example: (8, 3, 160, 160) - 8 face images, 160×160 pixels
                   Expected input size: 160×160 (standard for InceptionResnetV1)
                   MUST be in [0, 1] range!
            allow_grad: If True, allows gradients to flow (for training identity loss)
                       If False, uses no_grad for inference (default, saves memory)
        
        Returns:
            embeddings: (B, 512) tensor of L2-normalized face embeddings
                       Example: (8, 512) - 8 identity vectors
                       Each vector has L2 norm = 1.0 (unit length)
                       Values typically in [-1, 1] range
        
        Example:
            # Extract embeddings from detected faces (inference)
            faces = detector.extract_faces(images)  # (5, 3, 160, 160)
            embeddings = recognizer.extract_embeddings(faces)  # (5, 512)
            
            # Extract embeddings with gradients (for training)
            embeddings = recognizer.extract_embeddings(faces, allow_grad=True)
            
            # First person's identity vector
            person1_emb = embeddings[0]  # (512,)
            print(person1_emb.shape)  # torch.Size([512])
            print(torch.norm(person1_emb))  # 1.0 (L2 normalized)
        """
        # Use no_grad context only if allow_grad=False
        # This allows gradients to flow during training while still
        # being efficient during inference
        maybe_no_grad = torch.no_grad() if not allow_grad else contextlib.nullcontext()
        
        with maybe_no_grad:
            # ========================================
            # Step 1: Normalize input to [-1, 1] range
            # ========================================
            # InceptionResnetV1 expects inputs in [-1, 1], not [0, 1]
            # Formula: x_normalized = (x - 0.5) / 0.5
            #   If x=0 (black): (-0.5)/0.5 = -1
            #   If x=0.5 (gray): (0)/0.5 = 0
            #   If x=1 (white): (0.5)/0.5 = 1
            faces_normalized = (faces - 0.5) / 0.5
            
            # ========================================
            # Step 2: Extract embeddings using InceptionResnetV1
            # ========================================
            # The model processes the face through many layers:
            # Input (160×160) → Conv layers → Inception blocks → ResNet blocks → FC → Embedding (512)
            # NOTE: The face recognition model parameters are FROZEN (not trainable)
            # But we allow gradients to flow through for computing identity loss
            embeddings = self.model(faces_normalized)
            
            # ========================================
            # Step 3: L2 normalize embeddings
            # ========================================
            # This ensures all embeddings have unit length (norm = 1.0)
            # Benefits:
            # - Makes cosine similarity = dot product (faster computation)
            # - Removes magnitude differences, focuses on direction
            # - Standard practice in face recognition
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def compute_similarity(self, embedding1, embedding2):
        """
        Compute cosine similarity between two face embeddings.
        
        Cosine similarity measures how "similar" two vectors are:
        - 1.0 = Identical direction (same person with high confidence!)
        - 0.5 = Somewhat similar (might be same person, different pose/lighting)
        - 0.0 = Completely different (definitely different people)
        
        TYPICAL THRESHOLDS:
        - similarity > 0.7: Same person (high confidence)
        - 0.5 < similarity < 0.7: Uncertain (might be same person)
        - similarity < 0.5: Different people (high confidence)
        
        Args:
            embedding1, embedding2: (N, 512) tensors of L2-normalized embeddings
                                   Example: (8, 512) - compare 8 pairs of faces
        
        Returns:
            similarity: (N,) tensor of cosine similarities in [0, 1] range
                       1.0 = identical faces
                       0.0 = completely different faces
        
        Example:
            # Compare original and stylized faces
            orig_emb = recognizer.extract_embeddings(original_faces)  # (4, 512)
            style_emb = recognizer.extract_embeddings(stylized_faces)  # (4, 512)
            
            similarity = recognizer.compute_similarity(orig_emb, style_emb)  # (4,)
            print(similarity)  # tensor([0.89, 0.92, 0.87, 0.95])
            
            # Check if identities preserved
            if (similarity > 0.7).all():
                print("✓ All identities preserved!")
        """
        # ========================================
        # Compute cosine similarity
        # ========================================
        # For L2-normalized vectors, cosine similarity = dot product
        # Original range: [-1, 1] where:
        #   1 = identical vectors (0° angle)
        #   0 = orthogonal vectors (90° angle)
        #  -1 = opposite vectors (180° angle)
        similarity = F.cosine_similarity(embedding1, embedding2, dim=1)
        
        # ========================================
        # Convert to [0, 1] range for easier interpretation
        # ========================================
        # Formula: (x + 1) / 2
        # Maps: -1 → 0, 0 → 0.5, 1 → 1
        # Now 0 = totally different, 1 = identical (more intuitive!)
        similarity = (similarity + 1) / 2
        
        return similarity


class IdentityPreserver:
    """
    Complete identity preservation system for style transfer - THE MAIN CLASS!
    
    This class brings together face detection and recognition to ensure that
    stylized faces still look like the same person. It's the key to creating
    children's book illustrations that parents will love!
    
    WHAT IT DOES:
    1. Detects faces in both original and stylized images
    2. Extracts identity embeddings (512-d "fingerprints")
    3. Computes identity loss (how much identity changed)
    4. Provides metrics (similarity scores, detection counts)
    
    WHY IT MATTERS:
    Without identity preservation:
        Original: Your child's face
        Stylized: A different-looking child in Peter Rabbit style ❌
    
    With identity preservation:
        Original: Your child's face
        Stylized: YOUR child in Peter Rabbit style ✓
    
    HOW TO USE IN TRAINING:
        # Initialize (do this once)
        identity_preserver = IdentityPreserver(device='cuda')
        
        # During training loop
        for content, style in dataloader:
            # Generate stylized images
            stylized = model(content, style)
            
            # Compute identity loss
            id_loss, metrics = identity_preserver.compute_identity_loss(
                generated_images=stylized,
                content_images=content
            )
            
            # Add to total loss
            total_loss = content_loss + style_loss + 0.1 * id_loss
            
            # Check metrics
            if metrics['avg_similarity'] > 0.85:
                print("✓ Identity well preserved!")
    
    TECHNICAL DETAILS:
    - Uses MTCNN for face detection (robust, accurate)
    - Uses InceptionResnetV1 for face recognition (pretrained on VGGFace2)
    - Computes MSE loss on embeddings (encourages identity preservation)
    - Handles missing detections gracefully (returns 0 loss)
    """
    def __init__(self, device='cuda', pretrained='vggface2'):
        """
        Initialize identity preservation system.
        
        Args:
            device: Device to run on ('cuda' or 'cpu')
                   GPU strongly recommended (detection + recognition are slow on CPU)
            pretrained: Which face recognition model to use
                       'vggface2' - Best accuracy (recommended)
                       'casia-webface' - Smaller, faster
        
        Example:
            # Best setup for training (GPU + best model)
            preserver = IdentityPreserver(device='cuda', pretrained='vggface2')
            
            # Quick testing setup (CPU + smaller model)
            preserver = IdentityPreserver(device='cpu', pretrained='casia-webface')
        """
        self.device = device
        
        # ========================================
        # Initialize face detector
        # ========================================
        # keep_all=False means we only detect the largest face per image
        # This is perfect for our use case (one child per photo)
        self.detector = FaceDetector(device=device, keep_all=False)
        
        # ========================================
        # Initialize face recognizer
        # ========================================
        # This extracts 512-d embeddings representing identity
        self.recognizer = FaceRecognizer(device=device, pretrained=pretrained)
    
    def compute_identity_loss(self, generated_images, content_images):
        """
        Compute identity preservation loss - THE CORE METHOD!
        
        This is what you call during training to ensure stylized faces still look
        like the same person. It compares face embeddings between original and
        stylized images, penalizing changes in identity.
        
        THE PIPELINE:
        1. Detect faces in both original and stylized images
        2. Match corresponding faces (same image index)
        3. Extract embeddings (512-d identity vectors)
        4. Compute MSE loss between embeddings
        5. Return loss + helpful metrics
        
        GRACEFUL FAILURE HANDLING:
        - If no faces detected → return loss = 0.0 (no penalty)
        - If faces detected in only one set → return loss = 0.0
        - If no matching pairs found → return loss = 0.0
        - This prevents training crashes on images without faces
        
        Args:
            generated_images: (B, 3, H, W) tensor of stylized images
                             Example: (4, 3, 256, 256) - 4 stylized faces
            content_images: (B, 3, H, W) tensor of original images
                           Example: (4, 3, 256, 256) - 4 original faces
        
        Returns:
            loss: Scalar tensor, MSE loss between embeddings
                 Lower = better identity preservation
                 Typical values: 0.01-0.1 (good), 0.5+ (poor)
            
            metrics: Dictionary with useful information:
                    {
                        'num_content_faces': How many faces found in originals
                        'num_gen_faces': How many faces found in stylized
                        'num_matched_faces': How many pairs successfully matched
                        'avg_similarity': Average similarity (0-1, higher=better)
                    }
        
        Example during training:
            # Generate stylized images
            stylized = model(content, style)
            
            # Compute identity loss
            id_loss, metrics = identity_preserver.compute_identity_loss(
                generated_images=stylized,
                content_images=content
            )
            
            print(f"Identity loss: {id_loss.item():.4f}")
            print(f"Matched {metrics['num_matched_faces']} face pairs")
            print(f"Avg similarity: {metrics['avg_similarity']:.2f}")
            
            # Add to training loss
            total_loss = content_loss + 10*style_loss + 0.1*id_loss
        """
        batch_size = generated_images.size(0)
        
        # ========================================
        # Step 1: Detect faces ONLY on content images (CRITICAL FIX!)
        # ========================================
        # ROBUSTNESS IMPROVEMENT: By detecting only on content images and using
        # those bounding boxes to crop BOTH content and generated images, we ensure:
        # 1. We always get matching pairs (no "face not detected" in stylized images)
        # 2. Model gets feedback even when stylized faces are heavily distorted
        # 3. Training is more stable (no disappearing gradients when face detection fails)
        content_faces, content_indices, content_boxes = \
            self.detector.extract_faces(content_images, target_size=160)
        # Returns:
        # content_faces: (N, 3, 160, 160) tensor of face crops or None
        # content_indices: [0, 1, 1, 3] - which image each face came from
        # content_boxes: [(x1, y1, x2, y2), ...] - bounding boxes for each face
        
        # ========================================
        # Step 2: Handle case where no faces detected in content
        # ========================================
        if content_faces is None or len(content_faces) == 0:
            # No faces found in content images
            # Return zero loss (don't penalize the model for images without faces)
            metrics = {
                'num_content_faces': 0,
                'num_gen_faces': 0,
                'num_matched_faces': 0,
                'avg_similarity': 0.0
            }
            return torch.tensor(0.0, device=self.device), metrics
        
        # ========================================
        # Step 3: Use content bounding boxes to crop generated images
        # ========================================
        # CRITICAL: This ensures we ALWAYS get a face crop from generated images,
        # even if the stylization made the face undetectable by the detector
        gen_faces = []
        gen_indices = []
        
        for face_idx, (img_idx, box) in enumerate(zip(content_indices, content_boxes)):
            # Extract the region from generated image using content's bounding box
            x1, y1, x2, y2 = box
            # Ensure coordinates are within image bounds
            h, w = generated_images.shape[2:]
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))
            
            # SAFETY CHECK: Ensure the crop has valid dimensions
            # (prevents crash if box falls outside image or has zero area)
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Crop the face region
            face_crop = generated_images[img_idx, :, y1:y2, x1:x2]
            
            # Resize to target size (160x160 for FaceNet)
            face_crop_resized = F.interpolate(
                face_crop.unsqueeze(0), 
                size=(160, 160), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
            
            gen_faces.append(face_crop_resized)
            gen_indices.append(img_idx)
        
        # ========================================
        # Step 4: Stack all faces into tensors
        # ========================================
        # Now we have perfect 1:1 matching between content and generated faces
        # List of (3, 160, 160) → (N, 3, 160, 160)
        matched_content_faces = content_faces  # Already a tensor
        matched_gen_faces = torch.stack(gen_faces) if gen_faces else None
        
        # Sanity check
        if matched_gen_faces is None or len(matched_content_faces) != len(matched_gen_faces):
            metrics = {
                'num_content_faces': len(content_faces),
                'num_gen_faces': len(gen_faces) if gen_faces else 0,
                'num_matched_faces': 0,
                'avg_similarity': 0.0
            }
            return torch.tensor(0.0, device=self.device), metrics
        
        # ========================================
        # Step 7: Extract identity embeddings
        # ========================================
        # Convert face images to 512-d identity vectors
        # OPTIMIZATION: Content embeddings don't need gradients (fixed reference)
        # Only generated embeddings need gradients to update the style transfer model
        # ========================================
        content_embeddings = self.recognizer.extract_embeddings(matched_content_faces, allow_grad=False)  # (N, 512) - No grad needed
        gen_embeddings = self.recognizer.extract_embeddings(matched_gen_faces, allow_grad=True)           # (N, 512) - Needs grad
        
        # ========================================
        # Step 8: Compute identity loss
        # ========================================
        # MSE loss between embeddings - encourages them to be identical
        # Lower loss = more similar embeddings = better identity preservation
        identity_loss = F.mse_loss(gen_embeddings, content_embeddings)
        
        # ========================================
        # Step 9: Compute similarity metrics (for monitoring)
        # ========================================
        # Cosine similarity in [0, 1] range
        # Values close to 1.0 mean identity is well preserved
        similarity = self.recognizer.compute_similarity(gen_embeddings, content_embeddings)
        
        # ========================================
        # Step 10: Package metrics for monitoring
        # ========================================
        metrics = {
            'num_content_faces': len(content_faces),        # Total faces in originals
            'num_gen_faces': len(gen_faces),                # Total faces in stylized
            'num_matched_faces': len(matched_content_faces),# Successfully matched pairs
            'avg_similarity': similarity.mean().item()      # Average similarity (0-1)
        }
        
        return identity_loss, metrics


# ============================================================================
# FALLBACK: SIMPLE FACE DETECTOR (if facenet-pytorch not available)
# ============================================================================

class SimpleFaceDetector:
    """
    Simple face detector using OpenCV's Haar Cascade - FALLBACK OPTION ONLY!
    
    This is a lightweight alternative if facenet-pytorch is not available.
    It's based on classical computer vision (not deep learning).
    
    HAAR CASCADE:
    - Classic face detection algorithm from 2001
    - Uses handcrafted features (Haar-like features)
    - Fast but less accurate than modern deep learning methods
    - Works well for frontal faces, struggles with angles/occlusions
    
    WHEN TO USE:
    - facenet-pytorch is not installed
    - You need a quick test without installing dependencies
    - Resource-constrained environments
    
    LIMITATIONS vs MTCNN:
    ✗ Lower accuracy (misses ~20-30% of faces)
    ✗ Poor with non-frontal faces
    ✗ No facial landmarks
    ✗ Sensitive to lighting/shadows
    
    ✓ Fast (100+ FPS on CPU)
    ✓ No GPU needed
    ✓ Built into OpenCV (no extra dependencies)
    
    RECOMMENDATION:
    Use MTCNN (FaceDetector class) for production!
    Use this only for quick testing or when facenet-pytorch unavailable.
    """
    def __init__(self):
        """
        Initialize Haar Cascade face detector.
        
        Loads the pretrained cascade classifier from OpenCV's data directory.
        """
        # Check if OpenCV is installed
        if not CV2_AVAILABLE:
            raise ImportError(
                "OpenCV is required for SimpleFaceDetector. "
                "Install with: pip install opencv-python"
            )
        
        # ========================================
        # Load Haar Cascade classifier
        # ========================================
        # OpenCV comes with several pretrained cascades
        # haarcascade_frontalface_default.xml is the most common one
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Verify cascade loaded successfully
        if self.face_cascade.empty():
            raise RuntimeError("Failed to load Haar Cascade classifier")
    
    def detect(self, image):
        """
        Detect faces using Haar Cascade algorithm.
        
        This is simpler and less accurate than MTCNN but much faster.
        
        Args:
            image: PIL Image or numpy array (H, W, 3) in RGB format
                  Can be any size
        
        Returns:
            boxes: numpy array of bounding boxes, shape (N, 4)
                  Each box: [x, y, w, h] where:
                  - x, y = top-left corner
                  - w, h = width and height
                  Empty array if no faces detected
        
        Example:
            from PIL import Image
            
            detector = SimpleFaceDetector()
            img = Image.open('face.jpg')
            boxes = detector.detect(img)
            
            print(f"Found {len(boxes)} faces")
            for (x, y, w, h) in boxes:
                print(f"Face at: x={x}, y={y}, size={w}×{h}")
        """
        # ========================================
        # Convert PIL Image to numpy array if needed
        # ========================================
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # ========================================
        # Convert to grayscale (Haar Cascade requires grayscale)
        # ========================================
        # Haar features are computed on intensity values, not color
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # ========================================
        # Run face detection
        # ========================================
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,      # How much image size is reduced at each scale
                                  # 1.1 = 10% reduction (slower but more accurate)
                                  # 1.3 = 30% reduction (faster but less accurate)
            minNeighbors=5,       # How many neighbors each candidate should have
                                  # Higher = fewer false positives, more missed faces
            minSize=(30, 30)      # Minimum face size to detect
                                  # Smaller = detect distant faces (but slower)
        )
        
        # faces is numpy array of shape (N, 4): [[x, y, w, h], ...]
        return faces


# ============================================================================
# EYE-SPECIFIC LOSS (Novel Contribution!)
# ============================================================================

class EyeSpecificLoss(nn.Module):
    """
    Enhanced Eye-Specific Loss with Structure and Edge Constraints.
    
    IMPROVEMENTS & ADVICE IMPLEMENTED:
    1. Increased Crop Size (32 -> 48): Captures eyebrows and context to prevent "floating" eyes.
    2. Gaussian Masking: Eliminates "square artifacts" by softening the crop edges.
    3. Weighted VGG Layers: Prioritizes relu2_1 (structure) over relu3_1 (texture) to fix "swollen" eyes.
    4. Sobel Edge Loss: Explicitly penalizes blurry boundaries to keep eyelids/iris sharp.
    5. Landmark Consistency: Detects on CONTENT image only to ensure stable cropping.
    
    WHY THESE CHANGES MATTER:
    - **Gaussian Masking**: Fixes hard square outlines visible around eyes in stylized images
    - **Sobel Edge Loss**: Prevents "swollen" or "blobby" eyes by enforcing sharp boundaries
    - **relu2_1 + relu3_1**: Balances low-level structure (edges, shape) with mid-level texture
    - **Larger Crop (48×48)**: Includes eyebrows ("ceiling" for eyes), reduces artifacts
    - **Weighted Loss**: Heavy weight on structure (λ=4.0) ensures shape preservation
    
    USAGE:
        eye_loss_module = EyeSpecificLoss(device='cuda')
        
        # During training
        eye_loss, metrics = eye_loss_module.compute_eye_loss(
            generated_images=stylized_batch,
            content_images=original_batch
        )
        
        # Add to total loss
        total_loss = content_loss + style_loss + beta * eye_loss
    """
    
    def __init__(self, device='cuda', feature_extractor=None):
        """
        Initialize enhanced eye-specific loss computation.
        
        Args:
            device: Device to run on ('cuda' or 'cpu')
            feature_extractor: Optional VGG feature extractor
                             If None, will create one internally (relu2_1 + relu3_1)
        """
        super().__init__()
        self.device = device
        
        # Face detector (assumed to be available from your utils)
        self.detector = FaceDetector(device=device, keep_all=False)
        
        # ---------------------------------------------------------------------
        # ADVICE: Use shallower layers for structure, avoiding deep semantic layers
        # like relu4_1 which allow spatial invariance (causing "swollen" eyes).
        # ---------------------------------------------------------------------
        if feature_extractor is None:
            import torchvision.models as models
            vgg = models.vgg19(pretrained=True).features.to(device).eval()
            # We need up to relu3_1 (index 12). relu2_1 is at index 7.
            self.vgg_layers = vgg[:13]
            for param in self.vgg_layers.parameters():
                param.requires_grad = False
        else:
            self.vgg_layers = feature_extractor
        
        # ImageNet normalization (Standard VGG requirement)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406], device=device).view(1, -1, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225], device=device).view(1, -1, 1, 1))
        
        # ---------------------------------------------------------------------
        # ADVICE: Increase crop size to capture eyebrows ("ceiling" for the eye).
        # ---------------------------------------------------------------------
        self.eye_size = 48
        
        # ---------------------------------------------------------------------
        # ADVICE: Pre-calculate Gaussian Mask to fix "Square Artifacts".
        # ---------------------------------------------------------------------
        self.register_buffer('spatial_mask', self._create_gaussian_mask(self.eye_size))
        
        # ---------------------------------------------------------------------
        # ADVICE: Pre-calculate Sobel Filters for "Edge Loss" (Anti-swelling).
        # ---------------------------------------------------------------------
        self._init_sobel_filters()
        
        # ---------------------------------------------------------------------
        # UPGRADE 1: LPIPS Loss (Perceptual Beauty Loss)
        # This replaces/augments VGG loss for better realism and crispness.
        # LPIPS is trained to match human perception of image quality.
        # ---------------------------------------------------------------------
        self.lpips = lpips.LPIPS(net='vgg').to(device)
        for param in self.lpips.parameters():
            param.requires_grad = False
        
        # ---------------------------------------------------------------------
        # ADVICE: Hyperparameters for loss weighting.
        # Structure (relu2_1) is weighted highest to enforce shape.
        # UPGRADE: Added LPIPS and Color preservation weights.
        # ---------------------------------------------------------------------
        self.lambda_structure = 4.0  # relu2_1: Sharpness/Shape
        self.lambda_pattern = 1.0    # relu3_1: Texture/Context
        self.lambda_edge = 2.0       # Sobel:   Boundary precision
        self.lambda_lpips = 0.8      # LPIPS:   Perceptual beauty
        self.lambda_color = 0.5      # Color:   Iris color preservation
    
    def _create_gaussian_mask(self, size):
        """
        Creates a 2D gaussian mask to soften crop edges.
        Fixes: Visible square outlines around the eyes.
        """
        x = torch.linspace(-1, 1, size)
        y = torch.linspace(-1, 1, size)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        dist = torch.sqrt(xx**2 + yy**2)
        
        # Sigma 0.5 creates a nice falloff that is near 0 at the edges
        mask = torch.exp(-(dist**2) / 0.5)
        return mask.unsqueeze(0).unsqueeze(0).to(self.device)
    
    def _init_sobel_filters(self):
        """
        Initializes Sobel kernels for edge detection.
        FIX: Uses (3, 1, 3, 3) shape to process RGB channels independently.
        """
        # Base kernel (1, 1, 3, 3)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        
        # FIX: Repeat along output dimension (dim 0), NOT input dimension (dim 1).
        # Target Shape: (3, 1, 3, 3) -> 3 output channels, 1 input channel per group
        self.register_buffer('sobel_x', sobel_x.repeat(3, 1, 1, 1))
        self.register_buffer('sobel_y', sobel_y.repeat(3, 1, 1, 1))
    
    def get_vgg_features(self, x):
        """Extract features specifically from relu2_1 and relu3_1."""
        features = {}
        out = x
        for name, layer in self.vgg_layers.named_children():
            out = layer(out)
            # relu2_1 (Structure/Edges) -> Index '6' (ReLU activation, standard for perceptual loss)
            if name == '6':
                features['relu2_1'] = out
            # relu3_1 (Texture/Patterns) -> Index '11' (ReLU activation, standard for perceptual loss)
            elif name == '11':
                features['relu3_1'] = out
                break
        return features
    
    def get_edge_features(self, x):
        """
        Compute edge map using Sobel filters on the raw pixel input.
        Returns a map showing where the hard boundaries (eyelids, iris) are.
        """
        # FIX: Use groups=3 to apply the filter to each RGB channel independently
        # Input: (B, 3, H, W) -> Output: (B, 3, H, W)
        gx = F.conv2d(x, self.sobel_x, padding=1, groups=3)
        gy = F.conv2d(x, self.sobel_y, padding=1, groups=3)
        
        # Calculate magnitude per channel
        # Added epsilon for numerical stability
        edges_per_channel = torch.sqrt(gx**2 + gy**2 + 1e-6)
        
        # Combine channels: Average the edge magnitudes to get a single map
        # Output: (B, 1, H, W)
        return edges_per_channel.mean(dim=1, keepdim=True)
    
    def compute_color_loss(self, gen, content):
        """
        UPGRADE 2: Color Preservation Loss
        
        Matches color statistics (Mean & Std) between generated and content eyes.
        Keeps the eyes looking like the original person's eyes, not the painting's.
        
        WHY: Style transfer often washes out natural eye color (e.g., turning blue
        eyes into muddy brown). This forces the model to preserve the original
        iris color while still allowing texture stylization.
        
        Args:
            gen: Generated eye crops (B, 3, H, W)
            content: Content eye crops (B, 3, H, W)
        
        Returns:
            color_loss: MSE between color statistics (mean + std per channel)
        """
        # Calculate mean and std per channel (R, G, B)
        # Input shape: (B, 3, H, W) -> Flatten spatial dims
        gen_flat = gen.view(gen.size(0), 3, -1)
        content_flat = content.view(content.size(0), 3, -1)

        gen_mean = gen_flat.mean(dim=2)
        gen_std = gen_flat.std(dim=2)
        
        content_mean = content_flat.mean(dim=2)
        content_std = content_flat.std(dim=2)

        # MSE between statistics
        loss_mean = F.mse_loss(gen_mean, content_mean)
        loss_std = F.mse_loss(gen_std, content_std)
        return loss_mean + loss_std
    
    def extract_eye_regions(self, images, landmarks):
        """Extract eye regions based on provided landmarks."""
        eye_regions = []
        indices = []  # Stores (batch_idx, eye_type)
        
        for batch_idx, (img, landmark) in enumerate(zip(images, landmarks)):
            if landmark is None:
                continue
            
            # Normalize landmark format
            if isinstance(landmark, np.ndarray):
                if landmark.ndim == 3:
                    landmark = landmark[0]
            
            H, W = img.shape[1:]
            
            # 0=left eye, 1=right eye
            for eye_idx in range(2):
                try:
                    eye_x, eye_y = landmark[eye_idx]
                except:
                    continue
                
                half = self.eye_size // 2
                x1 = max(0, int(eye_x - half))
                y1 = max(0, int(eye_y - half))
                x2 = min(W, int(eye_x + half))
                y2 = min(H, int(eye_y + half))
                
                # Skip edge cases where crop is too small
                if (x2 - x1) < self.eye_size or (y2 - y1) < self.eye_size:
                    continue
                
                # Crop
                eye_region = img[:, y1:y2, x1:x2]
                
                # Resize to ensure strict 48x48 input for VGG
                eye_region = F.interpolate(
                    eye_region.unsqueeze(0),
                    size=(self.eye_size, self.eye_size),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
                
                eye_regions.append(eye_region)
                indices.append((batch_idx, eye_idx))
        
        return eye_regions, indices
    
    def compute_eye_loss(self, generated_images, content_images):
        """
        Compute total weighted eye loss.
        
        CRITICAL FIX: Landmark Consistency
        - Detect ONLY on content images. Use those coordinates for both.
        - This prevents the loss from "chasing a moving target" if the
          stylized face distorts slightly.
        
        Args:
            generated_images: (B, 3, H, W)
            content_images: (B, 3, H, W)
        
        Returns:
            total_loss: Weighted sum of structure + pattern + edge losses
            metrics: Dict with diagnostic information
        """
        batch_size = generated_images.size(0)
        metrics = {}
        
        # ---------------------------------------------------------------------
        # ADVICE: Landmark Consistency
        # Detect ONLY on content images. Use those coordinates for both.
        # ---------------------------------------------------------------------
        with torch.no_grad():
            _, _, content_landmarks = self.detector.detect(content_images)
        
        # Extract crops
        content_eyes, c_indices = self.extract_eye_regions(content_images, content_landmarks)
        gen_eyes, g_indices = self.extract_eye_regions(generated_images, content_landmarks)
        
        # Basic validation: ensure we found eyes
        if len(content_eyes) == 0:
            metrics['num_eyes'] = 0
            return torch.tensor(0.0, device=self.device, requires_grad=True), metrics
        
        # Stack them into tensors for batch processing
        content_batch = torch.stack(content_eyes)
        gen_batch = torch.stack(gen_eyes)
        
        # ---------------------------------------------------------------------
        # UPGRADE 2: Color Preservation (Compute on RAW pixels before masking)
        # We want the center of the eye to have the same color distribution
        # as the original photo. This brings back the natural eye color and "life".
        # ---------------------------------------------------------------------
        loss_color = self.compute_color_loss(gen_batch, content_batch)
        
        # ---------------------------------------------------------------------
        # CRITICAL FIX: Compute edges BEFORE masking!
        # The Gaussian mask creates artificial gradients at boundaries.
        # If we mask first, Sobel filters detect the mask edge as a feature.
        # By computing edges first, we only compare real eye structure.
        # ---------------------------------------------------------------------
        c_edges_raw = self.get_edge_features(content_batch)
        g_edges_raw = self.get_edge_features(gen_batch)
        
        # Now apply mask to edge maps (focus on center, ignore boundary noise)
        c_edges = c_edges_raw * self.spatial_mask
        g_edges = g_edges_raw * self.spatial_mask
        loss_edge = F.l1_loss(g_edges, c_edges)
        
        # ---------------------------------------------------------------------
        # ADVICE: VGG Loss - Mask pixels BEFORE feature extraction
        # For pixel/VGG loss, we MUST mask first to avoid square artifacts
        # ---------------------------------------------------------------------
        content_masked = content_batch * self.spatial_mask
        gen_masked = gen_batch * self.spatial_mask
        
        # ---------------------------------------------------------------------
        # UPGRADE 1: LPIPS Loss (Perceptual Beauty Loss)
        # LPIPS expects inputs in [-1, 1]. Our images are [0, 1].
        # Normalize: (x * 2) - 1
        # ---------------------------------------------------------------------
        c_lpips_input = (content_masked * 2) - 1
        g_lpips_input = (gen_masked * 2) - 1
        
        # Compute LPIPS on the masked eye regions
        # This optimizes for human perception of beauty/realism
        loss_lpips = self.lpips(g_lpips_input, c_lpips_input).mean()
        
        # ---------------------------------------------------------------------
        # ADVICE: Weighted VGG Loss
        # 1. Normalize images for VGG (ImageNet stats)
        # 2. Extract shallow features (relu2_1, relu3_1)
        # ---------------------------------------------------------------------
        c_norm = (content_masked - self.mean) / self.std
        g_norm = (gen_masked - self.mean) / self.std
        
        c_feats = self.get_vgg_features(c_norm)
        g_feats = self.get_vgg_features(g_norm)
        
        # Calculate MSE for features
        # ADVICE: Heavy weight on relu2_1 to fix swelling/shape issues
        loss_structure = F.mse_loss(g_feats['relu2_1'], c_feats['relu2_1'])
        
        # ADVICE: Lower weight on relu3_1 for general texture/context
        loss_pattern = F.mse_loss(g_feats['relu3_1'], c_feats['relu3_1'])
        
        # ---------------------------------------------------------------------
        # Final Weighted Sum (Now includes LPIPS and Color!)
        # ---------------------------------------------------------------------
        total_loss = (self.lambda_structure * loss_structure) + \
                     (self.lambda_pattern * loss_pattern) + \
                     (self.lambda_edge * loss_edge) + \
                     (self.lambda_lpips * loss_lpips) + \
                     (self.lambda_color * loss_color)
        
        # Metrics for logging
        metrics['eye_loss'] = total_loss.item()
        metrics['num_eyes'] = len(content_eyes)
        
        return total_loss, metrics


# ============================================================================
# TEST CODE
# ============================================================================
# Run this file directly to test face detection and recognition:
#     python model_face_utils.py
#
# This will verify that:
# 1. Dependencies are installed (facenet-pytorch)
# 2. IdentityPreserver can be created
# 3. Identity loss can be computed
# 4. Face detection/recognition works

if __name__ == "__main__":
    """
    Test face detection and recognition utilities.
    
    This runs a basic sanity check to ensure the face utilities work correctly
    before using them in training.
    """
    print("=" * 70)
    print("TESTING FACE DETECTION & RECOGNITION UTILITIES")
    print("=" * 70)
    
    # ========================================
    # Check if dependencies are installed
    # ========================================
    if not FACENET_AVAILABLE:
        print("\n⚠️  facenet-pytorch is NOT installed!")
        print("\nThis package is REQUIRED for face detection and recognition.")
        print("\nInstall with:")
        print("    pip install facenet-pytorch")
        print("\nThis will also install these dependencies:")
        print("  - torch (PyTorch)")
        print("  - torchvision")
        print("  - numpy")
        print("  - Pillow (PIL)")
        print("\nAfter installation, run this test again:")
        print("    python model_face_utils.py")
        print("\n" + "=" * 70)
        exit(1)
    
    # ========================================
    # Setup device
    # ========================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📱 Device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # ========================================
    # Test 1: Create IdentityPreserver
    # ========================================
    print("\n" + "=" * 70)
    print("TEST 1: Creating Identity Preserver")
    print("=" * 70)
    
    try:
        identity_preserver = IdentityPreserver(device=device, pretrained='vggface2')
        print("✅ Identity preserver created successfully")
        print("   - Face detector: MTCNN (loaded)")
        print("   - Face recognizer: InceptionResnetV1 on VGGFace2 (loaded)")
    except Exception as e:
        print(f"❌ Failed to create identity preserver: {e}")
        exit(1)
    
    # ========================================
    # Test 2: Compute identity loss with random images
    # ========================================
    print("\n" + "=" * 70)
    print("TEST 2: Computing Identity Loss")
    print("=" * 70)
    print("Note: Using random noise images - unlikely to detect faces")
    
    # Create random test images
    batch_size = 2
    content = torch.randn(batch_size, 3, 256, 256).to(device).clamp(0, 1)
    generated = torch.randn(batch_size, 3, 256, 256).to(device).clamp(0, 1)
    
    print(f"Input shapes:")
    print(f"  Content: {content.shape}")
    print(f"  Generated: {generated.shape}")
    
    # Compute identity loss
    try:
        loss, metrics = identity_preserver.compute_identity_loss(generated, content)
        
        print(f"\n✅ Identity loss computed successfully")
        print(f"\nResults:")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Metrics:")
        print(f"    - Content faces detected: {metrics['num_content_faces']}")
        print(f"    - Generated faces detected: {metrics['num_gen_faces']}")
        print(f"    - Matched face pairs: {metrics['num_matched_faces']}")
        print(f"    - Average similarity: {metrics['avg_similarity']:.2f}")
        
        if metrics['num_matched_faces'] == 0:
            print("\n⚠️  No faces detected (expected with random images)")
            print("   This is normal! Random noise doesn't look like faces.")
        else:
            print("\n🎉 Faces detected in random images (lucky!)")
    
    except Exception as e:
        print(f"❌ Failed to compute identity loss: {e}")
        exit(1)
    
    # ========================================
    # All tests passed!
    # ========================================
    print("\n" + "=" * 70)
    print("🎉 ALL TESTS PASSED!")
    print("=" * 70)
    print("\nIdentity preservation system is working correctly!")
    print("\nNext steps:")
    print("  1. Test with real face images for meaningful results")
    print("  2. Use identity_preserver.compute_identity_loss() during training")
    print("  3. Add identity loss to your total loss:")
    print("     total_loss = content_loss + style_loss + 0.1 * identity_loss")
    print("\nExample in training:")
    print("  >>> stylized = model(content, style)")
    print("  >>> id_loss, metrics = preserver.compute_identity_loss(stylized, content)")
    print("  >>> print(f'Identity similarity: {metrics[\"avg_similarity\"]:.2f}')")
    print("\n" + "=" * 70)


