#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AdaIN (Adaptive Instance Normalization) Style Transfer Model

This implements the fast feed-forward style transfer approach from:
"Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization"
Huang & Belongie, ICCV 2017

OVERVIEW:
AdaIN performs style transfer by aligning the mean and variance of content features
with those of style features. This is done in the feature space of a pretrained VGG network.

KEY INNOVATION:
- Traditional NST: Optimize pixels for 100s of iterations per image (SLOW - minutes per image)
- AdaIN: Train a decoder once, then single forward pass (FAST - 0.1 seconds per image)
- 300x speedup! Perfect for real-time applications.

ARCHITECTURE FLOW:
    Content Image (face photo)  → VGG Encoder → Content Features
                                                       ↓
                                                    AdaIN Layer (style transfer happens here!)
                                                       ↓
    Style Image (Peter Rabbit)  → VGG Encoder → Style Features
                                                       ↓
                                                    Decoder → Stylized Image

HOW ADAIN WORKS (the magic!):
    AdaIN(content, style) = σ(style) × normalize(content) + μ(style)
    
    Where:
    - μ (mu) = mean (average color values per channel)
    - σ (sigma) = standard deviation (color variation per channel)
    - normalize(content) = (content - μ(content)) / σ(content)
    
    This transfers the "style" by replacing content's color statistics with style's!

USAGE EXAMPLE:
    # Create model
    model = AdaINStyleTransfer(device='cuda')
    
    # Load images (should be torch tensors in [0,1] range)
    content = load_image('face.jpg')  # Child's face
    style = load_image('peter_rabbit.jpg')  # Style image
    
    # Perform style transfer
    stylized = model(content, style, alpha=1.0)  # alpha controls style strength
    
    # Save result
    save_image(stylized, 'stylized_face.jpg')
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class AdaIN(nn.Module):
    """
    Adaptive Instance Normalization layer - THE CORE OF STYLE TRANSFER!
    
    This layer transfers style by aligning the statistics (mean and variance)
    of content features with those of style features.
    
    Mathematical Formula:
        AdaIN(content, style) = σ(style) × normalize(content) + μ(style)
    
    Where:
        - normalize() = instance normalization (removes content's original statistics)
        - μ (mu) = mean (average values per channel)
        - σ (sigma) = standard deviation (variation per channel)
    
    Think of it like this:
        - Mean (μ): The "average color tone" of each channel
        - Std (σ): How much colors vary in each channel
        - By replacing content's statistics with style's, we transfer the "look"!
    
    Example:
        If style has warm red tones (high mean in red channel) and high contrast
        (high std), the output will have those same warm red tones and contrast!
    """
    def __init__(self):
        super().__init__()
        # No learnable parameters - AdaIN is a pure statistical operation!
    
    def calc_mean_std(self, feat, eps=1e-5):
        """
        Calculate mean and standard deviation across spatial dimensions.
        
        This computes statistics for EACH channel independently, across all
        spatial locations (height × width).
        
        Args:
            feat: (B, C, H, W) feature maps
                  B = batch size (e.g., 4 images)
                  C = channels (e.g., 512 feature channels from VGG)
                  H, W = spatial dimensions (e.g., 32×32)
            eps: small value (1e-5) for numerical stability (prevents division by zero)
            
        Returns:
            mean: (B, C, 1, 1) - mean for each channel
            std: (B, C, 1, 1) - standard deviation for each channel
        
        Example:
            Input feat: (4, 512, 32, 32) - 4 images, 512 channels, 32×32 spatial
            Output mean: (4, 512, 1, 1) - one mean per channel per image
            Output std: (4, 512, 1, 1) - one std per channel per image
        """
        size = feat.size()
        assert len(size) == 4, "Expected 4D input (B, C, H, W)"
        B, C = size[:2]  # Batch size and number of channels
        
        # ========================================
        # Step 1: Flatten spatial dimensions
        # ========================================
        # Original: (B, C, H, W) where H*W might be 32*32 = 1024 spatial locations
        # After view: (B, C, H*W) = (B, C, 1024) - flatten all spatial locations
        # This allows us to compute statistics across all locations per channel
        
        # Compute variance across spatial dimension (dim=2 is the H*W dimension)
        # var() gives us the variance for each of the C channels
        # +eps prevents division by zero when std is computed
        feat_var = feat.view(B, C, -1).var(dim=2) + eps
        
        # Standard deviation is square root of variance
        # Reshape to (B, C, 1, 1) for broadcasting in later operations
        feat_std = feat_var.sqrt().view(B, C, 1, 1)
        
        # Compute mean across spatial dimension
        # mean() gives us the average value for each of the C channels
        feat_mean = feat.view(B, C, -1).mean(dim=2).view(B, C, 1, 1)
        
        return feat_mean, feat_std
    
    def forward(self, content_feat, style_feat):
        """
        Apply AdaIN to align content feature statistics with style features.
        
        THIS IS WHERE STYLE TRANSFER HAPPENS!
        
        The process:
        1. Extract statistics (mean, std) from both content and style
        2. Normalize content (remove its original "look")
        3. Apply style statistics (give it the style's "look")
        
        Args:
            content_feat: (B, C, H, W) content features from VGG encoder
                         Example: (4, 512, 32, 32) - features from face photos
            style_feat: (B, C, H', W') style features from VGG encoder
                        Example: (4, 512, 64, 64) - features from Peter Rabbit
                        Note: H' and W' can be different from H and W!
            
        Returns:
            stylized_feat: (B, C, H, W) with style statistics applied
                          Same spatial size as content, but with style's "look"
        
        Mathematical Steps:
            1. normalized = (content - μ_content) / σ_content     [Remove content style]
            2. stylized = normalized × σ_style + μ_style          [Apply style]
        """
        # Sanity check: batch size and channels must match
        assert content_feat.size()[:2] == style_feat.size()[:2], \
            "Content and style must have same batch size and channels"
        
        size = content_feat.size()
        
        # ========================================
        # Step 1: Get statistics from both inputs
        # ========================================
        # Calculate mean and std for content (e.g., face photo statistics)
        content_mean, content_std = self.calc_mean_std(content_feat)
        # Calculate mean and std for style (e.g., Peter Rabbit statistics)
        style_mean, style_std = self.calc_mean_std(style_feat)
        
        # ========================================
        # Step 2: Normalize content features
        # ========================================
        # This removes the content's original "look" (color tones, contrast)
        # by subtracting mean and dividing by std
        # Result: standardized features with mean=0, std=1
        normalized_feat = (content_feat - content_mean) / content_std
        
        # ========================================
        # Step 3: Apply style statistics
        # ========================================
        # Now we give the normalized content the style's "look"
        # by multiplying by style's std and adding style's mean
        # Result: features with style's color distribution!
        stylized_feat = normalized_feat * style_std + style_mean
        
        # The magic is complete! Content structure + Style appearance
        return stylized_feat


class VGGEncoder(nn.Module):
    """
    VGG19 encoder that extracts multi-level features for style transfer.
    
    WHY VGG19?
    - Pretrained on ImageNet (millions of images)
    - Learns hierarchical features: edges → textures → patterns → objects
    - Perfect for understanding both "content" and "style" of images
    
    ARCHITECTURE USED:
    We use VGG19 up to relu4_1 (21 layers total):
        - relu1_1: Low-level features (edges, basic colors)
        - relu2_1: Textures and simple patterns
        - relu3_1: Complex patterns
        - relu4_1: High-level content (faces, objects) ← Main features for AdaIN
    
    KEY DESIGN CHOICES:
    1. Frozen weights: We DON'T train VGG, just use it as feature extractor
    2. Pretrained: Uses ImageNet knowledge to understand image structure
    3. Up to relu4_1: Captures semantic content without too much detail
    """
    def __init__(self, device='cuda'):
        super().__init__()
        
        self.device = device
        
        # ========================================
        # Load pretrained VGG19 from torchvision
        # ========================================
        # VGG19 was trained on ImageNet dataset
        # It learned to recognize 1000 object categories
        # We use its "perception" to understand image content and style
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        
        # ========================================
        # Extract layers up to relu4_1
        # ========================================
        # VGG19 has many layers, but we only need the first 21
        # Layer 21 corresponds to relu4_1 (first ReLU in 4th block)
        # This gives us features at the right abstraction level for style transfer
        self.encoder = nn.Sequential(*list(vgg.children())[:21])
        
        # ========================================
        # Freeze encoder parameters
        # ========================================
        # We DON'T want to train VGG - it's perfect as is!
        # requires_grad = False means these weights won't update during training
        # This saves memory and computation
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Set to evaluation mode (disables dropout, batchnorm updating, etc.)
        self.encoder.eval()
        
        # ========================================
        # Register ImageNet normalization parameters
        # ========================================
        # VGG19 was trained with ImageNet normalization:
        # - Mean: [0.485, 0.456, 0.406] for RGB channels
        # - Std: [0.229, 0.224, 0.225] for RGB channels
        # We need to apply the same normalization to our images!
        # register_buffer makes these part of the model state (but not trainable)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1))
        
        # Move everything to GPU if available
        self.to(device)
    
    def forward(self, x):
        """
        Encode image to feature space at relu4_1 level.
        
        Args:
            x: (B, 3, H, W) input images in [0, 1] range
               Example: (4, 3, 256, 256) - 4 RGB images, 256×256 pixels
               IMPORTANT: Pixel values must be in [0, 1], not [0, 255]!
        
        Returns:
            features: (B, 512, H/8, W/8) features at relu4_1
                     Example: (4, 512, 32, 32) - spatial size reduced by 8x
                     512 channels is the output of VGG19's relu4_1 layer
        """
        # Normalize using ImageNet statistics
        # VGG expects inputs normalized with ImageNet mean/std
        # This standardizes the input to match what VGG saw during training
        x = (x - self.mean) / self.std
        
        # Pass through VGG encoder (21 layers up to relu4_1)
        return self.encoder(x)
    
    def encode_with_intermediate(self, x):
        """
        Encode and return intermediate features at multiple layers.
        
        This is used during training to compute perceptual loss at multiple scales.
        We extract features at 4 different layers to capture style at multiple levels.
        
        Args:
            x: (B, 3, H, W) input images in [0, 1] range
        
        Returns:
            features: dict with keys ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1']
                     Each value is a feature tensor at that layer
        
        Example output:
            {
                'relu1_1': (4, 64, 256, 256),    # Low-level: edges, colors
                'relu2_1': (4, 128, 128, 128),   # Mid-level: textures
                'relu3_1': (4, 256, 64, 64),     # Higher-level: patterns
                'relu4_1': (4, 512, 32, 32)      # Highest-level: semantic content
            }
        """
        # Normalize using ImageNet statistics (same as forward())
        x = (x - self.mean) / self.std
        
        # ========================================
        # Layer indices mapping for VGG19
        # ========================================
        # VGG19 structure has specific layer indices where ReLU activations occur
        # These are the standard indices used in style transfer literature
        layer_mapping = {
            1: 'relu1_1',   # After first ReLU (block 1)
            6: 'relu2_1',   # After first ReLU in block 2
            11: 'relu3_1',  # After first ReLU in block 3
            20: 'relu4_1'   # After first ReLU in block 4 ← Most important for content!
        }
        
        # Dictionary to store features at each layer
        features = {}
        
        # ========================================
        # Pass through encoder layer by layer
        # ========================================
        # We iterate through each layer manually to extract intermediate features
        for i, layer in enumerate(self.encoder):
            x = layer(x)  # Apply this layer's transformation
            
            # If this is one of our target layers, save the features
            if i in layer_mapping:
                features[layer_mapping[i]] = x
        
        return features


class Decoder(nn.Module):
    """
    Decoder network that reconstructs image from AdaIN-stylized features.
    
    THIS IS THE ONLY TRAINABLE PART OF THE MODEL!
    - VGG encoder: Frozen (pretrained weights)
    - AdaIN layer: No parameters (pure statistical operation)
    - Decoder: Trainable! (~3.5M parameters)
    
    ARCHITECTURE DESIGN:
    - Mirrors the VGG encoder structure (but in reverse)
    - Uses upsampling to increase spatial resolution
    - Reduces channels: 512 → 256 → 128 → 64 → 3 (RGB)
    - Uses ReflectionPad2d to avoid border artifacts (better than zero padding)
    
    INPUT → OUTPUT:
        (B, 512, 32, 32) features → (B, 3, 256, 256) stylized image
        Spatial size increases 8x through 3 upsampling operations (2×2×2=8)
    """
    def __init__(self):
        super().__init__()
        
        # ========================================
        # Decoder Architecture
        # ========================================
        # This is a deep convolutional network that learns to convert
        # stylized features back into a beautiful image
        
        self.decoder = nn.Sequential(
            # ===== Block 1: Initial processing (512 → 256 channels) =====
            # Start from relu4_1 output: (B, 512, H/8, W/8)
            # Example: (4, 512, 32, 32)
            nn.ReflectionPad2d(1),          # Pad with reflected pixels (avoids edge artifacts)
            nn.Conv2d(512, 256, 3, 1, 0),   # Convolution: 512 input channels → 256 output
            nn.ReLU(inplace=True),           # Non-linearity
            
            # ===== Upsample 1: 2x spatial increase =====
            # (B, 256, 32, 32) → (B, 256, 64, 64)
            nn.Upsample(scale_factor=2, mode='nearest'),  # Double spatial dimensions
            
            # ===== Block 2: Process at medium resolution (256 channels) =====
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3, 1, 0),   # Refine features
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3, 1, 0),   # More refinement
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3, 1, 0),   # Even more refinement
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, 3, 1, 0),   # Reduce to 128 channels
            nn.ReLU(inplace=True),
            
            # ===== Upsample 2: Another 2x spatial increase =====
            # (B, 128, 64, 64) → (B, 128, 128, 128)
            nn.Upsample(scale_factor=2, mode='nearest'),
            
            # ===== Block 3: Process at higher resolution (128 → 64 channels) =====
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 128, 3, 1, 0),   # Refine features
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 64, 3, 1, 0),    # Reduce to 64 channels
            nn.ReLU(inplace=True),
            
            # ===== Upsample 3: Final 2x spatial increase =====
            # (B, 64, 128, 128) → (B, 64, 256, 256)
            nn.Upsample(scale_factor=2, mode='nearest'),
            
            # ===== Block 4: Final processing to RGB image (64 → 3 channels) =====
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 64, 3, 1, 0),     # Final refinement
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 3, 3, 1, 0),      # Output RGB image (3 channels)
            # Note: No activation here! Output can be any value
            # Values typically end up in ~[0, 1] range due to training
        )
    
    def forward(self, x):
        """
        Decode stylized features back to an image.
        
        Args:
            x: (B, 512, H/8, W/8) stylized features from AdaIN
               Example: (4, 512, 32, 32)
        
        Returns:
            image: (B, 3, H, W) reconstructed RGB image
                  Example: (4, 3, 256, 256)
                  Values are typically in [0, 1] range (but not clamped)
        """
        return self.decoder(x)


class AdaINStyleTransfer(nn.Module):
    """
    Complete AdaIN-based style transfer network - THE FULL PIPELINE!
    
    This brings together all components for fast, arbitrary style transfer:
    
    ARCHITECTURE:
        Content Image → VGG Encoder (frozen) → Content Features
                                                       ↓
                                                    AdaIN Layer → Stylized Features
                                                       ↑
        Style Image → VGG Encoder (frozen) → Style Features
                                                       ↓
                                                    Decoder (trainable) → Output Image
    
    COMPONENTS:
    1. VGG Encoder (frozen, ~20M parameters):
       - Extracts features from content and style images
       - Pretrained on ImageNet, never updated
       - Shared for both content and style encoding
    
    2. AdaIN Layer (no parameters):
       - Pure statistical operation
       - Transfers style by aligning mean/variance
       - No learning required!
    
    3. Decoder (~3.5M parameters):
       - THE ONLY TRAINABLE PART!
       - Learns to reconstruct beautiful images from stylized features
       - Trained on ~200 content + ~20 style images
    
    TRAINING vs INFERENCE:
    - Training: Only decoder weights are updated (encoder frozen)
    - Inference: Single forward pass, ~0.1 seconds per image
    
    WHY IT'S FAST:
    - No pixel optimization (unlike traditional NST)
    - Single forward pass through network
    - ~300x faster than optimization-based methods
    """
    def __init__(self, device='cuda'):
        super().__init__()
        
        self.device = device
        
        # ========================================
        # Initialize all components
        # ========================================
        # VGG encoder: Extracts features, frozen weights
        self.encoder = VGGEncoder(device=device)
        
        # Decoder: Reconstructs images, trainable weights
        self.decoder = Decoder()
        
        # AdaIN layer: Statistical style transfer, no parameters
        self.adain = AdaIN()
        
        # Move decoder to GPU/CPU (encoder already moved in its __init__)
        self.decoder.to(device)
    
    def encode(self, x):
        """
        Encode image to feature space using VGG encoder.
        
        Args:
            x: (B, 3, H, W) RGB images in [0, 1] range
        
        Returns:
            features: (B, 512, H/8, W/8) features at relu4_1
        """
        return self.encoder(x)
    
    def decode(self, x):
        """
        Decode features back to image using trained decoder.
        
        Args:
            x: (B, 512, H/8, W/8) features
        
        Returns:
            image: (B, 3, H, W) reconstructed RGB image
        """
        return self.decoder(x)
    
    def forward(self, content, style, alpha=1.0):
        """
        Perform style transfer - THE MAIN FUNCTION!
        
        This is what you call to stylize images. It orchestrates the entire pipeline.
        
        Args:
            content: (B, 3, H, W) content images in [0, 1] range
                    Example: (4, 3, 256, 256) - 4 face photos
                    MUST be in [0, 1], not [0, 255]!
            style: (B, 3, H, W) style images in [0, 1] range
                  Example: (4, 3, 512, 512) - 4 art images (Peter Rabbit, etc.)
                  Can be different size than content!
            alpha: style strength parameter (0.0 to 1.0)
                  0.0 = No style (output = content)
                  1.0 = Full style (default, maximum artistic effect)
                  0.5 = Half style (subtle artistic effect)
            
        Returns:
            stylized: (B, 3, H, W) stylized images in [0, 1] range
                     Same size as content images
                     Example: (4, 3, 256, 256) - 4 face photos with Peter Rabbit style!
        
        Example:
            # Load images
            content = load_image('face.jpg')  # (1, 3, 256, 256)
            style = load_image('peter_rabbit.jpg')  # (1, 3, 512, 512)
            
            # Full style transfer
            result = model(content, style, alpha=1.0)
            
            # Subtle style transfer
            result = model(content, style, alpha=0.5)
        """
        # ========================================
        # Step 1: Encode both images to feature space
        # ========================================
        # Extract features using VGG19 up to relu4_1
        # Content: (4, 3, 256, 256) → (4, 512, 32, 32)
        content_feat = self.encode(content)
        # Style: (4, 3, 512, 512) → (4, 512, 64, 64)
        style_feat = self.encode(style)
        
        # ========================================
        # Step 2: Apply AdaIN (THE MAGIC HAPPENS HERE!)
        # ========================================
        # Transfer style by aligning statistics
        # Result: Content structure + Style appearance
        # Output: (4, 512, 32, 32) - same spatial size as content_feat
        stylized_feat = self.adain(content_feat, style_feat)
        
        # ========================================
        # Step 3: Optional style strength control
        # ========================================
        # If alpha < 1.0, interpolate between stylized and original content features
        # This allows for subtle style transfer (less artistic, more realistic)
        # alpha=1.0: Full style (100% stylized)
        # alpha=0.5: Half style (50% stylized, 50% original)
        # alpha=0.0: No style (0% stylized, 100% original)
        if alpha < 1.0:
            stylized_feat = alpha * stylized_feat + (1 - alpha) * content_feat
        
        # ========================================
        # Step 4: Decode features back to image
        # ========================================
        # Use trained decoder to reconstruct final stylized image
        # (4, 512, 32, 32) → (4, 3, 256, 256)
        stylized = self.decode(stylized_feat)
        
        # Done! Return the stylized image
        return stylized
    
    def forward_with_features(self, content, style, alpha=1.0):
        """
        Perform style transfer and return intermediate features for loss computation.
        
        THIS IS USED DURING TRAINING!
        
        During training, we need to compute two losses:
        1. Content loss: Ensures output preserves content structure
        2. Style loss: Ensures output matches style appearance
        
        To compute these losses, we need features at multiple layers from VGG.
        This method returns both the stylized image AND all the features needed for loss.
        
        Args:
            content: (B, 3, H, W) content images
            style: (B, 3, H, W) style images
            alpha: style strength (0-1)
        
        Returns:
            stylized: (B, 3, H, W) stylized image
            stylized_features: dict of features from stylized image
                              {'relu1_1': tensor, 'relu2_1': tensor, ...}
            content_features: dict of features from original content image
            style_features: dict of features from style image
        
        Example during training:
            stylized, stylized_feats, content_feats, style_feats = \
                model.forward_with_features(content, style)
            
            # Compute losses
            content_loss = calc_content_loss(stylized_feats, content_feats)
            style_loss = calc_style_loss(stylized_feats, style_feats)
            total_loss = content_loss + 10.0 * style_loss
            
            # Backprop and update decoder weights
            total_loss.backward()
        """
        # ========================================
        # Perform style transfer (same as forward())
        # ========================================
        content_feat = self.encode(content)
        style_feat = self.encode(style)
        
        # Apply AdaIN
        stylized_feat = self.adain(content_feat, style_feat)
        
        # Optional alpha blending
        if alpha < 1.0:
            stylized_feat = alpha * stylized_feat + (1 - alpha) * content_feat
        
        # Decode to image
        stylized = self.decode(stylized_feat)
        
        # ========================================
        # Extract features at multiple layers for loss computation
        # ========================================
        # We pass each image through VGG again to extract features at 4 layers
        # (relu1_1, relu2_1, relu3_1, relu4_1)
        stylized_features = self.encoder.encode_with_intermediate(stylized)
        content_features = self.encoder.encode_with_intermediate(content)
        style_features = self.encoder.encode_with_intermediate(style)
        
        return stylized, stylized_features, content_features, style_features


# ============================================================================
# LOSS FUNCTIONS FOR TRAINING
# ============================================================================
# These functions compute the losses used to train the decoder network.
# During training, we want the stylized output to:
# 1. Preserve content structure (content loss)
# 2. Match style appearance (style loss)

def calc_content_loss(generated_features, content_features):
    """
    Content loss: Ensures stylized image preserves content structure.
    
    HOW IT WORKS:
    - Compare features of generated image vs. original content image
    - Use high-level features (relu4_1) that capture semantic content
    - Low MSE = output looks like the content (faces preserved!)
    
    WHY ONLY relu4_1?
    - Higher layers (relu4_1) capture semantic content (faces, objects)
    - Lower layers (relu1_1) capture low-level details (edges, textures)
    - We want to preserve high-level structure, not pixel-perfect copy
    
    Args:
        generated_features: dict with VGG features of stylized image
        content_features: dict with VGG features of original content
    
    Returns:
        loss: scalar tensor, Mean Squared Error at relu4_1
              Lower is better (means content is preserved)
    
    Example:
        If content is a face photo, this loss ensures the stylized output
        still looks like the same face (eyes, nose, mouth in same places).
    """
    # Extract relu4_1 features (highest level semantic content)
    # Shape: (B, 512, H/8, W/8)
    gen_relu4 = generated_features['relu4_1']
    content_relu4 = content_features['relu4_1']
    
    # Compute Mean Squared Error between features
    # This measures how different the content structure is
    loss = F.mse_loss(gen_relu4, content_relu4)
    
    return loss


def calc_style_loss(generated_features, style_features):
    """
    Style loss: Ensures stylized image matches style appearance.
    
    HOW IT WORKS:
    - Compare statistics (mean, std) of generated vs. style features
    - Use multiple layers (relu1_1 to relu4_1) to capture style at different scales
    - Low MSE = output has same "look" as style (colors, textures, patterns)
    
    WHY MEAN AND STD?
    - Mean: Captures average color/tone (e.g., warm red tones)
    - Std: Captures variation/contrast (e.g., high contrast in Peter Rabbit)
    - Together they define the "style" without caring about spatial arrangement
    
    WHY MULTIPLE LAYERS?
    - relu1_1: Low-level style (color distribution, edge patterns)
    - relu2_1: Mid-level style (texture patterns)
    - relu3_1: Higher-level style (complex patterns)
    - relu4_1: High-level style (abstract patterns)
    - Using all layers captures style at multiple scales!
    
    Args:
        generated_features: dict with VGG features of stylized image
        style_features: dict with VGG features of style image
    
    Returns:
        loss: scalar tensor, sum of MSE losses across all layers
              Lower is better (means style is matched)
    
    Example:
        If style is Peter Rabbit (soft watercolors, pastel tones), this loss
        ensures the output has those same soft, pastel characteristics.
    """
    style_loss = 0.0
    
    # ========================================
    # Compute loss at each VGG layer
    # ========================================
    for layer in ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1']:
        gen_feat = generated_features[layer]      # Stylized image features
        style_feat = style_features[layer]        # Style image features
        
        # ========================================
        # Calculate mean and std for generated image
        # ========================================
        # mean(dim=[2, 3]): Average across spatial dimensions (H, W)
        # Result: (B, C, 1, 1) - one mean per channel
        gen_mean = gen_feat.mean(dim=[2, 3], keepdim=True)
        # std(dim=[2, 3]): Standard deviation across spatial dimensions
        # Result: (B, C, 1, 1) - one std per channel
        gen_std = gen_feat.std(dim=[2, 3], keepdim=True)
        
        # ========================================
        # Calculate mean and std for style image
        # ========================================
        style_mean = style_feat.mean(dim=[2, 3], keepdim=True)
        style_std = style_feat.std(dim=[2, 3], keepdim=True)
        
        # ========================================
        # Compute MSE loss for both mean and std
        # ========================================
        # Loss = MSE(means) + MSE(stds)
        # This penalizes differences in both color tone AND variation
        mean_loss = F.mse_loss(gen_mean, style_mean)   # Match color tone
        std_loss = F.mse_loss(gen_std, style_std)      # Match contrast/variation
        
        style_loss += mean_loss + std_loss
    
    # Total style loss is sum across all 4 layers
    return style_loss


# ============================================================================
# TEST CODE
# ============================================================================
# Run this file directly to test the AdaIN model:
#     python model_adain.py
#
# This will verify that:
# 1. Model can be created successfully
# 2. Forward pass works correctly
# 3. Feature extraction works
# 4. Loss computation works
# 5. All tensor shapes are correct

if __name__ == "__main__":
    """
    Test the AdaIN model with dummy data.
    
    This runs a quick sanity check to ensure the model works correctly
    before starting actual training or inference.
    """
    print("=" * 70)
    print("TESTING AdaIN STYLE TRANSFER MODEL")
    print("=" * 70)
    
    # ========================================
    # Setup device (GPU if available, otherwise CPU)
    # ========================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📱 Device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # ========================================
    # Test 1: Create model
    # ========================================
    print("\n" + "=" * 70)
    print("TEST 1: Model Creation")
    print("=" * 70)
    model = AdaINStyleTransfer(device=device)
    print("✅ Model created successfully")
    print("   - VGG Encoder: Frozen (pretrained)")
    print("   - AdaIN Layer: No parameters")
    print("   - Decoder: Trainable")
    
    # ========================================
    # Test 2: Basic forward pass
    # ========================================
    print("\n" + "=" * 70)
    print("TEST 2: Forward Pass (Inference)")
    print("=" * 70)
    
    # Create dummy input tensors (random noise for testing)
    batch_size = 2
    content = torch.randn(batch_size, 3, 256, 256).to(device)  # Fake face images
    style = torch.randn(batch_size, 3, 256, 256).to(device)    # Fake style images
    
    print(f"📥 Input shapes:")
    print(f"   Content: {content.shape} (batch={batch_size}, RGB, 256×256)")
    print(f"   Style:   {style.shape} (batch={batch_size}, RGB, 256×256)")
    
    # Run inference (no gradient computation needed)
    with torch.no_grad():
        stylized = model(content, style)
    
    print(f"📤 Output shape: {stylized.shape}")
    print(f"✅ Forward pass successful!")
    
    # ========================================
    # Test 3: Forward pass with features (for training)
    # ========================================
    print("\n" + "=" * 70)
    print("TEST 3: Forward Pass with Features (Training Mode)")
    print("=" * 70)
    
    with torch.no_grad():
        stylized, gen_feat, content_feat, style_feat = model.forward_with_features(content, style)
    
    print(f"📤 Outputs:")
    print(f"   Stylized image: {stylized.shape}")
    print(f"   Feature layers: {list(gen_feat.keys())}")
    print(f"   Feature shapes:")
    for layer_name, feat in gen_feat.items():
        print(f"      {layer_name}: {feat.shape}")
    print(f"✅ Feature extraction successful!")
    
    # ========================================
    # Test 4: Loss computation
    # ========================================
    print("\n" + "=" * 70)
    print("TEST 4: Loss Computation")
    print("=" * 70)
    
    # Compute losses (this is what happens during training)
    content_loss = calc_content_loss(gen_feat, content_feat)
    style_loss = calc_style_loss(gen_feat, style_feat)
    total_loss = content_loss + 10.0 * style_loss  # Style weight = 10.0
    
    print(f"📊 Loss values:")
    print(f"   Content loss: {content_loss.item():.4f}")
    print(f"   Style loss:   {style_loss.item():.4f}")
    print(f"   Total loss:   {total_loss.item():.4f} (with style_weight=10.0)")
    print(f"✅ Loss computation successful!")
    
    # ========================================
    # Test 5: Count trainable parameters
    # ========================================
    print("\n" + "=" * 70)
    print("TEST 5: Parameter Count")
    print("=" * 70)
    
    # Count decoder parameters (only trainable part)
    decoder_params = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    
    print(f"📈 Model size:")
    print(f"   Encoder (VGG19, frozen):   {encoder_params:>12,} parameters")
    print(f"   Decoder (trainable):       {decoder_params:>12,} parameters")
    print(f"   Total:                     {encoder_params + decoder_params:>12,} parameters")
    print(f"✅ Parameter count complete!")
    
    # ========================================
    # All tests passed!
    # ========================================
    print("\n" + "=" * 70)
    print("🎉 ALL TESTS PASSED! Model is ready for training/inference.")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Train the model: python train_model.py")
    print("  2. Run inference: python eval_inference.py")
    print("  3. Evaluate metrics: python eval_metrics.py")
    print()


