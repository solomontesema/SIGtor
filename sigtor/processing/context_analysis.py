"""
Context analysis for background and object characteristics.

This module analyzes background and object properties to guide
augmentation and blending decisions for realistic compositions.
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class LightingContext:
    """Lighting characteristics of an image."""
    brightness: float  # Average brightness (0-255)
    contrast: float  # Standard deviation of brightness
    color_temperature: float  # Estimated color temperature (Kelvin)
    direction: Optional[Tuple[float, float]] = None  # Light direction (if detectable)


@dataclass
class ColorContext:
    """Color characteristics of an image."""
    mean_color: np.ndarray  # Mean RGB color
    std_color: np.ndarray  # Standard deviation of RGB
    dominant_colors: List[np.ndarray]  # Dominant color palette
    saturation: float  # Average saturation


@dataclass
class TextureContext:
    """Texture characteristics of an image."""
    complexity: float  # Texture complexity score (0-1)
    edge_density: float  # Edge density (0-1)
    smoothness: float  # Smoothness score (0-1)


@dataclass
class ImageContext:
    """Complete context information for an image."""
    lighting: LightingContext
    color: ColorContext
    texture: TextureContext
    size: Tuple[int, int]  # Image dimensions


def analyze_lighting(image: np.ndarray, mask: Optional[np.ndarray] = None) -> LightingContext:
    """
    Analyze lighting characteristics of image.
    
    Args:
        image: Input image (RGB).
        mask: Optional mask to analyze specific region.
    
    Returns:
        LightingContext with lighting information.
    """
    if mask is not None:
        region = image[mask > 127] if len(mask.shape) == 2 else image
    else:
        region = image.reshape(-1, 3)
    
    # Convert to LAB for better lighting analysis
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel = lab[:, :, 0].astype(np.float32)
    
    if mask is not None and len(mask.shape) == 2:
        l_region = l_channel[mask > 127]
    else:
        l_region = l_channel.flatten()
    
    # Compute brightness and contrast
    brightness = np.mean(l_region)
    contrast = np.std(l_region)
    
    # Estimate color temperature
    from sigtor.processing.color_harmonization import estimate_color_temperature
    color_temp = estimate_color_temperature(image)
    
    # Try to estimate light direction (simplified)
    # This is a basic implementation - could be enhanced
    direction = None
    try:
        # Use gradient to estimate light direction
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Average gradient direction (simplified)
        avg_grad_x = np.mean(grad_x)
        avg_grad_y = np.mean(grad_y)
        
        if abs(avg_grad_x) > 1e-6 or abs(avg_grad_y) > 1e-6:
            direction = (float(avg_grad_x), float(avg_grad_y))
    except Exception:
        pass
    
    return LightingContext(
        brightness=brightness,
        contrast=contrast,
        color_temperature=color_temp,
        direction=direction
    )


def analyze_color(image: np.ndarray, mask: Optional[np.ndarray] = None, 
                  num_dominant: int = 5) -> ColorContext:
    """
    Analyze color characteristics of image.
    
    Args:
        image: Input image (RGB).
        mask: Optional mask to analyze specific region.
        num_dominant: Number of dominant colors to extract.
    
    Returns:
        ColorContext with color information.
    """
    if mask is not None and len(mask.shape) == 2:
        region = image[mask > 127]
    else:
        region = image.reshape(-1, 3)
    
    # Compute mean and std
    mean_color = np.mean(region, axis=0)
    std_color = np.std(region, axis=0)
    
    # Extract dominant colors using K-means (simplified)
    dominant_colors = extract_dominant_colors(image, mask, num_dominant)
    
    # Compute saturation
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    if mask is not None and len(mask.shape) == 2:
        saturation = np.mean(hsv[:, :, 1][mask > 127])
    else:
        saturation = np.mean(hsv[:, :, 1])
    
    return ColorContext(
        mean_color=mean_color,
        std_color=std_color,
        dominant_colors=dominant_colors,
        saturation=saturation
    )


def extract_dominant_colors(image: np.ndarray, mask: Optional[np.ndarray] = None,
                           num_colors: int = 5) -> List[np.ndarray]:
    """
    Extract dominant colors from image using K-means clustering.
    
    Args:
        image: Input image (RGB).
        mask: Optional mask to analyze specific region.
        num_colors: Number of dominant colors to extract.
    
    Returns:
        List of dominant colors (RGB).
    """
    if mask is not None and len(mask.shape) == 2:
        pixels = image[mask > 127]
    else:
        pixels = image.reshape(-1, 3)
    
    if len(pixels) < num_colors:
        # Not enough pixels, return mean color
        return [np.mean(pixels, axis=0).astype(np.uint8)]
    
    # Reshape for K-means
    pixels_float = pixels.astype(np.float32)
    
    # Apply K-means
    try:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(
            pixels_float, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )
        
        # Sort by frequency
        unique, counts = np.unique(labels, return_counts=True)
        sorted_indices = np.argsort(counts)[::-1]
        
        dominant = [centers[i].astype(np.uint8) for i in sorted_indices[:num_colors]]
        return dominant
    except Exception:
        # Fallback: return mean color
        return [np.mean(pixels, axis=0).astype(np.uint8)]


def analyze_texture(image: np.ndarray, mask: Optional[np.ndarray] = None) -> TextureContext:
    """
    Analyze texture characteristics of image.
    
    Args:
        image: Input image (RGB).
        mask: Optional mask to analyze specific region.
    
    Returns:
        TextureContext with texture information.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Compute edge density
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    # Compute texture complexity using variance of Laplacian
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Normalize to 0-1 range (rough approximation)
    complexity = min(1.0, laplacian_var / 1000.0)
    
    # Compute smoothness (inverse of complexity)
    smoothness = 1.0 - min(1.0, complexity)
    
    return TextureContext(
        complexity=complexity,
        edge_density=edge_density,
        smoothness=smoothness
    )


def analyze_image_context(image: np.ndarray, 
                         mask: Optional[np.ndarray] = None) -> ImageContext:
    """
    Comprehensive context analysis of image.
    
    Args:
        image: Input image (RGB).
        mask: Optional mask to analyze specific region.
    
    Returns:
        ImageContext with complete analysis.
    """
    lighting = analyze_lighting(image, mask)
    color = analyze_color(image, mask)
    texture = analyze_texture(image, mask)
    size = (image.shape[1], image.shape[0])
    
    return ImageContext(
        lighting=lighting,
        color=color,
        texture=texture,
        size=size
    )


def compute_compatibility_score(
    object_context: ImageContext,
    background_context: ImageContext
) -> Dict[str, float]:
    """
    Compute compatibility score between object and background contexts.
    
    Args:
        object_context: Context of object to be placed.
        background_context: Context of background.
    
    Returns:
        Dictionary with compatibility scores for different aspects.
    """
    scores = {}
    
    # Lighting compatibility
    brightness_diff = abs(object_context.lighting.brightness - 
                          background_context.lighting.brightness) / 255.0
    contrast_diff = abs(object_context.lighting.contrast - 
                        background_context.lighting.contrast) / 255.0
    temp_diff = abs(object_context.lighting.color_temperature - 
                   background_context.lighting.color_temperature) / 8000.0
    
    lighting_score = 1.0 - (brightness_diff * 0.4 + contrast_diff * 0.3 + temp_diff * 0.3)
    scores['lighting'] = max(0.0, min(1.0, lighting_score))
    
    # Color compatibility
    color_diff = np.linalg.norm(object_context.color.mean_color - 
                               background_context.color.mean_color) / (255.0 * np.sqrt(3))
    saturation_diff = abs(object_context.color.saturation - 
                          background_context.color.saturation) / 255.0
    
    color_score = 1.0 - (color_diff * 0.7 + saturation_diff * 0.3)
    scores['color'] = max(0.0, min(1.0, color_score))
    
    # Texture compatibility
    complexity_diff = abs(object_context.texture.complexity - 
                         background_context.texture.complexity)
    edge_diff = abs(object_context.texture.edge_density - 
                   background_context.texture.edge_density)
    
    texture_score = 1.0 - (complexity_diff * 0.6 + edge_diff * 0.4)
    scores['texture'] = max(0.0, min(1.0, texture_score))
    
    # Overall compatibility
    scores['overall'] = (scores['lighting'] * 0.4 + 
                        scores['color'] * 0.4 + 
                        scores['texture'] * 0.2)
    
    return scores


def suggest_augmentations(
    object_context: ImageContext,
    background_context: ImageContext,
    compatibility_threshold: float = 0.7
) -> Dict[str, any]:
    """
    Suggest augmentations to improve object-background compatibility.
    
    Args:
        object_context: Context of object to be placed.
        background_context: Context of background.
        compatibility_threshold: Threshold for acceptable compatibility.
    
    Returns:
        Dictionary with suggested augmentation parameters.
    """
    suggestions = {}
    compatibility = compute_compatibility_score(object_context, background_context)
    
    # Brightness adjustment
    brightness_diff = background_context.lighting.brightness - object_context.lighting.brightness
    if abs(brightness_diff) > 20:  # Significant difference
        suggestions['brightness_adjust'] = brightness_diff / 255.0
    
    # Contrast adjustment
    contrast_diff = background_context.lighting.contrast - object_context.lighting.contrast
    if abs(contrast_diff) > 15:
        suggestions['contrast_adjust'] = contrast_diff / 255.0
    
    # Color temperature adjustment
    temp_diff = background_context.lighting.color_temperature - object_context.lighting.color_temperature
    if abs(temp_diff) > 500:
        suggestions['color_temp_adjust'] = temp_diff
    
    # Saturation adjustment
    sat_diff = background_context.color.saturation - object_context.color.saturation
    if abs(sat_diff) > 20:
        suggestions['saturation_adjust'] = sat_diff / 255.0
    
    # Blur adjustment (match texture complexity)
    if object_context.texture.complexity > background_context.texture.complexity + 0.2:
        suggestions['blur'] = True
        suggestions['blur_amount'] = min(5, int((object_context.texture.complexity - 
                                               background_context.texture.complexity) * 10))
    
    return suggestions

