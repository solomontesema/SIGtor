"""
Color harmonization and matching for seamless object integration.

This module provides techniques to match object colors to background
context, ensuring realistic and artifact-free compositions.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from scipy import ndimage
from scipy.stats import gaussian_kde


def match_histogram_lab(
    source: np.ndarray,
    target: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Match histogram of source image to target in LAB color space.
    
    Args:
        source: Source image (RGB).
        target: Target/reference image (RGB).
        mask: Optional mask indicating region to match.
    
    Returns:
        Color-matched source image (RGB).
    """
    # Convert to LAB color space
    source_lab = cv2.cvtColor(source, cv2.COLOR_RGB2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target, cv2.COLOR_RGB2LAB).astype(np.float32)
    
    if mask is not None:
        # Extract masked regions
        mask_bool = mask > 127
        source_region = source_lab[mask_bool]
        target_region = target_lab[mask_bool] if mask.shape == target.shape[:2] else target_lab.reshape(-1, 3)
    else:
        source_region = source_lab.reshape(-1, 3)
        target_region = target_lab.reshape(-1, 3)
    
    # Match each channel
    matched_lab = source_lab.copy()
    for channel in range(3):
        source_channel = source_region[:, channel]
        target_channel = target_region[:, channel] if len(target_region.shape) == 1 else target_region[:, channel]
        
        # Compute statistics
        source_mean = np.mean(source_channel)
        source_std = np.std(source_channel)
        target_mean = np.mean(target_channel)
        target_std = np.std(target_channel)
        
        # Avoid division by zero
        if source_std > 1e-6:
            # Match mean and std
            matched_channel = (source_lab[:, :, channel] - source_mean) * (target_std / source_std) + target_mean
        else:
            matched_channel = source_lab[:, :, channel].copy()
        
        # Clip to valid LAB range
        if channel == 0:  # L channel: 0-100
            matched_channel = np.clip(matched_channel, 0, 100)
        else:  # A, B channels: -127 to 127
            matched_channel = np.clip(matched_channel, -127, 127)
        
        matched_lab[:, :, channel] = matched_channel
    
    # Convert back to RGB
    matched_lab = matched_lab.astype(np.uint8)
    matched_rgb = cv2.cvtColor(matched_lab, cv2.COLOR_LAB2RGB)
    
    return matched_rgb


def extract_local_color_statistics(
    image: np.ndarray,
    mask: np.ndarray,
    region_size: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract color statistics from background region around object.
    
    Args:
        image: Background image (RGB).
        mask: Object mask (0 and 255).
        region_size: Size of region around object to sample.
    
    Returns:
        Mean and standard deviation of colors in background region.
    """
    # Create dilated mask to get surrounding region
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (region_size, region_size))
    dilated = cv2.dilate(mask, kernel, iterations=1)
    background_region = dilated - mask
    
    # Extract background pixels
    bg_pixels = image[background_region > 127]
    
    if len(bg_pixels) == 0:
        # Fallback: use entire image
        bg_pixels = image.reshape(-1, 3)
    
    # Compute statistics
    mean_color = np.mean(bg_pixels, axis=0)
    std_color = np.std(bg_pixels, axis=0)
    
    return mean_color, std_color


def apply_local_color_transfer(
    source: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    transfer_radius: int = 10
) -> np.ndarray:
    """
    Apply local color transfer around object boundaries.
    
    Args:
        source: Source object image (RGB).
        target: Target background image (RGB).
        mask: Object mask (0 and 255).
        transfer_radius: Radius for local color transfer.
    
    Returns:
        Color-transferred source image (RGB).
    """
    # Extract background statistics
    bg_mean, bg_std = extract_local_color_statistics(target, mask, region_size=transfer_radius * 2)
    
    # Extract object statistics
    obj_pixels = source[mask > 127]
    if len(obj_pixels) == 0:
        return source.copy()
    
    obj_mean = np.mean(obj_pixels, axis=0)
    obj_std = np.std(obj_pixels, axis=0)
    
    # Create color transfer map
    result = source.copy().astype(np.float32)
    
    # Compute distance from boundary for smooth transition
    from sigtor.processing.edge_refinement import compute_distance_transform
    dist = compute_distance_transform(mask)
    max_dist = np.max(dist) if np.max(dist) > 0 else 1.0
    transition = np.clip(dist / max_dist, 0, 1)
    transition = np.expand_dims(transition, axis=2)
    
    # Apply color transfer with smooth transition
    for c in range(3):
        channel = result[:, :, c]
        # Match mean and std
        if obj_std[c] > 1e-6:
            adjusted = (channel - obj_mean[c]) * (bg_std[c] / obj_std[c]) + bg_mean[c]
        else:
            adjusted = channel.copy()
        
        # Blend based on distance from boundary
        result[:, :, c] = transition[:, :, 0] * channel + (1 - transition[:, :, 0]) * adjusted
    
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result


def match_lighting_consistency(
    source: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray
) -> np.ndarray:
    """
    Match lighting consistency (brightness, contrast, color temperature).
    
    Args:
        source: Source object image (RGB).
        target: Target background image (RGB).
        mask: Object mask (0 and 255).
    
    Returns:
        Lighting-matched source image (RGB).
    """
    # Extract background lighting
    bg_mean, bg_std = extract_local_color_statistics(target, mask)
    
    # Extract object lighting
    obj_pixels = source[mask > 127]
    if len(obj_pixels) == 0:
        return source.copy()
    
    obj_mean = np.mean(obj_pixels, axis=0)
    obj_std = np.std(obj_pixels, axis=0)
    
    # Convert to LAB for better color temperature matching
    source_lab = cv2.cvtColor(source, cv2.COLOR_RGB2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target, cv2.COLOR_RGB2LAB).astype(np.float32)
    
    # Extract LAB statistics
    bg_lab_pixels = target_lab.reshape(-1, 3)
    bg_lab_mean = np.mean(bg_lab_pixels, axis=0)
    bg_lab_std = np.std(bg_lab_pixels, axis=0)
    
    obj_lab_pixels = source_lab[mask > 127]
    obj_lab_mean = np.mean(obj_lab_pixels, axis=0)
    obj_lab_std = np.std(obj_lab_pixels, axis=0)
    
    # Match L channel (brightness)
    if obj_lab_std[0] > 1e-6:
        source_lab[:, :, 0] = (source_lab[:, :, 0] - obj_lab_mean[0]) * (bg_lab_std[0] / obj_lab_std[0]) + bg_lab_mean[0]
    source_lab[:, :, 0] = np.clip(source_lab[:, :, 0], 0, 100)
    
    # Match A and B channels (color temperature)
    for c in [1, 2]:
        if obj_lab_std[c] > 1e-6:
            source_lab[:, :, c] = (source_lab[:, :, c] - obj_lab_mean[c]) * (bg_lab_std[c] / obj_lab_std[c]) + bg_lab_mean[c]
        source_lab[:, :, c] = np.clip(source_lab[:, :, c], -127, 127)
    
    # Convert back to RGB
    matched_lab = source_lab.astype(np.uint8)
    matched_rgb = cv2.cvtColor(matched_lab, cv2.COLOR_LAB2RGB)
    
    return matched_rgb


def harmonize_colors(
    source: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    method: str = 'histogram',
    boundary_blend: bool = True
) -> np.ndarray:
    """
    Comprehensive color harmonization pipeline.
    
    Args:
        source: Source object image (RGB).
        target: Target background image (RGB).
        mask: Object mask (0 and 255).
        method: Harmonization method ('histogram', 'local_transfer', 'lighting', 'combined').
        boundary_blend: Whether to blend colors at boundaries.
    
    Returns:
        Color-harmonized source image (RGB).
    """
    if method == 'histogram':
        result = match_histogram_lab(source, target, mask)
    elif method == 'local_transfer':
        result = apply_local_color_transfer(source, target, mask)
    elif method == 'lighting':
        result = match_lighting_consistency(source, target, mask)
    elif method == 'combined':
        # Apply multiple methods in sequence
        result = match_lighting_consistency(source, target, mask)
        result = apply_local_color_transfer(result, target, mask)
    else:
        result = source.copy()
    
    # Apply boundary blending if requested
    if boundary_blend:
        from sigtor.processing.edge_refinement import create_gradient_alpha_mask
        alpha = create_gradient_alpha_mask(mask, feather_radius=5)
        alpha_3d = np.expand_dims(alpha, axis=2) / 255.0
        
        # Blend original and harmonized at boundaries
        result = (alpha_3d * source.astype(np.float32) + 
                 (1 - alpha_3d) * result.astype(np.float32))
        result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result


def estimate_color_temperature(image: np.ndarray) -> float:
    """
    Estimate color temperature of image.
    
    Args:
        image: Input image (RGB).
    
    Returns:
        Estimated color temperature (Kelvin, approximate).
    """
    # Convert to LAB
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    
    # Extract A and B channels (color information)
    a_channel = lab[:, :, 1] - 128  # Center around 0
    b_channel = lab[:, :, 2] - 128
    
    # Compute average color shift
    avg_a = np.mean(a_channel)
    avg_b = np.mean(b_channel)
    
    # Estimate temperature (warm = positive b, cool = negative b)
    # This is a simplified approximation
    temp_shift = avg_b * 100  # Rough conversion
    
    # Base temperature (daylight ~5500K)
    base_temp = 5500.0
    estimated_temp = base_temp + temp_shift
    
    # Clip to reasonable range
    estimated_temp = np.clip(estimated_temp, 2000, 10000)
    
    return estimated_temp


def adjust_color_temperature(
    image: np.ndarray,
    target_temp: float,
    current_temp: Optional[float] = None
) -> np.ndarray:
    """
    Adjust color temperature of image.
    
    Args:
        image: Input image (RGB).
        target_temp: Target color temperature (Kelvin).
        current_temp: Current color temperature. If None, estimated.
    
    Returns:
        Color temperature adjusted image (RGB).
    """
    if current_temp is None:
        current_temp = estimate_color_temperature(image)
    
    # Compute adjustment factor
    temp_ratio = target_temp / current_temp if current_temp > 0 else 1.0
    
    # Convert to LAB
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    
    # Adjust B channel (yellow-blue axis) based on temperature
    # Warmer = more yellow (positive B), cooler = more blue (negative B)
    b_adjustment = (temp_ratio - 1.0) * 20  # Scale factor
    lab[:, :, 2] = np.clip(lab[:, :, 2] + b_adjustment, 0, 255)
    
    # Convert back to RGB
    adjusted_lab = lab.astype(np.uint8)
    adjusted_rgb = cv2.cvtColor(adjusted_lab, cv2.COLOR_LAB2RGB)
    
    return adjusted_rgb

