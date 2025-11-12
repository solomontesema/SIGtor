"""
Advanced edge detection and refinement for seamless object blending.

This module provides sophisticated edge processing techniques to eliminate
boundary artifacts in copy-paste augmentation.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from scipy import ndimage


def detect_edges_multi_scale(
    image: np.ndarray,
    low_threshold: Optional[int] = None,
    high_threshold: Optional[int] = None,
    scales: Tuple[float, ...] = (1.0, 0.5, 2.0)
) -> np.ndarray:
    """
    Detect edges using multi-scale Canny edge detection with adaptive thresholds.
    
    Args:
        image: Input image (grayscale or color).
        low_threshold: Lower threshold for Canny. If None, computed adaptively.
        high_threshold: Upper threshold for Canny. If None, computed adaptively.
        scales: Scale factors for multi-scale detection.
    
    Returns:
        Combined edge map from all scales.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Compute adaptive thresholds if not provided
    if low_threshold is None or high_threshold is None:
        median = np.median(gray)
        sigma = 0.33
        low_threshold = int(max(0, (1.0 - sigma) * median))
        high_threshold = int(min(255, (1.0 + sigma) * median))
    
    edge_maps = []
    for scale in scales:
        if scale != 1.0:
            h, w = gray.shape
            scaled = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            edges = cv2.Canny(scaled, low_threshold, high_threshold)
            edges = cv2.resize(edges, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            edges = cv2.Canny(gray, low_threshold, high_threshold)
        edge_maps.append(edges)
    
    # Combine edge maps
    combined = np.maximum.reduce(edge_maps)
    return combined


def compute_distance_transform(mask: np.ndarray) -> np.ndarray:
    """
    Compute distance transform from mask boundaries.
    
    Args:
        mask: Binary mask (0 and 255).
    
    Returns:
        Distance transform where values represent distance from boundary.
    """
    binary = (mask > 127).astype(np.uint8)
    
    # Compute distance from boundary
    dist_inner = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    dist_outer = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 5)
    
    # Combine to get distance from nearest boundary
    dist = np.minimum(dist_inner, dist_outer)
    
    return dist


def create_adaptive_feather_mask(
    mask: np.ndarray,
    edge_map: Optional[np.ndarray] = None,
    base_feather: int = 5,
    max_feather: int = 15,
    edge_sensitivity: float = 0.5
) -> np.ndarray:
    """
    Create adaptive feather mask based on edge strength and object size.
    
    Args:
        mask: Binary mask (0 and 255).
        edge_map: Optional edge map for edge-aware feathering.
        base_feather: Base feather radius.
        max_feather: Maximum feather radius.
        edge_sensitivity: How much edge strength affects feathering (0-1).
    
    Returns:
        Feather mask with values in [0, 255].
    """
    # Compute object size
    object_area = np.sum(mask > 127)
    image_area = mask.shape[0] * mask.shape[1]
    size_ratio = object_area / image_area if image_area > 0 else 0.0
    
    # Adjust feather based on size (larger objects need more feathering)
    size_factor = min(2.0, 1.0 + size_ratio * 2.0)
    adaptive_feather = int(base_feather * size_factor)
    adaptive_feather = min(max_feather, max(base_feather, adaptive_feather))
    
    # Create base feather mask
    dist = compute_distance_transform(mask)
    max_dist = np.max(dist) if np.max(dist) > 0 else 1.0
    normalized_dist = dist / max_dist
    
    # Adjust based on edge strength if edge map provided
    if edge_map is not None:
        edge_strength = edge_map.astype(np.float32) / 255.0
        # Stronger edges need less feathering
        edge_factor = 1.0 - edge_sensitivity * edge_strength
        normalized_dist = normalized_dist * edge_factor
    
    # Create feather mask
    feather_mask = (normalized_dist * 255).astype(np.uint8)
    
    # Apply Gaussian blur for smooth transition
    feather_mask = cv2.GaussianBlur(feather_mask, (adaptive_feather * 2 + 1, adaptive_feather * 2 + 1), 0)
    
    return feather_mask


def refine_mask_with_edges(
    mask: np.ndarray,
    edge_map: Optional[np.ndarray] = None,
    method: str = 'distance_transform'
) -> np.ndarray:
    """
    Refine mask using edge information for smoother boundaries.
    
    Args:
        mask: Binary mask (0 and 255).
        edge_map: Optional edge map for edge-aware refinement.
        method: Refinement method ('distance_transform', 'morphological', 'contour').
    
    Returns:
        Refined mask.
    """
    if method == 'distance_transform':
        # Use advanced morphological operations for high-quality edge refinement
        # This method preserves the object while creating smooth, refined boundaries
        binary = (mask > 127).astype(np.uint8)
        original_area = np.sum(binary)
        
        # Optimized: Use smaller kernels and fewer iterations for speed
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Stage 1: Close small holes and gaps (preserves object shape) - reduced iterations
        refined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small, iterations=1)  # Was 2
        
        # Stage 2: Smooth boundaries with opening (removes small protrusions)
        refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel_small, iterations=1)
        
        # Stage 3: Apply additional smoothing with medium kernel (only if needed)
        # Skip this stage for speed - can be enabled if quality is critical
        # refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel_medium, iterations=1)
        # refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel_medium, iterations=1)
        
        # Safety check: ensure we preserve at least 80% of the original mask area
        refined_area = np.sum(refined > 127)
        if refined_area < original_area * 0.8 and original_area > 0:
            # Fall back to less aggressive processing
            refined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small, iterations=1)  # Was 2
            refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel_small, iterations=1)
        
        # Final check: if still too much is lost, use original mask
        final_area = np.sum(refined > 127)
        if final_area < original_area * 0.7 and original_area > 0:
            refined = mask.copy()  # Preserve original if refinement is too aggressive
        
    elif method == 'morphological':
        # Edge-aware morphological operations
        if edge_map is not None:
            # Strong edges: preserve, weak edges: smooth
            edge_strength = edge_map.astype(np.float32) / 255.0
            kernel_size = int(3 + edge_strength.mean() * 5)
            kernel_size = kernel_size | 1  # Ensure odd
        else:
            kernel_size = 3
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        # Close small holes
        refined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        # Smooth boundaries
        refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel, iterations=1)
        
    elif method == 'contour':
        # Contour-based refinement
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        refined = np.zeros_like(mask)
        
        for contour in contours:
            # Approximate contour for smoother shape
            epsilon = 0.002 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            cv2.drawContours(refined, [approx], -1, 255, -1)
        
        # Fill holes
        refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, 
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), 
                                 iterations=2)
    else:
        refined = mask.copy()
    
    return refined


def apply_edge_aware_morphology(
    mask: np.ndarray,
    edge_map: np.ndarray,
    operation: str = 'erode',
    base_iterations: int = 2
) -> np.ndarray:
    """
    Apply morphological operations with edge-aware iteration count.
    
    Args:
        mask: Binary mask (0 and 255).
        edge_map: Edge map for edge-aware processing.
        operation: Morphological operation ('erode', 'dilate', 'open', 'close').
        base_iterations: Base number of iterations.
    
    Returns:
        Processed mask.
    """
    # Compute edge strength per region
    edge_strength = edge_map.astype(np.float32) / 255.0
    
    # Adjust iterations based on edge strength
    # Strong edges: fewer iterations (preserve), weak edges: more iterations (smooth)
    avg_edge_strength = np.mean(edge_strength[mask > 127]) if np.any(mask > 127) else 0.5
    iterations = max(1, int(base_iterations * (1.0 - avg_edge_strength * 0.5)))
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    if operation == 'erode':
        result = cv2.erode(mask, kernel, iterations=iterations)
    elif operation == 'dilate':
        result = cv2.dilate(mask, kernel, iterations=iterations)
    elif operation == 'open':
        result = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
    elif operation == 'close':
        result = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    else:
        result = mask.copy()
    
    return result


def _is_mask_already_smooth(mask: np.ndarray, threshold: float = 0.02) -> bool:
    """
    Check if mask boundaries are already smooth enough to skip refinement.
    
    Args:
        mask: Binary mask (0 and 255).
        threshold: Threshold for roughness (lower = stricter).
    
    Returns:
        True if mask is already smooth enough.
    """
    binary = (mask > 127).astype(np.uint8)
    if np.sum(binary) == 0:
        return True
    
    # Compute boundary roughness using gradient
    grad_x = cv2.Sobel(binary, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(binary, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Check if boundary is smooth (low gradient variance)
    boundary_pixels = gradient_magnitude[gradient_magnitude > 0]
    if len(boundary_pixels) == 0:
        return True
    
    # If gradient variance is low, mask is already smooth
    gradient_std = np.std(boundary_pixels)
    max_gradient = np.max(gradient_magnitude)
    roughness = gradient_std / max_gradient if max_gradient > 0 else 0
    
    return roughness < threshold


def refine_object_boundaries(
    mask: np.ndarray,
    image: Optional[np.ndarray] = None,
    refinement_level: str = 'medium'
) -> np.ndarray:
    """
    Comprehensive boundary refinement pipeline with optimizations.
    
    Args:
        mask: Binary mask (0 and 255).
        image: Optional source image for edge detection.
        refinement_level: Level of refinement ('low', 'medium', 'high').
    
    Returns:
        Refined mask with smooth boundaries.
    """
    # Fast path: check if mask is already smooth enough (skip refinement)
    if refinement_level == 'low' and _is_mask_already_smooth(mask, threshold=0.03):
        return mask.copy()
    
    # Determine parameters based on refinement level (optimized)
    if refinement_level == 'low':
        base_feather = 3
        use_edge_detection = False
        morph_method = 'morphological'
    elif refinement_level == 'high':
        base_feather = 8
        use_edge_detection = True
        morph_method = 'distance_transform'
        # For high level, consider downscaling for speed
        use_downscaling = True
    else:  # medium
        base_feather = 5
        use_edge_detection = False  # Skip edge detection for medium (faster)
        morph_method = 'morphological'
        use_downscaling = False
    
    # Downscale for high-level refinement (faster processing)
    original_mask = mask.copy()
    scale_factor = 0.5 if (refinement_level == 'high' and use_downscaling) else 1.0
    if scale_factor < 1.0:
        h, w = mask.shape
        mask = cv2.resize(mask, (int(w * scale_factor), int(h * scale_factor)), 
                         interpolation=cv2.INTER_AREA)
    
    # Detect edges if image provided (only for high level now)
    edge_map = None
    if use_edge_detection and image is not None:
        try:
            # For high level with downscaling, also downscale image
            if scale_factor < 1.0:
                img_h, img_w = image.shape[:2]
                image_small = cv2.resize(image, (int(img_w * scale_factor), int(img_h * scale_factor)),
                                        interpolation=cv2.INTER_AREA)
                edge_map = detect_edges_multi_scale(image_small, scales=(1.0, 0.5))  # Fewer scales
            else:
                edge_map = detect_edges_multi_scale(image, scales=(1.0, 0.5))  # Fewer scales for speed
        except Exception:
            edge_map = None
    
    # Refine mask
    refined_mask = refine_mask_with_edges(mask, edge_map, method=morph_method)
    
    # Upscale if we downscaled
    if scale_factor < 1.0:
        h, w = original_mask.shape
        refined_mask = cv2.resize(refined_mask, (w, h), interpolation=cv2.INTER_LINEAR)
        # Threshold to ensure binary
        refined_mask = (refined_mask > 127).astype(np.uint8) * 255
    
    # Apply edge-aware morphology if needed (with safety checks) - only for high
    if edge_map is not None and refinement_level == 'high':
        original_area = np.sum(original_mask > 127)
        refined_mask_before = refined_mask.copy()
        
        refined_mask = apply_edge_aware_morphology(refined_mask, None,  # Skip edge_map for speed
                                                   operation='close', 
                                                   base_iterations=1)  # Reduced iterations
        
        # Safety check: ensure edge-aware morphology doesn't remove too much
        refined_area = np.sum(refined_mask > 127)
        if original_area > 0 and refined_area < original_area * 0.85:
            # If too much is lost, use the mask before edge-aware morphology
            refined_mask = refined_mask_before
    
    return refined_mask


def create_gradient_alpha_mask(
    mask: np.ndarray,
    feather_radius: int = 5,
    edge_map: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Create gradient-based alpha mask for smooth blending.
    
    Args:
        mask: Binary mask (0 and 255).
        feather_radius: Radius for gradient falloff.
        edge_map: Optional edge map for edge-aware gradient.
    
    Returns:
        Alpha mask with values in [0, 255].
    """
    # Create distance-based gradient
    dist = compute_distance_transform(mask)
    
    # Normalize to [0, 1]
    if np.max(dist) > 0:
        normalized = dist / np.max(dist)
    else:
        normalized = (mask > 127).astype(np.float32)
    
    # Apply edge-aware adjustment
    if edge_map is not None:
        edge_strength = edge_map.astype(np.float32) / 255.0
        # Reduce gradient near strong edges for sharper boundaries
        normalized = normalized * (1.0 - 0.3 * edge_strength)
    
    # Create smooth gradient
    alpha = (normalized * 255).astype(np.uint8)
    
    # Apply Gaussian blur for smoothness
    blur_size = feather_radius * 2 + 1
    alpha = cv2.GaussianBlur(alpha, (blur_size, blur_size), feather_radius / 2.0)
    
    return alpha

