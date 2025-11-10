"""
Adaptive blending method selection based on context analysis.

This module selects optimal blending methods and parameters based on
object and background characteristics.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any
from PIL import Image

from sigtor.processing.context_analysis import (
    analyze_image_context, compute_compatibility_score, ImageContext
)
from sigtor.processing.edge_refinement import (
    detect_edges_multi_scale, create_adaptive_feather_mask
)


def select_optimal_blending_method(
    object_img: np.ndarray,
    background_img: np.ndarray,
    mask: np.ndarray,
    object_context: Optional[ImageContext] = None,
    background_context: Optional[ImageContext] = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Select optimal blending method based on object and background analysis.
    
    Args:
        object_img: Object image to blend (RGB).
        background_img: Background image (RGB).
        mask: Object mask (0 and 255).
        object_context: Optional pre-computed object context.
        background_context: Optional pre-computed background context.
    
    Returns:
        Tuple of (method_name, parameters_dict).
    """
    # Analyze contexts if not provided
    if object_context is None:
        object_context = analyze_image_context(object_img, mask)
    if background_context is None:
        background_context = analyze_image_context(background_img)
    
    # Compute compatibility
    compatibility = compute_compatibility_score(object_context, background_context)
    
    # Analyze edge characteristics
    object_edges = detect_edges_multi_scale(object_img)
    background_edges = detect_edges_multi_scale(background_img)
    
    object_edge_density = np.sum(object_edges > 0) / object_edges.size
    background_edge_density = np.sum(background_edges > 0) / background_edges.size
    
    # Analyze color characteristics
    object_color_var = np.var(cv2.cvtColor(object_img, cv2.COLOR_RGB2LAB))
    background_color_var = np.var(cv2.cvtColor(background_img, cv2.COLOR_RGB2LAB))
    
    # Decision logic
    method = 'SoftPaste'  # Default
    params = {}
    
    # High compatibility and similar edge density -> use seamless clone
    if compatibility['overall'] > 0.75 and abs(object_edge_density - background_edge_density) < 0.05:
        if object_edge_density > 0.01 and background_edge_density > 0.01:
            method = 'NormalClone'
            params = {'mode': cv2.NORMAL_CLONE}
        elif object_color_var > background_color_var * 0.5:
            method = 'MixedClone'
            params = {'mode': cv2.MIXED_CLONE}
    
    # High edge density difference -> use soft paste with adaptive feathering
    elif abs(object_edge_density - background_edge_density) > 0.1:
        method = 'SoftPaste'
        # Adaptive feather based on edge difference
        edge_diff = abs(object_edge_density - background_edge_density)
        feather = int(5 + edge_diff * 20)
        params = {'feather': min(15, max(3, feather))}
    
    # Low compatibility -> use color harmonization + soft paste
    elif compatibility['overall'] < 0.5:
        method = 'HarmonizedSoftPaste'
        params = {
            'feather': 8,
            'color_harmonize': True,
            'harmonize_method': 'combined'
        }
    
    # Similar texture complexity -> use seamless clone
    elif abs(object_context.texture.complexity - background_context.texture.complexity) < 0.2:
        if object_edge_density > 0.01:
            method = 'NormalClone'
            params = {'mode': cv2.NORMAL_CLONE}
        else:
            method = 'SoftPaste'
            params = {'feather': 5}
    
    # Default: soft paste with adaptive parameters
    else:
        method = 'SoftPaste'
        # Adjust feather based on object size
        object_area = np.sum(mask > 127)
        image_area = mask.shape[0] * mask.shape[1]
        size_ratio = object_area / image_area if image_area > 0 else 0.0
        feather = int(3 + size_ratio * 10)
        params = {'feather': min(12, max(3, feather))}
    
    return method, params


def compute_optimal_center(mask: np.ndarray) -> Tuple[int, int]:
    """
    Compute optimal center point for seamless cloning.
    
    Args:
        mask: Object mask (0 and 255).
    
    Returns:
        Center coordinates (x, y).
    """
    # Find centroid of mask
    moments = cv2.moments(mask)
    if moments['m00'] > 0:
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
    else:
        # Fallback to center of bounding box
        coords = np.where(mask > 127)
        if len(coords[0]) > 0:
            cy = int(np.mean(coords[0]))
            cx = int(np.mean(coords[1]))
        else:
            h, w = mask.shape
            cx, cy = w // 2, h // 2
    
    return cx, cy


def adaptive_blend(
    object_img: np.ndarray,
    background_img: np.ndarray,
    mask: np.ndarray,
    method: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None
) -> Image.Image:
    """
    Apply adaptive blending with optimal method and parameters.
    
    Args:
        object_img: Object image to blend (RGB).
        background_img: Background image (RGB).
        mask: Object mask (0 and 255).
        method: Optional specific method to use. If None, selected automatically.
        params: Optional parameters for blending method.
    
    Returns:
        Blended image as PIL Image.
    """
    # Select method if not provided
    if method is None:
        method, auto_params = select_optimal_blending_method(object_img, background_img, mask)
        if params is None:
            params = auto_params
        else:
            params = {**auto_params, **params}
    elif params is None:
        params = {}
    
    # Apply selected blending method
    if method == 'SoftPaste':
        from sigtor.processing.image_composition import soft_paste
        feather = params.get('feather', 5)
        result = soft_paste(object_img, background_img, mask, feather=feather)
        
    elif method == 'HarmonizedSoftPaste':
        from sigtor.processing.color_harmonization import harmonize_colors
        from sigtor.processing.image_composition import soft_paste
        
        # First harmonize colors
        harmonize_method = params.get('harmonize_method', 'combined')
        harmonized = harmonize_colors(object_img, background_img, mask, 
                                     method=harmonize_method, boundary_blend=True)
        
        # Then apply soft paste
        feather = params.get('feather', 8)
        result = soft_paste(harmonized, background_img, mask, feather=feather)
        
    elif method in ['NormalClone', 'MixedClone', 'MonochromeTransfer']:
        from sigtor.processing.image_composition import seamlessclone
        
        # Get center point
        center = compute_optimal_center(mask)
        
        # Get mode
        mode = params.get('mode', cv2.NORMAL_CLONE)
        if method == 'MixedClone':
            mode = cv2.MIXED_CLONE
        elif method == 'MonochromeTransfer':
            mode = cv2.MONOCHROME_TRANSFER
        
        result = seamlessclone(object_img, background_img, mask, center, mode)
        
    elif method == 'AlphaBlend':
        from sigtor.processing.image_composition import simple_paste_using_alpha
        alpha = params.get('alpha', 0.9)
        result = simple_paste_using_alpha(object_img, background_img, mask, alpha=alpha)
        
    else:
        # Fallback to soft paste
        from sigtor.processing.image_composition import soft_paste
        result = soft_paste(object_img, background_img, mask, feather=5)
    
    return result


def get_adaptive_feather_radius(
    mask: np.ndarray,
    object_img: Optional[np.ndarray] = None,
    background_img: Optional[np.ndarray] = None,
    base_radius: int = 5
) -> int:
    """
    Compute adaptive feather radius based on object and background characteristics.
    
    Args:
        mask: Object mask (0 and 255).
        object_img: Optional object image for analysis.
        background_img: Optional background image for analysis.
        base_radius: Base feather radius.
    
    Returns:
        Adaptive feather radius.
    """
    # Compute object size
    object_area = np.sum(mask > 127)
    image_area = mask.shape[0] * mask.shape[1]
    size_ratio = object_area / image_area if image_area > 0 else 0.0
    
    # Adjust based on size
    radius = int(base_radius * (1.0 + size_ratio))
    
    # Adjust based on edge characteristics if images provided
    if object_img is not None and background_img is not None:
        try:
            object_edges = detect_edges_multi_scale(object_img)
            background_edges = detect_edges_multi_scale(background_img)
            
            object_edge_density = np.sum(object_edges > 0) / object_edges.size
            background_edge_density = np.sum(background_edges > 0) / background_edges.size
            
            # Higher edge density difference needs more feathering
            edge_diff = abs(object_edge_density - background_edge_density)
            radius = int(radius * (1.0 + edge_diff * 2))
        except Exception:
            pass
    
    # Clip to reasonable range
    return max(3, min(20, radius))


def should_apply_color_harmonization(
    object_context: ImageContext,
    background_context: ImageContext,
    threshold: float = 0.6
) -> bool:
    """
    Determine if color harmonization should be applied.
    
    Args:
        object_context: Object context.
        background_context: Background context.
        threshold: Compatibility threshold.
    
    Returns:
        True if harmonization should be applied.
    """
    compatibility = compute_compatibility_score(object_context, background_context)
    return compatibility['color'] < threshold or compatibility['overall'] < threshold

