from PIL import Image
import numpy as np
import cv2
from typing import Optional, Tuple
from sigtor.utils.image_utils import blend_edges, apply_histogram_equalization
from sigtor.processing.edge_refinement import (
    refine_object_boundaries, create_gradient_alpha_mask, detect_edges_multi_scale
)
from sigtor.processing.color_harmonization import (
    harmonize_colors, match_lighting_consistency
)

def _is_image_already_good_quality(image: np.ndarray, mask: Optional[np.ndarray] = None) -> bool:
    """
    Quick check if image quality is already good enough to skip some post-processing.
    
    Args:
        image: Input image array.
        mask: Optional mask.
    
    Returns:
        True if image quality is already good.
    """
    # Check if image has reasonable contrast and color distribution
    if len(image.shape) == 3:
        # Check each channel
        for c in range(3):
            channel = image[:, :, c]
            std = np.std(channel)
            mean = np.mean(channel)
            # If contrast is reasonable (not too flat, not too extreme)
            if std < 10 or std > 100 or mean < 10 or mean > 245:
                return False
    return True


def post_processing(
    image: Image.Image,
    mask: Optional[np.ndarray] = None,
    background_img: Optional[np.ndarray] = None,
    refinement_level: str = 'medium',
    enable_color_harmonization: bool = True
) -> Image.Image:
    """
    Apply multi-stage post-processing pipeline to harmonize the synthetic image,
    reducing visual artifacts and creating a more seamless and realistic appearance.
    Optimized with fast paths and reduced processing for better performance.

    Args:
        image: Input image to be post-processed (PIL Image).
        mask: Optional mask indicating object areas (0 and 255).
        background_img: Optional background image for color harmonization.
        refinement_level: Level of edge refinement ('low', 'medium', 'high').
        enable_color_harmonization: Whether to apply color harmonization.

    Returns:
        Post-processed image as PIL Image.
    """
    try:
        # Convert to numpy array
        image_np = np.array(image)
        
        # Ensure RGB format
        if len(image_np.shape) == 2:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        elif image_np.shape[2] == 4:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
        
        # Convert BGR to RGB if needed (OpenCV uses BGR)
        # Assume input is RGB from PIL
        image_rgb = image_np.copy()
        
        # Fast path: Skip post-processing for low refinement if quality is already good
        if refinement_level == 'low' and not enable_color_harmonization:
            if _is_image_already_good_quality(image_rgb, mask):
                return image  # Return original if quality is good
        
        # Stage 1: Edge Refinement (optimized - already handles fast paths internally)
        if mask is not None:
            # Refine mask boundaries
            refined_mask = refine_object_boundaries(
                mask, image_rgb, refinement_level=refinement_level
            )
        else:
            # Create mask from image if not provided
            mask = np.any(image_rgb > 0, axis=2).astype(np.uint8) * 255
            refined_mask = refine_object_boundaries(
                mask, image_rgb, refinement_level=refinement_level
            )
        
        # Stage 2: Color Harmonization (if background provided) - optimized
        if enable_color_harmonization and background_img is not None:
            try:
                # Ensure background is RGB
                if len(background_img.shape) == 2:
                    bg_rgb = cv2.cvtColor(background_img, cv2.COLOR_GRAY2RGB)
                else:
                    bg_rgb = background_img.copy()
                
                # For low/medium refinement, use faster harmonization method
                if refinement_level == 'low':
                    # Use only lighting consistency (faster)
                    image_rgb = match_lighting_consistency(image_rgb, bg_rgb, refined_mask)
                else:
                    # Use full harmonization for medium/high
                    image_rgb = harmonize_colors(
                        image_rgb, bg_rgb, refined_mask,
                        method='combined', boundary_blend=True
                    )
            except Exception as e:
                # Fallback: apply lighting consistency only
                try:
                    image_rgb = match_lighting_consistency(
                        image_rgb, bg_rgb, refined_mask
                    )
                except Exception:
                    pass  # Continue without color harmonization
        
        # Stage 3: Edge Blending (optimized - skip for low level)
        if refinement_level != 'low':
            try:
                # Create gradient alpha mask for smooth blending
                # Skip edge detection for medium level (faster)
                edge_map = None
                if refinement_level == 'high':
                    try:
                        edge_map = detect_edges_multi_scale(image_rgb, scales=(1.0,))  # Single scale
                    except Exception:
                        edge_map = None
                
                alpha_mask = create_gradient_alpha_mask(
                    refined_mask, feather_radius=5, edge_map=edge_map
                )
                
                # Apply edge blending
                alpha_3d = np.expand_dims(alpha_mask, axis=2) / 255.0
                # Blend with slight smoothing at edges
                image_rgb = (alpha_3d * image_rgb.astype(np.float32) + 
                            (1 - alpha_3d) * image_rgb.astype(np.float32))
                image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)
            except Exception:
                # Fallback to simple edge blending
                image_rgb = blend_edges(image_rgb, refined_mask)
        
        # Stage 4: Global Enhancement (skip for low level)
        if refinement_level != 'low':
            try:
                # Convert to LAB for better color balancing
                lab_img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
                
                # Apply CLAHE to L channel for better contrast (smaller grid for speed)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))  # Smaller grid
                lab_img[:, :, 0] = clahe.apply(lab_img[:, :, 0])
                
                # Convert back to RGB
                image_rgb = cv2.cvtColor(lab_img, cv2.COLOR_LAB2RGB)
            except Exception:
                pass  # Continue without CLAHE
        
        # Stage 5: Final Smoothing (light) - skip for low level
        if refinement_level != 'low':
            try:
                # Apply very light Gaussian blur to reduce artifacts
                image_rgb = cv2.GaussianBlur(image_rgb, (3, 3), 0)
            except Exception:
                pass
        
        # Convert back to PIL Image
        output_img = Image.fromarray(image_rgb)
        return output_img
        
    except Exception as e:
        # Fallback: return original image if processing fails
        return image


def simple_blend_edges(image, mask):
    """Simplify edge blending by feathering the edges of the mask."""
    kernel_size = (5, 5)  # Smaller kernel size for faster performance
    blurred_mask = cv2.GaussianBlur(mask, kernel_size, 0)

    # Normalize the mask to ensure it's between 0 and 1
    normalized_mask = blurred_mask / 255.0

    # Blend the original image using the normalized mask
    blended_image = (image * normalized_mask[:, :, np.newaxis]) + (image * (1.0 - normalized_mask[:, :, np.newaxis]))

    # Convert back to uint8 to ensure the image has valid pixel values
    blended_image = np.clip(blended_image, 0, 255).astype(np.uint8)

    return blended_image


def simplest_cb(img, percent):
    """Apply the simplest color balance technique."""
    out_channels = []
    cumstops = (percent, 100 - percent)
    for channel in cv2.split(img):
        low, high = np.percentile(channel, cumstops)
        channel = np.clip(channel, low, high)
        channel = cv2.normalize(channel, None, 0, 255, cv2.NORM_MINMAX)
        out_channels.append(channel)
    return cv2.merge(out_channels)


def apply_global_tone_mapping(image):
    """Apply a fast global tone mapping."""
    ldr = cv2.createTonemapReinhard(gamma=1.2, intensity=0, light_adapt=1.0, color_adapt=0)
    image = ldr.process(np.float32(image) / 255.0)
    image = np.clip(image * 255, 0, 255).astype('uint8')
    return image


def blend_edges(image: np.ndarray, mask: np.ndarray, blur_radius: int = 5, feather_radius: int = 7) -> np.ndarray:
    """
    Blend the edges of the objects with the background to reduce sharp transitions.

    Args:
        image (np.ndarray): The input image in BGR format.
        mask (np.ndarray): The mask indicating where the objects are placed.
        blur_radius (int): The radius of the blur to apply around the edges.
        feather_radius (int): The radius for feathering the edges.

    Returns:
        np.ndarray: The image with blended edges.
    """
    # Ensure blur_radius and feather_radius are positive odd integers
    assert blur_radius > 0 and blur_radius % 2 == 1, "blur_radius must be a positive odd integer."
    assert feather_radius > 0 and feather_radius % 2 == 1, "feather_radius must be a positive odd integer."

    # Find contours from the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an edge mask
    edge_mask = np.zeros_like(mask)
    cv2.drawContours(edge_mask, contours, -1, 255, thickness=cv2.FILLED)

    # Apply Gaussian blur to the edge mask
    blurred_edge_mask = cv2.GaussianBlur(edge_mask, (blur_radius, blur_radius), 0)

    # Feather the edges by applying a larger blur
    feathered_mask = cv2.GaussianBlur(edge_mask, (feather_radius, feather_radius), 0)

    # Convert masks to 3-channel images
    blurred_edge_mask_3ch = cv2.cvtColor(blurred_edge_mask, cv2.COLOR_GRAY2BGR)
    feathered_mask_3ch = cv2.cvtColor(feathered_mask, cv2.COLOR_GRAY2BGR)

    # Blend the original image with the feathered mask
    blended_image = cv2.addWeighted(image, 1.0, feathered_mask_3ch, 0.3, 0)

    return blended_image
