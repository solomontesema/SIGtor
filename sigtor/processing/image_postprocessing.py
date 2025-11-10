from PIL import Image
import numpy as np
import cv2
from sigtor.utils.image_utils import blend_edges, apply_histogram_equalization

def post_processing(image: Image.Image, mask=None) -> Image.Image:
    """
    Apply optimized post-processing techniques to harmonize the synthetic image,
    reducing visual artifacts and creating a more seamless and realistic appearance.

    Args:
        image (Image.Image): The input image to be post-processed.
        mask (np.ndarray or None): Optional mask indicating areas to focus on for blending.

    Returns:
        Image.Image: The post-processed image.
    """
    image = np.array(image)

    # # Convert to LAB color space for more effective color balancing
    lab_img = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab_img[:, :, 0] = clahe.apply(lab_img[:, :, 0])
    image = cv2.cvtColor(lab_img, cv2.COLOR_LAB2BGR)

     # Apply histogram equalization to improve contrast
    # image = apply_histogram_equalization(image)

    # Reduce Gaussian blur kernel size for faster edge smoothing
    image = cv2.GaussianBlur(image, (3, 3), 0)

    # Simplify edge blending using a basic feathering approach
    if mask is None:
        mask = np.any(image > 0, axis=2).astype(np.uint8) * 255
    
    # image = simple_blend_edges(image, mask)
    image = blend_edges(image, mask)

    # Apply simplest color balance with lower precision for speed
    # image = simplest_cb(image, percent=5)

    # Apply a fast global tone mapping
    # image = apply_global_tone_mapping(image)

    # Convert back to PIL Image for consistency
    output_img = Image.fromarray(image)

    return output_img


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
