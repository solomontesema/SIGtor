import copy
import os
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, List, Optional, Dict, Any
from collections import OrderedDict
import hashlib

import cv2
import numpy as np
from PIL import Image
from skimage import img_as_float, img_as_ubyte
from skimage.restoration import inpaint

from sigtor.processing.augmentation import preprocess_mask, preprocess_source_image
from sigtor.processing.data_processing import get_realcoords, convert_instance_to_binary, get_tightfit_targetsize
from sigtor.utils.image_utils import mask_to_RGB, get_colors
from sigtor.processing.image_postprocessing import post_processing
from sigtor.processing.adaptive_blending import adaptive_blend, select_optimal_blending_method
from sigtor.processing.edge_refinement import refine_object_boundaries
from sigtor.processing.context_analysis import analyze_image_context, ImageContext

# Context cache for image composition
_COMPOSITION_CONTEXT_CACHE_SIZE = 50
_composition_context_cache = OrderedDict()


def _get_image_hash(image: np.ndarray) -> str:
    """Generate hash for image for caching."""
    # Use a small sample of the image for hashing (faster)
    sample = image[::10, ::10]  # Sample every 10th pixel
    return hashlib.md5(sample.tobytes()).hexdigest()


def _get_cached_composition_context(image: np.ndarray, mask: Optional[np.ndarray] = None) -> Optional[ImageContext]:
    """Get cached context or analyze and cache."""
    img_hash = _get_image_hash(image)
    cache_key = f"{img_hash}:{mask is not None}"
    
    # Check cache
    if cache_key in _composition_context_cache:
        _composition_context_cache.move_to_end(cache_key)
        return _composition_context_cache[cache_key]
    
    # Analyze and cache
    try:
        context = analyze_image_context(image, mask)
        if len(_composition_context_cache) >= _COMPOSITION_CONTEXT_CACHE_SIZE:
            _composition_context_cache.popitem(last=False)
        _composition_context_cache[cache_key] = context
        return context
    except Exception:
        return None


def get_backgrnd_image(bckgrnd_imgs_dir: str, target_img_size: Tuple[int, int]) -> np.ndarray:
    """
    Return a background image from the specified directory, resized to the target size.
    If the directory is empty or does not exist, return a randomly colored plain RGB image.
    """
    image_paths = get_image_paths(bckgrnd_imgs_dir)

    if image_paths:
        chosen_image_path = random.choice(image_paths)
        background_image = load_and_resize_image(chosen_image_path, target_img_size)
    else:
        background_image = get_random_color_image(target_img_size)

    return background_image


def handle_background_image(bckgrnd_imgs_dir: str, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Load the background image for the composition.

    Args:
        bckgrnd_imgs_dir (str): Directory containing background images.
        target_size (Tuple[int, int]): The target size for the background image.

    Returns:
        np.ndarray: The background image.
    """
    background_img = get_backgrnd_image(bckgrnd_imgs_dir, target_size)
    return background_img


def get_random_color_image(size: Tuple[int, int]) -> np.ndarray:
    """Generate a plain RGB image of random color with the given size."""
    random_color = np.random.randint(0, 256, size=(1, 1, 3), dtype=np.uint8)
    color_image = np.tile(random_color, (size[1], size[0], 1))
    return color_image


def get_image_paths(directory: str) -> Optional[list]:
    """Retrieve all image paths from the given directory and its subdirectories."""
    if not os.path.exists(directory):
        return None
    image_formats = ('.jpg', '.png', '.jpeg')
    image_paths = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(image_formats):
                image_paths.append(os.path.join(root, f))
    return image_paths


def load_and_resize_image(image_path: str, target_size: Tuple[int, int]) -> np.ndarray:
    """Load an image using OpenCV, convert it to RGB, and resize it to the target size."""
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not load image from path: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(image_rgb, target_size[:2])
    return resized_image


def paste_object(canvas: np.ndarray, obj_img: np.ndarray, coord: np.ndarray) -> np.ndarray:
    """
    Paste an object image onto the canvas at the specified coordinates.

    Args:
        canvas (np.ndarray): The canvas where the object image will be pasted.
        obj_img (np.ndarray): The object image to be pasted.
        coord (np.ndarray): The coordinates [x1, y1, x2, y2] where the object will be placed.

    Returns:
        np.ndarray: The updated canvas with the object pasted.
    """
    x1, y1, x2, y2 = coord.astype(np.int32)
    canvas[y1:y2, x1:x2, :] = obj_img
    return canvas


def soft_paste(src, dst, mask, feather=5):
    """
    Paste the source image onto the destination with smooth feathered edges.

    Args:
        src (numpy.ndarray): Source object image.
        dst (numpy.ndarray): Destination background image.
        mask (numpy.ndarray): Object mask.
        feather (int): Feathering radius to blend edges.

    Returns:
        Pasted image as a PIL Image.
    """
    # Smooth the mask to remove hard edges
    mask = cv2.GaussianBlur(mask.astype(np.uint8), (feather, feather), 0)  
    mask = mask.astype(np.float32) / 255.0  

    # Soft alpha blending
    blended = (mask[..., None] * src) + ((1 - mask[..., None]) * dst)
    return Image.fromarray(np.uint8(blended))


def match_histogram(src, ref):
    """
    Match the histogram of the source image to the reference background.

    Args:
        src (numpy.ndarray): Source image (object).
        ref (numpy.ndarray): Reference background image.

    Returns:
        Color-matched image.
    """
    # Convert images to LAB color space
    src_lab = cv2.cvtColor(src, cv2.COLOR_RGB2LAB)
    ref_lab = cv2.cvtColor(ref, cv2.COLOR_RGB2LAB)

    # Calculate the mean and standard deviation of each channel
    src_mean, src_std = cv2.meanStdDev(src_lab)
    ref_mean, ref_std = cv2.meanStdDev(ref_lab)

    # Adjust each channel of the source image
    for i in range(3):
        src_lab[..., i] = ((src_lab[..., i] - src_mean[i]) * (ref_std[i] / src_std[i])) + ref_mean[i]

    # Clip values to valid range and convert back to RGB
    src_lab = np.clip(src_lab, 0, 255).astype(np.uint8)
    matched = cv2.cvtColor(src_lab, cv2.COLOR_LAB2RGB)

    return matched


def add_shadow(image, mask, intensity=0.5):
    """
    Adds synthetic shadows under objects.

    Args:
        image (numpy.ndarray): The object image.
        mask (numpy.ndarray): The binary mask of the object.
        intensity (float): Shadow darkness.

    Returns:
        Image with shadows applied.
    """
    shadow = np.zeros_like(image, dtype=np.uint8)
    shadow[mask != 0] = (0, 0, 0)  # Black shadow layer
    shadow = cv2.GaussianBlur(shadow, (15, 15), 10)  # Blur for softness

    return cv2.addWeighted(image, 1.0, shadow, -intensity, 0)



def match_blur(obj, bg, mask):
    """
    Matches object sharpness to background depth.

    Args:
        obj (numpy.ndarray): Object image.
        bg (numpy.ndarray): Background image.
        mask (numpy.ndarray): Object mask.

    Returns:
        Blur-matched object.
    """
    bg_blur = cv2.Laplacian(bg, cv2.CV_64F).var()
    obj_blur = cv2.Laplacian(obj, cv2.CV_64F).var()

    if obj_blur > bg_blur:  # If object is sharper than background
        blur_amount = int(abs(obj_blur - bg_blur) // 2) | 1  # Ensure odd kernel size
        obj = cv2.GaussianBlur(obj, (blur_amount, blur_amount), 0)

    return obj


def paste_objs(targetsize: Tuple[int, int], all_obj_imgs: List[Image.Image],
               selected_obj_coords: Dict[int, np.ndarray]) -> np.ndarray:
    """
    Create a new image by stitching the cutout objects onto a fresh black canvas.

    Args:
        targetsize (Tuple[int, int]): The size of the target image (width, height).
        all_obj_imgs (List[Image.Image]): List of object images to be pasted.
        selected_obj_coords (Dict[int, np.ndarray]): Dictionary of object coordinates [x1, y1, x2, y2].

    Returns:
        np.ndarray: The final image with all objects stitched together on the black canvas.
    """
    # Create a black canvas
    canvas = np.zeros((targetsize[1], targetsize[0], 3), dtype=np.uint8)

    for obj_id, coord in selected_obj_coords.items():
        obj_img = np.array(all_obj_imgs[obj_id])
        canvas = paste_object(canvas, obj_img, coord)

    return canvas


def prepare_mask_for_pasting(mask: np.ndarray) -> Image.Image:
    """
    Prepare the mask for pasting by converting it to a binary format and applying a morphological close operation.

    Args:
        mask (np.ndarray): The mask to be processed.

    Returns:
        Image.Image: The processed mask ready for pasting.
    """
    _mask = copy.deepcopy(mask).astype('uint8')
    _mask[_mask != 0] = 255

    # Apply a morphological close operation to fill small holes in the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    _mask = cv2.morphologyEx(_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return Image.fromarray(_mask)


def paste_mask(new_img_mask: np.ndarray, current_mask: np.ndarray, coord: np.ndarray, new_inst_id: int) -> np.ndarray:
    """
    Paste the current mask into the new image mask at the specified coordinates, updating instance IDs.

    Args:
        new_img_mask (np.ndarray): The image mask being built.
        current_mask (np.ndarray): The current mask to be pasted.
        coord (np.ndarray): The coordinates where the mask should be placed [x1, y1, x2, y2].
        new_inst_id (int): The starting instance ID for the current mask.

    Returns:
        np.ndarray: The updated image mask.
    """
    x1, y1, x2, y2 = coord.astype(np.int32)
    mask_region = new_img_mask[y1:y2, x1:x2]

    unique_pixel_instances = np.unique(current_mask)
    for inst_id in unique_pixel_instances:
        if inst_id == 0:  # Skip the background and border pixels
            continue
        if inst_id == 255:  # Make the the border instances white
            mask_region[current_mask == inst_id] = 255
            continue
        mask_region[current_mask == inst_id] = new_inst_id
        new_inst_id += 1

    new_img_mask[y1:y2, x1:x2] = mask_region
    return new_img_mask


def paste_masks(targetsize: Tuple[int, int], all_obj_masks: List[Image.Image],
                selected_obj_coords: Dict[int, np.ndarray]) -> np.ndarray:
    """
    Create a new segmentation mask by stitching individual object masks into their correct locations.

    Args:
        targetsize (Tuple[int, int]): The size of the target image (width, height).
        all_obj_masks (List[Image.Image]): List of object masks to be pasted.
        selected_obj_coords (Dict[int, np.ndarray]): Dictionary of object coordinates [x1, y1, x2, y2].

    Returns:
        np.ndarray: The final stitched segmentation mask.
    """
    new_img_mask = np.zeros((targetsize[1], targetsize[0]), dtype=np.uint8)
    new_inst_id = 1

    for obj_id, coord in selected_obj_coords.items():
        current_mask = np.array(all_obj_masks[obj_id])
        new_img_mask = paste_mask(new_img_mask, current_mask, coord, new_inst_id)

        # Calculate the number of unique IDs, excluding 0 and 254 if present
        unique_pixel_instances = np.unique(current_mask)
        adjustment = 0
        if 0 in unique_pixel_instances:  # Exclude the background value 0
            adjustment += 1
        if 254 in unique_pixel_instances:  # Exclude the border value 254
            adjustment += 1

        new_inst_id += len(unique_pixel_instances) - adjustment

    return new_img_mask


def calculate_new_dimensions(oh: int, ow: int, scale_min: float, scale_max: float) -> Tuple[int, int]:
    """
    Calculate new dimensions for the image based on random scaling factors.

    Args:
        oh (int): Original height of the image.
        ow (int): Original width of the image.
        scale_min (float): Minimum scaling factor.
        scale_max (float): Maximum scaling factor.

    Returns:
        Tuple[int, int]: New width and height for the image.
    """
    nw = int(np.random.uniform(scale_min, scale_max) * ow)
    nh = int(np.random.uniform(scale_min, scale_max) * oh)

    # Ensure dimensions are even
    nw = nw + 1 if nw % 2 != 0 else nw
    nh = nh + 1 if nh % 2 != 0 else nh

    return nw, nh


def calculate_offsets(nw: int, nh: int, ow: int, oh: int) -> Tuple[int, int]:
    """
    Calculate random offsets for the image placement.

    Args:
        nw (int): New width of the image.
        nh (int): New height of the image.
        ow (int): Original width of the image.
        oh (int): Original height of the image.

    Returns:
        Tuple[int, int]: x and y offsets for the image placement.
    """
    px = int(np.random.uniform(0, nw - ow))
    py = int(np.random.uniform(0, nh - oh))

    return px, py


def create_offset_canvas(nw: int, nh: int, src_img: np.ndarray, src_msk: np.ndarray, px: int, py: int) -> Tuple[
    np.ndarray, np.ndarray]:
    """
    Create a new canvas and place the source image and mask at the random offset.

    Args:
        nw (int): New width of the image.
        nh (int): New height of the image.
        src_img (np.ndarray): The source image.
        src_msk (np.ndarray): The source mask.
        px (int): x offset.
        py (int): y offset.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The offset image and mask.
    """
    offsetted_mask = np.zeros((nh, nw), dtype=np.uint8)
    offsetted_src_img = np.zeros((nh, nw, 3), dtype=np.uint8)

    oh, ow = src_img.shape[:2]
    offsetted_mask[py:py + oh, px: px + ow] = src_msk
    offsetted_src_img[py:py + oh, px:px + ow] = src_img

    return offsetted_src_img, offsetted_mask


def paste_at_random_pivot(src_img: np.ndarray, src_msk: np.ndarray, src_boxes: np.ndarray, scale_min: float = 1.0,
                          scale_max: float = 1.5, use_random_backgrnd: bool = True) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray]:
    """
    Pivot the newly created artificial source image by some random offset.

    Args:
        src_img (np.ndarray): The source image to be pasted.
        src_msk (np.ndarray): The mask of the source image.
        src_boxes (np.ndarray): The bounding boxes of objects in the source image.
        scale_min (float): Minimum scaling factor for the image.
        scale_max (float): Maximum scaling factor for the image.
        use_random_backgrnd (bool): Whether to use a random background color.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: The offsetted image, mask, and updated bounding boxes.
    """
    oh, ow = src_img.shape[:2]

    # Calculate new dimensions and offsets
    nw, nh = calculate_new_dimensions(oh, ow, scale_min, scale_max)
    px, py = calculate_offsets(nw, nh, ow, oh)

    # Create the offset canvas
    offsetted_src_img, offsetted_mask = create_offset_canvas(nw, nh, src_img, src_msk, px, py)

    if use_random_backgrnd:
        # Fill the background with a random color
        bcgrnd_color = np.random.randint(0, 256, size=3)
        new_backgrnd = np.full((nh, nw, 3), bcgrnd_color, dtype=np.uint8)
        offsetted_src_img = simple_paste(offsetted_src_img, new_backgrnd, offsetted_mask)
        offsetted_src_img = np.array(offsetted_src_img)

    # Update the bounding box coordinates
    src_boxes[..., 0:4] += np.array([px, py, px, py]).reshape(-1, 4)

    return offsetted_src_img, offsetted_mask, src_boxes


def alpha_blend(src: np.ndarray, dst: np.ndarray, mask: np.ndarray, alpha: float = 0.9) -> np.ndarray:
    """
    Perform alpha blending of the source and destination images using the mask.

    Args:
        src (np.ndarray): The source image to paste.
        dst (np.ndarray): The destination image where the source will be pasted.
        mask (np.ndarray): The mask defining the area to paste.
        alpha (float): The alpha value for blending. Higher values give more weight to the source.

    Returns:
        np.ndarray: The resulting blended image.
    """

    # Normalize the mask to range [0, 1]
    _mask = mask / 255.0

    # Expand mask dimensions to match the shape of the source and destination images
    if len(_mask.shape) == 2:
        _mask = np.expand_dims(_mask, axis=-1)  # Shape: (height, width) -> (height, width, 1)

    # Perform alpha blending
    blended = alpha * src * _mask + (1 - alpha) * dst * (1 - _mask)

    # Convert to uint8 and clip values
    blended = np.clip(blended, 0, 255).astype('uint8')

    return blended


def simple_paste(src: np.ndarray, dst: np.ndarray, mask: np.ndarray) -> Image.Image:
    """
    Perform a simple paste operation, copying the source image onto the destination image using the mask.

    Args:
        src (np.ndarray): The source image to paste.
        dst (np.ndarray): The destination image where the source will be pasted.
        mask (np.ndarray): The mask defining the area to paste.

    Returns:
        Image.Image: The resulting image after the paste operation.
    """
    # Prepare the mask for pasting
    _mask = prepare_mask_for_pasting(mask)

    # Convert the source and destination images to PIL format
    _src = Image.fromarray(src.astype('uint8'))
    dst = Image.fromarray(dst.astype('uint8'))

    # Composite the images using the mask
    result = Image.composite(_src, dst, _mask)

    return result


def simple_paste_using_alpha(src: np.ndarray, dst: np.ndarray, mask: np.ndarray, alpha: float = 0.9) -> Image.Image:
    """
    Perform a simple paste operation with alpha blending, copying the source image onto the destination image using the mask.

    Args:
        src (np.ndarray): The source image to paste.
        dst (np.ndarray): The destination image where the source will be pasted.
        mask (np.ndarray): The mask defining the area to paste.
        alpha (float): The alpha value for blending. Higher values give more weight to the source.

    Returns:
        Image.Image: The resulting image after the paste operation.
    """
    # Ensure mask is binary (0 or 255)
    mask = np.array(prepare_mask_for_pasting(mask), dtype=np.uint8)
    # Perform alpha blending
    blended_image = alpha_blend(src, dst, mask, alpha)

    # Convert the blended result back to PIL Image format
    result = Image.fromarray(blended_image)

    return result


def seamlessclone(src: np.ndarray, dst: np.ndarray, mask: np.ndarray, center: Tuple[int, int],
                  mode: int = cv2.NORMAL_CLONE) -> Image.Image:
    """
    Perform a seamless cloning operation, blending the source image onto the destination image using the mask.

    Args:
        src (np.ndarray): The source image to paste.
        dst (np.ndarray): The destination image where the source will be pasted.
        mask (np.ndarray): The mask defining the area to paste.
        center (Tuple[int, int]): The center point for the seamless cloning operation.
        mode (int, optional): The cloning mode (NORMAL_CLONE, MIXED_CLONE, or MONOCHROME_TRANSFER).

    Returns:
        Image.Image: The resulting image after the seamless cloning operation.
    """
    # Prepare the mask for seamless cloning

    _mask = prepare_mask_for_pasting(mask)

    # Perform the seamless cloning operation
    result = cv2.seamlessClone(src, dst, np.array(_mask), center, mode)

    return Image.fromarray(result)


def select_seamless_clone_method(src_img: np.ndarray, destn_img: np.ndarray) -> str:
    """
    Select the best seamless cloning method based on a simplified analysis of the source and destination images,
    focusing on edge preservation, color homogeneity, and avoiding over-blending.

    Args:
        src_img (np.ndarray): The source image to be blended.
        destn_img (np.ndarray): The destination background image.

    Returns:
        str: The recommended cloning method ('NormalClone', 'MixedClone', 'MonochromeTransfer').
    """
    # Convert images to grayscale for edge detection
    src_gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    destn_gray = cv2.cvtColor(destn_img, cv2.COLOR_BGR2GRAY)

    # Perform edge detection to assess edge density
    src_edges = cv2.Canny(src_gray, 100, 200)
    destn_edges = cv2.Canny(destn_gray, 100, 200)
    src_edge_density = np.sum(src_edges) / src_edges.size
    destn_edge_density = np.sum(destn_edges) / destn_edges.size

    # Calculate color homogeneity (variance) of the destination and source
    destn_color_var = np.var(cv2.cvtColor(destn_img, cv2.COLOR_BGR2LAB))
    src_color_var = np.var(cv2.cvtColor(src_img, cv2.COLOR_BGR2LAB))

    # Initialize conditions
    normal_clone_condition = destn_edge_density > 0.01 and src_edge_density > 0.01
    mixed_clone_condition = src_edge_density > destn_edge_density * 0.5 and src_color_var > destn_color_var * 0.5
    monochrome_transfer_condition = src_color_var < destn_color_var * 0.3

    # print(f"Src Edge Density: {src_edge_density}, Destn Edge Density: {destn_edge_density}")
    # print(f"Src Color Variance: {src_color_var}, Destn Color Variance: {destn_color_var}")

    # Decision logic with simplified conditions
    if normal_clone_condition:
        if mixed_clone_condition:
            return 'MixedClone'
        return 'NormalClone'

    if mixed_clone_condition:
        return 'MixedClone'

    if monochrome_transfer_condition:
        return 'MonochromeTransfer'

    # Default case, prioritizing normal clone if conditions are not met
    return 'NormalClone'


def poisson_blending(src_img: np.ndarray, dest_img: np.ndarray, mask: np.ndarray,
                     scale_factor: float = 0.5) -> np.ndarray:
    """
    Perform Poisson image blending of src_img into dest_img using the given binary mask with optimizations.

    Args:
        src_img (np.ndarray): The source image to be blended.
        dest_img (np.ndarray): The destination (background) image.
        mask (np.ndarray): A binary mask where 255 represents the area to be blended from src_img to dest_img.
        scale_factor (float): Factor by which to downscale the images for faster processing. Default is 0.5.

    Returns:
        np.ndarray: The blended image.
    """
    # Downscale images and mask to speed up processing
    src_img_small = cv2.resize(src_img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
    dest_img_small = cv2.resize(dest_img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
    mask_small = cv2.resize(mask, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)

    # Convert images to float (skimage works better with floats)
    src_img_small = img_as_float(src_img_small)
    dest_img_small = img_as_float(dest_img_small)

    # Normalize mask to range [0, 1]
    mask_small = mask_small / 255.0

    # Perform inpainting on each channel separately using parallelization
    blended_img_small = np.zeros_like(dest_img_small)

    def inpaint_channel(channel_idx):
        return inpaint.inpaint_biharmonic(dest_img_small[..., channel_idx], mask_small)

    with ThreadPoolExecutor() as executor:
        results = executor.map(inpaint_channel, range(src_img_small.shape[2]))

    for i, result in enumerate(results):
        blended_img_small[..., i] = result

    # Combine the source and blended images based on the mask
    result_small = mask_small[:, :, np.newaxis] * src_img_small + (1 - mask_small[:, :, np.newaxis]) * blended_img_small

    # Upscale the result back to original size
    result = cv2.resize(result_small, (dest_img.shape[1], dest_img.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Clip the result to be in the valid range [0, 1]
    result = np.clip(result, 0, 1)

    return img_as_ubyte(result)


def create_composite_image(
        src_img: np.ndarray,
        destn_image: np.ndarray,
        mask: np.ndarray,
        clone_choice: Optional[str] = 'auto',
        center: Optional[Tuple[int, int]] = None
) -> Image.Image:
    """
    Create a composite image by blending the source image with the destination image using the specified method.

    Args:
        src_img (np.ndarray): The source image to be blended.
        destn_image (np.ndarray): The destination background image.
        mask (np.ndarray): The mask defining the area of the source image to be blended.
        clone_choice (str, optional): The blending method ('auto', 'SP', 'SoftPaste', 'NormalClone', 'MixedClone', 'MonochromeTransfer').
                                      Defaults to 'auto', which randomly selects a method.
        center (Tuple[int, int], optional): The center point for seamless cloning methods. Required for seamless cloning.

    Returns:
        Image.Image: The resulting composite image.
    """
    # If clone_choice is 'auto', randomly select a method
    if clone_choice == 'auto':
        clone_choice = random.choice(['SP', 'SoftPaste', 'NormalClone', 'MixedClone', 'MonochromeTransfer'])

    if clone_choice == 'SP':
        return simple_paste_using_alpha(src_img, destn_image, mask, alpha=0.6)
    elif clone_choice == 'SoftPaste':
        return soft_paste(src_img, destn_image, mask, feather=5)
    elif clone_choice in ['NormalClone', 'MixedClone', 'MonochromeTransfer']:
        if center is None:
            center = (src_img.shape[1] // 2, src_img.shape[0] // 2)
        mode = {
            'NormalClone': cv2.NORMAL_CLONE,
            'MixedClone': cv2.MIXED_CLONE,
            'MonochromeTransfer': cv2.MONOCHROME_TRANSFER
        }[clone_choice]
        return seamlessclone(src_img, destn_image, mask, center, mode)
    else:
        raise ValueError(f"Invalid clone_choice: {clone_choice}.")


def compose_final_image(
        args: Any,
        selected_objs_coords: List[np.ndarray],
        cutout_objs_coords: List[np.ndarray],
        cutout_objs_inner_coords: List[np.ndarray],
        cutout_objs_images: List[Image.Image],
        cutout_objs_masks: List[Image.Image]
) -> Tuple[Image.Image, Image.Image, np.ndarray]:
    """
    Compose the final artificial image by stitching selected objects onto a background.

    Args:
        args (Any): Arguments containing paths and other configurations.
        selected_objs_coords (List[np.ndarray]): Coordinates of the selected objects.
        cutout_objs_coords (List[np.ndarray]): Original outer bounding box coordinates of the cutout objects.
        cutout_objs_inner_coords (List[np.ndarray]): Inner bounding box coordinates of the cutout objects.
        cutout_objs_images (List[Image.Image]): Images of the cutout objects.
        cutout_objs_masks (List[Image.Image]): Masks of the cutout objects.

    Returns:
        Tuple[Image.Image, Image.Image, np.ndarray]: The final composed image, the final mask, and the final bounding boxes.
    """
    # Determine the tight-fitting target size based on selected object coordinates
    target_image_size = get_tightfit_targetsize(selected_objs_coords)

    # Stitch the selected object masks to create a new mask
    new_mask = paste_masks(target_image_size, cutout_objs_masks, selected_objs_coords)

    # Stitch the selected object images to create a new image
    new_img = paste_objs(target_image_size, cutout_objs_images, selected_objs_coords)

    # Get the real tight-fit coordinates of the objects
    final_boxes = get_realcoords(cutout_objs_coords, cutout_objs_inner_coords, selected_objs_coords)

    # Paste the selected objects onto a random pivot on the target image
    new_img2, new_mask2, new_boxes2 = paste_at_random_pivot(new_img, new_mask, final_boxes, scale_min=1.0,
                                                            scale_max=1.15)
    
    # Handle the background image
    background_img = handle_background_image(args.bckgrnd_imgs_dir, (new_img2.shape[1], new_img2.shape[0], 3))
    
    # Convert to numpy arrays for processing
    new_img2_np = np.array(new_img2)
    background_img_np = np.array(background_img) if isinstance(background_img, Image.Image) else background_img
    
    # Get configuration options with defaults
    blending_method = getattr(args, 'blending_method', 'auto')
    enable_post_processing = getattr(args, 'enable_post_processing', True)
    edge_refinement_level = getattr(args, 'edge_refinement_level', 'medium')
    color_harmonization = getattr(args, 'color_harmonization', True)
    context_aware_augmentations = getattr(args, 'context_aware_augmentations', True)
    
    try:
        # Refine mask boundaries using advanced edge refinement
        binary_mask = convert_instance_to_binary(new_mask2)
        refined_mask = refine_object_boundaries(
            binary_mask, new_img2_np, refinement_level=edge_refinement_level
        )
    except Exception:
        # Fallback to simple preprocessing
        refined_mask = preprocess_mask(binary_mask, method='erode')
    
    # Analyze contexts for adaptive blending (with caching and optimization)
    object_context = None
    background_context = None
    # Only do context analysis if blending_method is 'auto' (skip if specific method is chosen)
    if blending_method == 'auto':
        try:
            # Use cached context analysis
            object_context = _get_cached_composition_context(new_img2_np, refined_mask)
            background_context = _get_cached_composition_context(background_img_np, None)
        except Exception:
            pass  # Continue without context analysis
    
    # Create the composite image using adaptive blending
    try:
        if blending_method == 'auto' and object_context is not None and background_context is not None:
            # Use adaptive blending with context analysis
            final_image = adaptive_blend(
                new_img2_np, background_img_np, refined_mask,
                method=None, params=None
            )
        else:
            # Use specified method or fallback
            if blending_method == 'auto':
                blending_method = 'SoftPaste'  # Fallback
            
            final_image = create_composite_image(
                new_img2_np, background_img_np, refined_mask,
                clone_choice=blending_method
            )
    except Exception as e:
        # Fallback to simple paste if advanced blending fails
        try:
            final_image = create_composite_image(
                new_img2_np, background_img_np, refined_mask,
                clone_choice='SoftPaste'
            )
        except Exception:
            # Last resort: simple paste
            from sigtor.processing.image_composition import simple_paste
            final_image = simple_paste(new_img2_np, background_img_np, refined_mask)
    
    # Perform post-processing to minimize visual artifacts
    if enable_post_processing:
        try:
            final_image = post_processing(
                final_image,
                mask=refined_mask,
                background_img=background_img_np,
                refinement_level=edge_refinement_level,
                enable_color_harmonization=color_harmonization
            )
        except Exception:
            pass  # Continue without post-processing if it fails
    
    # Convert the mask into a colored RGB segmentation mask
    final_colored_mask = mask_to_RGB(Image.fromarray(new_mask2), get_colors(256))
    return final_image, final_colored_mask, new_boxes2
