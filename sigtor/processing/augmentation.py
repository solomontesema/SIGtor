import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from PIL import Image, ImageFilter
from sigtor.utils.data_utils import (
    random_blur, random_grayscale, random_brightness, random_sharpness,
    random_chroma, random_contrast, random_hsv_distort, random_vertical_flip,
    random_horizontal_flip, random_size, rand
)
from sigtor.utils.image_utils import adaptive_adjust_color
from sigtor.processing.context_analysis import (
    analyze_image_context, suggest_augmentations, ImageContext
)
from sigtor.processing.color_harmonization import (
    adjust_color_temperature, estimate_color_temperature
)

# Constants for size thresholds (using COCO's setup)
SMALL_OBJECT_THRESHOLD = 32 * 32  # Area < 32 * 32 pixels
MEDIUM_OBJECT_THRESHOLD = 96 * 96  # Area < 96 * 96 pixels
LARGE_OBJECT_THRESHOLD = 96 * 96  # Area >= 96 * 96 pixels

# Constants for brightness, contrast, chroma, and sharpness thresholds
LOW_BRIGHTNESS_THRESHOLD = 50  # Example threshold for low brightness
HIGH_BRIGHTNESS_THRESHOLD = 200  # Example threshold for high brightness

LOW_CONTRAST_THRESHOLD = 20  # Example threshold for low contrast (standard deviation)
HIGH_CONTRAST_THRESHOLD = 100  # Example threshold for high contrast (standard deviation)

LOW_CHROMA_THRESHOLD = 10  # Example threshold for low chroma/saturation (standard deviation)
HIGH_CHROMA_THRESHOLD = 50  # Example threshold for high chroma/saturation (standard deviation)

LOW_SHARPNESS_THRESHOLD = 10  # Example threshold for low sharpness (standard deviation)
HIGH_SHARPNESS_THRESHOLD = 50  # Example threshold for high sharpness (standard deviation)


def apply_augmentation(obj_img: Image.Image, obj_mask: Image.Image, outerbox: np.ndarray,
                       inner_boxes: np.ndarray, aug_type: str) -> Tuple[
    Image.Image, Image.Image, np.ndarray, np.ndarray]:
    """
    Apply a specific augmentation to the object image, mask, and bounding boxes.

    Args:
        obj_img (Image.Image): The object image to augment.
        obj_mask (Image.Image): The object mask to augment.
        outerbox (np.ndarray): The outer bounding box.
        inner_boxes (np.ndarray): The inner bounding boxes.
        aug_type (str): The type of augmentation to apply.

    Returns:
        Tuple[Image.Image, Image.Image, np.ndarray, np.ndarray]: The augmented image, mask, and bounding boxes.
    """
    if aug_type == 'blur':
        obj_img = random_blur(obj_img)
    elif aug_type == 'grayscale':
        obj_img = random_grayscale(obj_img)
    elif aug_type == 'brightness':
        obj_img = random_brightness(obj_img)
    elif aug_type == 'sharpness':
        obj_img = random_sharpness(obj_img)
    elif aug_type == 'chroma':
        obj_img = random_chroma(obj_img)
    elif aug_type == 'contrast':
        obj_img = random_contrast(obj_img)
    elif aug_type == 'hsv':
        obj_img = random_hsv_distort(obj_img)
    elif aug_type == 'vflip':
        obj_img, obj_mask, outerbox, inner_boxes = apply_vertical_flip(obj_img, obj_mask, outerbox, inner_boxes)
    elif aug_type == 'hflip':
        obj_img, obj_mask, outerbox, inner_boxes = apply_horizontal_flip(obj_img, obj_mask, outerbox, inner_boxes)
    elif aug_type == 'rescale':
        obj_img, obj_mask, outerbox, inner_boxes = apply_rescale(obj_img, obj_mask, outerbox, inner_boxes,
                                                                 rescale_factor=rand(0.5, 2.5))

    return obj_img, obj_mask, outerbox, inner_boxes


def apply_vertical_flip(obj_img: Image.Image, obj_mask: Image.Image, outerbox: np.ndarray, inner_boxes: np.ndarray) -> \
        Tuple[Image.Image, Image.Image, np.ndarray, np.ndarray]:
    """
    Apply vertical flip augmentation to the image, mask, and bounding boxes.

    Args:
        obj_img (Image.Image): The object image to flip.
        obj_mask (Image.Image): The object mask to flip.
        outerbox (np.ndarray): The outer bounding box.
        inner_boxes (np.ndarray): The inner bounding boxes.

    Returns:
        Tuple[Image.Image, Image.Image, np.ndarray, np.ndarray]: The flipped image, mask, and bounding boxes.
    """
    obj_img, _ = random_vertical_flip(obj_img, prob=1.0)
    obj_mask, _ = random_vertical_flip(obj_mask, prob=1.0)
    x1, y1, x2, y2 = outerbox.flatten()
    h = y2 - y1
    outerbox[0, [1, 3]] = h - outerbox[0, [3, 1]]
    inner_boxes[..., [1, 3]] = h - inner_boxes[..., [3, 1]]
    return obj_img, obj_mask, outerbox, inner_boxes


def apply_horizontal_flip(obj_img: Image.Image, obj_mask: Image.Image, outerbox: np.ndarray, inner_boxes: np.ndarray) -> \
        Tuple[Image.Image, Image.Image, np.ndarray, np.ndarray]:
    """
    Apply horizontal flip augmentation to the image, mask, and bounding boxes.

    Args:
        obj_img (Image.Image): The object image to flip.
        obj_mask (Image.Image): The object mask to flip.
        outerbox (np.ndarray): The outer bounding box.
        inner_boxes (np.ndarray): The inner bounding boxes.

    Returns:
        Tuple[Image.Image, Image.Image, np.ndarray, np.ndarray]: The flipped image, mask, and bounding boxes.
    """
    obj_img, _ = random_horizontal_flip(obj_img, prob=1.0)
    obj_mask, _ = random_horizontal_flip(obj_mask, prob=1.0)
    x1, y1, x2, y2 = outerbox.flatten()
    w = x2 - x1
    outerbox[0, [0, 2]] = w - outerbox[0, [2, 0]]
    inner_boxes[..., [0, 2]] = w - inner_boxes[..., [2, 0]]
    return obj_img, obj_mask, outerbox, inner_boxes


def apply_rescale(obj_img: Image.Image, obj_mask: Image.Image, outerbox: np.ndarray, inner_boxes: np.ndarray, rescale_factor: float) -> Tuple[
    Image.Image, Image.Image, np.ndarray, np.ndarray]:
    """
    Apply rescale augmentation to the image, mask, and bounding boxes.

    Args:
        obj_img (Image.Image): The object image to rescale.
        obj_mask (Image.Image): The object mask to rescale.
        outerbox (np.ndarray): The outer bounding box.
        inner_boxes (np.ndarray): The inner bounding boxes.

    Returns:
        Tuple[Image.Image, Image.Image, np.ndarray, np.ndarray]: The rescaled image, mask, and bounding boxes.
    """
    x1, y1, x2, y2 = outerbox.flatten()
    outerbox = np.array([x1, y1, x2, y2, -1]).reshape(1, 5)
    boxes = np.concatenate((outerbox, inner_boxes), axis=0)

    scale_w_or_h = rand()
    obj_img, obj_mask, boxes = random_size(obj_img, mask=obj_mask, boxes=boxes, scale=rescale_factor, prob=scale_w_or_h,
                                           keepaspectratio=True)
    outerbox = boxes[0, :4].reshape(1, 4)
    inner_boxes = boxes[1:, :5].reshape(-1, 5)

    return obj_img, obj_mask, outerbox, inner_boxes


def random_augmentations(obj_img: Image.Image, obj_mask: Image.Image, outerbox: np.ndarray, inner_boxes: np.ndarray,
                         max_augs: int = 2, random_aug_nums: bool = False) -> Tuple[
    Image.Image, Image.Image, np.ndarray, np.ndarray]:
    """
    Apply random augmentations to the object image, mask, and bounding boxes.

    Args:
        obj_img (Image.Image): The object image to augment.
        obj_mask (Image.Image): The object mask to augment.
        outerbox (np.ndarray): The outer bounding box.
        inner_boxes (np.ndarray): The inner bounding boxes.
        max_augs (int, optional): The maximum number of augmentations to apply. Defaults to 2.
        random_aug_nums (bool, optional): Whether to apply a random number of augmentations. Defaults to False.

    Returns:
        Tuple[Image.Image, Image.Image, np.ndarray, np.ndarray]: The augmented image, mask, and bounding boxes.
    """
    # List of available augmentation types
    augtypes = [
        'rescale', 'blur', 'brightness', 'chroma', 'contrast', 'hsv',
        'hflip', 'sharpness', 'grayscale', 'vflip'
    ]

    if random_aug_nums:
        max_augs = np.random.randint(1, len(augtypes) + 1)
    max_augs = min(max_augs, len(augtypes))

    # Randomly select augmentations to apply
    selected_augtypes = list(np.random.choice(augtypes, size=max_augs, replace=False))

    if 'rescale' not in selected_augtypes:
        selected_augtypes.append('rescale')  # Make 'rescale' mandatory

    # Apply selected augmentations
    for aug_type in selected_augtypes:
        obj_img, obj_mask, outerbox, inner_boxes = apply_augmentation(
            obj_img, obj_mask, outerbox, inner_boxes, aug_type
        )

    return obj_img, obj_mask, outerbox, inner_boxes


import random

def heuristic_augmentations(obj_img: Image.Image, obj_mask: Image.Image, outerbox: np.ndarray, inner_boxes: np.ndarray) -> Tuple[
    Image.Image, Image.Image, np.ndarray, np.ndarray]:
    """
    Apply heuristic-based augmentations to the object image, mask, and bounding boxes.

    Args:
        obj_img (Image.Image): The object image to augment.
        obj_mask (Image.Image): The object mask to augment.
        outerbox (np.ndarray): The outer bounding box.
        inner_boxes (np.ndarray): The inner bounding boxes with class labels.

    Returns:
        Tuple[Image.Image, Image.Image, np.ndarray, np.ndarray]: The augmented image, mask, and bounding boxes.
    """
    # Calculate object characteristics
    object_area = (outerbox[0, 2] - outerbox[0, 0]) * (outerbox[0, 3] - outerbox[0, 1])
    avg_brightness = np.mean(np.array(obj_img))
    avg_contrast = np.std(np.array(obj_img))  # Simple proxy for contrast
    avg_chroma = np.std(np.array(obj_img.convert("HSV").getchannel("S")))  # Proxy for saturation
    avg_sharpness = np.std(np.array(obj_img.filter(ImageFilter.FIND_EDGES)))  # Proxy for sharpness

    # Determine rescale factor based on object size
    if object_area < SMALL_OBJECT_THRESHOLD:
        rescale_factor = rand(1.5, 3.0)  # Rescale range for small objects
    elif object_area > LARGE_OBJECT_THRESHOLD:
        rescale_factor = rand(0.5, 1.5)  # Rescale range for large objects
    else:
        rescale_factor = rand(0.75, 2.0)  # Rescale range for medium-sized objects

    # Determine scale factors based on object properties using random ranges
    brightness_scale = 1.0
    contrast_scale = 1.0
    chroma_scale = 1.0
    sharpness_scale = 1.0

    # Adjust brightness
    if avg_brightness < LOW_BRIGHTNESS_THRESHOLD:
        brightness_scale = rand(1.0, 1.5)  # Randomly increase brightness within the range
    elif avg_brightness > HIGH_BRIGHTNESS_THRESHOLD:
        brightness_scale = rand(0.5, 0.9)  # Randomly decrease brightness within the range

    # Adjust contrast
    if avg_contrast < LOW_CONTRAST_THRESHOLD:
        contrast_scale = rand(1.0, 1.5)  # Randomly increase contrast within the range
    elif avg_contrast > HIGH_CONTRAST_THRESHOLD:
        contrast_scale = rand(0.5, 0.9)  # Randomly decrease contrast within the range

    # Adjust chroma (saturation)
    if avg_chroma < LOW_CHROMA_THRESHOLD:
        chroma_scale = rand(1.0, 1.5)  # Randomly increase chroma (saturation) within the range
    elif avg_chroma > HIGH_CHROMA_THRESHOLD:
        chroma_scale = rand(0.5, 0.9)  # Randomly decrease chroma (saturation) within the range

    # Adjust sharpness
    if avg_sharpness < LOW_SHARPNESS_THRESHOLD:
        sharpness_scale = rand(1.0, 1.5)  # Randomly increase sharpness within the range
    elif avg_sharpness > HIGH_SHARPNESS_THRESHOLD:
        sharpness_scale = rand(0.5, 0.9)  # Randomly decrease sharpness within the range

    # Initialize augmentation list with rescale as mandatory
    augmentations = [
        ('rescale', rescale_factor)
    ]

    # Add other augmentations based on the analysis
    augmentations.extend([
        ('brightness', brightness_scale),
        ('contrast', contrast_scale),
        ('chroma', chroma_scale),
        ('sharpness', sharpness_scale),
        ('hsv_distort', (0.0, chroma_scale, brightness_scale))  # Adjust hue, saturation, value
    ])

    # Add random horizontal and vertical flips
    if rand() < 0.5:
        augmentations.append(('hflip', 1.0))
    if rand() < 0.5:
        augmentations.append(('vflip', 1.0))

    # Apply selected augmentations
    for aug_type, scale in augmentations:
        if aug_type == 'rescale':
            obj_img, obj_mask, outerbox, inner_boxes = apply_rescale(obj_img, obj_mask, outerbox, inner_boxes,
                                                                     rescale_factor)
        elif aug_type == 'brightness':
            obj_img = random_brightness(obj_img, scale=scale)
        elif aug_type == 'contrast':
            obj_img = random_contrast(obj_img, scale=scale)
        elif aug_type == 'chroma':
            obj_img = random_chroma(obj_img, scale=scale)
        elif aug_type == 'sharpness':
            obj_img = random_sharpness(obj_img, scale=scale)
        elif aug_type == 'hsv_distort':
            hue, sat, val = scale  # Unpacking the tuple
            obj_img = random_hsv_distort(obj_img, hue=hue, sat=sat, val=val)
        elif aug_type == 'hflip':
            obj_img, obj_mask, outerbox, inner_boxes = apply_horizontal_flip(obj_img, obj_mask, outerbox, inner_boxes)
        elif aug_type == 'vflip':
            obj_img, obj_mask, outerbox, inner_boxes = apply_vertical_flip(obj_img, obj_mask, outerbox, inner_boxes)

    return obj_img, obj_mask, outerbox, inner_boxes



def preprocess_mask(mask: np.ndarray, method: str = 'none') -> np.ndarray:
    """
    Preprocess the mask to reduce artifacts during composition.

    Args:
        mask (np.ndarray): The input mask.
        method (str): The preprocessing method to apply ('none', 'erode', 'dilate', 'blur').

    Returns:
        np.ndarray: The preprocessed mask.
    """
    if method == 'erode':
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        preprocessed_mask = cv2.erode(mask, kernel, iterations=2)
    elif method == 'dilate':
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        preprocessed_mask = cv2.dilate(mask, kernel, iterations=2)
    elif method == 'blur':
        preprocessed_mask = cv2.GaussianBlur(mask, (5, 5), 0)
    else:  # 'none' or any other method
        preprocessed_mask = mask

    return preprocessed_mask


def preprocess_source_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess the source image by applying adaptive color adjustment.

    Args:
        image (np.ndarray): The source image.

    Returns:
        np.ndarray: The preprocessed source image.
    """
    preprocessed_image = adaptive_adjust_color(image)
    return preprocessed_image


def context_aware_augmentations(
    obj_img: Image.Image,
    obj_mask: Image.Image,
    outerbox: np.ndarray,
    inner_boxes: np.ndarray,
    background_img: Optional[np.ndarray] = None,
    background_context: Optional[ImageContext] = None
) -> Tuple[Image.Image, Image.Image, np.ndarray, np.ndarray]:
    """
    Apply context-aware augmentations based on background analysis.
    
    Args:
        obj_img: Object image to augment.
        obj_mask: Object mask.
        outerbox: Outer bounding box.
        inner_boxes: Inner bounding boxes.
        background_img: Optional background image for context analysis.
        background_context: Optional pre-computed background context.
    
    Returns:
        Augmented image, mask, and bounding boxes.
    """
    # Convert PIL to numpy for analysis
    obj_img_np = np.array(obj_img)
    obj_mask_np = np.array(obj_mask)
    
    # Analyze object context
    object_context = analyze_image_context(obj_img_np, obj_mask_np)
    
    # Analyze background context if provided
    if background_context is None and background_img is not None:
        background_context = analyze_image_context(background_img)
    
    # Get augmentation suggestions if background context available
    if background_context is not None:
        suggestions = suggest_augmentations(object_context, background_context)
        
        # Apply suggested augmentations
        if 'brightness_adjust' in suggestions:
            brightness_factor = 1.0 + suggestions['brightness_adjust']
            obj_img = random_brightness(obj_img, scale=brightness_factor)
        
        if 'contrast_adjust' in suggestions:
            contrast_factor = 1.0 + suggestions['contrast_adjust']
            obj_img = random_contrast(obj_img, scale=contrast_factor)
        
        if 'saturation_adjust' in suggestions:
            saturation_factor = 1.0 + suggestions['saturation_adjust']
            obj_img = random_chroma(obj_img, scale=saturation_factor)
        
        if 'color_temp_adjust' in suggestions:
            # Adjust color temperature
            obj_img_np = np.array(obj_img)
            current_temp = estimate_color_temperature(obj_img_np)
            target_temp = current_temp + suggestions['color_temp_adjust']
            adjusted_np = adjust_color_temperature(obj_img_np, target_temp, current_temp)
            obj_img = Image.fromarray(adjusted_np)
        
        if suggestions.get('blur', False):
            blur_amount = suggestions.get('blur_amount', 3)
            # Apply blur using OpenCV
            obj_img_np = np.array(obj_img)
            blurred = cv2.GaussianBlur(obj_img_np, (blur_amount * 2 + 1, blur_amount * 2 + 1), blur_amount)
            obj_img = Image.fromarray(blurred)
    
    # Always apply rescale (mandatory)
    object_area = (outerbox[0, 2] - outerbox[0, 0]) * (outerbox[0, 3] - outerbox[0, 1])
    if object_area < SMALL_OBJECT_THRESHOLD:
        rescale_factor = rand(1.5, 3.0)
    elif object_area > LARGE_OBJECT_THRESHOLD:
        rescale_factor = rand(0.5, 1.5)
    else:
        rescale_factor = rand(0.75, 2.0)
    
    obj_img, obj_mask, outerbox, inner_boxes = apply_rescale(
        obj_img, obj_mask, outerbox, inner_boxes, rescale_factor
    )
    
    # Apply random flips (not context-dependent)
    if rand() < 0.5:
        obj_img, obj_mask, outerbox, inner_boxes = apply_horizontal_flip(
            obj_img, obj_mask, outerbox, inner_boxes
        )
    if rand() < 0.5:
        obj_img, obj_mask, outerbox, inner_boxes = apply_vertical_flip(
            obj_img, obj_mask, outerbox, inner_boxes
        )
    
    return obj_img, obj_mask, outerbox, inner_boxes
