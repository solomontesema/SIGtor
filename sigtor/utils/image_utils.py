import colorsys
import math
import os
import random
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image


# Random number generation
def rand(a=0.0, b=1.0):
    """
    Generate a random float between a and b.

    Args:
        a (float): The lower bound (inclusive).
        b (float): The upper bound (exclusive).

    Returns:
        float: A random float between a and b.
    """
    return np.random.rand() * (b - a) + a


# Annotation file reading
def read_ann(ann_file, shuffle=True):
    """
    Read and optionally shuffle lines from an annotation file.

    Args:
        ann_file (str): Path to the annotation file.
        shuffle (bool): Whether to shuffle the lines.

    Returns:
        list: A list of lines from the file.
    """
    with open(ann_file) as f:
        lines = f.readlines()
    if shuffle:
        np.random.shuffle(lines)
    return lines


# Get file paths with filtering
def get_file_paths(filepath, file_format=None, shuffle=False):
    """
    Retrieve file paths from a directory with optional filtering and shuffling.

    Args:
        filepath (str): The directory to search for files.
        file_format (list of str, optional): List of file extensions to filter by.
        shuffle (bool): Whether to shuffle the file paths.

    Returns:
        list: A list of file paths.
    """
    if file_format:
        file_paths = [os.path.join(filepath, filename) for filename in os.listdir(filepath)
                      if os.path.splitext(filename)[1].lower() in file_format]
    else:
        file_paths = [os.path.join(filepath, filename) for filename in os.listdir(filepath)]

    if shuffle:
        np.random.shuffle(file_paths)
    return file_paths


# Extract ground truth data from an annotation line
def get_ground_truth_data(annotation_line):
    """
    Extract the image path and bounding boxes from a single annotation line.

    Args:
        annotation_line (str): A single line from the annotation file.

    Returns:
        tuple: The image path and bounding boxes as a numpy array.
    """
    line = annotation_line.split()
    img_path = line[0]
    boxes = np.array([np.array(list(map(float, box.split(',')))) for box in line[1:]]).astype('int32')
    return img_path, boxes


# Get class names from a file
def get_classes(classes_path):
    """
    Load class names from a file.

    Args:
        classes_path (str): Path to the file containing class names.

    Returns:
        list: A list of class names.
    """
    with open(os.path.expanduser(classes_path)) as f:
        class_names = [c.strip() for c in f.readlines()]
    return class_names


# Generate distinct colors for bounding boxes
def get_colors(number=100, bright=True):
    """
    Generate a list of distinct colors for drawing bounding boxes.

    Args:
        number (int): Number of colors to generate.
        bright (bool): Whether the colors should be bright.

    Returns:
        list: A list of RGB tuples.
    """
    if number <= 0:
        return []

    brightness = 1.0 if bright else 0.7
    hsv_tuples = [(x / number, 1., brightness) for x in range(number)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    np.random.seed(10101)  # Fixed seed for consistent colors across runs.
    np.random.shuffle(colors)  # Shuffle colors to decorate adjacent classes.
    np.random.seed(None)  # Reset seed to default.
    return colors


# Draw a label on an image
def draw_label(image, text, color, coords):
    """
    Draw a label with text on an image.

    Args:
        image (np.ndarray): The image to draw on.
        text (str): The text of the label.
        color (tuple): The color of the label's background.
        coords (tuple): The (x, y) coordinates for the label's top-left corner.

    Returns:
        np.ndarray: The image with the label drawn.
    """
    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 1.0
    padding = 5

    text_size = cv2.getTextSize(text, font, fontScale=font_scale, thickness=2)[0]
    text_width, text_height = text_size

    rect_width = text_width + padding * 2
    rect_height = text_height + padding * 2

    x, y = coords

    # Draw the rectangle behind the text
    cv2.rectangle(image, (x, y), (x + rect_width, y - rect_height), color, cv2.FILLED)

    # Draw the text on top of the rectangle
    cv2.putText(image, text, (x + padding, y - text_height + padding), font,
                fontScale=font_scale, color=(255, 255, 255), lineType=cv2.LINE_AA)

    return image


# Draw bounding boxes on an image
def draw_boxes(image, boxes, classes, class_names, colors, show_label=True):
    """
    Draw bounding boxes and class labels on an image.

    Args:
        image (np.ndarray): The image to draw on.
        boxes (np.ndarray): Array of bounding boxes in (xmin, ymin, xmax, ymax) format.
        classes (np.ndarray): Array of class indices corresponding to the boxes.
        class_names (list): List of class names.
        colors (list): List of colors corresponding to each class.
        show_label (bool): Whether to show labels above the boxes.

    Returns:
        np.ndarray: The image with bounding boxes and labels drawn.
    """
    if boxes is None or len(boxes) == 0:
        return image
    if classes is None or len(classes) == 0:
        return image

    for box, cls in zip(boxes, classes):
        xmin, ymin, xmax, ymax = map(int, box)
        class_name = class_names[cls]
        color = colors[cls] if colors else (0, 0, 0)

        if show_label:
            image = draw_label(image, class_name, color, (xmin, ymin))

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2, cv2.LINE_AA)

    return image


# Measure overlap between bounding boxes
def overlap_measure(box1, box2, expand_dim=False):
    """
    Measure the overlap between two sets of bounding boxes.

    Args:
        box1 (np.ndarray): Array of bounding boxes [m, 4].
        box2 (np.ndarray): Array of bounding boxes [n, 4].
        expand_dim (bool): Whether to expand the dimensions of box2 for broadcasting.

    Returns:
        list: List containing IoU, IoL, and IoS values.
    """
    if expand_dim:
        box2 = np.expand_dims(box2, axis=1)

    intersect_mins = np.maximum(box1[..., 0:2], box2[..., 0:2])
    intersect_maxs = np.minimum(box1[..., 2:4], box2[..., 2:4])
    intersect_wh = np.maximum(intersect_maxs - intersect_mins, 0)

    box1_wh = box1[..., 2:4] - box1[..., 0:2]
    box2_wh = box2[..., 2:4] - box2[..., 0:2]
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    box1_area = box1_wh[..., 0] * box1_wh[..., 1]
    box2_area = box2_wh[..., 0] * box2_wh[..., 1]

    iou = intersect_area / (box1_area + box2_area - intersect_area)
    iol = intersect_area / np.maximum(box1_area, box2_area)
    ios = intersect_area / np.minimum(box1_area, box2_area)

    return [iou, iol, ios]


# Convert bounding boxes to Pascal VOC annotation line
def convert_to_ann_line(img_path: str, obj_boxes: np.ndarray) -> str:
    """
    Convert image path and object bounding boxes (with class labels) to a Pascal VOC format annotation line.

    Args:
        img_path (str): The image path.
        obj_boxes (np.ndarray): Array of bounding boxes with class labels.

    Returns:
        str: Pascal VOC format annotation line.
    """
    # Create a string representation of each bounding box with class labels and concatenate them
    box_strings = ["{},{},{},{},{}".format(int(x1), int(y1), int(x2), int(y2), int(cls))
                   for x1, y1, x2, y2, cls in obj_boxes]

    # Combine the image path with the bounding box strings
    ann_line = "{} {}\n".format(img_path, " ".join(box_strings))

    return ann_line


# Simple image color balancing implementation
def simplest_cb(img: np.ndarray, percent: float) -> np.ndarray:
    """
    Perform simplest color balancing on an image.

    Args:
        img (np.ndarray): The input image, expected to be in BGR format with 3 channels.
        percent (float): The percentage for color balancing. Must be between 0 and 100.

    Returns:
        np.ndarray: The color-balanced image.
    """
    assert img.shape[2] == 3, "Input image must have 3 channels (BGR format)."
    assert 0 < percent < 100, "Percent must be between 0 and 100."

    half_percent = percent / 200.0
    out_channels = []

    for channel in cv2.split(img):
        # Flatten the channel and sort it to find the percentile values
        flat_channel = np.sort(channel.flatten())

        # Calculate the low and high percentile values
        low_val = flat_channel[int(math.floor(len(flat_channel) * half_percent))]
        high_val = flat_channel[int(math.ceil(len(flat_channel) * (1.0 - half_percent)))]

        # Apply thresholding by clipping the values
        channel = np.clip(channel, low_val, high_val)

        # Normalize the channel to stretch the pixel values to the full range (0-255)
        normalized_channel = cv2.normalize(channel, None, 0, 255, cv2.NORM_MINMAX)
        out_channels.append(normalized_channel)

    # Merge the processed channels back together
    return cv2.merge(out_channels)


def adjust_color(image: np.ndarray, alpha: float = 1.0, beta: int = 0) -> np.ndarray:
    """
    Adjust the brightness and contrast of an image.

    Args:
        image (np.ndarray): The input image in BGR format.
        alpha (float): Contrast control (1.0-3.0). 1.0 means no change.
        beta (int): Brightness control (0-100). 0 means no change.

    Returns:
        np.ndarray: The color-adjusted image.
    """
    assert alpha >= 0, "Alpha (contrast) must be non-negative."
    assert isinstance(beta, int), "Beta (brightness) must be an integer."

    # Apply the brightness and contrast adjustment
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted_image


def adaptive_adjust_color(image: np.ndarray, target_brightness: int = 128, target_contrast: int = 64) -> np.ndarray:
    """
    Adjust the image brightness and contrast adaptively based on the histogram.

    Args:
        image (np.ndarray): The input image in BGR format.
        target_brightness (int): Desired average brightness of the output image.
        target_contrast (int): Desired standard deviation (contrast) of the output image.

    Returns:
        np.ndarray: The adaptively color-adjusted image.
    """
    assert 0 <= target_brightness <= 255, "Target brightness must be in the range 0-255."
    assert target_contrast >= 0, "Target contrast must be non-negative."

    # Calculate current brightness and contrast (standard deviation)
    current_brightness = np.mean(image)
    current_contrast = np.std(image)

    # Calculate alpha and beta for adjustment
    if current_contrast > 0:
        alpha = target_contrast / current_contrast
    else:
        alpha = 1.0  # Avoid division by zero, no contrast change
    beta = target_brightness - np.mean(image * alpha)

    # Apply the adaptive brightness and contrast adjustment
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted_image


def apply_mask(matrix: np.ndarray, mask: np.ndarray, fill_value: float) -> np.ndarray:
    """
    Apply a binary mask to a matrix, replacing masked-out values with a specified fill value.

    Args:
        matrix (np.ndarray): The input matrix.
        mask (np.ndarray): The binary mask (same shape as matrix).
        fill_value (float): The value to replace masked elements with.

    Returns:
        np.ndarray: The masked matrix.
    """
    masked_matrix = np.where(mask, fill_value, matrix)
    return masked_matrix


def mask_to_RGB(mask: Image.Image, colors: List[List[int]]) -> Image.Image:
    """
    Convert a grayscale mask image to an RGB image, where each unique mask value is mapped to a specific color.

    Args:
        mask (Image.Image): The grayscale mask image.
        colors (List[List[int]]): A list of RGB color values corresponding to each unique mask value.

    Returns:
        Image.Image: The RGB image.
    """
    mask_array = np.array(mask)
    h, w = mask_array.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)

    unique_values = np.unique(mask_array)
    for i, value in enumerate(unique_values):
        if value == 0:  # Skip the background
            continue
        elif value == 255 or value == 254:  # Leave the mask borders white
            colored_mask[mask_array == value] = (255, 255, 255)
            continue
        # Assign colors to other unique values
        color = colors[i % len(colors)]
        colored_mask[mask_array == value] = color

    return Image.fromarray(colored_mask)


def apply_histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Apply histogram equalization to each channel of the image to improve contrast.

    Args:
        image (np.ndarray): The input image in BGR format.

    Returns:
        np.ndarray: The contrast-enhanced image.
    """
    channels = cv2.split(image)
    eq_channels = [cv2.equalizeHist(channel) for channel in channels]
    return cv2.merge(eq_channels)


def blend_edges(image: np.ndarray, mask: np.ndarray, blur_radius: int = 5, feather_radius: int = 10) -> np.ndarray:
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

    # TODO: Optionally adjust color balance to match the background
    # This can be done using color transfer techniques or histogram matching

    return blended_image


# def post_processing(image: Image.Image, mask=None) -> Image.Image:
#     """
#     Apply post-processing techniques to reduce visual artifacts in the artificially created image.
#
#     Args:
#         image (Image.Image): The input image to be post-processed.
#
#     Returns:
#         Image.Image: The post-processed image.
#     """
#     image = np.array(image)
#
#     # Apply histogram equalization to improve contrast
#     image = apply_histogram_equalization(image)
#
#     # Convert to YCrCb color space for color balancing
#     ycrcb_img = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
#     ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])
#     image = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)
#
#     # Optionally apply Gaussian blur to soften edges (can be adjusted)
#     image = cv2.GaussianBlur(image, (3, 3), 0)
#
#     # Blend edges using the mask (if available)
#     if mask is None:
#         mask = np.any(image > 0, axis=2).astype(np.uint8) * 255  # Create a binary mask from non-black pixels
#     image = blend_edges(image, mask)
#
#     # Apply simplest color balance to harmonize the overall color scheme
#     image = simplest_cb(image, percent=5)
#
#     # Convert back to PIL Image for consistency
#     output_img = Image.fromarray(image)
#
#     return output_img


def recalculate_targetsize(current_targetsize: Tuple[int, int], current_outerbox: np.ndarray) -> Tuple[int, int]:
    """
    Recalculate the target background size based on the size of the cutout object.
    If the cutout object is larger than the current target size in either width or height,
    adjust the target size accordingly.

    Args:
        current_targetsize (Tuple[int, int]): The current target background size (width, height).
        current_outerbox (np.ndarray): The outer bounding box of the cutout object, shaped (1, 4).

    Returns:
        Tuple[int, int]: The new target background size (width, height).
    """
    # Extract the current width and height of the target background
    target_width, target_height = current_targetsize

    # Extract the bounding box dimensions
    x1, y1, x2, y2 = current_outerbox.flatten()
    object_width = x2 - x1
    object_height = y2 - y1

    # Ensure the new dimensions are even numbers
    new_target_width = max(target_width, object_width + (object_width % 2))
    new_target_height = max(target_height, object_height + (object_height % 2))

    return new_target_width, new_target_height


def generate_random_even_size(min_value: int, max_value: int) -> int:
    """
    Generate a random even size within the specified range.

    Args:
        min_value (int): The minimum value for the size.
        max_value (int): The maximum value for the size.

    Returns:
        int: A random even size within the range.
    """
    size = random.randint(min_value, max_value)
    return size + 1 if size % 2 != 0 else size


def random_new_image_size(width_min_max: Tuple[int, int], height_min_max: Tuple[int, int]) -> Tuple[int, int]:
    """
    Generate a random even-sized image dimensions tuple within the specified width and height ranges.

    Args:
        width_min_max (Tuple[int, int]): A tuple containing the minimum and maximum width.
        height_min_max (Tuple[int, int]): A tuple containing the minimum and maximum height.

    Returns:
        Tuple[int, int]: A tuple containing the random even width and height.
    """
    # Generate random even width and height
    width = generate_random_even_size(*width_min_max)
    height = generate_random_even_size(*height_min_max)

    return width, height
