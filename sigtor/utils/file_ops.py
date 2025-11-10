import os
import numpy as np
from typing import Tuple, Any
from PIL import Image

from sigtor.utils.image_utils import convert_to_ann_line


def create_directory(path: str) -> None:
    """
    Create a directory if it does not already exist.

    Args:
        path (str): The directory path to create.
    """
    os.makedirs(path, exist_ok=True)


def initialize_directories(destn_dir: str) -> Tuple[str, str]:
    """
    Initialize the necessary directories for saving augmented images and masks.

    Args:
        destn_dir (str): The destination directory where the augmented images and masks will be saved.

    Returns:
        Tuple[str, str]: Paths to the created directories for augmented images and augmented masks.
    """
    # Define paths for augmented images and masks directories
    new_images_dir = os.path.join(destn_dir, "augmented_images")
    new_masks_dir = os.path.join(destn_dir, "augmented_masks")

    # Create the directories
    create_directory(new_images_dir)
    create_directory(new_masks_dir)

    return new_images_dir, new_masks_dir


def open_annotation_file(destn_dir: str, filename: str = 'sigtored_annotations.txt') -> Any:
    """
    Open the annotation file in write mode, ensuring that the destination directory exists.

    Args:
        destn_dir (str): The directory where the annotation file will be saved.
        filename (str, optional): The name of the annotation file. Default is 'sigtored_annotations.txt'.

    Returns:
        Any: The file object opened in write mode.
    """
    # Ensure the destination directory exists
    os.makedirs(destn_dir, exist_ok=True)

    # Construct the full path to the annotation file
    annotation_file_path = os.path.join(destn_dir, filename)

    # Open the file in write mode and return the file object
    return open(annotation_file_path, 'w')


def save_image(image: Image.Image, directory: str, filename: str) -> str:
    """
    Save an image to the specified directory with the given filename.

    Args:
        image (Image.Image): The image to save.
        directory (str): The directory where the image will be saved.
        filename (str): The filename for the image.

    Returns:
        str: The path to the saved image.
    
    Raises:
        OSError: If the directory cannot be created or the file cannot be written.
    """
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    try:
        image.save(path)
    except Exception as e:
        raise OSError(f"Failed to save image to {path}: {e}")
    return path


def save_mask(mask: Image.Image, directory: str, filename: str) -> str:
    """
    Save a mask to the specified directory with the given filename.

    Args:
        mask (Image.Image): The mask to save.
        directory (str): The directory where the mask will be saved.
        filename (str): The filename for the mask.

    Returns:
        str: The path to the saved mask.
    
    Raises:
        OSError: If the directory cannot be created or the file cannot be written.
    """
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    try:
        mask.save(path)
    except Exception as e:
        raise OSError(f"Failed to save mask to {path}: {e}")
    return path


def save_annotation(annotation: str, annotation_file: Any) -> None:
    """
    Save the Pascal VOC format annotation to the annotation file.

    Args:
        annotation (str): The annotation string in Pascal VOC format.
        annotation_file (Any): The file object where the annotation will be written.
    """
    annotation_file.write(annotation)
    annotation_file.flush()


def save_new_image_and_mask(
        final_image: Image.Image,
        final_mask: Image.Image,
        final_boxes: np.ndarray,
        new_dataset: Any,
        new_images_dir: str,
        new_masks_dir: str,
        count_new_images: int
) -> None:
    """
    Save the artificial image, its mask, and create a Pascal VOC format annotation.

    Args:
        final_image (Image.Image): The final composite image to save.
        final_mask (Image.Image): The final mask corresponding to the composite image.
        final_boxes (np.ndarray): The bounding boxes for the objects in the composite image.
        new_dataset (Any): The file object where the Pascal VOC annotations will be saved.
        new_images_dir (str): The directory where the images will be saved.
        new_masks_dir (str): The directory where the masks will be saved.
        count_new_images (int): The current count of saved images, used to generate filenames.
    """
    # Generate filenames for the image and mask
    image_filename = f"{count_new_images:08d}.jpg"
    mask_filename = f"{count_new_images:08d}.png"

    # Save the image and mask
    newimgpath = save_image(final_image, new_images_dir, image_filename)
    newmaskpath = save_mask(final_mask, new_masks_dir, mask_filename)

    # Create the Pascal VOC annotation line
    annotation_line = convert_to_ann_line(newimgpath, final_boxes)

    # Save the annotation
    save_annotation(annotation_line, new_dataset)

