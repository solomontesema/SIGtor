"""Core visualization functionality."""

import os
import cv2
import numpy as np

from sigtor.utils.image_utils import read_ann, get_colors, draw_boxes, get_classes


def visualize_annotations(ann_path, class_names=None, output_dir="./misc/images/", num_images="All"):
    """
    Visualize annotations by drawing bounding boxes on images.
    
    Args:
        ann_path (str): Path to the annotation file.
        class_names (list, optional): List of class names. If None, uses class indices.
        output_dir (str): Directory to save the visualized images.
        num_images (str or int): Number of images to process. "All" for all images, or a positive integer.
    """
    lines = read_ann(ann_path, shuffle=True)
    
    # Limit number of images if specified
    if num_images != "All":
        try:
            num_images = int(num_images)
            if num_images > 0:
                lines = lines[:num_images]
        except (ValueError, TypeError):
            pass
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    for line in lines:
        line_content = line.split()
        if len(line_content) < 2:
            continue
            
        img_path = line_content[0]
        filename = os.path.basename(img_path)
        
        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            continue
        
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"Warning: Could not read image: {img_path}")
            continue
        
        try:
            boxes = np.array([np.array(list(map(float, box.split(',')))) for box in line_content[1:]]).astype(
                'int32').reshape(-1, 5)
            classes = boxes[..., 4] + 1
            if class_names is None:
                class_names = classes
            colors = get_colors()
            image = draw_boxes(image, boxes[..., 0:4], classes, class_names, colors)
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, image)
        except (ValueError, IndexError) as e:
            print(f"Warning: Error processing annotation for {img_path}: {e}")
            continue

