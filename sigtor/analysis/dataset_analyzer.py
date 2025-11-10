"""
Dataset analysis module for extracting statistics from YOLO-format annotations.

This module provides comprehensive analysis of object detection datasets including
class distribution, object size distribution, spatial analysis, and imbalance metrics.
"""

import os
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, field
from PIL import Image

from sigtor.utils.image_utils import read_ann, get_ground_truth_data, get_classes


# COCO size thresholds
SMALL_AREA_THRESHOLD = 32 * 32  # 1024 pixels
MEDIUM_AREA_THRESHOLD = 96 * 96  # 9216 pixels


@dataclass
class ClassStatistics:
    """Statistics for a single class."""
    class_id: int
    class_name: str
    total_count: int = 0
    percentage: float = 0.0
    avg_per_image: float = 0.0
    images_with_class: int = 0
    small_count: int = 0
    medium_count: int = 0
    large_count: int = 0


@dataclass
class SizeDistribution:
    """Object size distribution statistics."""
    small_count: int = 0
    medium_count: int = 0
    large_count: int = 0
    small_percentage: float = 0.0
    medium_percentage: float = 0.0
    large_percentage: float = 0.0


@dataclass
class DatasetStatistics:
    """Complete dataset statistics."""
    dataset_name: str
    total_images: int = 0
    total_objects: int = 0
    num_classes: int = 0
    class_stats: Dict[int, ClassStatistics] = field(default_factory=dict)
    size_distribution: SizeDistribution = field(default_factory=SizeDistribution)
    objects_per_image: List[int] = field(default_factory=list)
    image_dimensions: List[Tuple[int, int]] = field(default_factory=list)
    box_areas: List[float] = field(default_factory=list)
    box_aspect_ratios: List[float] = field(default_factory=list)
    spatial_distribution: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    imbalance_ratio: float = 0.0
    underrepresented_classes: List[int] = field(default_factory=list)
    overrepresented_classes: List[int] = field(default_factory=list)


def calculate_box_area(box: np.ndarray) -> float:
    """Calculate area of a bounding box."""
    x1, y1, x2, y2 = box[:4]
    return float((x2 - x1) * (y2 - y1))


def calculate_aspect_ratio(box: np.ndarray) -> float:
    """Calculate aspect ratio (width/height) of a bounding box."""
    x1, y1, x2, y2 = box[:4]
    width = x2 - x1
    height = y2 - y1
    if height == 0:
        return 0.0
    return float(width / height)


def get_size_category(area: float) -> str:
    """Categorize object size based on COCO thresholds."""
    if area < SMALL_AREA_THRESHOLD:
        return 'small'
    elif area < MEDIUM_AREA_THRESHOLD:
        return 'medium'
    else:
        return 'large'


def get_spatial_position(box: np.ndarray, img_width: int, img_height: int) -> Tuple[int, int]:
    """
    Determine spatial position of object in 3x3 grid.
    
    Returns:
        Tuple of (row, col) in 3x3 grid (0-2, 0-2)
    """
    x1, y1, x2, y2 = box[:4]
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    
    # Normalize to [0, 1]
    norm_x = center_x / img_width if img_width > 0 else 0.5
    norm_y = center_y / img_height if img_height > 0 else 0.5
    
    # Map to 3x3 grid
    col = min(2, int(norm_x * 3))
    row = min(2, int(norm_y * 3))
    
    return row, col


def analyze_dataset(
    annotation_file: str,
    classnames_file: Optional[str] = None,
    dataset_name: Optional[str] = None
) -> DatasetStatistics:
    """
    Analyze a YOLO-format annotation file and extract comprehensive statistics.
    
    Args:
        annotation_file: Path to annotation file in YOLO format
        classnames_file: Optional path to class names file
        dataset_name: Optional name for the dataset
    
    Returns:
        DatasetStatistics object with all extracted statistics
    """
    if dataset_name is None:
        dataset_name = os.path.basename(annotation_file).replace('.txt', '')
    
    # Load class names if provided
    class_names = {}
    if classnames_file and os.path.exists(classnames_file):
        try:
            classes_list = get_classes(classnames_file)
            class_names = {i: name for i, name in enumerate(classes_list)}
        except Exception:
            pass
    
    # Read annotations
    annotation_lines = read_ann(annotation_file, shuffle=False)
    
    # Initialize statistics
    stats = DatasetStatistics(dataset_name=dataset_name)
    class_counts = defaultdict(int)
    class_image_counts = defaultdict(set)
    objects_per_image = []
    image_dims = []
    box_areas = []
    box_aspect_ratios = []
    spatial_grid = np.zeros((3, 3))
    size_counts = {'small': 0, 'medium': 0, 'large': 0}
    
    # Process each annotation line
    for line in annotation_lines:
        try:
            img_path, boxes = get_ground_truth_data(line)
            
            # Get image dimensions
            if os.path.exists(img_path):
                try:
                    with Image.open(img_path) as img:
                        width, height = img.size
                        image_dims.append((width, height))
                except Exception:
                    # If image can't be loaded, skip dimension tracking
                    width, height = 800, 600  # Default
            else:
                width, height = 800, 600  # Default
            
            num_objects = len(boxes)
            objects_per_image.append(num_objects)
            
            # Process each bounding box
            for box in boxes:
                x1, y1, x2, y2, class_id = box
                class_id = int(class_id)
                
                # Class statistics
                class_counts[class_id] += 1
                class_image_counts[class_id].add(img_path)
                
                # Size statistics
                area = calculate_box_area(box)
                box_areas.append(area)
                size_cat = get_size_category(area)
                size_counts[size_cat] += 1
                
                # Aspect ratio
                aspect_ratio = calculate_aspect_ratio(box)
                box_aspect_ratios.append(aspect_ratio)
                
                # Spatial distribution
                row, col = get_spatial_position(box, width, height)
                spatial_grid[row, col] += 1
                
        except Exception as e:
            # Skip malformed lines
            continue
    
    # Calculate totals
    stats.total_images = len(annotation_lines)
    stats.total_objects = sum(class_counts.values())
    stats.num_classes = len(class_counts)
    stats.objects_per_image = objects_per_image
    stats.image_dimensions = image_dims
    stats.box_areas = box_areas
    stats.box_aspect_ratios = box_aspect_ratios
    stats.spatial_distribution = spatial_grid
    
    # Size distribution
    total_size_count = sum(size_counts.values())
    if total_size_count > 0:
        stats.size_distribution.small_count = size_counts['small']
        stats.size_distribution.medium_count = size_counts['medium']
        stats.size_distribution.large_count = size_counts['large']
        stats.size_distribution.small_percentage = (size_counts['small'] / total_size_count) * 100
        stats.size_distribution.medium_percentage = (size_counts['medium'] / total_size_count) * 100
        stats.size_distribution.large_percentage = (size_counts['large'] / total_size_count) * 100
    
    # Class statistics
    if stats.total_objects > 0:
        for class_id, count in class_counts.items():
            class_name = class_names.get(class_id, f"Class_{class_id}")
            images_with = len(class_image_counts[class_id])
            avg_per_image = count / stats.total_images if stats.total_images > 0 else 0.0
            
            # Size distribution per class
            small_count = 0
            medium_count = 0
            large_count = 0
            
            # Re-process to get size distribution per class (simplified - could be optimized)
            for line in annotation_lines:
                try:
                    _, boxes = get_ground_truth_data(line)
                    for box in boxes:
                        if int(box[4]) == class_id:
                            area = calculate_box_area(box)
                            size_cat = get_size_category(area)
                            if size_cat == 'small':
                                small_count += 1
                            elif size_cat == 'medium':
                                medium_count += 1
                            else:
                                large_count += 1
                except Exception:
                    continue
            
            class_stat = ClassStatistics(
                class_id=class_id,
                class_name=class_name,
                total_count=count,
                percentage=(count / stats.total_objects) * 100,
                avg_per_image=avg_per_image,
                images_with_class=images_with,
                small_count=small_count,
                medium_count=medium_count,
                large_count=large_count
            )
            stats.class_stats[class_id] = class_stat
    
    # Imbalance analysis
    if len(class_counts) > 0:
        counts = list(class_counts.values())
        max_count = max(counts)
        min_count = min(counts)
        mean_count = np.mean(counts)
        std_count = np.std(counts)
        
        if min_count > 0:
            stats.imbalance_ratio = max_count / min_count
        
        # Identify underrepresented and overrepresented classes
        for class_id, count in class_counts.items():
            percentage = (count / stats.total_objects) * 100
            
            # Underrepresented: < 5% of total or < mean - 2*std
            if percentage < 5.0 or count < (mean_count - 2 * std_count):
                stats.underrepresented_classes.append(class_id)
            
            # Overrepresented: > 20% of total or > mean + 2*std
            if percentage > 20.0 or count > (mean_count + 2 * std_count):
                stats.overrepresented_classes.append(class_id)
    
    return stats


def compare_datasets(
    stats_before: DatasetStatistics,
    stats_after: DatasetStatistics
) -> Dict:
    """
    Compare two datasets and compute improvement metrics.
    
    Args:
        stats_before: Statistics from source dataset
        stats_after: Statistics from SIGtored dataset
    
    Returns:
        Dictionary with comparison metrics
    """
    comparison = {
        'images_growth': ((stats_after.total_images - stats_before.total_images) / 
                          stats_before.total_images * 100) if stats_before.total_images > 0 else 0,
        'objects_growth': ((stats_after.total_objects - stats_before.total_objects) / 
                          stats_before.total_objects * 100) if stats_before.total_objects > 0 else 0,
        'imbalance_improvement': stats_before.imbalance_ratio - stats_after.imbalance_ratio,
        'class_balance_changes': {},
        'size_distribution_changes': {}
    }
    
    # Class balance changes
    all_classes = set(stats_before.class_stats.keys()) | set(stats_after.class_stats.keys())
    for class_id in all_classes:
        before_stat = stats_before.class_stats.get(class_id)
        after_stat = stats_after.class_stats.get(class_id)
        
        if before_stat and after_stat:
            comparison['class_balance_changes'][class_id] = {
                'before_percentage': before_stat.percentage,
                'after_percentage': after_stat.percentage,
                'change': after_stat.percentage - before_stat.percentage,
                'before_count': before_stat.total_count,
                'after_count': after_stat.total_count
            }
    
    # Size distribution changes
    comparison['size_distribution_changes'] = {
        'small': {
            'before': stats_before.size_distribution.small_percentage,
            'after': stats_after.size_distribution.small_percentage,
            'change': stats_after.size_distribution.small_percentage - stats_before.size_distribution.small_percentage
        },
        'medium': {
            'before': stats_before.size_distribution.medium_percentage,
            'after': stats_after.size_distribution.medium_percentage,
            'change': stats_after.size_distribution.medium_percentage - stats_before.size_distribution.medium_percentage
        },
        'large': {
            'before': stats_before.size_distribution.large_percentage,
            'after': stats_after.size_distribution.large_percentage,
            'change': stats_after.size_distribution.large_percentage - stats_before.size_distribution.large_percentage
        }
    }
    
    return comparison

