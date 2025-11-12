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
from tqdm import tqdm

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
    dataset_name: Optional[str] = None,
    load_image_dims: bool = False
) -> DatasetStatistics:
    """
    Analyze a YOLO-format annotation file and extract comprehensive statistics.
    
    Args:
        annotation_file: Path to annotation file in YOLO format
        classnames_file: Optional path to class names file
        dataset_name: Optional name for the dataset
        load_image_dims: Whether to load actual image dimensions (slow for large datasets)
    
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
    print(f"Reading annotation file: {annotation_file}")
    annotation_lines = read_ann(annotation_file, shuffle=False)
    total_lines = len(annotation_lines)
    print(f"Found {total_lines:,} annotation lines")
    
    # Initialize statistics
    stats = DatasetStatistics(dataset_name=dataset_name)
    class_counts = defaultdict(int)
    class_image_counts = defaultdict(set)
    objects_per_image = []
    image_dims = []
    
    # Pre-allocate numpy arrays for vectorized operations
    all_boxes = []
    all_class_ids = []
    all_img_paths = []
    all_img_indices = []
    
    # First pass: collect all data (vectorized where possible)
    print("Collecting annotation data...")
    img_idx = 0
    for line in tqdm(annotation_lines, desc="Parsing annotations", unit="lines"):
        try:
            img_path, boxes = get_ground_truth_data(line)
            
            # Get image dimensions (optional, can be slow)
            if load_image_dims:
                if os.path.exists(img_path):
                    try:
                        with Image.open(img_path) as img:
                            width, height = img.size
                            image_dims.append((width, height))
                    except Exception:
                        width, height = 800, 600  # Default
                        image_dims.append((width, height))
                else:
                    width, height = 800, 600  # Default
                    image_dims.append((width, height))
            else:
                # Use default dimensions for spatial calculations (faster)
                width, height = 800, 600
                image_dims.append((width, height))
            
            num_objects = len(boxes)
            objects_per_image.append(num_objects)
            
            # Collect all boxes for vectorized processing
            if len(boxes) > 0:
                all_boxes.append(boxes)
                all_class_ids.extend(boxes[:, 4].astype(int))
                all_img_paths.extend([img_path] * len(boxes))
                all_img_indices.extend([img_idx] * len(boxes))
            
            img_idx += 1
                
        except Exception:
            # Skip malformed lines
            continue
    
    # Vectorized processing of all boxes
    if len(all_boxes) > 0:
        print("Processing bounding boxes (vectorized)...")
        # Flatten all boxes into single array
        all_boxes_flat = np.vstack(all_boxes)
        all_class_ids = np.array(all_class_ids)
        
        # Vectorized area calculation
        widths = all_boxes_flat[:, 2] - all_boxes_flat[:, 0]
        heights = all_boxes_flat[:, 3] - all_boxes_flat[:, 1]
        box_areas = (widths * heights).astype(float)
        
        # Vectorized aspect ratio calculation
        box_aspect_ratios = np.divide(widths, heights, out=np.zeros_like(widths, dtype=float), 
                                     where=heights != 0)
        
        # Vectorized size categorization
        small_mask = box_areas < SMALL_AREA_THRESHOLD
        medium_mask = (box_areas >= SMALL_AREA_THRESHOLD) & (box_areas < MEDIUM_AREA_THRESHOLD)
        large_mask = box_areas >= MEDIUM_AREA_THRESHOLD
        
        size_counts = {
            'small': int(np.sum(small_mask)),
            'medium': int(np.sum(medium_mask)),
            'large': int(np.sum(large_mask))
        }
        
        # Vectorized class counting
        unique_classes, class_counts_array = np.unique(all_class_ids, return_counts=True)
        for class_id, count in zip(unique_classes, class_counts_array):
            class_counts[int(class_id)] = int(count)
        
        # Class-image mapping (vectorized where possible)
        for class_id in unique_classes:
            class_mask = all_class_ids == class_id
            unique_images = set([all_img_paths[i] for i in np.where(class_mask)[0]])
            class_image_counts[int(class_id)] = unique_images
        
        # Vectorized spatial distribution (using default dimensions for speed)
        # Calculate centers
        center_x = (all_boxes_flat[:, 0] + all_boxes_flat[:, 2]) / 2.0
        center_y = (all_boxes_flat[:, 1] + all_boxes_flat[:, 3]) / 2.0
        
        # Normalize to [0, 1] (using default dimensions)
        norm_x = center_x / 800.0
        norm_y = center_y / 600.0
        
        # Map to 3x3 grid (vectorized)
        cols = np.clip((norm_x * 3).astype(int), 0, 2)
        rows = np.clip((norm_y * 3).astype(int), 0, 2)
        
        # Count spatial distribution (vectorized using bincount)
        spatial_grid = np.zeros((3, 3))
        # Flatten 2D indices to 1D for bincount
        flat_indices = rows * 3 + cols
        counts = np.bincount(flat_indices, minlength=9)
        spatial_grid = counts.reshape(3, 3)
        
        # Per-class size distribution (vectorized)
        class_size_stats = defaultdict(lambda: {'small': 0, 'medium': 0, 'large': 0})
        for class_id in unique_classes:
            class_mask = all_class_ids == class_id
            class_areas = box_areas[class_mask]
            
            class_size_stats[int(class_id)]['small'] = int(np.sum(class_areas < SMALL_AREA_THRESHOLD))
            class_size_stats[int(class_id)]['medium'] = int(np.sum(
                (class_areas >= SMALL_AREA_THRESHOLD) & (class_areas < MEDIUM_AREA_THRESHOLD)
            ))
            class_size_stats[int(class_id)]['large'] = int(np.sum(class_areas >= MEDIUM_AREA_THRESHOLD))
    else:
        box_areas = []
        box_aspect_ratios = []
        spatial_grid = np.zeros((3, 3))
        size_counts = {'small': 0, 'medium': 0, 'large': 0}
        class_size_stats = {}
    
    # Calculate totals
    stats.total_images = len(annotation_lines)
    stats.total_objects = sum(class_counts.values())
    stats.num_classes = len(class_counts)
    stats.objects_per_image = objects_per_image
    stats.image_dimensions = image_dims
    stats.box_areas = box_areas.tolist() if isinstance(box_areas, np.ndarray) else box_areas
    stats.box_aspect_ratios = box_aspect_ratios.tolist() if isinstance(box_aspect_ratios, np.ndarray) else box_aspect_ratios
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
    
    # Class statistics (no double pass needed!)
    print("Computing class statistics...")
    if stats.total_objects > 0:
        for class_id in tqdm(class_counts.keys(), desc="Processing classes", unit="classes"):
            count = class_counts[class_id]
            class_name = class_names.get(class_id, f"Class_{class_id}")
            images_with = len(class_image_counts[class_id])
            avg_per_image = count / stats.total_images if stats.total_images > 0 else 0.0
            
            # Get size distribution from pre-computed stats
            size_stats = class_size_stats.get(class_id, {'small': 0, 'medium': 0, 'large': 0})
            
            class_stat = ClassStatistics(
                class_id=class_id,
                class_name=class_name,
                total_count=count,
                percentage=(count / stats.total_objects) * 100,
                avg_per_image=avg_per_image,
                images_with_class=images_with,
                small_count=size_stats['small'],
                medium_count=size_stats['medium'],
                large_count=size_stats['large']
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

