"""
Visualization module for dataset analysis.

Generates comprehensive plots and charts for dataset statistics.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, List, Tuple
from pathlib import Path

from sigtor.analysis.dataset_analyzer import DatasetStatistics, SMALL_AREA_THRESHOLD, MEDIUM_AREA_THRESHOLD

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def plot_class_distribution(
    stats: DatasetStatistics,
    output_path: str,
    figsize: Tuple[int, int] = (12, 8)
):
    """
    Plot class distribution as horizontal bar chart.
    
    Args:
        stats: DatasetStatistics object
        output_path: Path to save the plot
        figsize: Figure size (width, height)
    """
    if not stats.class_stats:
        return
    
    # Sort classes by count
    sorted_classes = sorted(
        stats.class_stats.items(),
        key=lambda x: x[1].total_count,
        reverse=True
    )
    
    class_names = [stat.class_name for _, stat in sorted_classes]
    counts = [stat.total_count for _, stat in sorted_classes]
    percentages = [stat.percentage for _, stat in sorted_classes]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Bar chart with counts
    y_pos = np.arange(len(class_names))
    ax1.barh(y_pos, counts, color='steelblue')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(class_names)
    ax1.set_xlabel('Object Count')
    ax1.set_title('Class Distribution (Count)')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (count, pct) in enumerate(zip(counts, percentages)):
        ax1.text(count, i, f' {count} ({pct:.1f}%)', 
                va='center', fontsize=9)
    
    # Pie chart
    ax2.pie(counts, labels=class_names, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Class Distribution (Percentage)')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()


def plot_size_distribution(
    stats: DatasetStatistics,
    output_path: str,
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Plot object size distribution.
    
    Args:
        stats: DatasetStatistics object
        output_path: Path to save the plot
        figsize: Figure size
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Overall size distribution
    sizes = ['Small', 'Medium', 'Large']
    counts = [
        stats.size_distribution.small_count,
        stats.size_distribution.medium_count,
        stats.size_distribution.large_count
    ]
    percentages = [
        stats.size_distribution.small_percentage,
        stats.size_distribution.medium_percentage,
        stats.size_distribution.large_percentage
    ]
    
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    ax1.bar(sizes, counts, color=colors)
    ax1.set_ylabel('Count')
    ax1.set_title('Size Distribution (Count)')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add percentage labels
    for i, (count, pct) in enumerate(zip(counts, percentages)):
        ax1.text(i, count, f'\n{count}\n({pct:.1f}%)', 
                ha='center', va='bottom', fontsize=9)
    
    # Pie chart
    ax2.pie(counts, labels=sizes, autopct='%1.1f%%', 
           colors=colors, startangle=90)
    ax2.set_title('Size Distribution (Percentage)')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()


def plot_objects_per_image(
    stats: DatasetStatistics,
    output_path: str,
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Plot histogram of objects per image.
    
    Args:
        stats: DatasetStatistics object
        output_path: Path to save the plot
        figsize: Figure size
    """
    if not stats.objects_per_image:
        return
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.hist(stats.objects_per_image, bins=min(50, max(10, len(set(stats.objects_per_image)))),
           color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Objects per Image')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Objects per Image')
    ax.grid(axis='y', alpha=0.3)
    
    # Add statistics
    mean_objs = np.mean(stats.objects_per_image)
    median_objs = np.median(stats.objects_per_image)
    ax.axvline(mean_objs, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_objs:.2f}')
    ax.axvline(median_objs, color='green', linestyle='--', linewidth=2, label=f'Median: {median_objs:.2f}')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()


def plot_image_dimensions(
    stats: DatasetStatistics,
    output_path: str,
    figsize: Tuple[int, int] = (10, 8)
):
    """
    Plot scatter plot of image dimensions.
    
    Args:
        stats: DatasetStatistics object
        output_path: Path to save the plot
        figsize: Figure size
    """
    if not stats.image_dimensions:
        return
    
    widths = [w for w, h in stats.image_dimensions]
    heights = [h for w, h in stats.image_dimensions]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.scatter(widths, heights, alpha=0.5, s=20, color='steelblue')
    ax.set_xlabel('Image Width (pixels)')
    ax.set_ylabel('Image Height (pixels)')
    ax.set_title('Image Dimension Distribution')
    ax.grid(True, alpha=0.3)
    
    # Add aspect ratio lines
    if widths and heights:
        max_dim = max(max(widths), max(heights))
        ax.plot([0, max_dim], [0, max_dim], 'r--', alpha=0.5, label='1:1 Aspect Ratio')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()


def plot_box_area_distribution(
    stats: DatasetStatistics,
    output_path: str,
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Plot distribution of bounding box areas (log scale).
    
    Args:
        stats: DatasetStatistics object
        output_path: Path to save the plot
        figsize: Figure size
    """
    if not stats.box_areas:
        return
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use log scale for better visualization
    log_areas = np.log10(np.array(stats.box_areas) + 1)
    
    ax.hist(log_areas, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Log10(Box Area + 1)')
    ax.set_ylabel('Frequency')
    ax.set_title('Bounding Box Area Distribution (Log Scale)')
    ax.grid(axis='y', alpha=0.3)
    
    # Add size category thresholds
    small_thresh = np.log10(SMALL_AREA_THRESHOLD + 1)
    medium_thresh = np.log10(MEDIUM_AREA_THRESHOLD + 1)
    ax.axvline(small_thresh, color='red', linestyle='--', alpha=0.7, label='Small/Medium threshold')
    ax.axvline(medium_thresh, color='orange', linestyle='--', alpha=0.7, label='Medium/Large threshold')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()


def plot_aspect_ratio_distribution(
    stats: DatasetStatistics,
    output_path: str,
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Plot distribution of bounding box aspect ratios.
    
    Args:
        stats: DatasetStatistics object
        output_path: Path to save the plot
        figsize: Figure size
    """
    if not stats.box_aspect_ratios:
        return
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Filter out extreme values for better visualization
    ratios = np.array(stats.box_aspect_ratios)
    ratios = ratios[(ratios > 0) & (ratios < 10)]  # Reasonable range
    
    ax.hist(ratios, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Aspect Ratio (Width/Height)')
    ax.set_ylabel('Frequency')
    ax.set_title('Bounding Box Aspect Ratio Distribution')
    ax.grid(axis='y', alpha=0.3)
    
    # Add 1:1 line
    ax.axvline(1.0, color='red', linestyle='--', alpha=0.7, label='1:1 (Square)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()


def plot_spatial_distribution(
    stats: DatasetStatistics,
    output_path: str,
    figsize: Tuple[int, int] = (8, 8)
):
    """
    Plot spatial heatmap showing where objects appear in images.
    
    Args:
        stats: DatasetStatistics object
        output_path: Path to save the plot
        figsize: Figure size
    """
    if stats.spatial_distribution.sum() == 0:
        return
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Normalize for better visualization
    normalized = stats.spatial_distribution / stats.spatial_distribution.sum() * 100
    
    im = ax.imshow(normalized, cmap='YlOrRd', aspect='auto')
    
    # Add text annotations
    for i in range(3):
        for j in range(3):
            text = ax.text(j, i, f'{normalized[i, j]:.1f}%',
                          ha="center", va="center", color="black", fontweight='bold')
    
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(['Left', 'Center', 'Right'])
    ax.set_yticklabels(['Top', 'Middle', 'Bottom'])
    ax.set_title('Spatial Distribution of Objects\n(3x3 Grid)')
    
    plt.colorbar(im, ax=ax, label='Percentage of Objects')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()


def plot_size_by_class(
    stats: DatasetStatistics,
    output_path: str,
    figsize: Tuple[int, int] = (14, 8)
):
    """
    Plot stacked bar chart showing size distribution per class.
    
    Args:
        stats: DatasetStatistics object
        output_path: Path to save the plot
        figsize: Figure size
    """
    if not stats.class_stats:
        return
    
    # Sort classes by total count
    sorted_classes = sorted(
        stats.class_stats.items(),
        key=lambda x: x[1].total_count,
        reverse=True
    )
    
    class_names = [stat.class_name for _, stat in sorted_classes]
    small_counts = [stat.small_count for _, stat in sorted_classes]
    medium_counts = [stat.medium_count for _, stat in sorted_classes]
    large_counts = [stat.large_count for _, stat in sorted_classes]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(class_names))
    width = 0.6
    
    ax.bar(x, small_counts, width, label='Small', color='#ff9999')
    ax.bar(x, medium_counts, width, bottom=small_counts, label='Medium', color='#66b3ff')
    ax.bar(x, large_counts, width, 
          bottom=np.array(small_counts) + np.array(medium_counts), 
          label='Large', color='#99ff99')
    
    ax.set_ylabel('Object Count')
    ax.set_title('Size Distribution by Class')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()


def plot_comparison(
    stats_before: DatasetStatistics,
    stats_after: DatasetStatistics,
    output_dir: str
):
    """
    Generate comparison plots between two datasets.
    
    Args:
        stats_before: Statistics from source dataset
        stats_after: Statistics from SIGtored dataset
        output_dir: Directory to save comparison plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Comparison: Class distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Before
    if stats_before.class_stats:
        sorted_before = sorted(
            stats_before.class_stats.items(),
            key=lambda x: x[1].total_count,
            reverse=True
        )
        names_before = [stat.class_name for _, stat in sorted_before]
        counts_before = [stat.total_count for _, stat in sorted_before]
        
        ax1.barh(range(len(names_before)), counts_before, color='steelblue')
        ax1.set_yticks(range(len(names_before)))
        ax1.set_yticklabels(names_before)
        ax1.set_xlabel('Object Count')
        ax1.set_title(f'Before SIGtoring\n({stats_before.dataset_name})')
        ax1.invert_yaxis()
        ax1.grid(axis='x', alpha=0.3)
    
    # After
    if stats_after.class_stats:
        sorted_after = sorted(
            stats_after.class_stats.items(),
            key=lambda x: x[1].total_count,
            reverse=True
        )
        names_after = [stat.class_name for _, stat in sorted_after]
        counts_after = [stat.total_count for _, stat in sorted_after]
        
        ax2.barh(range(len(names_after)), counts_after, color='green')
        ax2.set_yticks(range(len(names_after)))
        ax2.set_yticklabels(names_after)
        ax2.set_xlabel('Object Count')
        ax2.set_title(f'After SIGtoring\n({stats_after.dataset_name})')
        ax2.invert_yaxis()
        ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_class_distribution.png'), 
               bbox_inches='tight', dpi=300)
    plt.close()
    
    # Comparison: Size distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    sizes = ['Small', 'Medium', 'Large']
    before_counts = [
        stats_before.size_distribution.small_count,
        stats_before.size_distribution.medium_count,
        stats_before.size_distribution.large_count
    ]
    after_counts = [
        stats_after.size_distribution.small_count,
        stats_after.size_distribution.medium_count,
        stats_after.size_distribution.large_count
    ]
    
    ax1.bar(sizes, before_counts, color=['#ff9999', '#66b3ff', '#99ff99'])
    ax1.set_ylabel('Count')
    ax1.set_title('Before SIGtoring')
    ax1.grid(axis='y', alpha=0.3)
    
    ax2.bar(sizes, after_counts, color=['#ff9999', '#66b3ff', '#99ff99'])
    ax2.set_ylabel('Count')
    ax2.set_title('After SIGtoring')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_size_distribution.png'), 
               bbox_inches='tight', dpi=300)
    plt.close()


def generate_all_plots(
    stats: DatasetStatistics,
    output_dir: str,
    comparison_stats: Optional[DatasetStatistics] = None
):
    """
    Generate all visualization plots for a dataset.
    
    Args:
        stats: DatasetStatistics object
        output_dir: Directory to save plots
        comparison_stats: Optional second dataset for comparison
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plot_class_distribution(stats, os.path.join(output_dir, 'class_distribution.png'))
    plot_size_distribution(stats, os.path.join(output_dir, 'size_distribution.png'))
    plot_objects_per_image(stats, os.path.join(output_dir, 'objects_per_image.png'))
    plot_image_dimensions(stats, os.path.join(output_dir, 'image_dimensions.png'))
    plot_box_area_distribution(stats, os.path.join(output_dir, 'box_area_distribution.png'))
    plot_aspect_ratio_distribution(stats, os.path.join(output_dir, 'aspect_ratio_distribution.png'))
    plot_spatial_distribution(stats, os.path.join(output_dir, 'spatial_distribution.png'))
    plot_size_by_class(stats, os.path.join(output_dir, 'size_by_class.png'))
    
    if comparison_stats:
        plot_comparison(stats, comparison_stats, output_dir)

