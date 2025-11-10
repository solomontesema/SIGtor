"""
Report generation module for dataset analysis.

Generates text and JSON reports with insights and recommendations.
"""

import os
import json
import numpy as np
from typing import Optional, Dict
from datetime import datetime

from sigtor.analysis.dataset_analyzer import (
    DatasetStatistics, compare_datasets, ClassStatistics
)


def generate_text_report(
    stats: DatasetStatistics,
    output_path: str,
    comparison_stats: Optional[DatasetStatistics] = None
):
    """
    Generate a comprehensive text report.
    
    Args:
        stats: DatasetStatistics object
        output_path: Path to save the text report
        comparison_stats: Optional second dataset for comparison
    """
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("DATASET ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: {stats.dataset_name}\n\n")
        
        # Executive Summary
        f.write("-" * 80 + "\n")
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Images: {stats.total_images:,}\n")
        f.write(f"Total Objects: {stats.total_objects:,}\n")
        f.write(f"Number of Classes: {stats.num_classes}\n")
        if stats.total_images > 0:
            f.write(f"Average Objects per Image: {stats.total_objects / stats.total_images:.2f}\n")
        f.write(f"Imbalance Ratio: {stats.imbalance_ratio:.2f}\n")
        f.write("\n")
        
        # Class Distribution
        f.write("-" * 80 + "\n")
        f.write("CLASS DISTRIBUTION\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Class':<20} {'Count':<12} {'Percentage':<12} {'Avg/Image':<12} {'Images':<10}\n")
        f.write("-" * 80 + "\n")
        
        if stats.class_stats:
            sorted_classes = sorted(
                stats.class_stats.items(),
                key=lambda x: x[1].total_count,
                reverse=True
            )
            
            for _, class_stat in sorted_classes:
                f.write(f"{class_stat.class_name:<20} "
                       f"{class_stat.total_count:<12,} "
                       f"{class_stat.percentage:<12.2f}% "
                       f"{class_stat.avg_per_image:<12.2f} "
                       f"{class_stat.images_with_class:<10}\n")
        f.write("\n")
        
        # Size Distribution
        f.write("-" * 80 + "\n")
        f.write("SIZE DISTRIBUTION\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Category':<15} {'Count':<12} {'Percentage':<12}\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Small':<15} "
               f"{stats.size_distribution.small_count:<12,} "
               f"{stats.size_distribution.small_percentage:<12.2f}%\n")
        f.write(f"{'Medium':<15} "
               f"{stats.size_distribution.medium_count:<12,} "
               f"{stats.size_distribution.medium_percentage:<12.2f}%\n")
        f.write(f"{'Large':<15} "
               f"{stats.size_distribution.large_count:<12,} "
               f"{stats.size_distribution.large_percentage:<12.2f}%\n")
        f.write("\n")
        
        # Objects per Image Statistics
        if stats.objects_per_image:
            import numpy as np
            f.write("-" * 80 + "\n")
            f.write("OBJECTS PER IMAGE STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Mean: {np.mean(stats.objects_per_image):.2f}\n")
            f.write(f"Median: {np.median(stats.objects_per_image):.2f}\n")
            f.write(f"Min: {np.min(stats.objects_per_image)}\n")
            f.write(f"Max: {np.max(stats.objects_per_image)}\n")
            f.write(f"Std Dev: {np.std(stats.objects_per_image):.2f}\n")
            f.write("\n")
        
        # Imbalance Analysis
        f.write("-" * 80 + "\n")
        f.write("IMBALANCE ANALYSIS\n")
        f.write("-" * 80 + "\n")
        
        if stats.class_stats:
            sorted_by_count = sorted(
                stats.class_stats.items(),
                key=lambda x: x[1].total_count,
                reverse=True
            )
            
            if sorted_by_count:
                most_over = sorted_by_count[0][1]
                f.write(f"Most Overrepresented: {most_over.class_name} "
                       f"({most_over.total_count:,} objects, {most_over.percentage:.2f}%)\n")
            
            if stats.overrepresented_classes:
                f.write(f"\nOverrepresented Classes (>20% or >mean+2*std):\n")
                for class_id in stats.overrepresented_classes:
                    class_stat = stats.class_stats.get(class_id)
                    if class_stat:
                        f.write(f"  - {class_stat.class_name}: {class_stat.total_count:,} "
                               f"({class_stat.percentage:.2f}%)\n")
            
            if stats.underrepresented_classes:
                f.write(f"\nUnderrepresented Classes (<5% or <mean-2*std):\n")
                for class_id in stats.underrepresented_classes:
                    class_stat = stats.class_stats.get(class_id)
                    if class_stat:
                        f.write(f"  - {class_stat.class_name}: {class_stat.total_count:,} "
                               f"({class_stat.percentage:.2f}%)\n")
        f.write("\n")
        
        # Recommendations
        f.write("-" * 80 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("-" * 80 + "\n")
        
        recommendations = generate_recommendations(stats)
        for i, rec in enumerate(recommendations, 1):
            f.write(f"{i}. {rec}\n")
        f.write("\n")
        
        # Comparison section if provided
        if comparison_stats:
            f.write("=" * 80 + "\n")
            f.write("COMPARISON: BEFORE vs AFTER SIGTORING\n")
            f.write("=" * 80 + "\n")
            
            comparison = compare_datasets(stats, comparison_stats)
            
            f.write(f"\nDataset Growth:\n")
            f.write(f"  Images: {comparison['images_growth']:+.2f}%\n")
            f.write(f"  Objects: {comparison['objects_growth']:+.2f}%\n")
            f.write(f"  Imbalance Improvement: {comparison['imbalance_improvement']:+.2f}\n")
            
            f.write(f"\nSize Distribution Changes:\n")
            for size_cat, changes in comparison['size_distribution_changes'].items():
                f.write(f"  {size_cat.capitalize()}: "
                       f"{changes['before']:.2f}% -> {changes['after']:.2f}% "
                       f"({changes['change']:+.2f}%)\n")
            
            f.write("\n")


def generate_recommendations(stats: DatasetStatistics) -> list:
    """
    Generate recommendations based on dataset analysis.
    
    Args:
        stats: DatasetStatistics object
    
    Returns:
        List of recommendation strings
    """
    recommendations = []
    
    if not stats.class_stats:
        return recommendations
    
    # Class balance recommendations
    if stats.underrepresented_classes:
        under_classes = [stats.class_stats[cid].class_name 
                        for cid in stats.underrepresented_classes 
                        if cid in stats.class_stats]
        if under_classes:
            recommendations.append(
                f"Consider generating more images for underrepresented classes: "
                f"{', '.join(under_classes[:5])}"
            )
    
    if stats.overrepresented_classes:
        over_classes = [stats.class_stats[cid].class_name 
                       for cid in stats.overrepresented_classes 
                       if cid in stats.class_stats]
        if over_classes:
            recommendations.append(
                f"Consider undersampling or reducing generation for overrepresented classes: "
                f"{', '.join(over_classes[:5])}"
            )
    
    # Size distribution recommendations
    if stats.size_distribution.small_percentage < 20:
        recommendations.append(
            "Low percentage of small objects. Consider generating more small objects "
            "or applying scale augmentations to create smaller instances."
        )
    
    if stats.size_distribution.large_percentage < 20:
        recommendations.append(
            "Low percentage of large objects. Consider including more large objects "
            "in the source dataset or using different scaling factors."
        )
    
    # Imbalance recommendations
    if stats.imbalance_ratio > 10:
        recommendations.append(
            f"High class imbalance detected (ratio: {stats.imbalance_ratio:.2f}). "
            "Focus SIGtoring on underrepresented classes to improve balance."
        )
    
    # Objects per image recommendations
    if stats.objects_per_image:
        import numpy as np
        mean_objs = np.mean(stats.objects_per_image)
        if mean_objs < 2:
            recommendations.append(
                "Low average objects per image. Consider generating images with "
                "more objects per image to increase diversity."
            )
        elif mean_objs > 10:
            recommendations.append(
                "High average objects per image. Ensure object placement algorithm "
                "handles high-density scenarios correctly."
            )
    
    if not recommendations:
        recommendations.append(
            "Dataset appears well-balanced. Continue monitoring as dataset grows."
        )
    
    return recommendations


def generate_json_report(
    stats: DatasetStatistics,
    output_path: str,
    comparison_stats: Optional[DatasetStatistics] = None
):
    """
    Generate a JSON report for programmatic access.
    
    Args:
        stats: DatasetStatistics object
        output_path: Path to save the JSON report
        comparison_stats: Optional second dataset for comparison
    """
    report = {
        'dataset_name': stats.dataset_name,
        'generated_at': datetime.now().isoformat(),
        'summary': {
            'total_images': stats.total_images,
            'total_objects': stats.total_objects,
            'num_classes': stats.num_classes,
            'avg_objects_per_image': stats.total_objects / stats.total_images if stats.total_images > 0 else 0,
            'imbalance_ratio': stats.imbalance_ratio
        },
        'class_distribution': {},
        'size_distribution': {
            'small': {
                'count': stats.size_distribution.small_count,
                'percentage': stats.size_distribution.small_percentage
            },
            'medium': {
                'count': stats.size_distribution.medium_count,
                'percentage': stats.size_distribution.medium_percentage
            },
            'large': {
                'count': stats.size_distribution.large_count,
                'percentage': stats.size_distribution.large_percentage
            }
        },
        'objects_per_image': {
            'mean': float(np.mean(stats.objects_per_image)) if stats.objects_per_image else 0,
            'median': float(np.median(stats.objects_per_image)) if stats.objects_per_image else 0,
            'min': int(np.min(stats.objects_per_image)) if stats.objects_per_image else 0,
            'max': int(np.max(stats.objects_per_image)) if stats.objects_per_image else 0,
            'std': float(np.std(stats.objects_per_image)) if stats.objects_per_image else 0
        },
        'imbalance_analysis': {
            'underrepresented_classes': [
                {
                    'class_id': cid,
                    'class_name': stats.class_stats[cid].class_name,
                    'count': stats.class_stats[cid].total_count,
                    'percentage': stats.class_stats[cid].percentage
                }
                for cid in stats.underrepresented_classes if cid in stats.class_stats
            ],
            'overrepresented_classes': [
                {
                    'class_id': cid,
                    'class_name': stats.class_stats[cid].class_name,
                    'count': stats.class_stats[cid].total_count,
                    'percentage': stats.class_stats[cid].percentage
                }
                for cid in stats.overrepresented_classes if cid in stats.class_stats
            ]
        },
        'recommendations': generate_recommendations(stats)
    }
    
    # Add class distribution details
    for class_id, class_stat in stats.class_stats.items():
        report['class_distribution'][str(class_id)] = {
            'class_name': class_stat.class_name,
            'total_count': class_stat.total_count,
            'percentage': class_stat.percentage,
            'avg_per_image': class_stat.avg_per_image,
            'images_with_class': class_stat.images_with_class,
            'size_distribution': {
                'small': class_stat.small_count,
                'medium': class_stat.medium_count,
                'large': class_stat.large_count
            }
        }
    
    # Add comparison if provided
    if comparison_stats:
        comparison = compare_datasets(stats, comparison_stats)
        report['comparison'] = comparison
    
    # Write JSON file
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)


def generate_all_reports(
    stats: DatasetStatistics,
    output_dir: str,
    comparison_stats: Optional[DatasetStatistics] = None
):
    """
    Generate both text and JSON reports.
    
    Args:
        stats: DatasetStatistics object
        output_dir: Directory to save reports
        comparison_stats: Optional second dataset for comparison
    """
    os.makedirs(output_dir, exist_ok=True)
    
    text_path = os.path.join(output_dir, f'{stats.dataset_name}_report.txt')
    json_path = os.path.join(output_dir, f'{stats.dataset_name}_report.json')
    
    generate_text_report(stats, text_path, comparison_stats)
    generate_json_report(stats, json_path, comparison_stats)

