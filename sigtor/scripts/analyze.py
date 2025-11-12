"""CLI script for dataset analysis."""

import argparse
import os
from tqdm import tqdm

from sigtor.utils.config import load_config, get_config_section
from sigtor.analysis.dataset_analyzer import analyze_dataset, compare_datasets
from sigtor.analysis.visualizer import generate_all_plots
from sigtor.analysis.report_generator import generate_all_reports


def main():
    """Main entry point for the analyze script."""
    # Load configuration from YAML file
    config_path = "./config.yaml"
    try:
        config = load_config(config_path)
        analysis_config = get_config_section(config, 'Analysis')
    except (ValueError, FileNotFoundError) as e:
        analysis_config = {}
    
    parser = argparse.ArgumentParser(
        description='Dataset Analysis Tool for SIGtor - Analyze YOLO-format annotation files'
    )
    parser.add_argument('--config', type=str, default=config_path,
                        help='Path to the YAML configuration file')
    parser.add_argument('--source_ann_file', type=str, required=False,
                        default=analysis_config.get('source_ann_file', ''),
                        help='Path to source annotation file (before SIGtoring)')
    parser.add_argument('--sigtored_ann_file', type=str, required=False,
                        default=analysis_config.get('sigtored_ann_file', ''),
                        help='Path to SIGtored annotation file (after SIGtoring)')
    parser.add_argument('--classnames_file', type=str, required=False,
                        default=analysis_config.get('classnames_file', ''),
                        help='Path to class names file')
    parser.add_argument('--output_dir', type=str, required=False,
                        default=analysis_config.get('output_dir', './analysis_reports/'),
                        help='Directory to save analysis reports and plots')
    parser.add_argument('--generate_plots', type=lambda x: (str(x).lower() == 'true'),
                        required=False, default=analysis_config.get('generate_plots', True),
                        help='Generate visualization plots (true/false)')
    parser.add_argument('--generate_report', type=lambda x: (str(x).lower() == 'true'),
                        required=False, default=analysis_config.get('generate_report', True),
                        help='Generate text and JSON reports (true/false)')
    parser.add_argument('--comparison_mode', type=lambda x: (str(x).lower() == 'true'),
                        required=False, default=analysis_config.get('comparison_mode', False),
                        help='Compare source vs SIGtored datasets (true/false)')
    
    args = parser.parse_args()
    
    # Reload config if specified
    if os.path.exists(args.config):
        config = load_config(args.config)
        analysis_config = get_config_section(config, 'Analysis')
        
        # Override with config values if not provided via command line
        if not args.source_ann_file and analysis_config.get('source_ann_file'):
            args.source_ann_file = analysis_config['source_ann_file']
        if not args.sigtored_ann_file and analysis_config.get('sigtored_ann_file'):
            args.sigtored_ann_file = analysis_config['sigtored_ann_file']
        if not args.classnames_file and analysis_config.get('classnames_file'):
            args.classnames_file = analysis_config['classnames_file']
        if args.output_dir == './analysis_reports/' and analysis_config.get('output_dir'):
            args.output_dir = analysis_config['output_dir']
        if analysis_config.get('generate_plots') is not None:
            args.generate_plots = analysis_config['generate_plots']
        if analysis_config.get('generate_report') is not None:
            args.generate_report = analysis_config['generate_report']
        if analysis_config.get('comparison_mode') is not None:
            args.comparison_mode = analysis_config['comparison_mode']
    
    # Validate inputs
    if not args.source_ann_file and not args.sigtored_ann_file:
        print("Error: At least one annotation file (--source_ann_file or --sigtored_ann_file) must be provided.")
        return
    
    if args.comparison_mode and (not args.source_ann_file or not args.sigtored_ann_file):
        print("Error: Both --source_ann_file and --sigtored_ann_file are required for comparison mode.")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    stats_before = None
    stats_after = None
    
    # Analyze source dataset
    if args.source_ann_file and os.path.exists(args.source_ann_file):
        print(f"\n{'='*80}")
        print(f"Analyzing source dataset: {args.source_ann_file}")
        print(f"{'='*80}")
        stats_before = analyze_dataset(
            args.source_ann_file,
            args.classnames_file,
            dataset_name="source",
            load_image_dims=False  # Skip image loading for speed (use defaults for spatial calc)
        )
        print(f"\n✓ Found {stats_before.total_images:,} images with {stats_before.total_objects:,} objects")
    elif args.comparison_mode:
        print("Warning: Source annotation file not found, skipping source analysis")
    
    # Analyze SIGtored dataset
    if args.sigtored_ann_file and os.path.exists(args.sigtored_ann_file):
        print(f"\n{'='*80}")
        print(f"Analyzing SIGtored dataset: {args.sigtored_ann_file}")
        print(f"{'='*80}")
        stats_after = analyze_dataset(
            args.sigtored_ann_file,
            args.classnames_file,
            dataset_name="sigtored",
            load_image_dims=False  # Skip image loading for speed (use defaults for spatial calc)
        )
        print(f"\n✓ Found {stats_after.total_images:,} images with {stats_after.total_objects:,} objects")
    elif args.comparison_mode:
        print("Warning: SIGtored annotation file not found, skipping SIGtored analysis")
    
    # Determine which dataset to analyze/visualize
    if args.comparison_mode and stats_before and stats_after:
        # Comparison mode
        print("\nGenerating comparison analysis...")
        
        if args.generate_plots:
            print("  Generating comparison plots...")
            generate_all_plots(stats_after, args.output_dir, comparison_stats=stats_before)
            print(f"  Plots saved to: {args.output_dir}")
        
        if args.generate_report:
            print("  Generating comparison reports...")
            generate_all_reports(stats_after, args.output_dir, comparison_stats=stats_before)
            print(f"  Reports saved to: {args.output_dir}")
        
        # Print summary
        comparison = compare_datasets(stats_before, stats_after)
        print("\n" + "=" * 80)
        print("COMPARISON SUMMARY")
        print("=" * 80)
        print(f"Images Growth: {comparison['images_growth']:+.2f}%")
        print(f"Objects Growth: {comparison['objects_growth']:+.2f}%")
        print(f"Imbalance Improvement: {comparison['imbalance_improvement']:+.2f}")
        print("=" * 80)
        
    elif stats_after:
        # Analyze SIGtored only
        print("\nGenerating analysis for SIGtored dataset...")
        
        if args.generate_plots:
            print("  Generating plots...")
            generate_all_plots(stats_after, args.output_dir)
            print(f"  Plots saved to: {args.output_dir}")
        
        if args.generate_report:
            print("  Generating reports...")
            generate_all_reports(stats_after, args.output_dir)
            print(f"  Reports saved to: {args.output_dir}")
        
        print_summary(stats_after)
        
    elif stats_before:
        # Analyze source only
        print("\nGenerating analysis for source dataset...")
        
        if args.generate_plots:
            print("  Generating plots...")
            generate_all_plots(stats_before, args.output_dir)
            print(f"  Plots saved to: {args.output_dir}")
        
        if args.generate_report:
            print("  Generating reports...")
            generate_all_reports(stats_before, args.output_dir)
            print(f"  Reports saved to: {args.output_dir}")
        
        print_summary(stats_before)
    
    print("\nAnalysis complete!")


def print_summary(stats):
    """Print a quick summary of dataset statistics."""
    print("\n" + "=" * 80)
    print("DATASET SUMMARY")
    print("=" * 80)
    print(f"Dataset: {stats.dataset_name}")
    print(f"Total Images: {stats.total_images:,}")
    print(f"Total Objects: {stats.total_objects:,}")
    print(f"Number of Classes: {stats.num_classes}")
    print(f"Imbalance Ratio: {stats.imbalance_ratio:.2f}")
    print(f"\nSize Distribution:")
    print(f"  Small: {stats.size_distribution.small_count:,} ({stats.size_distribution.small_percentage:.2f}%)")
    print(f"  Medium: {stats.size_distribution.medium_count:,} ({stats.size_distribution.medium_percentage:.2f}%)")
    print(f"  Large: {stats.size_distribution.large_count:,} ({stats.size_distribution.large_percentage:.2f}%)")
    if stats.underrepresented_classes:
        under = [stats.class_stats[cid].class_name for cid in stats.underrepresented_classes[:5] 
                if cid in stats.class_stats]
        print(f"\nUnderrepresented Classes: {', '.join(under)}")
    if stats.overrepresented_classes:
        over = [stats.class_stats[cid].class_name for cid in stats.overrepresented_classes[:5] 
               if cid in stats.class_stats]
        print(f"Overrepresented Classes: {', '.join(over)}")
    print("=" * 80)


if __name__ == '__main__':
    main()

