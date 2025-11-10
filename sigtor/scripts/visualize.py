"""CLI script for visualizing annotations."""

import argparse
import os

from sigtor.utils.config import load_config, get_config_section
from sigtor.utils.image_utils import get_classes
from sigtor.core.visualizer import visualize_annotations


def main():
    """Main entry point for the visualize script."""
    # Load configuration from YAML file
    config_path = "./config.yaml"
    try:
        config = load_config(config_path)
        test_config = get_config_section(config, 'Test')
    except ValueError as e:
        print(f"Warning: {e}")
        print("Falling back to command-line arguments only.")
        test_config = {}

    parser = argparse.ArgumentParser(
        description='Supplementary Synthetic Image Generation for Object Detection and Segmentation')
    parser.add_argument('--config', type=str, default=config_path,
                        help='Path to the YAML configuration file')
    parser.add_argument('--source_ann_file', type=str, required=False,
                        default=test_config.get('source_ann_file', ''),
                        help='YOLO format annotation txt file as a source dataset')
    parser.add_argument('--classnames_file', type=str, required=False,
                        default=test_config.get('classnames_file', ''),
                        help='Dataset object classes')
    parser.add_argument('--output_dir', type=str, required=False,
                        default=test_config.get('output_dir', './misc/images/'),
                        help='Directory to save the visualized images')
    parser.add_argument('--num_test_images', type=str, required=False,
                        default=test_config.get('num_test_images', 'All'),
                        help='Number of images to test: positive numbers or "All"')

    args = parser.parse_args()
    
    # If config file was specified via command line, reload it
    if args.config != config_path and os.path.exists(args.config):
        config = load_config(args.config)
        test_config = get_config_section(config, 'Test')
        if not args.source_ann_file and test_config.get('source_ann_file'):
            args.source_ann_file = test_config['source_ann_file']
        if not args.classnames_file and test_config.get('classnames_file'):
            args.classnames_file = test_config['classnames_file']
        if not args.output_dir and test_config.get('output_dir'):
            args.output_dir = test_config['output_dir']
        if args.num_test_images == 'All' and test_config.get('num_test_images'):
            args.num_test_images = test_config['num_test_images']
    
    if not args.classnames_file:
        raise ValueError("classnames_file must be provided either via config or command line")
    if not args.source_ann_file:
        raise ValueError("source_ann_file must be provided either via config or command line")
    
    class_names = get_classes(args.classnames_file)
    visualize_annotations(args.source_ann_file, class_names, args.output_dir, args.num_test_images)


if __name__ == '__main__':
    main()

