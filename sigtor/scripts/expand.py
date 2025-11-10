"""CLI script for expanding annotations."""

import argparse
import os

from sigtor.utils.config import load_config, get_config_section
from sigtor.core.expander import expand_annotations


def main():
    """Main entry point for the expand script."""
    # Load configuration from YAML file
    config_path = "./config.yaml"
    try:
        config = load_config(config_path)
        expand_config = get_config_section(config, 'Expanding_Annotation')
    except ValueError as e:
        print(f"Warning: {e}")
        print("Falling back to command-line arguments only.")
        expand_config = {}

    parser = argparse.ArgumentParser(
        description='Object re-annotator or annotation expander for synthetic image generation.')
    parser.add_argument('--config', type=str, default=config_path,
                        help='Path to the YAML configuration file')
    parser.add_argument('--source_ann_file', type=str, required=False,
                        default=expand_config.get('source_ann_file', ''),
                        help='YOLO format annotation txt file')
    parser.add_argument('--iou_threshold', type=float, required=False,
                        default=expand_config.get('iou_threshold', 0.1),
                        help='IoU threshold for determining inner bounding boxes')
    
    args = parser.parse_args()
    
    # If config file was specified via command line, reload it
    if args.config != config_path and os.path.exists(args.config):
        config = load_config(args.config)
        expand_config = get_config_section(config, 'Expanding_Annotation')
        if not args.source_ann_file and expand_config.get('source_ann_file'):
            args.source_ann_file = expand_config['source_ann_file']
        if args.iou_threshold == 0.1 and expand_config.get('iou_threshold'):
            args.iou_threshold = expand_config['iou_threshold']
    
    expand_annotations(args.source_ann_file, args.iou_threshold)


if __name__ == "__main__":
    main()

