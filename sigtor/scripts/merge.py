"""CLI script for merging annotation files."""

import argparse
import os

from sigtor.utils.config import load_config, get_config_section
from sigtor.utils.annotation_utils import merge_annotations


def bool_from_config(value, default):
    """Helper function to convert YAML boolean/string to Python boolean."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ('true', '1', 'yes', 'on')
    return bool(value)


def main():
    """Main entry point for the merge script."""
    # Load configuration from YAML file
    config_path = "./config.yaml"
    try:
        config = load_config(config_path)
        merge_config = get_config_section(config, 'Merge')
    except (ValueError, FileNotFoundError) as e:
        print(f"Warning: {e}")
        print("Falling back to command-line arguments only.")
        merge_config = {}
    
    parser = argparse.ArgumentParser(
        description='Merge source and SIGtored annotation files into a combined file'
    )
    parser.add_argument('--config', type=str, default=config_path,
                        help='Path to the YAML configuration file')
    parser.add_argument('--source_ann_file', type=str, required=False,
                        default=merge_config.get('source_ann_file', ''),
                        help='Path to source annotation file (original dataset)')
    parser.add_argument('--sigtored_ann_file', type=str, required=False,
                        default=merge_config.get('sigtored_ann_file', ''),
                        help='Path to SIGtored annotation file (synthetic dataset)')
    parser.add_argument('--output_file', type=str, required=False,
                        default=merge_config.get('output_file', './Datasets/Combined/combined_annotations.txt'),
                        help='Output path for combined annotations')
    parser.add_argument('--shuffle', type=lambda x: (str(x).lower() == 'true'),
                        required=False, default=bool_from_config(merge_config.get('shuffle'), True),
                        help='Shuffle/randomize combined annotations (true/false)')
    
    args = parser.parse_args()
    
    # Reload config if specified
    if os.path.exists(args.config):
        config = load_config(args.config)
        merge_config = get_config_section(config, 'Merge')
        
        # Override with config values if not provided via command line
        if not args.source_ann_file and merge_config.get('source_ann_file'):
            args.source_ann_file = merge_config['source_ann_file']
        if not args.sigtored_ann_file and merge_config.get('sigtored_ann_file'):
            args.sigtored_ann_file = merge_config['sigtored_ann_file']
        if args.output_file == './Datasets/Combined/combined_annotations.txt' and merge_config.get('output_file'):
            args.output_file = merge_config['output_file']
        if merge_config.get('shuffle') is not None:
            args.shuffle = bool_from_config(merge_config['shuffle'], True)
    
    # Validate required arguments
    if not args.source_ann_file:
        print("Error: --source_ann_file is required. Provide it via command line or config file.")
        return
    
    if not args.sigtored_ann_file:
        print("Error: --sigtored_ann_file is required. Provide it via command line or config file.")
        return
    
    # Perform merge
    try:
        print(f"Merging annotations...")
        print(f"  Source: {args.source_ann_file}")
        print(f"  SIGtored: {args.sigtored_ann_file}")
        print(f"  Output: {args.output_file}")
        print(f"  Shuffle: {args.shuffle}")
        print()
        
        stats = merge_annotations(
            source_ann_file=args.source_ann_file,
            sigtored_ann_file=args.sigtored_ann_file,
            output_file=args.output_file,
            shuffle=args.shuffle,
            validate_paths=True
        )
        
        # Print statistics
        print("=" * 60)
        print("Merge Complete!")
        print("=" * 60)
        print(f"Source annotations:     {stats['source_count']:,} lines")
        print(f"SIGtored annotations:   {stats['sigtored_count']:,} lines")
        print(f"Total combined:         {stats['total_count']:,} lines")
        print(f"Output saved to:        {stats['output_path']}")
        print(f"Shuffled:               {'Yes' if stats['shuffled'] else 'No'}")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    except IOError as e:
        print(f"Error: {e}")
        return
    except Exception as e:
        print(f"Unexpected error: {e}")
        return


if __name__ == '__main__':
    main()

