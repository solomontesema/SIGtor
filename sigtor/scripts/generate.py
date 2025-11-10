"""CLI script for generating synthetic images."""

import argparse
import os

from sigtor.utils.config import load_config, get_config_section
from sigtor.core.generator import generate_images


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
    """Main entry point for the generate script."""
    # Load configuration from YAML file
    config_path = "./config.yaml"
    try:
        config = load_config(config_path)
        sigtor_config = get_config_section(config, 'SIGtor')
    except (ValueError, FileNotFoundError) as e:
        print(f"Warning: {e}")
        print("Falling back to command-line arguments only.")
        sigtor_config = {}

    parser = argparse.ArgumentParser(
        description='Supplementary Synthetic Image Generation for Object Detection and Segmentation'
    )
    parser.add_argument('--config', type=str, default=config_path,
                        help='Path to the YAML configuration file')
    parser.add_argument('--source_ann_file', type=str, required=False,
                        default=sigtor_config.get('source_ann_file', ''),
                        help='YOLO format annotation txt file as a source dataset')
    parser.add_argument('--destn_dir', type=str, required=False,
                        default=sigtor_config.get('destn_dir', './Datasets/SIGtored/'),
                        help='Directory to save the generated images, their ground_truth annotations, and masks')
    parser.add_argument('--mask_image_dirs', type=str, required=False,
                        default=sigtor_config.get('mask_image_dirs', ''),
                        help='Directory where the ground-truth masks are stored, if any')
    parser.add_argument('--bckgrnd_imgs_dir', type=str, required=False,
                        default=sigtor_config.get('bckgrnd_imgs_dir', './Datasets/BackgroundImages'),
                        help='Directory where the background images are stored, if any')
    parser.add_argument('--total_new_imgs', type=int, required=False,
                        default=sigtor_config.get('total_new_imgs', 100),
                        help='Total number of new images to generate from the total annotations.')
    parser.add_argument('--max_search_iterations', type=int, required=False,
                        default=sigtor_config.get('max_search_iterations', 5),
                        help='Maximum iterations for object selection')
    parser.add_argument('--blending_method', type=str, required=False,
                        default=sigtor_config.get('blending_method', 'auto'),
                        help='Blending method: auto, SoftPaste, NormalClone, MixedClone, etc.')
    parser.add_argument('--enable_post_processing', type=lambda x: (str(x).lower() == 'true'),
                        required=False, default=bool_from_config(sigtor_config.get('enable_post_processing'), True),
                        help='Enable post-processing (true/false)')
    parser.add_argument('--edge_refinement_level', type=str, required=False,
                        default=sigtor_config.get('edge_refinement_level', 'medium'),
                        choices=['low', 'medium', 'high'],
                        help='Edge refinement level: low, medium, or high')
    parser.add_argument('--color_harmonization', type=lambda x: (str(x).lower() == 'true'),
                        required=False, default=bool_from_config(sigtor_config.get('color_harmonization'), True),
                        help='Enable color harmonization (true/false)')
    parser.add_argument('--context_aware_augmentations', type=lambda x: (str(x).lower() == 'true'),
                        required=False, default=bool_from_config(sigtor_config.get('context_aware_augmentations'), True),
                        help='Enable context-aware augmentations (true/false)')
    parser.add_argument('--quality_validation', type=lambda x: (str(x).lower() == 'true'),
                        required=False, default=bool_from_config(sigtor_config.get('quality_validation'), False),
                        help='Enable quality validation (true/false)')
    parser.add_argument('--quality_reject_threshold', type=str, required=False,
                        default=sigtor_config.get('quality_reject_threshold', 'critical'),
                        choices=['none', 'critical', 'all'],
                        help='Quality rejection threshold: none (log only), critical (reject critical issues), all (reject all issues)')

    args = parser.parse_args()
    
    # Always reload config to ensure all values are properly loaded
    # This handles both default config and custom config file
    if os.path.exists(args.config):
        config = load_config(args.config)
        sigtor_config = get_config_section(config, 'SIGtor')
        
        # Override with config file values if not provided via command line
        # (only override if using default values or if explicitly in config)
        if not args.source_ann_file or (args.source_ann_file == sigtor_config.get('source_ann_file', '')):
            if sigtor_config.get('source_ann_file'):
                args.source_ann_file = sigtor_config['source_ann_file']
        if not args.destn_dir or args.destn_dir == './Datasets/SIGtored/':
            if sigtor_config.get('destn_dir'):
                args.destn_dir = sigtor_config['destn_dir']
        if not args.mask_image_dirs:
            if sigtor_config.get('mask_image_dirs'):
                args.mask_image_dirs = sigtor_config['mask_image_dirs']
        if args.bckgrnd_imgs_dir == './Datasets/BackgroundImages':
            if sigtor_config.get('bckgrnd_imgs_dir'):
                args.bckgrnd_imgs_dir = sigtor_config['bckgrnd_imgs_dir']
        if args.total_new_imgs == 100:
            if sigtor_config.get('total_new_imgs'):
                args.total_new_imgs = sigtor_config['total_new_imgs']
        
        # Always load advanced configuration options from config file
        if 'max_search_iterations' in sigtor_config:
            args.max_search_iterations = sigtor_config['max_search_iterations']
        if 'blending_method' in sigtor_config:
            args.blending_method = sigtor_config['blending_method']
        if 'enable_post_processing' in sigtor_config:
            args.enable_post_processing = bool_from_config(sigtor_config['enable_post_processing'], True)
        if 'edge_refinement_level' in sigtor_config:
            args.edge_refinement_level = sigtor_config['edge_refinement_level']
        if 'color_harmonization' in sigtor_config:
            args.color_harmonization = bool_from_config(sigtor_config['color_harmonization'], True)
        if 'context_aware_augmentations' in sigtor_config:
            args.context_aware_augmentations = bool_from_config(sigtor_config['context_aware_augmentations'], True)
        if 'quality_validation' in sigtor_config:
            args.quality_validation = bool_from_config(sigtor_config['quality_validation'], False)
        if 'quality_reject_threshold' in sigtor_config:
            args.quality_reject_threshold = sigtor_config['quality_reject_threshold']
    
    generate_images(args)


if __name__ == '__main__':
    main()

