"""CLI script for generating synthetic images."""

import argparse
import os

from sigtor.utils.config import load_config, get_config_section
from sigtor.core.generator import generate_images


def main():
    """Main entry point for the generate script."""
    # Load configuration from YAML file
    config_path = "./config.yaml"
    try:
        config = load_config(config_path)
        sigtor_config = get_config_section(config, 'SIGtor')
    except ValueError as e:
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

    args = parser.parse_args()
    
    # If config file was specified via command line, reload it
    if args.config != config_path and os.path.exists(args.config):
        config = load_config(args.config)
        sigtor_config = get_config_section(config, 'SIGtor')
        # Override with config file values if not provided via command line
        if not args.source_ann_file and sigtor_config.get('source_ann_file'):
            args.source_ann_file = sigtor_config['source_ann_file']
        if not args.destn_dir and sigtor_config.get('destn_dir'):
            args.destn_dir = sigtor_config['destn_dir']
        if not args.mask_image_dirs and sigtor_config.get('mask_image_dirs'):
            args.mask_image_dirs = sigtor_config['mask_image_dirs']
        if not args.bckgrnd_imgs_dir and sigtor_config.get('bckgrnd_imgs_dir'):
            args.bckgrnd_imgs_dir = sigtor_config['bckgrnd_imgs_dir']
        if args.total_new_imgs == 100 and sigtor_config.get('total_new_imgs'):
            args.total_new_imgs = sigtor_config['total_new_imgs']
    
    generate_images(args)


if __name__ == '__main__':
    main()

