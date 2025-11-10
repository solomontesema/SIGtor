"""Core image generation functionality."""

import logging
import os
from tqdm import tqdm
import numpy as np

from sigtor.processing.data_processing import select_objects_for_image
from sigtor.utils.file_ops import initialize_directories, open_annotation_file, save_new_image_and_mask
from sigtor.processing.image_composition import compose_final_image
from sigtor.processing.image_composition import handle_background_image
from sigtor.utils.index_generator import SourceIndexGenerator
from sigtor.utils.image_utils import read_ann, random_new_image_size
from sigtor.processing.context_analysis import analyze_image_context

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_image_quality(image, mask=None, min_size=(100, 100)):
    """
    Validate generated image quality.
    
    Args:
        image: Generated image (PIL Image or numpy array).
        mask: Optional mask for validation.
        min_size: Minimum image size (width, height).
    
    Returns:
        Tuple of (is_valid, issues_list).
    """
    issues = []
    
    try:
        if hasattr(image, 'size'):
            # PIL Image
            width, height = image.size
            img_array = np.array(image)
        else:
            # Numpy array
            img_array = image
            if len(img_array.shape) == 3:
                height, width = img_array.shape[:2]
            else:
                height, width = img_array.shape
        
        # Check minimum size
        if width < min_size[0] or height < min_size[1]:
            issues.append(f"Image too small: {width}x{height}")
        
        # Check for empty image
        if img_array.size == 0:
            issues.append("Empty image")
            return False, issues
        
        # Check for all black/white (might indicate failure)
        if len(img_array.shape) == 3:
            mean_val = np.mean(img_array)
            if mean_val < 5 or mean_val > 250:
                issues.append(f"Suspicious mean value: {mean_val}")
        else:
            mean_val = np.mean(img_array)
            if mean_val < 5 or mean_val > 250:
                issues.append(f"Suspicious mean value: {mean_val}")
        
        # Check for NaN or Inf values
        if np.any(np.isnan(img_array)) or np.any(np.isinf(img_array)):
            issues.append("Image contains NaN or Inf values")
            return False, issues
        
        # Check data type
        if img_array.dtype != np.uint8:
            issues.append(f"Unexpected dtype: {img_array.dtype}")
        
    except Exception as e:
        issues.append(f"Validation error: {str(e)}")
        return False, issues
    
    is_valid = len(issues) == 0
    return is_valid, issues


def generate_images(args):
    """
    Generate artificial images by selecting objects, creating composite images,
    and saving the results along with their annotations.

    Args:
        args: Configuration object with the following attributes:
            - source_ann_file: Path to source annotation file
            - destn_dir: Destination directory for output
            - mask_image_dirs: Directory containing masks (optional)
            - bckgrnd_imgs_dir: Directory containing background images
            - total_new_imgs: Number of images to generate
            - max_search_iterations: Maximum search iterations
            - blending_method: Blending method ('auto' or specific)
            - enable_post_processing: Enable post-processing
            - edge_refinement_level: Edge refinement level
            - color_harmonization: Enable color harmonization
            - context_aware_augmentations: Enable context-aware augmentations
            - quality_validation: Enable quality validation
    """
    try:
        # Read all ground-truth annotations from the source annotation file
        source_ann = read_ann(args.source_ann_file)
        if not source_ann:
            logger.error("No annotations found in source file")
            return
    except Exception as e:
        logger.error(f"Failed to read annotations: {e}")
        return

    try:
        # Initialize the necessary directories
        new_images_dir, new_masks_dir = initialize_directories(args.destn_dir)
    except Exception as e:
        logger.error(f"Failed to initialize directories: {e}")
        return

    try:
        # Open the annotation file for writing the new annotations
        new_dataset = open_annotation_file(args.destn_dir)
    except Exception as e:
        logger.error(f"Failed to open annotation file: {e}")
        return

    # Initialize the source index generator for iterating over the source annotations
    index_generator = SourceIndexGenerator(len(source_ann), shuffle=False)

    count_new_images = 0
    failed_images = 0
    rejected_images = 0
    quality_issues_log = []
    max_search_iterations = getattr(args, 'max_search_iterations', 5)
    quality_validation = getattr(args, 'quality_validation', False)
    quality_reject_threshold = getattr(args, 'quality_reject_threshold', 'critical')  # 'none', 'critical', 'all'
    context_aware_aug = getattr(args, 'context_aware_augmentations', True)

    # Generate the required number of new images
    for img_idx in tqdm(range(args.total_new_imgs), desc='Generating artificial images', leave=True):
        try:
            # Randomly determine the new image size
            target_image_size = random_new_image_size((400, 600), (400, 600))
            
            # Get background context if context-aware augmentations are enabled
            background_context = None
            if context_aware_aug:
                try:
                    # Sample a background to analyze
                    background_sample = handle_background_image(
                        args.bckgrnd_imgs_dir, target_image_size
                    )
                    if background_sample is not None:
                        background_context = analyze_image_context(background_sample)
                except Exception as e:
                    logger.debug(f"Could not analyze background context: {e}")
                    background_context = None

            # Select objects from the source annotations to create the new image
            try:
                (
                    source_img_paths, cutout_objs_images, cutout_objs_masks,
                    cutout_objs_coords, cutout_objs_inner_coords,
                    selected_objs_coords, target_image_size
                ) = select_objects_for_image(
                    max_search_iterations, target_image_size, source_ann, 
                    index_generator, args.mask_image_dirs,
                    background_context=background_context,
                    use_context_aware_aug=context_aware_aug
                )
            except Exception as e:
                logger.warning(f"Failed to select objects for image {img_idx}: {e}")
                failed_images += 1
                continue

            # Compose the final artificial image
            try:
                final_image, final_mask, final_boxes = compose_final_image(
                    args, selected_objs_coords, cutout_objs_coords,
                    cutout_objs_inner_coords, cutout_objs_images, cutout_objs_masks
                )
            except Exception as e:
                logger.warning(f"Failed to compose image {img_idx}: {e}")
                failed_images += 1
                continue

            # Validate image quality if enabled
            should_reject = False
            if quality_validation:
                try:
                    is_valid, issues = validate_image_quality(final_image)
                    if not is_valid:
                        # Determine if image should be rejected based on threshold
                        critical_issues = [issue for issue in issues if any(keyword in issue.lower() 
                                           for keyword in ['empty', 'nan', 'inf', 'dtype'])]
                        
                        if quality_reject_threshold == 'critical' and critical_issues:
                            should_reject = True
                            logger.warning(f"Image {img_idx} REJECTED due to critical quality issues: {', '.join(critical_issues)}")
                        elif quality_reject_threshold == 'all':
                            should_reject = True
                            logger.warning(f"Image {img_idx} REJECTED due to quality issues: {', '.join(issues)}")
                        else:
                            logger.warning(f"Image {img_idx} quality issues (non-critical): {', '.join(issues)}")
                        
                        # Log quality issues for report
                        quality_issues_log.append({
                            'image_idx': img_idx,
                            'issues': issues,
                            'critical': len(critical_issues) > 0,
                            'rejected': should_reject
                        })
                except Exception as e:
                    logger.debug(f"Quality validation failed for image {img_idx}: {e}")
                    # On validation error, optionally reject
                    if quality_reject_threshold in ['critical', 'all']:
                        should_reject = True
                        quality_issues_log.append({
                            'image_idx': img_idx,
                            'issues': [f"Validation error: {str(e)}"],
                            'critical': True,
                            'rejected': True
                        })
            
            # Skip saving if image was rejected
            if should_reject:
                rejected_images += 1
                continue

            # Save the generated image, mask, and their corresponding annotation
            try:
                save_new_image_and_mask(
                    final_image, final_mask, final_boxes, new_dataset,
                    new_images_dir, new_masks_dir, count_new_images
                )
                count_new_images += 1
            except Exception as e:
                logger.warning(f"Failed to save image {img_idx}: {e}")
                failed_images += 1
                continue

        except Exception as e:
            logger.error(f"Unexpected error generating image {img_idx}: {e}")
            failed_images += 1
            continue

    # Close the annotation file
    try:
        new_dataset.close()
    except Exception as e:
        logger.warning(f"Error closing annotation file: {e}")

    # Save quality report if validation was enabled
    if quality_validation and quality_issues_log:
        try:
            import json
            quality_report_path = os.path.join(args.destn_dir, 'quality_report.json')
            with open(quality_report_path, 'w') as f:
                json.dump({
                    'total_validated': len(quality_issues_log),
                    'total_issues_found': sum(len(item['issues']) for item in quality_issues_log),
                    'critical_issues': sum(1 for item in quality_issues_log if item['critical']),
                    'rejected_images': sum(1 for item in quality_issues_log if item['rejected']),
                    'issues_by_type': {},
                    'detailed_log': quality_issues_log
                }, f, indent=2)
            logger.info(f"Quality report saved to: {quality_report_path}")
        except Exception as e:
            logger.warning(f"Failed to save quality report: {e}")
    
    # Log summary
    summary_msg = f"Generation complete: {count_new_images} images generated, {failed_images} failed"
    if quality_validation and rejected_images > 0:
        summary_msg += f", {rejected_images} rejected due to quality issues"
    logger.info(summary_msg)

