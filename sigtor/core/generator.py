"""Core image generation functionality."""

from tqdm import tqdm

from sigtor.processing.data_processing import select_objects_for_image
from sigtor.utils.file_ops import initialize_directories, open_annotation_file, save_new_image_and_mask
from sigtor.processing.image_composition import compose_final_image
from sigtor.utils.index_generator import SourceIndexGenerator
from sigtor.utils.image_utils import read_ann, random_new_image_size


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
    """
    # Read all ground-truth annotations from the source annotation file
    source_ann = read_ann(args.source_ann_file)

    # Initialize the necessary directories
    new_images_dir, new_masks_dir = initialize_directories(args.destn_dir)

    # Open the annotation file for writing the new annotations
    new_dataset = open_annotation_file(args.destn_dir)

    # Initialize the source index generator for iterating over the source annotations
    index_generator = SourceIndexGenerator(len(source_ann), shuffle=False)

    count_new_images = 0
    max_search_iterations = 5

    # Generate the required number of new images
    for _ in tqdm(range(args.total_new_imgs), desc='Generating artificial images', leave=True):
        # Randomly determine the new image size
        target_image_size = random_new_image_size((400, 600), (400, 600))

        # Select objects from the source annotations to create the new image
        (
            source_img_paths, cutout_objs_images, cutout_objs_masks,
            cutout_objs_coords, cutout_objs_inner_coords,
            selected_objs_coords, target_image_size
        ) = select_objects_for_image(
            max_search_iterations, target_image_size, source_ann, index_generator, args.mask_image_dirs
        )

        # Compose the final artificial image
        final_image, final_mask, final_boxes = compose_final_image(
            args, selected_objs_coords, cutout_objs_coords,
            cutout_objs_inner_coords, cutout_objs_images, cutout_objs_masks
        )

        # Save the generated image, mask, and their corresponding annotation
        save_new_image_and_mask(
            final_image, final_mask, final_boxes, new_dataset,
            new_images_dir, new_masks_dir, count_new_images
        )

        count_new_images += 1

    # Close the annotation file
    new_dataset.close()

