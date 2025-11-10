import os
import numpy as np
from PIL import Image
from typing import Dict, Tuple, List, Any

from sigtor.processing.augmentation import random_augmentations, heuristic_augmentations
from sigtor.utils.image_utils import recalculate_targetsize, overlap_measure


def convert_instance_to_binary(mask: np.ndarray) -> np.ndarray:
    """
    Convert a colored segmentation mask to a binary mask where non-background pixels
    are set to 255 (white) and background pixels are set to 0 (black).

    Args:
        mask (np.ndarray): The input colored segmentation mask.

    Returns:
        np.ndarray: The binary mask image.
    """
    binary_mask = np.where(mask > 0, 255, 0).astype(np.uint8)

    return binary_mask


def get_outerbox(org_boxes: np.ndarray) -> np.ndarray:
    """
    Calculate the outer bounding box that encompasses all overlapping or partially overlapping objects.

    Args:
        org_boxes (np.ndarray): A numpy array of shape (n, 4) where each row represents
                                the coordinates [x1, y1, x2, y2] of an object's bounding box.

    Returns:
        np.ndarray: A numpy array of shape (1, 4) representing the coordinates [x1, y1, x2, y2]
                    of the outer bounding box that encompasses all the input bounding boxes.
    """
    if org_boxes.size == 0:
        raise ValueError("Input bounding boxes array is empty.")

    x1 = np.min(org_boxes[:, 0])
    y1 = np.min(org_boxes[:, 1])
    x2 = np.max(org_boxes[:, 2])
    y2 = np.max(org_boxes[:, 3])

    outerbox = np.array([x1, y1, x2, y2]).reshape(-1, 4)
    return outerbox


def get_area(box: np.ndarray) -> int:
    """
    Calculate the area of a bounding box.

    Args:
        box (np.ndarray): A bounding box in the format [x1, y1, x2, y2].

    Returns:
        int: The area of the bounding box.
    """
    x1, y1, x2, y2 = box
    return (x2 - x1) * (y2 - y1)


def sort_boxes_by_area(boxes: np.ndarray) -> List[int]:
    """
    Sort bounding boxes by their area in descending order.

    Args:
        boxes (np.ndarray): An array of bounding boxes in the format (n, 4).

    Returns:
        List[int]: Indices of the sorted bounding boxes.
    """
    areas = np.apply_along_axis(get_area, 1, boxes)
    return np.argsort(-areas)  # Sort in descending order


def is_overlap(box1: np.ndarray, box2: np.ndarray) -> bool:
    """
    Check if two bounding boxes overlap.

    Args:
        box1 (np.ndarray): The first bounding box [x1, y1, x2, y2].
        box2 (np.ndarray): The second bounding box [x1, y1, x2, y2].

    Returns:
        bool: True if the boxes overlap, False otherwise.
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    return not (x1_1 >= x2_2 or x1_2 >= x2_1 or y1_1 >= y2_2 or y1_2 >= y2_1)


def place_box(vertex_pool: List[Tuple[int, int]], box_size: Tuple[int, int], target_size: Tuple[int, int]) -> Tuple[
    np.ndarray, bool]:
    """
    Attempt to place a box on the target image using available vertices.

    Args:
        vertex_pool (List[Tuple[int, int]]): List of available vertices (x, y).
        box_size (Tuple[int, int]): The size of the box (width, height).
        target_size (Tuple[int, int]): The size of the target image (width, height).

    Returns:
        Tuple[np.ndarray, bool]: The new bounding box coordinates and a success flag.
    """
    w, h = box_size
    target_w, target_h = target_size

    for i, (vx, vy) in enumerate(vertex_pool):
        if vx + w <= target_w and vy + h <= target_h:
            new_box = np.array([vx, vy, vx + w, vy + h])
            return new_box, True

    return np.array([0, 0, 0, 0]), False


def update_vertex_pool(vertex_pool: List[Tuple[int, int]], new_box: np.ndarray) -> List[Tuple[int, int]]:
    """
    Update the vertex pool with new vertices after placing a box.

    Args:
        vertex_pool (List[Tuple[int, int]]): Current list of vertices.
        new_box (np.ndarray): The placed box coordinates [x1, y1, x2, y2].

    Returns:
        List[Tuple[int, int]]: Updated list of vertices.
    """
    x1, y1, x2, y2 = new_box
    vertex_pool.append((x2, y1))
    vertex_pool.append((x1, y2))
    vertex_pool.append((x2, y2))

    # Remove the vertex from which this box was placed
    vertex_pool = [(vx, vy) for (vx, vy) in vertex_pool if not (vx == x1 and vy == y1)]
    return vertex_pool


def get_pastecoords(target_size: Tuple[int, int], all_outer_boxes: np.ndarray) -> Dict[int, np.ndarray]:
    target_w, target_h = target_size
    sorted_indices = sort_boxes_by_area(all_outer_boxes)

    vertex_pool = [(0, 0)]
    paste_coords = {}

    for obj_id in sorted_indices:
        obj_w, obj_h = all_outer_boxes[obj_id, 2] - all_outer_boxes[obj_id, 0], all_outer_boxes[obj_id, 3] - \
                       all_outer_boxes[obj_id, 1]
        new_box, success = place_box(vertex_pool, (obj_w, obj_h), target_size)

        if success:
            overlap = False
            for prev_box in paste_coords.values():
                _, iol, _ = overlap_measure(new_box.reshape(-1, 4), prev_box)
                if iol != 0.0:
                    overlap = True
                    break

            if not overlap:
                paste_coords[obj_id] = new_box
                vertex_pool = update_vertex_pool(vertex_pool, new_box)

    return paste_coords


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1 (np.ndarray): The first bounding box [x1, y1, x2, y2].
        box2 (np.ndarray): The second bounding box [x1, y1, x2, y2].

    Returns:
        float: The IoU between the two bounding boxes.
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Determine the coordinates of the intersection rectangle
    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)

    # Calculate the area of the intersection rectangle
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Calculate the area of both bounding boxes
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)

    # Calculate the IoU
    iou = inter_area / (box1_area + box2_area - inter_area)
    return iou


def get_total_iol(target_image_size: Tuple[int, int], selected_objs_coords: Dict[int, np.ndarray]) -> float:
    """
    Calculate the total Intersection over Larger image (IoL) for all selected objects
    against the target background size.

    Args:
        target_image_size (Tuple[int, int]): The size of the target background image (width, height).
        selected_objs_coords (Dict[int, np.ndarray]): Dictionary of selected objects and their coordinates [x1, y1, x2, y2].

    Returns:
        float: The total IoL for all selected objects.
    """
    target_box = np.array([0, 0, target_image_size[0], target_image_size[1]]).reshape(1, 4)
    total_iou = 0.0

    for obj_id, obj_box in selected_objs_coords.items():
        iou = calculate_iou(target_box.flatten(), obj_box.flatten())
        total_iou += iou

    return total_iou


def parse_annotation_line(annotation_line: str) -> Tuple[str, np.ndarray]:
    """
    Parse a single annotation line to extract the image path and object bounding boxes.

    Args:
        annotation_line (str): A line from the annotation file in the format:
                               "image_path x1,y1,x2,y2,class_index ..."

    Returns:
        Tuple[str, np.ndarray]: The image path and a numpy array of shape (n, 5)
                                where each row contains [x1, y1, x2, y2, class_index].
    """
    parts = annotation_line.split()
    imgpath = parts[0]
    boxes = np.array([list(map(float, box.split(','))) for box in parts[1:]], dtype=np.float32).reshape(-1, 5)
    return imgpath, boxes


def crop_image(image: Image.Image, outerbox: np.ndarray) -> Image.Image:
    """
    Crop the image based on the outer bounding box coordinates.

    Args:
        image (Image.Image): The source image.
        outerbox (np.ndarray): The outer bounding box as a numpy array [x1, y1, x2, y2].

    Returns:
        Image.Image: The cropped image containing the object(s).
    """
    x1, y1, x2, y2 = outerbox.flatten()
    return image.crop((x1, y1, x2, y2))


def load_mask_image(mask_path: str, cutout_coord: Tuple[int, int, int, int]) -> Image.Image:
    """Load and crop the mask image using the provided coordinates, converting borders to a distinct value."""
    mask_img = Image.open(mask_path)
    obj_mask = mask_img.crop(box=cutout_coord)

    # Convert white border (255) to 254 for differentiation
    obj_mask_array = np.array(obj_mask)
    #obj_mask_array[obj_mask_array == 255] = 254

    return Image.fromarray(obj_mask_array)



def create_rectangular_mask(cutout_coord: Tuple[int, int, int, int]) -> Image.Image:
    """Create a simple rectangular mask using the bounding box coordinates."""
    x1, y1, x2, y2 = cutout_coord
    width, height = int(x2 - x1), int(y2 - y1)
    rect_mask = Image.fromarray(255 * np.ones((height, width), dtype=np.uint8))
    return rect_mask


def get_mask_file_path(src_imgpath: str, maskdir: str) -> str:
    """Generate the corresponding mask file path based on the source image path."""
    filename = os.path.splitext(os.path.basename(src_imgpath))[0]  # Extract '1234' from '1234.jpg'
    return os.path.join(maskdir, f"{filename}.png")


def get_objmask(src_imgpath: str, cutout_coord: Tuple[int, int, int, int], maskdir: str) -> Image.Image:
    """
    Retrieve the object mask for the given image. If the corresponding mask file
    is not found, create a simple rectangular mask using the cutout coordinates.
    """
    mask_path = get_mask_file_path(src_imgpath, maskdir)

    if os.path.exists(mask_path):
        obj_mask = load_mask_image(mask_path, cutout_coord)
    else:
        obj_mask = create_rectangular_mask(cutout_coord)

    return obj_mask


def get_data(annotation_line: str, maskdir: str) -> Tuple[str, Image.Image, Image.Image, np.ndarray, np.ndarray]:
    """
    Extract data from a single annotation line, including the image path, object cutout,
    segmentation mask cutout, and bounding boxes.

    Args:
        annotation_line (str): A line from the annotation file.
        maskdir (str): Directory containing the segmentation masks.

    Returns:
        Tuple[str, Image.Image, Image.Image, np.ndarray, np.ndarray]:
            - imgpath: The path to the source image.
            - obj_img: The cropped image containing the object(s).
            - obj_mask: The cropped segmentation mask.
            - outerbox: The outer bounding box encompassing all overlapping objects.
            - inner_boxes: The original bounding boxes for each object in the cropped area.
    """
    # Parse the annotation line to get the image path and bounding boxes
    imgpath, inner_boxes = parse_annotation_line(annotation_line)

    # Load the original image
    org_img = Image.open(imgpath)

    # Calculate the outer bounding box that encompasses all overlapping objects
    outerbox = get_outerbox(inner_boxes[:, :4])

    # Crop the image based on the outer bounding box
    obj_img = crop_image(org_img, outerbox)

    # Get the corresponding segmentation mask
    obj_mask = get_objmask(imgpath, tuple(outerbox.flatten()), maskdir)

    return imgpath, obj_img, obj_mask, outerbox, inner_boxes


def select_objects_for_image(
        max_search_iterations: int,
        new_img_size: Tuple[int, int],
        source_ann: List[str],
        index_generator: Any,
        maskdir: str
) -> Tuple[
    List[str], List[Any], List[Any], List[Any], List[Any],
    List[Any], Tuple[int, int]
]:
    """
    Select objects from source annotations to create a new artificial image.
    And pass each object through some random augmentations to increase object dataset variability.

    Args:
        max_search_iterations (int): Maximum number of iterations to search for objects.
        new_img_size (Tuple[int, int]): The initial target image size.
        source_ann (List[str]): List of source annotations.
        index_generator (SourceIndexGenerator): An instance of SourceIndexGenerator to provide indices.
        maskdir (str): Directory containing the masks.

    Returns:
        Tuple: A tuple containing lists of image paths, cutout images, masks, coordinates, and the final image size.
    """
    # Initialize variables to store data
    (source_img_paths, cutout_objs_images, cutout_objs_masks,
     cutout_objs_coords, cutout_objs_inner_coords, total_iol,
     selected_objs_coords) = prepare_image_data()

    target_image_size = new_img_size

    while total_iol <= 0.8 and max_search_iterations > 0:
        try:
            # Get the next source annotation index
            source_indx = next(index_generator)
        except StopIteration:
            # If StopIteration is raised, reset the generator and shuffle annotations
            index_generator.reset()
            np.random.shuffle(source_ann)
            source_indx = next(index_generator)

        # Extract data from the current annotation line
        annotation_line = source_ann[source_indx]
        imgpath, obj_img, obj_mask, outerbox, inner_boxes = get_data(annotation_line, maskdir)

        # Apply random augmentations to the cutout image

        # obj_img, obj_mask, outerbox, inner_boxes = random_augmentations(
        #     obj_img, obj_mask, outerbox, inner_boxes, max_augs=2
        # )
        obj_img, obj_mask, outerbox, inner_boxes = heuristic_augmentations(
            obj_img, obj_mask, outerbox, inner_boxes)

        # Store the data
        source_img_paths.append(imgpath)
        cutout_objs_images.append(obj_img)
        cutout_objs_masks.append(obj_mask)
        cutout_objs_coords.append(outerbox)
        cutout_objs_inner_coords.append(inner_boxes)
        # Recalculate the target image size
        target_image_size = recalculate_targetsize(target_image_size, outerbox)

        # Calculate paste coordinates for the selected objects
        selected_objs_coords = get_pastecoords(target_image_size, np.array(cutout_objs_coords).reshape(-1, 4))

        # Calculate the total intersection over larger image (IoL)
        total_iol = get_total_iol(target_image_size, selected_objs_coords)

        max_search_iterations -= 1

    return (
        source_img_paths, cutout_objs_images, cutout_objs_masks,
        cutout_objs_coords, cutout_objs_inner_coords,
        selected_objs_coords, target_image_size
    )


def adjust_inner_boxes(outer_coord: np.ndarray, inner_coords: np.ndarray, new_outer_coord: np.ndarray) -> np.ndarray:
    """
    Adjust the inner bounding boxes relative to the new outer bounding box coordinates.

    Args:
        outer_coord (np.ndarray): The original outer bounding box, shape (1, 4).
        inner_coords (np.ndarray): The inner bounding boxes within the original outer box, shape (n, 5).
        new_outer_coord (np.ndarray): The new outer bounding box coordinates, shape (4,).

    Returns:
        np.ndarray: The adjusted inner bounding boxes in the new coordinate system, shape (n, 5).
    """
    # Calculate the shift (delta) between the original and new outer bounding box coordinates
    vx1 = new_outer_coord[0] - outer_coord[0, 0]
    vy1 = new_outer_coord[1] - outer_coord[0, 1]
    vx2 = new_outer_coord[2] - outer_coord[0, 2]
    vy2 = new_outer_coord[3] - outer_coord[0, 3]

    # Create the shift array
    shift = np.array([vx1, vy1, vx2, vy2]).reshape(1, 4)

    # Adjust inner boxes by the delta
    adjusted_inner_coords = np.copy(inner_coords)
    adjusted_inner_coords[:, 0:4] += shift  # Adjust x1, y1, x2, y2

    return adjusted_inner_coords


def get_realcoords(cutout_objs_coords: Dict[int, np.ndarray], cutout_objs_inner_coords: Dict[int, np.ndarray],
                   selected_objs_coords: Dict[int, np.ndarray]) -> np.ndarray:
    """
    Translate the inner bounding boxes from their relative positions within the cutout
    to their absolute positions within the new composite image.

    Args:
        cutout_objs_coords (Dict[int, np.ndarray]): Original outer bounding boxes for each cutout object.
        cutout_objs_inner_coords (Dict[int, np.ndarray]): Inner bounding boxes within each cutout.
        selected_objs_coords (Dict[int, np.ndarray]): New outer bounding boxes for each selected object.

    Returns:
        np.ndarray: The absolute coordinates of the inner objects in the new image, with shape (m, 5).
    """
    realobj_params = []

    for obj_id, new_outer_coord in selected_objs_coords.items():
        old_outer_coord = cutout_objs_coords[obj_id]
        inner_coords = cutout_objs_inner_coords[obj_id]

        # Adjust inner boxes to the new outer box coordinates
        adjusted_inner_coords = adjust_inner_boxes(old_outer_coord, inner_coords, new_outer_coord)

        # Append the adjusted coordinates to the final list
        realobj_params.append(adjusted_inner_coords)

    # Concatenate all adjusted inner boxes into a single array
    realobj_params = np.vstack(realobj_params)

    return realobj_params


def prepare_image_data() -> Tuple[List[str], List[Any], List[Any], List[Any], List[Any], float, List[Any]]:
    """
    Initialize and organize image data variables in a single function.

    Returns:
        Tuple[List[str], List[Any], List[Any], List[Any], List[Any], float, List[Any]]:
            - source_img_paths: A list to store paths of source images.
            - cutout_objs_images: A list to store cutout object images.
            - cutout_objs_masks: A list to store cutout object masks.
            - cutout_objs_coords: A list to store coordinates of cutout objects.
            - cutout_objs_inner_coords: A list to store inner coordinates of cutout objects.
            - total_iol: A float to track the total Intersection over Larger image.
            - selected_objs_coords: A list to store coordinates of selected objects.
    """
    source_img_paths: List[str] = []
    cutout_objs_images: List[Any] = []
    cutout_objs_masks: List[Any] = []
    cutout_objs_coords: List[Any] = []
    cutout_objs_inner_coords: List[Any] = []
    total_iol: float = 0.0  # Intersection over Larger image
    selected_objs_coords: List[Any] = []

    return (source_img_paths, cutout_objs_images, cutout_objs_masks,
            cutout_objs_coords, cutout_objs_inner_coords,
            total_iol, selected_objs_coords)


def get_tightfit_targetsize(selected_obj_coords):
    """
    Calculate the tight-fitting target size for an image that encompasses all selected objects.
    Ensures the width and height are even numbers.

    Args:
        selected_obj_coords (dict): A dictionary of selected object coordinates.

    Returns:
        tuple: The tight-fitting target size (width, height) as even integers.
    """
    # Extract the coordinates of all selected objects
    boxes = np.array([coord for coord in selected_obj_coords.values()]).reshape(-1, 4)

    # Determine the outermost bounding box that encompasses all objects
    x1, y1, x2, y2 = get_outerbox(boxes)[0, 0:4]
    width = x2 - x1
    height = y2 - y1

    # Ensure the width and height are even
    if width % 2 != 0:
        width += 1
    if height % 2 != 0:
        height += 1

    return int(width), int(height)

