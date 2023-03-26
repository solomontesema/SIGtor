import argparse
import copy
import math
import os
import random

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from data_utils import random_blur, random_grayscale, random_brightness, random_sharpness, \
    random_chroma, random_contrast, random_hsv_distort, random_vertical_flip, random_horizontal_flip, \
    random_resize, enhance_image
from utils import read_ann, get_file_paths, rand, overlap_measure, convert_to_ann_line, get_colors, \
    parse_commandline_arguments


def get_next_source(current_indx, total_indexs, randomly=False):
    """
    Get the next source index given the current index and the total number of sources.

    Args:
        current_indx (int): The current source index.
        total_indexs (int): The total number of sources.
        randomly (bool): Whether to choose the next index randomly. Defaults to False.

    Returns:
        int: The index of the next source.

    Raises:
        ValueError: If current_indx is greater than or equal to total_indexs.

    """

    if current_indx >= total_indexs:
        raise ValueError("current_indx cannot be greater than or equal to total_indexs.")

    if randomly:
        return np.random.randint(0, total_indexs - 1)
    elif current_indx < total_indexs - 1:
        return current_indx + 1
    else:
        return 0


def get_backgrnd_image(bckgrnd_imgs_dir, target_img_size):
    """
    Returns a background image of the specified size, either randomly selected from a directory of images
    or a solid color if no images are found.

    Args:
    - bckgrnd_imgs_dir: str, the directory path containing background images
    - target_img_size: tuple, the desired size of the background image in (height, width, channels)

    Returns:
    - bcgrnd_img: numpy.ndarray, the background image

    Raises:
    -

    """

    bckgrnd_imgs_list = []
    if os.path.exists(bckgrnd_imgs_dir):
        bckgrnd_imgs_list = get_file_paths(bckgrnd_imgs_dir, file_format=['.jpg', '.png', '.jpeg'])

    h, w, c = target_img_size
    if len(bckgrnd_imgs_list) != 0:
        np.random.shuffle(bckgrnd_imgs_list)
        bcgrnd_img_path = np.random.choice(bckgrnd_imgs_list)
        bcgrnd_img = cv2.cvtColor(cv2.imread(bcgrnd_img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        bcgrnd_img = cv2.resize(bcgrnd_img, (h, w))
    else:
        bcgrnd_img = np.ones(w, h, c)
        random_color = np.random.randint(0, 255, size=3).reshape((1, 1, 3))
        bcgrnd_img *= random_color
        bcgrnd_img = np.array(bcgrnd_img).astype(np.uint8)

    bcgrnd_img = cv2.GaussianBlur(bcgrnd_img, (3, 3), 0)
    return bcgrnd_img


def get_objmask(src_imgpath, cutout_coord, maskdir, class_list=None):
    """
    Returns the object mask for the given image and cutout coordinates. If the mask file exists, it loads and returns the
    cropped mask. Otherwise, it creates a new mask of the same size as the cutout coordinates.

    Args:
        src_imgpath (str): Path to the source image.
        cutout_coord (tuple): Tuple of 4 integers representing the bounding box of the object in the image (x1, y1, x2, y2).
        maskdir (str): Path to the directory where mask files are found.
        class_list (list, optional): List of class names. Defaults to None.

    Returns:
        obj_mask (PIL.Image.Image): Cropped object mask.

    Raises:
        ValueError: If the cutout coordinates are invalid.
        IOError: If there is an error while opening or saving the mask image.

    """

    if len(cutout_coord) != 4:
        raise ValueError("cutout_coord must be a tuple of 4 integers representing the bounding box (x1, y1, x2, y2)")

    fname = os.path.splitext(os.path.basename(src_imgpath))[0]
    mfname = os.path.join(maskdir, fname + ".png")

    if os.path.exists(mfname):
        try:
            mask_img = Image.open(mfname)
            obj_mask = mask_img.crop(box=cutout_coord)
        except IOError as e:
            raise IOError("Error while opening mask file") from e
    else:
        x1, y1, x2, y2 = cutout_coord
        outerbox_width = (x2 - x1)
        outerbox_height = (y2 - y1)
        obj_mask = Image.fromarray(255 * np.ones((outerbox_height, outerbox_width), dtype=np.uint8))

    return obj_mask


def get_outerbox(org_boxes, img_size=(None, None), addrandomoffset=False, minoffset=1, maxoffset=5):
    """
    Returns the coordinates of the outer bounding box that contains all the boxes in org_boxes.

    Args:
    - org_boxes (ndarray): An array of shape (N, 4) where N is the number of boxes and each row
        represents a box in the format [xmin, ymin, xmax, ymax].
    - img_size (tuple): A tuple of two integers representing the width and height of the image.
        If not specified, the size will not be taken into account while computing the outer box.
    - addrandomoffset (bool): A flag indicating whether to add a random offset to the outer box.
    - minoffset (int): The minimum value of the random offset to be added.
    - maxoffset (int): The maximum value of the random offset to be added.

    Returns:
    - outerbox (ndarray): An array of shape (1, 4) representing the outer bounding box in the
        format [xmin, ymin, xmax, ymax].
    """
    if img_size[0] is not None and img_size[1] is not None:
        w, h = img_size
    else:
        w, h = np.max(org_boxes[..., 2]), np.max(org_boxes[..., 3])

    try:
        if addrandomoffset:
            x1 = max(np.min(org_boxes[..., 0]) - int(random.randint(minoffset, maxoffset)), 0)
            y1 = max(np.min(org_boxes[..., 1]) - int(random.randint(minoffset, maxoffset)), 0)
            x2 = min(np.max(org_boxes[..., 2]) + int(random.randint(minoffset, maxoffset)), w)
            y2 = min(np.max(org_boxes[..., 3]) + int(random.randint(minoffset, maxoffset)), h)
        else:
            x1 = np.min(org_boxes[..., 0])
            y1 = np.min(org_boxes[..., 1])
            x2 = np.max(org_boxes[..., 2])
            y2 = np.max(org_boxes[..., 3])

        outerbox = np.array([x1, y1, x2, y2, -1]).reshape(-1, 5)
    except:
        outerbox = np.zeros((1, 5))

    return outerbox


def get_data(annotation_line: str, mask_dir: str):
    """
    Extracts image, object image, object mask, outer box, and inner boxes from an annotation line.

    Parameters:
        annotation_line (str): A string containing the image path and object boxes information.
        mask_dir (str): The directory where the image masks are stored.

    Returns:
        imgpath (str): The path of the original image.
        obj_img (PIL.Image.Image): The object image extracted from the original image.
        obj_mask (PIL.Image.Image): The object mask extracted from the original mask..
        outerbox (numpy.ndarray): The outer bounding box that contains all object boxes.
        inner_boxes (numpy.ndarray): The object boxes contained in the outer box.
    """

    try:
        line = annotation_line.split()
        imgpath = line[0]
        org_img = Image.open(imgpath)
        org_boxes = np.array([np.array(list(map(float, box.split(',')))) for box in line[1:]]).astype('int32').reshape(
            -1, 5)
        outerbox = get_outerbox(org_boxes, org_img.size, addrandomoffset=False, minoffset=0, maxoffset=15)
        inner_boxes = org_boxes
        obj_classes = list(inner_boxes[:, 4]) if len(inner_boxes[:, 4]) != 0 else None
        obj_img = org_img.crop(box=outerbox[0, 0:4])
        obj_mask = get_objmask(imgpath, outerbox[0, 0:4], mask_dir, class_list=obj_classes)
    except Exception as e:
        print(f"Error occurred while processing annotation line: {annotation_line}")
        raise e

    return imgpath, obj_img, obj_mask, outerbox, inner_boxes


def random_augmentations(obj_img, obj_mask, outerbox, inner_boxes):
    augtypes = {
        'rescale': random_resize,
        'blur': random_blur,
        'brightness': random_brightness,
        'chroma': random_chroma,
        'contrast': random_contrast,
        'hsv': random_hsv_distort,
        'hflip': random_horizontal_flip,
        'sharpness': random_sharpness,
        'grayscale': random_grayscale,
        'vflip': random_vertical_flip
    }

    num_augs = random.randint(1, len(augtypes))
    selected_augtypes = random.sample(augtypes.keys(), num_augs)
    if 'rescale' not in selected_augtypes:
        selected_augtypes.append('rescale')

    new_obj_img, new_obj_mask = obj_img, obj_mask
    new_outerbox, new_innerboxes = outerbox, inner_boxes
    boxes = np.concatenate((outerbox, inner_boxes), axis=0)

    for augtype in selected_augtypes:
        if augtype == 'rescale':
            new_obj_img, new_obj_mask, boxes = random_resize(
                new_obj_img, mask=new_obj_mask, boxes=boxes, scale=1.0,
                scale_w_or_h=np.random.choice([0, 1]),
                keep_aspect_ratio=True
            )
        elif augtype in ('vflip', 'hflip'):
            new_obj_img = augtypes[augtype](new_obj_img, prob=1.0)
            new_obj_mask = augtypes[augtype](new_obj_mask, prob=1.0)
            w, h = boxes[0, 2:4] - boxes[0, :2]
            if augtype == 'vflip':
                boxes[1:, [1, 3]] = h - boxes[1:, [3, 1]]
            else:
                boxes[1:, [0, 2]] = w - boxes[1:, [2, 0]]
        else:
            new_obj_img = augtypes[augtype](new_obj_img, prob=1.0)

    return new_obj_img, new_obj_mask, boxes[:1], boxes[1:]


def recalculate_targetsize(current_targetsize, current_outerbox):
    """
    Calculates a new target size based on the current target size and the outer box of the object of interest.

    Args:
    - current_targetsize: tuple of two integers representing the current target size (width, height)
    - current_outerbox: numpy array of shape (1, 5) representing the coordinates of the outer box of the object of interest
                        (x1, y1, x2, y2, class_id)

    Returns:
    - target_image_size: tuple of two integers representing the new target size (width, height)

    Note: the width and height of the new target size are adjusted to be greater than or equal to the width and height
          of the outer box, respectively.
    """
    x1, y1, x2, y2 = current_outerbox[0, :4]
    bw, bh = (x2 - x1 + 1), (y2 - y1 + 1)
    w, h = current_targetsize

    if bw > w and bh > h:
        return bw, bh
    elif bw > w:
        return bw, h
    elif bh > h:
        return w, bh
    else:
        return current_targetsize


def get_tightfit_target_size(selected_obj_coords):
    """
    Calculate the target size for a tight fit of the selected objects.

    Args:
    - selected_obj_coords: a dictionary of object coordinates in the format {key: coordinates}

    Returns:
    - tightfit_target_size: a tuple containing the calculated width and height
    """
    boxes = np.array([coord[0, :4] for coord in selected_obj_coords.values()])
    x1, y1, x2, y2 = get_outerbox(boxes)[0, :4]
    w, h = x2 - x1 + 1, y2 - y1 + 1
    if w % 2 != 0:
        w += 1
    if h % 2 != 0:
        h += 1
    tightfit_target_size = (w, h)
    return tightfit_target_size


def get_pastecoords(target_size, all_outer_boxes):
    """
    Given a target size and a list of object outer boxes, returns a dictionary of paste coordinates
    for each object, optimized to fit as tightly as possible in the target size.

    Args:
        target_size (tuple): A tuple of integers representing the target image size.
        all_outer_boxes (list): A list of numpy arrays representing the outer boxes of each object.

    Returns:
        A dictionary where the keys are the object IDs and the values are numpy arrays representing
        the paste coordinates of each object.
    """
    w, h = target_size
    all_outer_boxes = np.array(all_outer_boxes).reshape(-1, 5)
    wh = all_outer_boxes[..., 2:4] - all_outer_boxes[..., 0:2]
    area = wh[..., 0] * wh[..., 1]
    decending_order = np.argsort(-area, axis=-1)
    vertex_pool = [(0, 0)]
    paste_coords = {}

    for obj_id in decending_order:
        obj_w, obj_h = tuple(wh[obj_id, :2])
        for j, (vx, vy) in enumerate(vertex_pool):
            if (vx + obj_w) <= w and (vy + obj_h) <= h:
                obj_newcoord = np.array([vx, vy, vx + obj_w, vy + obj_h]).reshape(-1, 4)
                overlap = any([overlap_measure(obj_newcoord, prev_paste_coord)[1] != 0.0
                               for prev_paste_coord in paste_coords.values()])
                if not overlap:
                    paste_coords[obj_id] = obj_newcoord
                    vertex_pool.pop(j)
                    vertex_pool.extend([(vx + obj_w, vy), (vx, vy + obj_h), (vx + obj_w, vy + obj_h)])
                    break
        else:
            continue
        break

    return paste_coords


def get_total_iol(target_image_size, selected_objs_coords):
    total_iol = 0.0
    imgcord = np.array([0, 0, target_image_size[0], target_image_size[1]]).reshape(-1, 4)
    for obj_id, box in selected_objs_coords.items():
        _, iol, _ = overlap_measure(imgcord, box, expand_dim=False)
        total_iol += iol
    return total_iol


def get_realcoords(cutout_objs_coords, cutout_objs_inner_coords, selected_objs_coords):
    """
    Calculates the real coordinates of objects in the original image based on the cutout objects coordinates and
    selected objects coordinates.

    Args:
        cutout_objs_coords (dict): A dictionary containing the outer coordinates of cutout objects.
        cutout_objs_inner_coords (dict): A dictionary containing the inner coordinates of cutout objects.
        selected_objs_coords (dict): A dictionary containing the selected outer coordinates of objects.

    Returns:
        np.ndarray: An array of the real object parameters (x1, y1, x2, y2, label).
    """
    realobj_params = []
    for obj_id, new_outer_coord in selected_objs_coords.items():
        old_outer_coord = cutout_objs_coords[obj_id]
        shift = new_outer_coord[0, 0:4] - old_outer_coord[0, 0:4]
        all_inner_obs = cutout_objs_inner_coords[obj_id]
        all_inner_obs[..., 0:4] += shift
        realobj_params.extend(all_inner_obs)

    return np.array(realobj_params)


def paste_masks(target_size, all_obj_masks, selected_obj_coords):
    """
    Paste object masks onto a new image with the specified target size, using selected object coordinates.

    Args: target_size (tuple): a tuple of two integers specifying the target size of the new image all_obj_masks (
    dict): a dictionary of all object masks, where keys are object IDs and values are corresponding masks
    selected_obj_coords (dict): a dictionary of selected object coordinates, where keys are object IDs and values are
    corresponding coordinates

    Returns:
        new_img_mask (numpy.ndarray): a new image mask with the pasted object masks
    """

    # Initialize a new image mask with zeros
    new_img_mask = np.zeros((target_size[1], target_size[0]), dtype=np.uint8)

    # Initialize a new instance ID
    new_inst_id = 1

    # Loop through selected object coordinates
    for obj_id, coord in selected_obj_coords.items():

        # Get the current object mask
        current_mask = np.array(all_obj_masks[obj_id])

        # Get unique pixel instances in the current object mask
        unique_pixel_instances = np.unique(current_mask)

        # Loop through unique pixel instances
        for inst_id in unique_pixel_instances:

            # Ignore invalid instance IDs
            if inst_id == 0 or inst_id == 255:
                continue

            # Update the instance ID in the current object mask
            current_mask[current_mask == inst_id] = new_inst_id

            # Increment the instance ID
            new_inst_id += 1

        # Get the coordinates of the current object
        x1, y1, x2, y2 = coord[0, 0:4]

        # Paste the current object mask onto the new image mask
        new_img_mask[y1:y2, x1:x2] = current_mask

    return new_img_mask


def paste_objs(target_size, all_obj_imgs, selected_obj_coords):
    """
    Pastes objects onto a new image.

    Args:
        target_size (tuple): The size of the new image (width, height).
        all_obj_imgs (dict): A dictionary mapping object IDs to object images.
        selected_obj_coords (dict): A dictionary mapping object IDs to their new coordinates.

    Returns:
        np.ndarray: The new image with the objects pasted onto it.
    """
    new_img = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    for obj_id, coords in selected_obj_coords.items():
        x1, y1, x2, y2 = coords[0, :4].astype(int)
        new_img[y1:y2, x1:x2, :] = all_obj_imgs[obj_id]
    return new_img


def simple_paste(src, dst, mask):
    """
    Paste the source image onto the destination image using a binary mask.

    Args:
        src (numpy.ndarray): The source image to be pasted.
        dst (numpy.ndarray): The destination image where the source image is pasted onto.
        mask (numpy.ndarray): The binary mask indicating the region where the source image is pasted.

    Returns:
        The pasted image as a PIL Image object.
    """
    # Create a binary mask
    _mask = np.uint8(copy.deepcopy(mask) != 0) * 255
    # Apply morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    _mask = cv2.morphologyEx(_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    # Convert mask to PIL Image
    _mask = Image.fromarray(_mask)

    # Convert source to PIL Image
    _src = Image.fromarray(np.uint8(copy.deepcopy(src)))

    # Convert destination to PIL Image
    dst = Image.fromarray(np.uint8(dst))

    # Paste the source onto the destination using the binary mask
    dst = Image.composite(_src, dst, _mask)

    return dst


def seamlessclone(src, dst, mask, center, mode=cv2.NORMAL_CLONE):
    _mask = copy.deepcopy(mask).astype('uint8')
    _mask[_mask != 0] = 255
    dst = cv2.seamlessClone(src, dst, _mask, center, mode)
    dst = Image.fromarray(dst)
    return dst


def paste_at_random_pivot(src_img, src_msk, src_boxes, scale_min=1.0, scale_max=1.5, use_random_backgrnd=True):
    """
    Paste the source image and mask at a random position with random scale and optional random background color.

    Args:
        src_img (numpy.ndarray): The source image.
        src_msk (numpy.ndarray): The source mask.
        src_boxes (numpy.ndarray): The source boxes.
        scale_min (float): The minimum scaling factor for the image. Default is 1.0.
        scale_max (float): The maximum scaling factor for the image. Default is 1.5.
        use_random_backgrnd (bool): Whether to use a random background color. Default is True.

    Returns:
        Tuple of offsetted source image, offsetted mask, and source boxes.
    """
    # Get source image dimensions
    oh, ow = src_img.shape[:2]

    # Calculate new dimensions based on random scaling factor
    nw = int(rand(scale_min, scale_max) * ow)
    nh = int(rand(scale_min, scale_max) * oh)

    # Ensure new dimensions are odd numbers
    nw = nw + 1 if nw % 2 != 0 else nw
    nh = nh + 1 if nh % 2 != 0 else nh

    # Calculate random position for source image
    px = int(rand(0, (nw - ow)))
    py = int(rand(0, (nh - oh)))

    # Create offsetted mask and source image
    offsetted_mask = np.zeros((nh, nw), dtype=np.uint8)
    offsetted_mask[py:py + oh, px:px + ow] = src_msk

    offsetted_src_img = np.zeros((nh, nw, 3))
    offsetted_src_img[py:py + oh, px:px + ow] = src_img

    # Paste on a random background color if specified
    if use_random_backgrnd:
        bcgrnd_color = np.random.randint(0, 255, size=3)
        new_backgrnd = np.array(Image.new(mode="RGB", size=(nw, nh), color=tuple(bcgrnd_color))).astype('uint8')
        offsetted_src_img = simple_paste(offsetted_src_img, new_backgrnd, offsetted_mask)
        offsetted_src_img = np.array(offsetted_src_img)

    # Update source boxes to reflect new position
    src_boxes[..., 0:4] = src_boxes[..., 0:4] + np.array([px, py, px, py]).reshape(-1, 4)

    return offsetted_src_img, offsetted_mask, src_boxes


def mask_to_RGB(mask, colors):
    """
    Convert a grayscale mask to RGB format using the specified colors for each instance.
    The mask is expected to have 0 as the background and 255 as the void.
    Any other pixel value represents a different instance, and its corresponding color
    will be assigned according to the index in the `colors` list.

    Args:
        mask (PIL.Image): The grayscale mask to convert.
        colors (List[Tuple[int, int, int]]): A list of RGB tuples representing the color
            for each instance.

    Returns:
        PIL.Image: The mask converted to RGB format.

    """
    h, w = mask.height, mask.width
    mask_array = np.array(mask)
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for i, color in enumerate(colors):
        inst_mask = np.where(mask_array == i + 1)
        colored_mask[inst_mask] = color

    # Assign white color to void pixels
    colored_mask[mask_array == 255] = [255, 255, 255]

    return Image.fromarray(colored_mask)


def main(args):
    source_ann = read_ann(args.source_ann_file)  # All ground-truth annotations
    maskdir = args.mask_image_dirs

    os.makedirs(args.destn_dir, exist_ok=True)
    new_images_dir = os.path.join(args.destn_dir, "augmented_images")
    new_masks_dir = os.path.join(args.destn_dir, "augmented_masks")
    os.makedirs(new_images_dir, exist_ok=True)
    os.makedirs(new_masks_dir, exist_ok=True)
    new_images_ann_path = os.path.join(args.destn_dir, 'sigtored_annotations.txt')
    new_dataset = open(new_images_ann_path, 'w')

    total_source_anns = len(source_ann)
    source_indx = -1

    count_new_images = 0

    for _ in tqdm(range(args.total_new_imgs), desc='Generating artificial images', leave=True):
        target_img_width = random.randint(400, 600)
        target_img_height = random.randint(400, 600)
        target_img_width = target_img_width if target_img_width % 2 == 0 else target_img_width + 1
        target_img_height = target_img_height if target_img_height % 2 == 0 else target_img_height + 1
        target_image_size = (target_img_width, target_img_height)

        source_img_path = []
        cutout_objs_images = []
        cutout_objs_masks = []
        cutout_objs_coords = []
        cutout_objs_inner_coords = []

        total_iol = 0.0

        selected_objs_coords = []

        max_search_iterations = 3
        while max_search_iterations > 0:
            if total_iol >= 0.8:
                break
            source_indx = get_next_source(source_indx, total_source_anns, randomly=False)
            if source_indx == 0:
                np.random.shuffle(source_ann)

            annotation_line = source_ann[source_indx]
            imgpath, obj_img, obj_mask, outerbox, inner_boxes = get_data(annotation_line, maskdir)
            obj_img, obj_mask, outerbox, inner_boxes = random_augmentations(obj_img, obj_mask, outerbox, inner_boxes)

            source_img_path.append(imgpath)
            cutout_objs_images.append(obj_img)
            cutout_objs_masks.append(obj_mask)
            cutout_objs_coords.append(outerbox)
            cutout_objs_inner_coords.append(inner_boxes)

            target_image_size = recalculate_targetsize(target_image_size, outerbox)
            selected_objs_coords = get_pastecoords(target_image_size, np.array(cutout_objs_coords).reshape(-1, 5))
            total_iol = get_total_iol(target_image_size, selected_objs_coords)
            max_search_iterations -= 1

        target_image_size = get_tightfit_target_size(selected_objs_coords)

        new_mask = paste_masks(target_image_size, cutout_objs_masks, selected_objs_coords)
        new_img = paste_objs(target_image_size, cutout_objs_images, selected_objs_coords)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        new_mask = cv2.erode(new_mask, kernel, iterations=2)

        new_boxes1 = get_realcoords(cutout_objs_coords, cutout_objs_inner_coords, selected_objs_coords)
        new_img2, new_mask2, new_boxes2 = paste_at_random_pivot(new_img, new_mask, new_boxes1, scale_min=1.0,
                                                                scale_max=1.15)

        bcgrnd_img = get_backgrnd_image(args.bckgrnd_imgs_dir, (new_img2.shape[1], new_img2.shape[0], 3))

        center = (new_img2.shape[1] // 2, new_img2.shape[0] // 2)
        cloneoptions = ['SP']  # 'NormalClone', 'MonochromeTransfer', 'MixedClone'
        clone_choice = np.random.choice(cloneoptions)
        if clone_choice == 'SP':
            new_artificial_img1 = simple_paste(new_img2, bcgrnd_img, new_mask2)
        elif clone_choice == 'NormalClone':
            new_artificial_img1 = seamlessclone(new_img2, bcgrnd_img, new_mask2, center, mode=cv2.NORMAL_CLONE)
        elif clone_choice == 'MixedClone':
            new_artificial_img1 = seamlessclone(new_img2, bcgrnd_img, new_mask2, center, mode=cv2.MIXED_CLONE)
        else:
            new_artificial_img1 = seamlessclone(new_img2, bcgrnd_img, new_mask2, center, mode=cv2.MONOCHROME_TRANSFER)

        new_artificial_img1 = enhance_image(new_artificial_img1)  # post_processing(new_artificial_img1)

        newimgpath = os.path.join(new_images_dir, "{:08d}.jpg".format(count_new_images))
        new_artificial_img1.save(newimgpath)

        newmaskpath = os.path.join(new_masks_dir, "{:08d}.png".format(count_new_images))
        new_mask2 = Image.fromarray(new_mask2)
        colored_mask = mask_to_RGB(new_mask2, get_colors(256))
        colored_mask.save(newmaskpath)

        annotation_line = convert_to_ann_line(newimgpath, new_boxes2)
        new_dataset.write(annotation_line)
        new_dataset.flush()

        count_new_images += 1


if __name__ == '__main__':
    config_path = "./sig_argument.txt"
    if os.path.exists(config_path):
        arguments = parse_commandline_arguments(config_path)
    else:
        raise ValueError("path {} containing general SIGtor arguments is not found.".format(config_path))

    parser = argparse.ArgumentParser(
        description='Supplementary Synthetic Image Generation for Object Detection and Segmentation')
    parser.add_argument('--source_ann_file', type=str, required=False, default=arguments['source_ann_file'][0],
                        help='YOLO format annotation txt file as a source dataset')
    parser.add_argument('--destn_dir', type=str, required=False, default=arguments['destn_dir'][0],
                        help='directory to save the generated images, their ground_truth annotations and masks')
    parser.add_argument('--mask_image_dirs', type=str, required=False, default=arguments['mask_image_dirs'][0],
                        help='directory of where to find the ground-truth masks, if there are any')
    parser.add_argument('--bckgrnd_imgs_dir', type=str, required=False, default=arguments['bckgrnd_imgs_dir'][0],
                        help='directory of where the background images are found if there are any')
    parser.add_argument('--total_new_imgs', type=int, required=False, default=int(arguments['total_new_imgs'][0]),
                        help='total number of new images to generate from the total annotations.')
    args = parser.parse_args()
    main(args)
