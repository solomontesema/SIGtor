"""Core annotation expansion functionality."""

import os
import numpy as np
from tqdm import tqdm

from sigtor.utils.image_utils import read_ann, get_ground_truth_data, overlap_measure, convert_to_ann_line


def get_bboxes_and_their_inner_bboxes(annotation_line, iou_threshold=0.2):
    """
    Identifies and returns bounding boxes and their associated inner bounding boxes based on a specified IoU threshold.

    Args:
        annotation_line (str): The annotation line containing the image path and bounding box details.
        iou_threshold (float): The threshold for determining inner bounding boxes based on IoU values.

    Returns:
        tuple: Contains the path to the image, a list of all bounding boxes, and a dictionary mapping each bounding box to its list of inner bounding boxes.
    """
    img_path, all_boxes = get_ground_truth_data(annotation_line)
    box_dims = all_boxes[..., 2:4] - all_boxes[..., 0:2]
    areas = box_dims[..., 0] * box_dims[..., 1]
    iou, _, ios = overlap_measure(all_boxes, all_boxes, expand_dim=True)
    associated_inner_boxes = {}

    for k, area_k in enumerate(areas):
        inner_boxes = []
        for i, area_i in enumerate(areas):
            if iou[k][i] >= iou_threshold or (iou_threshold > iou[k][i] > 0 and area_i < area_k and ios[k][i] >= 0.75):
                inner_boxes.append(i)
        associated_inner_boxes[k] = inner_boxes

    return img_path, all_boxes, associated_inner_boxes


def expand_annotations(source_ann_file, iou_threshold=0.1):
    """
    Expand annotations by identifying overlapping objects and creating separate annotation lines.

    Args:
        source_ann_file (str): Path to the source annotation file.
        iou_threshold (float): IoU threshold for determining inner bounding boxes.
    """
    source_ann = read_ann(source_ann_file)
    source_dir = os.path.dirname(source_ann_file)
    sour_ann_fname = os.path.basename(source_ann_file).split('.')[0]
    new_ann_path = os.path.join(source_dir, sour_ann_fname + '_expanded.txt')
    new_dataset = open(new_ann_path, 'w')
    
    for i in tqdm(range(len(source_ann)), desc='Creating new annotation file ...'):
        ann_line = source_ann[i]
        img_path, all_boxes, associated_inner_boxes = get_bboxes_and_their_inner_bboxes(ann_line, iou_threshold=iou_threshold)
        for key, value in associated_inner_boxes.items():
            boxes = []
            for indx in value:
                boxes.append(all_boxes[indx])
            boxes = np.array(boxes).reshape((-1, 5))
            new_ann_line = convert_to_ann_line(img_path, boxes)
            new_dataset.write(new_ann_line)
            new_dataset.flush()
    new_dataset.close()

