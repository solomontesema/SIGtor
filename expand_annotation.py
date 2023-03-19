import argparse
import os

import numpy as np
from tqdm import tqdm

from utils import read_ann, get_ground_truth_data, overlap_measure, convert_to_ann_line, \
    parse_commandline_arguments


def get_bboxes_and_their_inner_bboxes(annotation_line, iou_theshold=0.2):
    img_path, all_boxes = get_ground_truth_data(annotation_line)
    box_ids = list(range(len(all_boxes)))
    associated_inner_boxes = {}
    b_wh = (all_boxes[..., 2:4] - all_boxes[..., 0:2])
    area = b_wh[..., 0] * b_wh[..., 1]
    iou, iol, ios = overlap_measure(all_boxes, all_boxes, expand_dim=True)
    for k, indx in enumerate(box_ids):
        for i in range(len(all_boxes)):
            area_k = area[k]
            area_i = area[i]
            if iou[k][i] < iou_theshold:
                if iou[k][i] == 0:
                    """
                        No intersection, then Ak is not inner object of Ai and vice-versa. Thus skip.
                    """
                    continue
                elif area_i < area_k and ios[k][i] >= 0.75:
                    if k in associated_inner_boxes:
                        associated_inner_boxes[k].extend([i])
                    else:
                        associated_inner_boxes[k] = [i]
                else:
                    continue
            else:
                """
                    Ai is inner object of Ak, hence no skip instead add in the associated inner object dictionary.
                """
                if k in associated_inner_boxes:
                    associated_inner_boxes[k].extend([i])
                else:
                    associated_inner_boxes[k] = [i]
        if k not in associated_inner_boxes:
            associated_inner_boxes[k] = None
    return img_path, all_boxes, associated_inner_boxes


def group_overlapping_objs(sets):
    new_sets = {}
    for indx in range(len(sets)):
        if len(sets[indx]) == 1:
            new_sets[indx] = sets[indx]
            continue
        for k, v in sets.items():
            if k == indx:
                continue
            if indx in v:
                sets[indx].extend(v)
        new_sets[indx] = list(np.unique(np.array(sets[indx])))
    return new_sets


def get_unique_groups(sets):
    values = [v for k, v in sets.items()]
    total = len(values)
    unique = {}
    for i in range(total):
        same_value_keys = [i]
        for j in range(total):
            if i == j:
                continue
            if (values[i] == values[j]) and (len(values[i]) == len(values[j])):
                same_value_keys.append(j)
        if len(same_value_keys) == 1:
            unique[i] = values[i]
        else:
            flag = False
            for k in same_value_keys:
                if k in unique:
                    flag = True
            if not flag:
                unique[i] = values[i]
    return unique


def main(args):
    source_ann = read_ann(args.source_ann_file)
    source_dir = os.path.dirname(args.source_ann_file)
    sour_ann_fname = os.path.basename(args.source_ann_file).split('.')[0]
    new_ann_path = os.path.join(source_dir, sour_ann_fname + '_expanded.txt')
    new_dataset = open(new_ann_path, 'w')
    for i in tqdm(range(len(source_ann)), desc='Creating new annotation file ...'):
        ann_line = source_ann[i]
        img_path, all_boxes, associated_inner_boxes = get_bboxes_and_their_inner_bboxes(ann_line, iou_theshold=0.1)
        for key, value in associated_inner_boxes.items():
            boxes = []
            for indx in value:
                boxes.append(all_boxes[indx])
            boxes = np.array(boxes).reshape((-1, 5))
            new_ann_line = convert_to_ann_line(img_path, boxes)
            new_dataset.write(new_ann_line)
            new_dataset.flush()
    new_dataset.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Object re-annotator or annotation expander for synthetic image generation.')
    config_path = "./sig_argument.txt"
    if os.path.exists(config_path):
        arguments = parse_commandline_arguments(config_path)
    else:
        raise ValueError("path {} containing general YOLO arguments is not found.".format(config_path))
    parser.add_argument('--source_ann_file', type=str, required=False, default=arguments['source_ann_file'][0],
                        help='YOLO format annotation txt file')
    args = parser.parse_args()
    main(args)
