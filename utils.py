import colorsys
import os

import cv2
import numpy as np


def rand(a=0.0, b=1.0):
    return np.random.rand() * (b - a) + a


def parse_commandline_arguments(argument_filepath):
    arguments = {}
    if not os.path.exists(argument_filepath):
        raise ValueError("Unable to find {}".format(argument_filepath))
    else:
        with open(argument_filepath) as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith('#[') or line.isspace():
                    continue
                key, value = line.strip().split("=")
                arguments[key] = value.strip().split(',')
    return arguments


def read_ann(ann_file, shuffle=True):
    with open(ann_file) as f:
        lines = f.readlines()
    if shuffle:
        np.random.shuffle(lines)
    return lines


def get_file_paths(filepath, file_format=None, shuffle=False):
    if file_format is not None:
        file_paths = [os.path.join(filepath, filename) for filename in os.listdir(filepath)
                      if os.path.splitext(filename)[1].lower() in file_format]
    else:
        file_paths = [os.path.join(filepath, filename) for filename in os.listdir(filepath)]
    if shuffle:
        np.random.shuffle(file_paths)
    return file_paths


def get_ground_truth_data(annotation_line):
    """given single annotation line returns image path, image and bounding boxes on the image."""
    line = annotation_line.split()
    img_path = line[0]
    boxes = np.array([np.array(list(map(float, box.split(',')))) for box in line[1:]]).astype('int32')
    return img_path, boxes


def get_classes(classes_path):
    classes_path = os.path.expanduser(classes_path)
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_colors(number=100, bright=True):
    """
    Generate random colors for drawing bounding boxes.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    if number <= 0:
        return []

    brightness = 1.0 if bright else 0.7
    hsv_tuples = [(x / number, 1., brightness)
                  for x in range(number)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    np.random.seed(10101)  # Fixed seed for consistent colors across runs.
    np.random.shuffle(colors)  # Shuffle colors to decorate adjacent classes.
    np.random.seed(None)  # Reset seed to default.
    return colors


def draw_label(image, text, color, coords):
    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 1.
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=2)[0]

    padding = 5
    rect_height = text_height + padding * 2
    rect_width = text_width + padding * 2

    (x, y) = coords

    cv2.rectangle(image, (x, y), (x + rect_width, y - rect_height), color, cv2.FILLED)
    cv2.putText(image, text, (x + padding, y - text_height + padding), font,
                fontScale=font_scale,
                color=(255, 255, 255),
                lineType=cv2.LINE_AA)

    return image


def draw_boxes(image, boxes, classes, class_names, colors, show_label=True):
    if boxes is None or len(boxes) == 0:
        return image
    if classes is None or len(classes) == 0:
        return image

    for box, cls in zip(boxes, classes):
        xmin, ymin, xmax, ymax = map(int, box)
        class_name = class_names[cls]
        if colors == None:
            color = (0, 0, 0)
        else:
            color = colors[cls]
        if show_label:
            label = '{}'.format(class_name)
            image = draw_label(image, label, color, (xmin, ymin))
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2, cv2.LINE_AA)

    return image


def overlap_measure(box1, box2, expand_dim=False):
    """
        box1: [m, 4] numpy array
        box2: [n, 4] numpy array
        Given two numpy boxes in format [xmin, ymin, xmax, ymax] it returns their iou,iol, ios
    """
    if expand_dim:
        box2 = np.expand_dims(box2, axis=1)
    intersect_mins = np.maximum(box1[..., 0:2], box2[..., 0:2])
    intersect_maxs = np.minimum(box1[..., 2:4], box2[..., 2:4])
    intersect_wh = np.maximum(intersect_maxs - intersect_mins, 0)

    box1_wh = box1[..., 2:4] - box1[..., 0:2]
    box2_wh = box2[..., 2:4] - box2[..., 0:2]
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    box1_area = box1_wh[..., 0] * box1_wh[..., 1]
    box2_area = box2_wh[..., 0] * box2_wh[..., 1]

    iou = intersect_area / (box1_area + box2_area - intersect_area)
    iol = intersect_area / np.maximum(box1_area, box2_area)
    ios = intersect_area / np.minimum(box1_area, box2_area)
    return [iou, iol, ios]


def convert_to_ann_line(img_path, obj_boxes):
    new_ann_line = ""
    for obj_cord in obj_boxes:
        new_ann_line += " " + str(list(obj_cord)).strip('[]').replace(" ", "")
    new_ann_line = img_path + new_ann_line + "\n"
    return new_ann_line
