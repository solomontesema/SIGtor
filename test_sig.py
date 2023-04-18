import argparse
import os

import cv2
import numpy as np

from data_utils import is_image_bright
from utils import read_ann, get_colors, draw_boxes, parse_commandline_arguments, get_classes


def visualize_sigtor(ann_path, class_names=None):
    line = "./Datasets/SIGtored/augmented_images/00000003.jpg 2,327,368,457,10 206,232,278,457,4 2,29,314,375,14 246,154,306,224,15 60,119,78,158,4 2,116,41,178,14 368,29,631,148,3 419,73,433,89,14 411,77,424,90,14 465,66,491,89,14 494,76,514,89,14 535,76,550,88,14 574,73,588,86,14 368,148,621,263,3 559,206,573,221,14 568,205,580,217,14 503,206,528,228,14 481,206,500,218,14 447,207,461,218,14 410,209,424,221,14 556,263,594,297,2 498,271,530,308,2 510,266,551,339,2 534,326,555,360,2 480,350,516,390,2 392,338,431,379,2 368,316,424,349,2 368,390,504,542,7 504,390,635,536,7"
    # read_ann(ann_path, shuffle=True)[0]
    line_content = line.split()
    img_path = line_content[0]
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    boxes = np.array([np.array(list(map(float, box.split(',')))) for box in line_content[1:]]).astype('int32').reshape(
        -1, 5)
    classes = boxes[..., 4] + 1
    if class_names is None:
        class_names = classes
    colors = get_colors()
    image = draw_boxes(image, boxes[..., 0:4], classes, class_names, colors)
    cv2.imwrite("misc/images/test_image.jpg", image)


from PIL import Image


def get_brightness_scale(image):
    # Convert the image to grayscale
    image = image.convert('L')

    # Get the pixel values as a list of integers
    pixels = list(image.getdata())

    # Calculate the average pixel value
    avg_pixel = sum(pixels) / len(pixels)

    # Determine the brightness scale based on the average pixel value
    if avg_pixel < 85:
        return 0  # darkest image
    elif avg_pixel < 170:
        return 1
    elif avg_pixel < 255:
        return 2
    else:
        return 3  # brightest image


# Example usage:
image = Image.open("/home/solubuntu/Documents/SIGtor/Datasets/Source/Images/2009_005038.jpg")
brightness_scale = get_brightness_scale(image)
print(brightness_scale, is_image_bright(image))

# if __name__ == '__main__':
#     config_path = "./sig_argument.txt"
#     if os.path.exists(config_path):
#         arguments = parse_commandline_arguments(config_path)
#     else:
#         raise ValueError("path {} containing general SIGtor arguments is not found.".format(config_path))
#     parser = argparse.ArgumentParser(
#         description='Supplementary Synthetic Image Generation for Object Detection and Segmentation')
#     parser.add_argument('--source_ann_file', type=str, required=False, default=arguments['source_ann_file'][0],
#                         help='YOLO format annotation txt file as a source dataset')
#     parser.add_argument('--classnames_file', type=str, required=False, default=arguments['classnames_file'][0],
#                         help='Dataset object classes')
#
#     args = parser.parse_args()
#     class_names = get_classes(args.classnames_file)
#     visualize_sigtor(args.source_ann_file, class_names)
