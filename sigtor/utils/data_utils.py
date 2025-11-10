import random

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from sigtor.utils.image_utils import rand


def random_size(image, mask=None, boxes=None, scale=1.0, prob=0.2, keepaspectratio=True):
    old_w, old_h = image.size

    if keepaspectratio:
        if prob < 0.5:
            ratio = (old_w / old_h)
            new_h = np.round(scale * old_h).astype('int')
            new_w = np.round(ratio * new_h).astype('int')
        else:
            ratio = (old_h / old_w)
            new_w = np.round(scale * old_w).astype('int')
            new_h = np.round(ratio * new_w).astype('int')
    else:
        new_h = np.round(scale * old_h).astype('int')
        new_w = np.round(scale * old_w).astype('int')

    # Resize the image using BICUBIC interpolation
    image = image.resize((new_w, new_h), Image.BICUBIC)

    # Resize the mask using NEAREST interpolation to preserve original pixel values
    if mask is not None:
        mask = mask.resize((new_w, new_h), Image.NEAREST)

    # Adjust the bounding boxes
    if boxes is not None:
        boxes[:, [0, 2]] = np.floor(boxes[:, [0, 2]] * new_w / old_w).astype('int')
        boxes[:, [1, 3]] = np.floor(boxes[:, [1, 3]] * new_h / old_h).astype('int')
        boxes = np.array(boxes).reshape(-1, 5).astype('int')

    return image, mask, boxes


def random_hsv_distort(image, hue=0.0, sat=1.0, val=1.0):
    """
    Distort image in HSV color space with provided scales

    # Arguments
        image: origin image for HSV distort
            PIL Image object containing image data
        hue: scale for Hue adjustment
            scalar
        sat: scale for Saturation adjustment
            scalar
        val: scale for Value(Brightness) adjustment
            scalar

    # Returns
        new_image: distorted PIL Image object.
    """
    # transform color space from RGB to HSV
    x = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV)

    # distort image
    x = x.astype(np.float64)
    x[..., 0] = (x[..., 0] * (1 + hue)) % 180
    x[..., 1] = x[..., 1] * sat
    x[..., 2] = x[..., 2] * val
    x[..., 1:3][x[..., 1:3] > 255] = 255
    x[..., 1:3][x[..., 1:3] < 0] = 0
    x = x.astype(np.uint8)

    # back to PIL RGB distort image
    x = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)
    new_image = Image.fromarray(x)

    return new_image


def random_brightness(image, scale=1.0):
    """
    Adjust brightness of image with provided scale

    # Arguments
        image: origin image for brightness change
            PIL Image object containing image data
        scale: scale for brightness adjustment,
            scalar to control the brightness level.

    # Returns
        new_image: adjusted PIL Image object.
    """
    enh_bri = ImageEnhance.Brightness(image)
    new_image = enh_bri.enhance(scale)

    return new_image


def random_chroma(image, scale=1.0):
    """
    Adjust chroma (color level) of image with provided scale

    # Arguments
        image: origin image for chroma change
            PIL Image object containing image data
        scale: scale for chroma adjustment,
            scalar to control the color level.

    # Returns
        new_image: adjusted PIL Image object.
    """
    enh_col = ImageEnhance.Color(image)
    new_image = enh_col.enhance(scale)

    return new_image


def random_contrast(image, scale=1.0):
    """
    Adjust contrast of image with provided scale

    # Arguments
        image: origin image for contrast change
            PIL Image object containing image data
        scale: scale for contrast adjustment,
            scalar to control the contrast level.

    # Returns
        new_image: adjusted PIL Image object.
    """
    enh_con = ImageEnhance.Contrast(image)
    new_image = enh_con.enhance(scale)

    return new_image


def random_sharpness(image, scale=1.0):
    """
    Adjust sharpness of image with provided scale

    # Arguments
        image: origin image for sharpness change
            PIL Image object containing image data
        scale: scale for sharpness adjustment,
            scalar to control the sharpness level.

    # Returns
        new_image: adjusted PIL Image object.
    """
    enh_sha = ImageEnhance.Sharpness(image)
    new_image = enh_sha.enhance(scale)

    return new_image


def random_horizontal_flip(image, prob=0.2):
    """
    Random horizontal flip for image

    # Arguments
        image: origin image for horizontal flip, PIL Image object containing image data
        prob: probability for random flip, scalar to control the flip probability.
    # Returns
        image: adjusted PIL Image object.
        flip: boolean flag for horizontal flip action
    """
    flip = rand() < prob
    if flip:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    return image, flip


def random_vertical_flip(image, prob=0.2):
    """
    Random vertical flip for image

    # Arguments
        image: origin image for vertical flip
            PIL Image object containing image data
        prob: probability for random flip,
            scalar to control the flip probability.

    # Returns
        image: adjusted PIL Image object.
        flip: boolean flag for vertical flip action
    """
    flip = rand() < prob
    if flip:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
    return image, flip


def random_grayscale(image, prob=0.2):
    """
    Random convert image to grayscale

    # Arguments
        image: origin image for grayscale convert
            PIL Image object containing image data
        prob: probability for grayscale convert,
            scalar to control the convert probability.

    # Returns
        image: adjusted PIL Image object.
    """
    convert = rand() < prob
    if convert:
        # convert to grayscale first, and then
        # back to 3 channels fake RGB
        image = image.convert('L')
        image = image.convert('RGB')

    return image


def random_blur(image, prob=0.1):
    """
    Random add normal blur to image

    # Arguments
        image: origin image for blur
            PIL Image object containing image data
        prob: probability for blur,
            scalar to control the blur probability.

    # Returns
        image: adjusted PIL Image object.
    """
    blur = rand() < prob
    if blur:
        try:
            image = image.filter(ImageFilter.BLUR)
        except ValueError:
            image = image
    return image


def isbright(image, dim=10, thresh=0.5):
    # Resize image to 10x10
    image = cv2.resize(image, (dim, dim))
    # Convert color space to LAB format and extract L channel
    L, A, B = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2LAB))
    # Normalize L channel by dividing all pixel values with maximum pixel value
    L = L / np.max(L)
    # Return True if mean is greater than thresh else False
    return np.mean(L) > thresh
