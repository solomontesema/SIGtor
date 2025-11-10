"""Tests for data processing functions."""

import numpy as np
import pytest
from sigtor.processing.data_processing import (
    get_outerbox,
    get_area,
    calculate_iou,
    is_overlap,
    convert_instance_to_binary
)


def test_get_area():
    """Test calculating bounding box area."""
    box = np.array([10, 20, 50, 60])
    area = get_area(box)
    assert area == (50 - 10) * (60 - 20)
    assert area == 1600


def test_get_outerbox():
    """Test getting outer bounding box."""
    boxes = np.array([
        [10, 10, 30, 30],
        [20, 20, 40, 40],
        [5, 5, 15, 15]
    ])
    outerbox = get_outerbox(boxes)
    assert outerbox.shape == (1, 4)
    assert outerbox[0, 0] == 5  # min x1
    assert outerbox[0, 1] == 5  # min y1
    assert outerbox[0, 2] == 40  # max x2
    assert outerbox[0, 3] == 40  # max y2


def test_get_outerbox_empty():
    """Test getting outer box with empty input."""
    boxes = np.array([]).reshape(0, 4)
    with pytest.raises(ValueError, match="empty"):
        get_outerbox(boxes)


def test_calculate_iou():
    """Test IoU calculation."""
    box1 = np.array([10, 10, 30, 30])
    box2 = np.array([20, 20, 40, 40])
    
    # Overlapping boxes
    iou = calculate_iou(box1, box2)
    assert 0 <= iou <= 1
    
    # Same box
    iou_same = calculate_iou(box1, box1)
    assert iou_same == 1.0
    
    # Non-overlapping boxes
    box3 = np.array([50, 50, 70, 70])
    iou_no_overlap = calculate_iou(box1, box3)
    assert iou_no_overlap == 0.0


def test_is_overlap():
    """Test overlap detection."""
    box1 = np.array([10, 10, 30, 30])
    box2 = np.array([20, 20, 40, 40])
    box3 = np.array([50, 50, 70, 70])
    
    assert is_overlap(box1, box2) == True
    assert is_overlap(box1, box3) == False
    assert is_overlap(box1, box1) == True


def test_convert_instance_to_binary():
    """Test converting instance mask to binary."""
    # Create a test mask with instance IDs
    mask = np.array([
        [0, 0, 0],
        [0, 1, 2],
        [0, 3, 0]
    ], dtype=np.uint8)
    
    binary = convert_instance_to_binary(mask)
    assert binary.dtype == np.uint8
    assert np.all(binary[mask > 0] == 255)
    assert np.all(binary[mask == 0] == 0)

