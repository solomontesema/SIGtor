"""Tests for file operations."""

import os
import tempfile
import pytest
from PIL import Image
import numpy as np
from sigtor.utils.file_ops import save_image, save_mask, initialize_directories, create_directory


def test_create_directory():
    """Test directory creation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = os.path.join(tmpdir, 'test_subdir')
        create_directory(test_dir)
        assert os.path.exists(test_dir)
        assert os.path.isdir(test_dir)


def test_initialize_directories():
    """Test initialization of output directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        images_dir, masks_dir = initialize_directories(tmpdir)
        assert os.path.exists(images_dir)
        assert os.path.exists(masks_dir)
        assert 'augmented_images' in images_dir
        assert 'augmented_masks' in masks_dir


def test_save_image():
    """Test saving an image."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_image = Image.new('RGB', (100, 100), color='red')
        saved_path = save_image(test_image, tmpdir, 'test_image.jpg')
        assert os.path.exists(saved_path)
        assert saved_path.endswith('test_image.jpg')
        
        # Verify the image can be loaded
        loaded_image = Image.open(saved_path)
        assert loaded_image.size == (100, 100)


def test_save_mask():
    """Test saving a mask."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_mask = Image.new('L', (100, 100), color=255)
        saved_path = save_mask(test_mask, tmpdir, 'test_mask.png')
        assert os.path.exists(saved_path)
        assert saved_path.endswith('test_mask.png')
        
        # Verify the mask can be loaded
        loaded_mask = Image.open(saved_path)
        assert loaded_mask.size == (100, 100)


def test_save_image_creates_directory():
    """Test that save_image creates the directory if it doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = os.path.join(tmpdir, 'new_dir')
        test_image = Image.new('RGB', (50, 50), color='blue')
        saved_path = save_image(test_image, test_dir, 'image.jpg')
        assert os.path.exists(test_dir)
        assert os.path.exists(saved_path)

