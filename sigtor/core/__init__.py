"""Core functionality for SIGtor."""

from sigtor.core.generator import generate_images
from sigtor.core.expander import expand_annotations, get_bboxes_and_their_inner_bboxes
from sigtor.core.visualizer import visualize_annotations

__all__ = ['generate_images', 'expand_annotations', 'get_bboxes_and_their_inner_bboxes', 'visualize_annotations']

