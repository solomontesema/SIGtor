"""
Annotation utility functions for merging and processing annotation files.
"""

import os
import numpy as np
from typing import Dict, Any, List, Optional

from sigtor.utils.image_utils import read_ann


def merge_annotations(
    source_ann_file: str,
    sigtored_ann_file: str,
    output_file: str,
    shuffle: bool = True,
    validate_paths: bool = True
) -> Dict[str, Any]:
    """
    Merge two annotation files into one combined file.
    
    Args:
        source_ann_file: Path to source annotation file (original dataset)
        sigtored_ann_file: Path to SIGtored annotation file (synthetic dataset)
        output_file: Path to output combined annotation file
        shuffle: Whether to shuffle/randomize the combined annotations
        validate_paths: Whether to validate that input files exist
    
    Returns:
        Dictionary with statistics:
        - source_count: Number of lines from source file
        - sigtored_count: Number of lines from SIGtored file
        - total_count: Total combined lines
        - output_path: Path to output file
        - shuffled: Whether shuffling was applied
    """
    # Validate input files exist
    if validate_paths:
        if not os.path.exists(source_ann_file):
            raise FileNotFoundError(f"Source annotation file not found: {source_ann_file}")
        if not os.path.exists(sigtored_ann_file):
            raise FileNotFoundError(f"SIGtored annotation file not found: {sigtored_ann_file}")
    
    # Read source annotations (without shuffling to preserve order)
    source_lines = read_ann(source_ann_file, shuffle=False)
    # Filter out empty lines
    source_lines = [line.strip() for line in source_lines if line.strip()]
    
    # Read SIGtored annotations (without shuffling to preserve order)
    sigtored_lines = read_ann(sigtored_ann_file, shuffle=False)
    # Filter out empty lines
    sigtored_lines = [line.strip() for line in sigtored_lines if line.strip()]
    
    # Combine annotations
    combined_lines = source_lines + sigtored_lines
    
    # Shuffle if requested
    if shuffle and len(combined_lines) > 0:
        np.random.shuffle(combined_lines)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Write combined annotations to output file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for line in combined_lines:
                f.write(line + '\n')
    except IOError as e:
        raise IOError(f"Failed to write output file {output_file}: {e}")
    
    # Return statistics
    return {
        'source_count': len(source_lines),
        'sigtored_count': len(sigtored_lines),
        'total_count': len(combined_lines),
        'output_path': output_file,
        'shuffled': shuffle
    }


def validate_annotation_file(ann_file: str) -> bool:
    """
    Validate that an annotation file exists and is readable.
    
    Args:
        ann_file: Path to annotation file
    
    Returns:
        True if file is valid, False otherwise
    """
    if not os.path.exists(ann_file):
        return False
    
    try:
        # Try to read first line to check if file is readable
        with open(ann_file, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            return bool(first_line.strip())
    except (IOError, UnicodeDecodeError):
        return False

