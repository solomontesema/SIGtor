# SIGtor: Supplementary Synthetic Image Generation for Object Detection and Segmentation

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.txt)

SIGtor is a powerful tool for generating synthetic training datasets for object detection and segmentation tasks. It uses a copy-paste augmentation approach to create new images by combining objects from existing datasets with various backgrounds, while automatically generating accurate bounding boxes and segmentation masks.

## Quick Reference

**Run without installation** (from project root):
```bash
python3 -m sigtor.scripts.generate      # Generate images
python3 -m sigtor.scripts.expand        # Expand annotations
python3 -m sigtor.scripts.visualize     # Visualize results
```

**Run with installation** (after `pip install -e .`):
```bash
sigtor              # Generate images
sigtor-expand       # Expand annotations
sigtor-visualize    # Visualize results
```

## Features

- **Copy-Paste Augmentation**: Intelligently combines objects from source images onto new backgrounds
- **Automatic Annotation**: Generates YOLO-format annotations with bounding boxes and segmentation masks
- **Flexible Configuration**: YAML-based configuration system for easy customization
- **Multiple Blending Methods**: Supports seamless cloning, alpha blending, and soft pasting
- **Object Augmentation**: Applies geometric and morphological transformations to increase dataset diversity
- **Background Support**: Works with custom background images or generates plain backgrounds
- **Format Conversion Tools**: Includes utilities to convert between Pascal VOC, COCO, and YOLO formats

## Installation

### Requirements

- Python 3.7 or higher
- See `requirements.txt` for full list of dependencies

### Option 1: Run Without Installation (Recommended for Quick Start)

You can run SIGtor directly from the project directory without installing:

```bash
# Clone the repository
git clone https://github.com/solomontesema/sigtor.git
cd sigtor

# Install dependencies only
pip install -r requirements.txt

# Run scripts directly using Python module syntax (from project root)
python3 -m sigtor.scripts.generate
python3 -m sigtor.scripts.expand
python3 -m sigtor.scripts.visualize
```

**Important**: Always run these commands from the project root directory (where `setup.py` is located) so Python can properly import the `sigtor` package.

### Option 2: Install as Package (Recommended for Regular Use)

For convenient CLI commands, install the package:

```bash
# Clone the repository
git clone https://github.com/solomontesema/sigtor.git
cd sigtor

# Install dependencies and package
pip install -r requirements.txt
pip install -e .

# Then you can use CLI commands
sigtor
sigtor-expand
sigtor-visualize
```

**Note**: If CLI commands are not found after installation, add `~/.local/bin` to your PATH or use the Python module syntax from Option 1.

## Quick Start

### 1. Prepare Your Dataset

Your dataset should be annotated in YOLO format:
```
./Datasets/Source/Images/image1.jpg x1,y1,x2,y2,class_id x1,y1,x2,y2,class_id ...
```

### 2. Configure SIGtor

Copy the example configuration file and edit it:
```bash
cp config.yaml.example config.yaml
```

Edit `config.yaml` with your paths:
```yaml
SIGtor:
  source_ann_file: "./Datasets/Source/source_annotations.txt"
  destn_dir: "./Datasets/SIGtored/"
  mask_image_dirs: "./Datasets/Source/Masks"
  bckgrnd_imgs_dir: "./Datasets/BackgroundImages"
  classnames_file: "./data/classes.txt"
  total_new_imgs: 100
```

### 3. Expand Annotations (Optional but Recommended)

This step identifies overlapping objects and creates separate annotation lines.

**Without installation:**
```bash
python3 -m sigtor.scripts.expand
```

**With installation:**
```bash
sigtor-expand
```

Or with custom config:
```bash
# Without installation
python3 -m sigtor.scripts.expand --config config.yaml --source_ann_file ./Datasets/Source/annotations.txt

# With installation
sigtor-expand --config config.yaml --source_ann_file ./Datasets/Source/annotations.txt
```

### 4. Generate Synthetic Images

**Without installation:**
```bash
python3 -m sigtor.scripts.generate
```

**With installation:**
```bash
sigtor
```

Or with command-line arguments:
```bash
# Without installation
python3 -m sigtor.scripts.generate --source_ann_file ./Datasets/Source/annotations.txt --total_new_imgs 500

# With installation
sigtor --source_ann_file ./Datasets/Source/annotations.txt --total_new_imgs 500
```

### 5. Visualize Results

Test and visualize the generated images:

**Without installation:**
```bash
python3 -m sigtor.scripts.visualize --source_ann_file ./Datasets/SIGtored/sigtored_annotations.txt
```

**With installation:**
```bash
sigtor-visualize --source_ann_file ./Datasets/SIGtored/sigtored_annotations.txt
```

## Configuration

SIGtor uses a YAML configuration file (`config.yaml`) with three main sections:

### SIGtor Section
Main configuration for image generation:
- `source_ann_file`: Path to source annotation file
- `destn_dir`: Output directory for generated images
- `mask_image_dirs`: Directory containing segmentation masks (optional)
- `bckgrnd_imgs_dir`: Directory with background images (optional)
- `classnames_file`: Path to file with class names
- `total_new_imgs`: Number of synthetic images to generate
- `max_search_iterations`: Maximum iterations for object selection

### Test Section
Configuration for visualization:
- `source_ann_file`: Annotation file to visualize
- `classnames_file`: Class names file
- `output_dir`: Directory for output images
- `num_test_images`: Number of images to visualize ("All" or integer)

### Expanding_Annotation Section
Configuration for annotation expansion:
- `source_ann_file`: Source annotation file
- `iou_threshold`: IoU threshold for determining inner bounding boxes (0.0-1.0)

See `config.yaml.example` for a complete example with comments.

## How It Works

SIGtor generates synthetic images through the following process:

1. **Input Selection**: Randomly selects source images and their masks from the dataset
2. **Background Preparation**: Chooses a background image (or generates a plain one) and resizes it
3. **Object Selection**: Selects object cutouts, applies augmentations, and continues until coverage threshold is met (default 80% IoL)
4. **Object Placement**: Strategically places objects on the background to maximize space utilization without overlap
5. **Image Composition**: Blends objects onto the background using selected blending method (seamless cloning, alpha blending, etc.)
6. **Annotation Generation**: Creates YOLO-format annotations with accurate bounding boxes and segmentation masks
7. **Output**: Saves the composite image, mask, and annotation

## Directory Structure

```
SIGtor/
├── sigtor/                   # Main package
│   ├── core/                 # Core functionality
│   │   ├── generator.py      # Image generation logic
│   │   ├── expander.py       # Annotation expansion
│   │   └── visualizer.py     # Visualization
│   ├── processing/           # Image/data processing
│   │   ├── data_processing.py
│   │   ├── augmentation.py
│   │   ├── image_composition.py
│   │   └── image_postprocessing.py
│   ├── utils/                # Utility modules
│   │   ├── config.py
│   │   ├── file_ops.py
│   │   ├── image_utils.py
│   │   ├── data_utils.py
│   │   └── index_generator.py
│   └── scripts/              # CLI scripts
│       ├── generate.py
│       ├── expand.py
│       └── visualize.py
├── Datasets/                 # User data (not in repo)
│   ├── Source/
│   │   ├── Images/
│   │   ├── Masks/
│   │   └── annotations.txt
│   ├── BackgroundImages/
│   └── SIGtored/
├── tools/                     # Format conversion utilities
├── tests/                     # Test suite
├── docs/                      # Documentation and examples
│   └── misc/                  # Example images
├── examples/                  # Example notebooks and scripts
│   ├── demo.ipynb
│   └── download_background_images.py
├── config.yaml.example        # Configuration template
├── setup.py
├── requirements.txt
└── README.md
```

## Tools

The `tools/` directory contains utilities for format conversion:

- `voc_annotation.py`: Pascal VOC format utilities
- `coco_annotation.py`: COCO format utilities
- `voc_to_darknet.py`: Convert VOC to YOLO format
- `coco_to_pascal_voc.py`: Convert COCO to Pascal VOC
- `pascal_voc_to_coco.py`: Convert Pascal VOC to COCO

## Background Images

Background images are optional but recommended for more realistic results. You can:

1. Download images manually and place them in `Datasets/BackgroundImages/`
2. Use the provided script:
   ```bash
   python3 examples/download_background_images.py
   ```

**Important**: Manually review and remove any background images that contain objects from your dataset classes to avoid introducing unannotated objects.

## Command-Line Interface

SIGtor can be run in two ways:

### Method 1: Direct Execution (No Installation Required)

Run scripts directly using Python's module syntax from the project root directory:

```bash
# Make sure you're in the project root directory
cd /path/to/sigtor

# Generate synthetic images
python3 -m sigtor.scripts.generate [OPTIONS]

# Expand annotations
python3 -m sigtor.scripts.expand [OPTIONS]

# Visualize results
python3 -m sigtor.scripts.visualize [OPTIONS]
```

**Note**: Always run these commands from the project root directory (where `setup.py` and `README.md` are located) so Python can find the `sigtor` package.

### Method 2: CLI Commands (After Installation)

If you've installed the package (`pip install -e .`), you can use CLI commands:

```bash
# Generate synthetic images
sigtor [OPTIONS]

# Expand annotations
sigtor-expand [OPTIONS]

# Visualize results
sigtor-visualize [OPTIONS]
```

### Available Options

All commands support the following command-line arguments (override config file settings):

**Generate Images:**
```bash
[--config CONFIG] [--source_ann_file PATH] [--destn_dir PATH] 
[--mask_image_dirs PATH] [--bckgrnd_imgs_dir PATH] [--total_new_imgs N]
```

**Expand Annotations:**
```bash
[--config CONFIG] [--source_ann_file PATH] [--iou_threshold FLOAT]
```

**Visualize:**
```bash
[--config CONFIG] [--source_ann_file PATH] [--classnames_file PATH] 
[--output_dir PATH] [--num_test_images N]
```

### Using as Python Module

You can also use SIGtor as a Python package:

```python
from sigtor.core import generate_images, expand_annotations, visualize_annotations
from sigtor.utils.config import load_config, get_config_section

# Load config
config = load_config('config.yaml')
sigtor_config = get_config_section(config, 'SIGtor')

# Use core functions
expand_annotations('annotations.txt', iou_threshold=0.1)
generate_images(args)  # args object with configuration
visualize_annotations('annotations.txt', class_names, output_dir='./output/')
```

## Tips and Best Practices

1. **Annotation Expansion**: Run `sigtor-expand` first to handle overlapping objects properly
2. **Background Images**: Use diverse background images that don't contain objects from your classes
3. **Class Balance**: Monitor class distribution in generated images to avoid over-representation
4. **Dataset Size**: Start with a small number of images to test, then scale up
5. **Quality Check**: Use `sigtor-visualize` to visualize and verify generated images before training

## Limitations

- SIGtor is an offline dataset generator, not a real-time augmentation tool
- Generated images may have visible artifacts from the copy-paste process
- Contextual relationships between objects are not preserved (objects are placed independently)
- Requires YOLO-format annotations as input

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details.

## Citation

If you use SIGtor in your research, please cite:

```bibtex
@software{sigtor2024,
  title={SIGtor: Supplementary Synthetic Image Generation for Object Detection and Segmentation},
  author={Solomon Negussie Tesema},
  year={2024},
  url={https://github.com/solomontesema/sigtor}
}
```

## Acknowledgments

SIGtor was developed to address the challenge of creating large, diverse training datasets for object detection and segmentation tasks. The copy-paste augmentation approach has been shown to be effective in improving model performance.

## Support

For issues, questions, or contributions, please open an issue on the GitHub repository.
