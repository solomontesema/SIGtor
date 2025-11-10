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
python3 -m sigtor.scripts.analyze        # Analyze dataset quality
```

**Run with installation** (after `pip install -e .`):
```bash
sigtor              # Generate images
sigtor-expand       # Expand annotations
sigtor-visualize    # Visualize results
sigtor-analyze      # Analyze dataset quality
```

## Features

- **Advanced Copy-Paste Augmentation**: Intelligently combines objects from source images onto new backgrounds with seamless blending
- **Automatic Annotation**: Generates YOLO-format annotations with bounding boxes and segmentation masks
- **Flexible Configuration**: YAML-based configuration system for easy customization
- **Adaptive Blending Methods**: Context-aware selection of blending techniques (seamless cloning, Poisson blending, alpha blending, soft pasting)
- **Context-Aware Augmentations**: Dynamic augmentation selection based on object-background compatibility analysis
- **Advanced Edge Refinement**: Multi-scale edge detection and distance transform-based boundary smoothing
- **Color Harmonization**: Histogram matching and lighting consistency adjustment for realistic compositions
- **Multi-Stage Post-Processing**: Comprehensive pipeline for artifact reduction and visual quality enhancement
- **Object Augmentation**: Applies geometric and morphological transformations to increase dataset diversity
- **Background Support**: Works with custom background images or generates plain backgrounds
- **Quality Validation**: Optional image quality checks and validation
- **Dataset Analysis**: Comprehensive analysis tool for class distribution, size distribution, imbalance detection, and quality metrics
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
  classnames_file: "./Datasets/voc_classes.txt"
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
- `blending_method`: Blending method ('auto' for adaptive selection, or specific method)
- `enable_post_processing`: Enable multi-stage post-processing (true/false)
- `edge_refinement_level`: Edge refinement level ('low', 'medium', 'high')
- `color_harmonization`: Enable color harmonization between objects and background (true/false)
- `context_aware_augmentations`: Enable context-aware augmentations (true/false)
- `quality_validation`: Enable quality validation after generation (true/false)
- `quality_reject_threshold`: Quality rejection threshold ('none', 'critical', 'all')

### Test Section
Configuration for visualization:
- `source_ann_file`: Annotation file to visualize
- `classnames_file`: Class names file
- `output_dir`: Directory for output images
- `num_test_images`: Number of images to visualize ("All" or integer)

### Analysis Section
Configuration for dataset analysis:
- `source_ann_file`: Path to source annotation file (before SIGtoring)
- `sigtored_ann_file`: Path to SIGtored annotation file (after SIGtoring)
- `classnames_file`: Path to class names file
- `output_dir`: Directory to save analysis reports and visualizations
- `generate_plots`: Generate visualization plots (true/false)
- `generate_report`: Generate text and JSON reports (true/false)
- `comparison_mode`: Compare source vs SIGtored datasets (true/false)

### Expanding_Annotation Section
Configuration for annotation expansion:
- `source_ann_file`: Source annotation file
- `iou_threshold`: IoU threshold for determining inner bounding boxes (0.0-1.0)

See `config.yaml.example` for a complete example with comments.

## How It Works

SIGtor generates synthetic images through the following process:

1. **Input Selection**: Randomly selects source images and their masks from the dataset
2. **Background Preparation**: Chooses a background image (or generates a plain one) and resizes it
3. **Context Analysis**: Analyzes background characteristics (lighting, color, texture) for compatibility assessment
4. **Object Selection**: Selects object cutouts, applies context-aware augmentations, and continues until coverage threshold is met (default 80% IoL)
5. **Object Placement**: Strategically places objects on the background to maximize space utilization without overlap
6. **Edge Refinement**: Applies advanced edge detection and boundary smoothing to eliminate artifacts
7. **Adaptive Blending**: Selects optimal blending method based on object-background compatibility
8. **Color Harmonization**: Matches object colors and lighting to background context
9. **Post-Processing**: Multi-stage pipeline for artifact reduction and quality enhancement
10. **Annotation Generation**: Creates YOLO-format annotations with accurate bounding boxes and segmentation masks
11. **Quality Validation**: Optional validation of generated image quality
12. **Output**: Saves the composite image, mask, and annotation

## Directory Structure

```
SIGtor/
├── sigtor/                   # Main package
│   ├── core/                 # Core functionality
│   │   ├── generator.py      # Image generation logic with quality validation
│   │   ├── expander.py       # Annotation expansion
│   │   └── visualizer.py     # Visualization
│   ├── processing/           # Image/data processing
│   │   ├── data_processing.py      # Object selection and placement
│   │   ├── augmentation.py         # Context-aware augmentations
│   │   ├── image_composition.py    # Blending and composition
│   │   ├── image_postprocessing.py # Multi-stage post-processing
│   │   ├── edge_refinement.py     # Advanced edge processing
│   │   ├── color_harmonization.py # Color matching and harmonization
│   │   ├── context_analysis.py     # Background/object analysis
│   │   └── adaptive_blending.py   # Context-aware blending selection
│   ├── utils/                # Utility modules
│   │   ├── config.py
│   │   ├── file_ops.py
│   │   ├── image_utils.py
│   │   ├── data_utils.py
│   │   └── index_generator.py
│   ├── analysis/             # Dataset analysis
│   │   ├── dataset_analyzer.py   # Core statistics extraction
│   │   ├── report_generator.py   # Text and JSON report generation
│   │   └── visualizer.py         # Plot generation for analysis
│   └── scripts/              # CLI scripts
│       ├── analyze.py        # Dataset analysis CLI
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
6. **Quality Validation**: Enable `quality_validation` to automatically detect and optionally reject problematic images

### Quality Validation Feature

When `quality_validation` is enabled, SIGtor performs comprehensive quality checks on each generated image:

**What It Checks:**
- **Image Size**: Ensures images meet minimum size requirements (default: 100x100 pixels)
- **Empty Images**: Detects completely empty or corrupted images
- **Data Integrity**: Checks for NaN (Not a Number) or Inf (Infinity) values that indicate processing errors
- **Data Type**: Validates that images are in the correct format (uint8)
- **Suspicious Values**: Flags images that are entirely black (mean < 5) or white (mean > 250), which may indicate failures

**Rejection Behavior:**

The `quality_reject_threshold` parameter controls what happens when issues are found:

- **`none`**: Only logs issues to console and quality report. All images are saved regardless of issues.
- **`critical`** (default): Rejects images with critical issues (empty, NaN, Inf, wrong dtype). Non-critical issues (size warnings, suspicious values) are logged but images are still saved.
- **`all`**: Rejects images with any quality issues, including non-critical ones.

**Quality Report:**

When quality validation is enabled and issues are found, a `quality_report.json` file is automatically saved in the output directory containing:

- Total number of images validated
- Total issues found
- Number of critical issues
- Number of rejected images
- Detailed log of all issues by image index

**Example Usage:**

```yaml
# Log issues but keep all images
quality_validation: true
quality_reject_threshold: "none"

# Reject only critical failures (recommended)
quality_validation: true
quality_reject_threshold: "critical"

# Strict mode - reject any issues
quality_validation: true
quality_reject_threshold: "all"
```

**When to Use:**

- **Development/Testing**: Use `quality_reject_threshold: "none"` to see what issues occur without losing images
- **Production**: Use `quality_reject_threshold: "critical"` to automatically filter out corrupted images
- **High-Quality Datasets**: Use `quality_reject_threshold: "all"` for maximum quality control

## Technical Architecture

### Core Components

SIGtor employs a sophisticated multi-stage pipeline designed to produce high-quality synthetic images with minimal artifacts. The system consists of several interconnected modules:

#### 1. Context Analysis Module

The context analysis module (`context_analysis.py`) extracts comprehensive characteristics from both objects and backgrounds:

- **Lighting Analysis**: Computes brightness, contrast, and color temperature using LAB color space transformations
- **Color Analysis**: Extracts mean color, standard deviation, dominant color palette, and saturation metrics
- **Texture Analysis**: Quantifies texture complexity, edge density, and smoothness using Laplacian variance and Canny edge detection
- **Compatibility Scoring**: Computes multi-dimensional compatibility scores between objects and backgrounds based on lighting, color, and texture characteristics

The compatibility score $C$ is computed as:
$$C = 0.4 \cdot C_{lighting} + 0.4 \cdot C_{color} + 0.2 \cdot C_{texture}$$

where each component is normalized to [0, 1] based on feature differences.

#### 2. Edge Refinement Module

The edge refinement module (`edge_refinement.py`) implements advanced boundary processing techniques:

- **Multi-Scale Edge Detection**: Uses Canny edge detection at multiple scales (1.0x, 0.5x, 2.0x) with adaptive thresholding based on image statistics
- **Distance Transform**: Computes distance from boundaries using Euclidean distance transform to create smooth transition zones
- **Adaptive Feathering**: Dynamically adjusts feather radius based on object size and edge strength:
  $$r_{feather} = r_{base} \cdot (1 + \alpha \cdot \frac{A_{object}}{A_{image}}) \cdot (1 - \beta \cdot E_{strength})$$
  where $\alpha$ and $\beta$ are sensitivity parameters, $A$ represents area, and $E_{strength}$ is normalized edge strength
- **Morphological Refinement**: Applies edge-aware morphological operations (erosion, dilation) with iteration counts adjusted based on local edge characteristics

#### 3. Color Harmonization Module

The color harmonization module (`color_harmonization.py`) ensures visual consistency between objects and backgrounds:

- **Histogram Matching**: Performs histogram matching in LAB color space, preserving perceptual uniformity:
  $$L'_{obj} = (L_{obj} - \mu_{L,obj}) \cdot \frac{\sigma_{L,bg}}{\sigma_{L,obj}} + \mu_{L,bg}$$
  where $\mu$ and $\sigma$ represent mean and standard deviation of the L channel
- **Local Color Transfer**: Applies color transfer in boundary regions using distance-weighted blending:
  $$I_{final}(p) = w(p) \cdot I_{obj}(p) + (1-w(p)) \cdot I_{matched}(p)$$
  where $w(p)$ is a distance-based weight function
- **Lighting Consistency**: Matches brightness, contrast, and color temperature by adjusting LAB channels independently
- **Color Temperature Adjustment**: Estimates and adjusts color temperature using A and B channel statistics in LAB space

#### 4. Adaptive Blending Module

The adaptive blending module (`adaptive_blending.py`) selects optimal blending methods based on context analysis:

**Method Selection Algorithm**:
1. Compute compatibility score $C$ between object and background
2. Analyze edge density difference $\Delta E = |E_{obj} - E_{bg}|$
3. Compute color variance ratio $R_{var} = \frac{Var_{obj}}{Var_{bg}}$

**Decision Rules**:
- If $C > 0.75$ and $\Delta E < 0.05$: Use Normal Clone (high compatibility, similar edges)
- If $\Delta E > 0.1$: Use Soft Paste with adaptive feathering (high edge difference)
- If $C < 0.5$: Use Harmonized Soft Paste (low compatibility requires color adjustment)
- If $|C_{texture,obj} - C_{texture,bg}| < 0.2$: Use Normal Clone (similar texture complexity)
- Default: Soft Paste with size-adaptive feathering

**Available Blending Methods**:
- **Seamless Cloning**: OpenCV's seamless clone with Normal, Mixed, or Monochrome Transfer modes
- **Poisson Blending**: Gradient-domain blending using biharmonic inpainting
- **Soft Paste**: Gaussian-blurred alpha blending with adaptive feather radius
- **Alpha Blending**: Weighted combination with configurable alpha values

#### 5. Context-Aware Augmentation Module

The augmentation module (`augmentation.py`) applies intelligent transformations based on background context:

**Augmentation Selection Process**:
1. Analyze object and background contexts
2. Compute compatibility scores
3. Generate augmentation suggestions:
   - Brightness adjustment: $\Delta B = B_{bg} - B_{obj}$
   - Contrast adjustment: $\Delta C = C_{bg} - C_{obj}$
   - Color temperature adjustment: $\Delta T = T_{bg} - T_{obj}$
   - Saturation adjustment: $\Delta S = S_{bg} - S_{obj}$
4. Apply suggested augmentations with smooth transitions

**Augmentation Types**:
- Geometric: Scaling, flipping, rotation
- Photometric: Brightness, contrast, saturation, color temperature
- Morphological: Blur (texture complexity matching)
- Combined: Multi-stage augmentation pipeline

#### 6. Post-Processing Pipeline

The post-processing module (`image_postprocessing.py`) implements a five-stage enhancement pipeline:

**Stage 1: Edge Refinement**
- Applies multi-scale edge detection
- Refines mask boundaries using distance transform
- Creates smooth edge transitions

**Stage 2: Color Harmonization**
- Matches object colors to background (if enabled)
- Applies local color transfer at boundaries
- Adjusts lighting consistency

**Stage 3: Edge Blending**
- Creates gradient alpha masks
- Applies smooth blending at boundaries
- Reduces visible seams

**Stage 4: Global Enhancement**
- Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
- Enhances contrast while preserving natural appearance
- Operates in LAB color space for perceptual uniformity

**Stage 5: Final Smoothing**
- Applies light Gaussian blur to reduce artifacts
- Preserves image sharpness while smoothing transitions

### Algorithmic Pipeline

The complete SIGtor pipeline can be formalized as follows:

**Algorithm: SIGtor Image Generation**

1. **Input**: Source annotations $A$, background directory $B$, configuration $C$
2. **Initialize**: Target size $T$, object list $O = \emptyset$
3. **While** $IoL(O) < 0.8$ and iterations $< max\_iter$:
   - Select source annotation $a_i$ from $A$
   - Extract object $o_i$ with mask $m_i$
   - Analyze background context $\Gamma_{bg}$
   - Apply context-aware augmentations: $o_i' = Augment(o_i, \Gamma_{bg})$
   - Add to $O$: $O = O \cup \{o_i'\}$
   - Update target size: $T = Recalculate(O)$
4. **Place objects**: Compute coordinates $P = PlaceObjects(O, T)$
5. **Select background**: $bg = SelectBackground(B, T)$
6. **For each** object $o_i$ in $O$:
   - Refine mask: $m_i' = RefineEdges(m_i, o_i)$
   - Analyze contexts: $\Gamma_{obj}, \Gamma_{bg}$
   - Select blending method: $M = SelectBlending(\Gamma_{obj}, \Gamma_{bg})$
   - Harmonize colors: $o_i'' = Harmonize(o_i', bg, m_i')$
   - Blend: $I = Blend(o_i'', bg, m_i', M)$
7. **Post-process**: $I_{final} = PostProcess(I, m, bg)$
8. **Validate**: $Q = ValidateQuality(I_{final})$
9. **Output**: Image $I_{final}$, mask $m$, annotations $A_{final}$

### Performance Characteristics

- **Computational Complexity**: $O(n \cdot (E + H + B + P))$ where $n$ is number of objects, $E$ is edge refinement, $H$ is color harmonization, $B$ is blending, and $P$ is post-processing
- **Memory Requirements**: Linear with image size and number of objects
- **Quality Metrics**: Compatibility scores, edge smoothness, color consistency
- **Scalability**: Efficient for batch processing with configurable quality/performance trade-offs

### Configuration Parameters

Advanced users can fine-tune SIGtor behavior through configuration parameters:

- **Edge Refinement Level**: Controls computational intensity vs. quality trade-off
  - `low`: Fast processing, basic edge smoothing
  - `medium`: Balanced quality and performance (default)
  - `high`: Maximum quality, slower processing

- **Blending Method**: Manual override or adaptive selection
  - `auto`: Context-aware automatic selection (recommended)
  - Specific methods: `SoftPaste`, `NormalClone`, `MixedClone`, `MonochromeTransfer`, `AlphaBlend`

- **Post-Processing**: Multi-stage enhancement pipeline
  - Can be disabled for faster generation
  - Individual stages can be configured

- **Context-Aware Augmentations**: Enable/disable intelligent augmentation selection
  - When enabled, augmentations adapt to background characteristics
  - When disabled, uses heuristic-based augmentations

## Limitations

- SIGtor is an offline dataset generator, not a real-time augmentation tool
- Contextual relationships between objects are not preserved (objects are placed independently)
- Requires YOLO-format annotations as input
- Computational cost increases with higher refinement levels and post-processing enabled
- Quality depends on source image and mask quality

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

## Research and Academic Use

SIGtor represents a significant advancement in offline copy-paste augmentation for object detection and segmentation. The system addresses key limitations of traditional copy-paste methods:

1. **Artifact Reduction**: Advanced edge refinement and color harmonization eliminate visible boundaries that can cause model overfitting
2. **Context Awareness**: Dynamic augmentation and blending selection based on object-background compatibility ensures realistic compositions
3. **Quality Assurance**: Multi-stage post-processing pipeline and optional quality validation ensure production-ready outputs
4. **Flexibility**: Configurable pipeline allows balancing quality and performance for different use cases

### Key Innovations

- **Multi-Scale Edge Refinement**: Distance transform-based boundary smoothing with adaptive feathering
- **Context-Aware Augmentation**: Intelligent augmentation selection based on compatibility analysis
- **Adaptive Blending Selection**: Automatic method selection based on object-background characteristics
- **Color Harmonization Pipeline**: Histogram matching and lighting consistency in perceptually uniform color space
- **Multi-Stage Post-Processing**: Comprehensive artifact reduction while preserving image quality

### Future Directions (SIGtorV2)

Potential enhancements for future versions:
- Real-time augmentation during training
- Deep learning-based blending methods
- Semantic-aware object placement
- 3D-aware composition for perspective correction
- Style transfer integration
- GAN-based refinement for photorealistic results

## Acknowledgments

SIGtor was developed to address the challenge of creating large, diverse training datasets for object detection and segmentation tasks. The copy-paste augmentation approach has been shown to be effective in improving model performance. The advanced techniques implemented in SIGtor further enhance this effectiveness by reducing artifacts and ensuring visual consistency.

## Support

For issues, questions, or contributions, please open an issue on the GitHub repository.
