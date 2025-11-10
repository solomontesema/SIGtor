# Changelog

All notable changes to SIGtor will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-11-10

### Added
- Initial public release of SIGtor
- YAML-based configuration system
- Support for YOLO format annotations
- Multiple image blending methods (seamless cloning, alpha blending, soft paste)
- Automatic annotation generation with bounding boxes and segmentation masks
- Background image support
- Object augmentation capabilities
- Format conversion tools (VOC, COCO, YOLO)
- Comprehensive documentation
- Test suite
- Setup.py for package installation

### Changed
- Migrated from text-based config (`sig_argument.txt`) to YAML configuration
- Improved error handling throughout the codebase
- Enhanced code organization and documentation
- Updated dependencies to more flexible version requirements

### Fixed
- Removed duplicate imports
- Fixed hardcoded paths
- Improved error messages
- Enhanced file operation error handling

### Removed
- Deprecated `sig_argument.txt` configuration format (replaced by YAML)
- Old/unused files: `synthetic_image_generator.py`, `refactored_data_utils.py`, etc.
- Deprecated `google_images_download` script (replaced with icrawler-based script)

## [Unreleased]

### Planned
- Additional augmentation options
- Performance optimizations
- Extended format support
- GUI interface (potential)

