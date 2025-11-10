"""Setup script for SIGtor package."""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="sigtor",
    version="1.0.0",
    description="Supplementary Synthetic Image Generation for Object Detection and Segmentation Datasets",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Solomon Negussie Tesema",
    url="https://github.com/solomontesema/sigtor",
    packages=find_packages(exclude=['tests', 'tests.*']),
    python_requires=">=3.7",
    install_requires=read_requirements(),
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'black>=21.0',
            'flake8>=3.8',
        ],
    },
    entry_points={
        'console_scripts': [
            'sigtor=sigtor.scripts.generate:main',
            'sigtor-expand=sigtor.scripts.expand:main',
            'sigtor-visualize=sigtor.scripts.visualize:main',
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    keywords="computer-vision object-detection segmentation data-augmentation synthetic-data",
    include_package_data=True,
    zip_safe=False,
)

