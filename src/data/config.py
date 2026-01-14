"""
Configuration settings for the medimgprep package.

This module defines default constants used across medimgprep, such as default
directory names and processing modes.

Constants:
- DEFAULT_INCLUDE_MODE (bool): Whether to include subfolders by default. Default is False.
- DEFAULT_OUTPUT_DIR_NAME (str): Default name for the output directory. Default is "original".
"""

# organize.py
DEFAULT_INCLUDE_MODE: bool = False
DEFAULT_ORGANIZE_OUTPUT_DIR_NAME: str = "original"

MAX_WORKERS = 20
BATCH_SIZE = 1000

# base_image_seperator.py
MEAN_THRESHOLD = 10
BRIGHT_PIXEL_RATIO = 0.2
MAX_BRIGHTNESS = 50
