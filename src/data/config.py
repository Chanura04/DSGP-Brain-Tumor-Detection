"""
Configuration settings for the medimgprep package.

This module defines default constants used across medimgprep, such as default
directory names and processing modes.

Constants:
- DEFAULT_INCLUDE_MODE (bool): Whether to include subfolders by default. Default is False.
- DEFAULT_OUTPUT_DIR_NAME (str): Default name for the output directory. Default is "original".
"""

from typing import Final

# organize.py
DEFAULT_INCLUDE_MODE: Final[bool] = False
DEFAULT_ORGANIZE_OUTPUT_DIR_NAME: Final[str] = "original"

MAX_WORKERS: Final[int] = 20
BATCH_SIZE: Final[int] = 1000

# base_image_seperator.py
MEAN_THRESHOLD: Final[int] = 10
BRIGHT_PIXEL_RATIO: Final[float] = 0.2
MAX_BRIGHTNESS: Final[int] = 50
