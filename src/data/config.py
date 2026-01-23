"""
Configuration settings for the medimgprep package.

This module defines default constants used across medimgprep, such as default
directory names and processing modes.

Constants:
- DEFAULT_INCLUDE_MODE (bool): Whether to include subfolders by default. Default is False.
- DEFAULT_OUTPUT_DIR_NAME (str): Default name for the output directory. Default is "original".
"""

from typing import Final, Tuple, List

# organize.py
DEFAULT_INCLUDE_MODE: Final[bool] = False
DEFAULT_ORGANIZE_OUTPUT_DIR_NAME: Final[str] = "original"

MAX_WORKERS: Final[int] = 20
BATCH_SIZE: Final[int] = 1000

# base_image_seperator.py
MEAN_THRESHOLD: Final[int] = 10
BRIGHT_PIXEL_RATIO: Final[float] = 0.2
MAX_BRIGHTNESS: Final[int] = 50

DEFAULT_SEPARATOR_LOOKFOR_DIR_NAME: Final[str] = "original"
DEFAULT_SEPARATOR_OUTPUT_DIR_NAME: Final[str] = "no_black"
DEFAULT_SEPARATOR_SOURCE_DIR_NAME: Final[str] = "images"
DEFAULT_SEPARATOR_APPLY_TO_DIR_NAME: Final[str] = "mask"

IMG_TRANSFORM_RESIZE_SIZE: Final[Tuple[int, int]] = (224, 224)
IMG_TRANSFORM_BRIGHTNESS: Final[float] = 0.2
IMG_TRANSFORM_CONTRAST: Final[float] = 0.2
IMG_TRANSFORM_MEAN_VECTOR: Final[List[float]] = [0.485, 0.456, 0.406]
IMG_TRANSFORM_STD_VECTOR: Final[List[float]] = [0.229, 0.224, 0.225]

DEFAULT_TOPVIEW_LOOKFOR_DIR_NAME: Final[str] = "no_black"
DEFAULT_TOPVIEW_OUTPUT_DIR_NAME: Final[str] = "top_view"
DEFAULT_TOPVIEW_PREDICTIONS_FILE_NAME: Final[str] = "predictions.csv"
CONFIDENCE_THRESHOLD: Final[float] = 0.99
