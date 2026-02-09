"""
ImageSeparator Module

This module provides the `ImageSeparator` abstract class, which is designed to filter out / copy
low intensity images of the organized raw dataset to an interim dataset. It supports:

- Filtering images by a the mean threshold/ mean intensity of the image.
- Filtering images by a the bright pixel ratio of the image.
- Filtering images by a the max brightness of the image.
- Copying only valid image extensions.
- Logging progress, duplicates, and summary information.
- Dry-run mode to simulate file operations without writing files.
- Measuring execution time for performance monitoring (via the `get_time` decorator).

Dependencies:
- cv2
- numpy
- pathlib
- abc
- typing
- decorators: `get_time`, `deprecated`, `final`, `log_action`, `abstractmethod`, `staticmethod`
- config: `MEAN_THRESHOLD`, `BRIGHT_PIXEL_RATIO`, `MAX_BRIGHTNESS`, `DEFAULT_SEPARATOR_LOOKFOR_DIR_NAME`,
`DEFAULT_SEPARATOR_OUTPUT_DIR_NAME`

This module is useful for preparing datasets for machine learning, ensuring that
only valid images are copied and that file operations are tracked.
"""

import cv2
import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod
from typing import final, Optional, cast
from numpy.typing import NDArray

from src.utils.decorators import get_time, log_action, deprecated

from src.data.config import (
    MEAN_THRESHOLD,
    BRIGHT_PIXEL_RATIO,
    MAX_BRIGHTNESS,
    DEFAULT_SEPARATOR_LOOKFOR_DIR_NAME,
    DEFAULT_SEPARATOR_OUTPUT_DIR_NAME,
)


class ImageSeparator(ABC):
    """
    An abstract class to filter out low intensity image files from an original raw dataset to an interim dataset.

    This class supports filtering by mean imtensity, brightness and bright pixel ratio, copying only valid image
    extensions, logging progress, and dry-run mode for testing.

    Attributes:
        dataset_path (Path): Path to the original raw dataset folder.
        lookfor (str): A folder name or class to process.
        out (str): Subdirectory name for the filtered output.
        dry_run (bool): If True, simulate copying without writing files.
    """

    def __init__(
        self,
        dataset_path: str,
        lookfor: str = DEFAULT_SEPARATOR_LOOKFOR_DIR_NAME,
        out: str = DEFAULT_SEPARATOR_OUTPUT_DIR_NAME,
        dry_run: bool = False,
    ):
        self.dataset_path: Path = Path(dataset_path)
        self.source_word: str = lookfor
        self.out = out
        self.dry_run = dry_run

    def __repr__(self) -> str:
        """
        __repr__ is meant to provide an unambiguous string representation of the object.
        It's often for debugging and should ideally return a string that could be used
        to recreate the object.
        :return: a developer friendly representation of the object
        """
        return f"ImageSeparator(dataset_path={self.dataset_path}, source_word={self.source_word}, out={self.out})"

    def __str__(self) -> str:
        """
        __str__ is meant to provide a readable string representation of the object.
        It's what gets shown when you print the object or convert it to a string.
        :return: a user-friendly representation of the object
        """
        return f"Seperating Low Intensity Images from {self.dataset_path} to {self.out} (dry_run={self.dry_run})"

    @get_time
    @staticmethod
    @final
    def is_mostly_black(
        img_path: Path,
        mean_thresh: int = MEAN_THRESHOLD,
        bright_pixel_ratio: float = BRIGHT_PIXEL_RATIO,
    ) -> bool:
        """
        Determines whether an image is predominantly black or very dark.

        This method reads an image in grayscale, calculates the mean pixel intensity,
        and checks the proportion of bright pixels. An image is considered "mostly black"
        if either:
          1. The mean intensity is below `mean_thresh`, or
          2. The ratio of pixels brighter than a threshold (`MAX_BRIGHTNESS`) is less than `bright_pixel_ratio`.

          :param img_path: Path to the image file.
          :param mean_thresh: The mean intensity threshold below which the image is
          considered mostly black. Defaults to `MEAN_THRESHOLD`.
          :param bright_pixel_ratio: Maximum allowed ratio of bright pixels for the
          image to be considered mostly black. Defaults to `BRIGHT_PIXEL_RATIO`.
          :return: True if the image is mostly black or if reading the image fails, False otherwise.
        """
        img = cast(
            Optional[NDArray[np.generic]],
            cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE),
        )
        if img is None:
            return True  # Image reading failed

        mean_intensity: float = img.mean()

        # Ratio of pixels brighter than 50 (adjustable)
        img = cast(NDArray[np.uint8], img)
        bright_pixels: int = int(np.sum(img > MAX_BRIGHTNESS))
        ratio: float = bright_pixels / img.size

        # Mostly black if mean very low OR almost all pixels are dark
        return mean_intensity < mean_thresh or ratio < bright_pixel_ratio

    @log_action
    def make_directory(self, name: Path) -> Path:
        """
        Create the filtered directory in the interim dataset.
        If the directory already exists, it does nothing.
        :param name: Subdirectory name to create inside interim dataset.
        :return: Full path to the filtered directory.
        """
        no_black_folder: Path = name / self.out
        no_black_folder.mkdir(parents=True, exist_ok=True)
        return no_black_folder

    @deprecated("Use filter_low_intensity_images instead")
    @log_action
    @get_time
    @abstractmethod
    def process_images(self) -> None:
        pass

    @abstractmethod
    def filter_low_intensity_images(self) -> None:
        pass
