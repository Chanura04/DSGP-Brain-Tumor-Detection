"""
ImageSeparator Module

This module provides the `ImageSeparator` abstract class, which is designed to filter out / copy
low intensity images of the organized raw dataset to an interim dataset. It supports:

- Filtering images by the mean threshold/ mean intensity of the image.
- Filtering images by the bright pixel ratio of the image.
- Filtering images by the max brightness of the image.
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

from pathlib import Path
from abc import ABC, abstractmethod

from src.utils.decorators import get_time, log_action, deprecated
from src.data.config import DEFAULT_SEPARATOR_LOOKFOR_DIR_NAME, DEFAULT_SEPARATOR_OUTPUT_DIR_NAME


class ImageSeparator(ABC):
    """
    An abstract class to filter out low intensity image files from an original raw dataset to an interim dataset.

    This class supports filtering by mean intensity, brightness and bright pixel ratio, copying only valid image
    extensions, logging progress, and dry-run mode for testing.

    Attributes:
        dataset_path (Path): Path to the original raw dataset folder.
        lookfor (str): A folder name or class to process.
        out (str): Subdirectory name for the filtered output.
        dry_run (bool): If True, simulate copying without writing files.
    """

    def __init__(
            self,
            dataset_path: Path,
            lookfor: str = DEFAULT_SEPARATOR_LOOKFOR_DIR_NAME,
            out: str = DEFAULT_SEPARATOR_OUTPUT_DIR_NAME,
            dry_run: bool = False,
    ):
        self.dataset_path: Path = Path(dataset_path)
        self.lookfor: str = lookfor
        self.out = out
        self.dry_run = dry_run

    def __str__(self) -> str:
        """
        __str__ is meant to provide a readable string representation of the object.
        It's what gets shown when you print the object or convert it to a string.
        :return: a user-friendly representation of the object
        """
        return f"Separating Low Intensity Images from {self.dataset_path} to {self.out} (dry_run={self.dry_run})"

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
    def process_images(self) -> None:
        pass

    @abstractmethod
    def filter_low_intensity_images(self) -> None:
        pass
