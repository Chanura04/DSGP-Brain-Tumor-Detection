import cv2
import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod
from typing import final, Optional, cast, Generator, List
from numpy.typing import NDArray

from utils.decorators import get_time, log_calls, deprecated

from data.config import (
    MEAN_THRESHOLD,
    BRIGHT_PIXEL_RATIO,
    MAX_BRIGHTNESS,
    DEFAULT_SEPARATOR_LOOKFOR_DIR_NAME,
    DEFAULT_SEPARATOR_OUTPUT_DIR_NAME,
)


class ImageSeparator(ABC):
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

    @log_calls
    def make_directory(self, name: Path) -> Path:
        no_black_folder: Path = name / self.out
        no_black_folder.mkdir(parents=True, exist_ok=True)
        return no_black_folder

    @deprecated("Use filter_low_intensity_images instead")
    @log_calls
    @get_time
    @abstractmethod
    def process_images(self) -> None:
        pass

    @staticmethod
    def batch(iterable: List[Path], n: int) -> Generator[List[Path], None, None]:
        """
        Makes batches of the total images to reduce cpu overload,
        memory usage and have control over certain operation.
        e.g. to increase efficiency, reduce downtime, and improve consistency.

        :param iterable: image list to make batches
        :param n: number of images in a batch
        """
        batch_list: List[Path] = []
        for item in iterable:
            batch_list.append(item)
            if len(batch_list) == n:
                yield batch_list
                batch_list = []
        if batch_list:
            yield batch_list

    @abstractmethod
    def filter_low_intensity_images(self) -> None:
        pass
