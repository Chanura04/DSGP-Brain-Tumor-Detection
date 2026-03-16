import io

from PIL import Image

from src.data.config import MEAN_BLACK_THRESHOLD, BRIGHT_BLACK_PIXEL_RATIO, MEAN_WHITE_THRESHOLD, \
    BRIGHT_WHITE_PIXEL_RATIO, MAX_BRIGHTNESS
import cv2
import numpy as np

from src.utils.decorators import get_time
from pathlib import Path
from numpy.typing import NDArray
from typing import Optional, cast

HEAD_DETECTION_IMG_SIZE = (128, 128)
CT_MRI_TUMOR_IMG_SIZE = (224, 224)
SEGMENTATION_TUMOR_IMG_SIZE = (256, 256)

IMAGE_DISPLAY_SIZE = (512, 512)


@get_time
def is_mostly_black(
        img_path: Path,
        mean_thresh: int = MEAN_BLACK_THRESHOLD,
        bright_pixel_ratio: float = BRIGHT_BLACK_PIXEL_RATIO,
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
    return bool(mean_intensity < mean_thresh or ratio < bright_pixel_ratio)


def is_too_black(img_bytes, mean_thresh: int = MEAN_BLACK_THRESHOLD,
                 bright_pixel_ratio: float = BRIGHT_BLACK_PIXEL_RATIO) -> bool:
    img = np.array(Image.open(io.BytesIO(img_bytes)).convert("L"))

    mean_intensity = img.mean()

    # Ratio of pixels brighter than 50 (adjustable)
    img = cast(NDArray[np.uint8], img)
    bright_pixels = int(np.sum(img > MAX_BRIGHTNESS))
    ratio = bright_pixels / img.size

    # Mostly black if mean very low OR almost all pixels are dark
    return bool(mean_intensity < mean_thresh or ratio < bright_pixel_ratio)


def is_too_white(img_bytes, mean_thresh: int = MEAN_WHITE_THRESHOLD,
                 bright_pixel_ratio: float = BRIGHT_WHITE_PIXEL_RATIO) -> bool:
    img = np.array(Image.open(io.BytesIO(img_bytes)).convert("L"))

    mean_intensity = img.mean()

    # Ratio of pixels brighter than 50 (adjustable)
    img = cast(NDArray[np.uint8], img)
    bright_pixels = int(np.sum(img > MAX_BRIGHTNESS))
    ratio = bright_pixels / img.size

    # Mostly black if mean very low OR almost all pixels are dark
    return bool(mean_intensity > mean_thresh or ratio > bright_pixel_ratio)
