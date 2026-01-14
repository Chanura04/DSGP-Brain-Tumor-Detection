import cv2
import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod
import logging

from utils.decorators import get_time, log_calls, deprecated

from data.config import MEAN_THRESHOLD, BRIGHT_PIXEL_RATIO, MAX_BRIGHTNESS

logger = logging.getLogger(__name__)


class ImageSeperator(ABC):
    def __init__(self, dataset_path, lookfor, out):
        self.dataset_path = Path(dataset_path)
        self.source_word = lookfor
        self.out = out

    @deprecated("")
    @get_time
    @staticmethod
    def is_mostly_black(
        img_path, mean_thresh=MEAN_THRESHOLD, bright_pixel_ratio=BRIGHT_PIXEL_RATIO
    ):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return True  # skip unreadable images

        mean_intensity = img.mean()

        # Ratio of pixels brighter than 50 (adjustable)
        bright_pixels = np.sum(img > MAX_BRIGHTNESS)
        ratio = bright_pixels / img.size

        # Mostly black if mean very low OR almost all pixels are dark
        return mean_intensity < mean_thresh or ratio < bright_pixel_ratio

    @log_calls
    def make_directory(self, name):
        no_black_folder = name / self.out
        no_black_folder.mkdir(parents=True, exist_ok=True)
        return no_black_folder

    @log_calls(logger)
    @get_time
    @abstractmethod
    def process_images(self):
        pass
