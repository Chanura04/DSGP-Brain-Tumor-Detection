"""
SegmentationImageSeparator Module

This module provides the `SegmentationImageSeparator` concrete class, which is designed to filter out / copy
low intensity images of a segmentation dataset of the organized raw dataset to an interim dataset. It supports:

- Filtering images by a the mean threshold/ mean intensity of the image.
- Filtering images by a the bright pixel ratio of the image.
- Filtering images by a the max brightness of the image.
- Copying only valid image extensions.
- Logging progress, duplicates, and summary information.
- Dry-run mode to simulate file operations without writing files.
- Measuring execution time for performance monitoring (via the `get_time` decorator).

Typical usage:

    from segmentation_image_separator import SegmentationImageSeparator

    image_separator = SegmentationImageSeparator(
        dataset_path="path/to/(mri, ct)"
        lookfor="original",
        out="no_black",
        dry_run=False,
        source=DEFAULT_SEPARATOR_SOURCE_DIR_NAME,
        apply_to=DEFAULT_SEPARATOR_APPLY_TO_DIR_NAME
    )
    image_separator.filter_low_intensity_images()

Dependencies:
- pathlib
- concurrent
- shutil
- logging
- typing
- config: `MAX_WORKERS`, `BATCH_SIZE`, `DEFAULT_SEPARATOR_SOURCE_DIR_NAME`, `DEFAULT_SEPARATOR_APPLY_TO_DIR_NAME`
- utils: `VALID_IMAGE_EXTENSIONS`

This module is useful for preparing datasets for machine learning, ensuring that
only valid images are copied and that file operations are tracked.
"""

from pathlib import Path
from typing import List
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
import shutil
import logging

from src.data.base_image_separator import ImageSeparator
from src.utils.utils_config import VALID_IMAGE_EXTENSIONS
from src.data.config import (
    MAX_WORKERS,
    BATCH_SIZE,
    DEFAULT_SEPARATOR_LOOKFOR_DIR_NAME,
    DEFAULT_SEPARATOR_OUTPUT_DIR_NAME,
    DEFAULT_SEPARATOR_SOURCE_DIR_NAME,
    DEFAULT_SEPARATOR_APPLY_TO_DIR_NAME,
)
from src.utils.batching import create_batch

from src.data.image_seperator_schema import SegmentationImageSeparatorConfig

logger = logging.getLogger(__name__)


class SegmentationImageSeparator(ImageSeparator):
    """
    A concrete class to filter out low intensity image files of a segmentation dataset from an original raw
    dataset to an interim dataset.

    This class supports filtering by mean imtensity, brightness and bright pixel ratio, copying only valid image
    extensions, logging progress, and dry-run mode for testing.

    Attributes:
        dataset_path (Path): Path to the original raw dataset folder.
        lookfor (str): A folder name or class to process.
        out (str): Subdirectory name for the filtered output.
        dry_run (bool): If True, simulate copying without writing files.
        source (str): Image path of the segmentation dataset
        apply_to (str): Mask path of the segmentation dataset
    """

    def __init__(self, config: SegmentationImageSeparatorConfig):
        super().__init__(config.dataset_path, config.lookfor, config.out, config.dry_run)
        self.source = config.source
        self.apply_to = config.apply_to

    def process_images(self, source: str, apply_to: str) -> None:
        logger.error("Use filter_low_intensity_images instead")
        raise NotImplementedError("Use filter_low_intensity_images instead")

    def _process_pair_images(
        self, img: Path, img_mask: Path, dest: Path, dest_mask: Path
    ) -> bool:
        """
        Process a pair of images (image and corresponding mask) by filtering low-intensity images
        and optionally copying them to destination paths.

        The method performs the following:
            1. Checks if the mask image exists. If not, logs a warning and considers the pair removed.
            2. Checks if the main image is mostly black using `ImageSeparator.is_mostly_black`.
            - If the image is mostly black, it is considered removed.
            3. If `self.dry_run` is True, logs the intended copy action without copying.
            4. If `self.dry_run` is False, copies both the main image and its mask to the specified
            destination paths.

        :param: img: Path to the main source image.
        :param: img_mask: Path to the corresponding mask image.
        :param: dest: Destination path for the main image.
        :param: dest_mask: Destination path for the mask image.
        :return: True if the image pair was removed (missing mask, mostly black, or error),
                False if both files were successfully copied or dry-run logged.
        """
        if img_mask.exists():
            try:
                if ImageSeparator.is_mostly_black(img):
                    return True  # removed

                if self.dry_run:
                    logger.info(
                        "Copying %s to %s and %s to %s", img, dest, img_mask, dest_mask
                    )
                    return False
                else:
                    shutil.copy2(img, dest)
                    shutil.copy2(img_mask, dest_mask)
                    return False  # copied

            except Exception:
                logger.exception("File processing failed")
                return True

        logger.warning("Missing mask for %s", img)
        return True

    def filter_low_intensity_images(self) -> None:
        """
        Filters and processes low-intensity (mostly black) image pairs in a dataset.

        Workflow:
            1. Determines source and output folders based on `self.dataset_path`, `self.source`,
            `self.apply_to`, and `self.source_word`.
            2. Skips processing if the source folder does not exist or if the source and destination
            paths are the same.
            3. Collects all valid image files (based on `VALID_IMAGE_EXTENSIONS`) from the source folder.
            4. Generates corresponding mask paths for each image in `apply_path` by replacing
            ".jpg" with "_m.jpg".
            5. Processes images in batches using a ThreadPoolExecutor:
                - Uses `_process_pair_images` to copy valid images and masks or mark them as removed.
            6. Tracks progress and logs updates every 50 processed files.
            7. Logs a summary of how many images were copied versus removed at the end of processing.

        :return: None
        """
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

            source_path: Path = self.dataset_path / self.source / self.source_word

            if not source_path.exists():
                logger.debug(
                    "Skipping (no '%s' folder): %s", self.source_word, source_path
                )

            out_folder: Path = self.make_directory(self.source)

            if source_path.resolve() == out_folder.resolve():
                logger.debug("Source and destination are the same, skipping")

            apply_path: Path = self.dataset_path / self.apply_to / self.source_word
            out_apply_to: Path = self.make_directory(self.apply_to)

            logger.info("Processing from: %s", source_path)
            logger.info("Applying to: %s", apply_path)
            logger.info("Outputting to: %s", out_folder)
            logger.info("Outputting to: %s", out_apply_to)

            # Process all images in source folder
            processed: int = 0
            copied_count: int = 0
            removed_count: int = 0
            images: List[Path] = [
                image
                for image in source_path.glob("*")
                if image.is_file() and image.suffix.lower() in VALID_IMAGE_EXTENSIONS
            ]

            for batches in create_batch(images, BATCH_SIZE):
                futures: List[Future[bool]] = [
                    executor.submit(
                        self._process_pair_images,
                        img,
                        apply_path / img.name.replace(".jpg", "_m.jpg"),
                        out_folder / img.name,
                        out_apply_to / img.name.replace(".jpg", "_m.jpg"),
                    )
                    for img in batches
                ]
                for future in as_completed(futures):
                    processed += 1
                    if future.result():
                        removed_count += 1
                    else:
                        copied_count += 1

                    # Log progress every 50 files
                    if processed % 50 == 0:
                        logger.info(
                            "Processed %d files so far in folder %s",
                            processed,
                            source_path,
                        )

            logger.info(
                "Look at '%s': copied %d images, removed %d mostly black images",
                self.source_word,
                copied_count,
                removed_count,
            )
