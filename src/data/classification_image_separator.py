"""
ClassificationImageSeparator Module

This module provides the `ClassificationImageSeparator` concrete class, which is designed to filter out / copy
low intensity images of a classification dataset of the organized raw dataset to an interim dataset. It supports:

- Filtering images by the mean threshold/ mean intensity of the image.
- Filtering images by the bright pixel ratio of the image.
- Filtering images by the max brightness of the image.
- Copying only valid image extensions.
- Logging progress, duplicates, and summary information.
- Dry-run mode to simulate file operations without writing files.
- Measuring execution time for performance monitoring (via the `get_time` decorator).

Typical usage:

    from classification_image_separator import ClassificationImageSeparator

    image_separator = ClassificationImageSeparator(
        dataset_path="path/to/(mri, ct)"
        lookfor="original",
        out="no_black",
        dry_run=False
    )
    image_separator.filter_low_intensity_images()

Dependencies:
- pathlib
- concurrent
- shutil
- logging
- typing
- config: `MAX_WORKERS`, `BATCH_SIZE`
- utils: `VALID_IMAGE_EXTENSIONS`

This module is useful for preparing datasets for machine learning, ensuring that
only valid images are copied and that file operations are tracked.
"""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
import shutil
import logging

from typing import List

from src.data.base_image_separator import ImageSeparator
from src.utils.utils_config import VALID_IMAGE_EXTENSIONS
from src.data.config import MAX_WORKERS, BATCH_SIZE

from src.utils.batching import create_batch

from src.data.image_seperator_schema import ClassificationImageSeparatorConfig

logger = logging.getLogger(__name__)


class ClassificationImageSeparator(ImageSeparator):
    """
    A concrete class to filter out low intensity image files of a classification dataset from an original raw
    dataset to an interim dataset.

    This class supports filtering by mean intensity, brightness and bright pixel ratio, copying only valid image
    extensions, logging progress, and dry-run mode for testing.

    Attributes:
        dataset_path (Path): Path to the original raw dataset folder.
        lookfor (str): A folder name or class to process.
        out (str): Subdirectory name for the filtered output.
        dry_run (bool): If True, simulate copying without writing files.
    """

    def __init__(self, config: ClassificationImageSeparatorConfig):
        super().__init__(config.dataset_path, config.lookfor, config.out, config.dry_run)

        self.source_folders: List[Path] = [
            f for f in self.dataset_path.iterdir() if f.is_dir()
        ]

    def process_images(self) -> None:
        for source in self.source_folders:
            source_path: Path = Path(source) / self.lookfor

            if not source_path.exists():
                logger.debug(
                    "Skipping (no '%s' folder): %s", self.lookfor, source_path
                )
                continue

            out_folder: Path = self.make_directory(source)

            logger.info("Processing from: %s", source_path)
            logger.info("Outputting to: %s", out_folder)

            if source_path.resolve() == Path(out_folder).resolve():
                logger.debug("Source and destination are the same, skipping")
                continue

            # Process all images in source folder
            count_total: int = 0
            count_removed: int = 0
            for image in source_path.glob("*.*"):
                if image.is_file() and image.suffix.lower() in VALID_IMAGE_EXTENSIONS:
                    count_total += 1
                    if ClassificationImageSeparator.is_mostly_black(img_path=image):
                        count_removed += 1
            logger.info(
                "Processed %d images, removed %d mostly black images.",
                count_total,
                count_removed,
            )

    def _process_single_image(self, img: Path, dest: Path) -> bool:
        """
        Process a single image by checking its intensity and optionally copying it.

        This method performs the following steps:
            1. Checks if the image is mostly black using `ImageSeparator.is_mostly_black`.
            - If the image is mostly black or reading fails, it is considered "removed".
            2. If `self.dry_run` is True, it logs the copy operation without actually copying.
            3. If `self.dry_run` is False, it copies the image to the destination path.

        :param img: Path to the source image file.
        :param dest: Path to the destination where the image should be copied.

        returns: True if the image was removed (mostly black or failed), False if it was copied or dry-run logged.
        """
        try:
            if ImageSeparator.is_mostly_black(img):
                return True  # removed

            if self.dry_run:
                logger.info("Copying %s to %s", img, dest)
                return False
            else:
                shutil.copy2(img, dest)
                return False  # copied
        except OSError:
            logger.exception("File processing failed")
            return True

    def filter_low_intensity_images(self) -> None:
        """
        Filters out low-intensity (mostly black) images from source folders.

        This method:
            1. Iterates over each folder in `self.source_folders`.
            2. Skips folders that do not exist or if the source and destination folders are the same.
            3. Collects all valid image files (based on `VALID_IMAGE_EXTENSIONS`) from the source folder.
            4. Processes images in batches using a ThreadPoolExecutor for parallel execution.
            - Uses `_process_single_image` to determine if an image should be copied or removed.
            5. Tracks and logs progress every 50 files.
            6. Logs a summary of how many images were copied versus removed at the end of processing each folder.

        return: None
        """
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

            for source in self.source_folders:
                source_path: Path = Path(source) / self.lookfor

                if not source_path.exists():
                    logger.debug(
                        "Skipping (no '%s' folder): %s", self.lookfor, source_path
                    )
                    continue

                out_folder: Path = self.make_directory(source)

                logger.info("Processing from: %s", source_path)
                logger.info("Outputting to: %s", out_folder)

                if source_path.resolve() == Path(out_folder).resolve():
                    logger.debug("Source and destination are the same, skipping")
                    continue

                # Process all images in source folder
                processed: int = 0
                copied_count: int = 0
                removed_count: int = 0
                images: List[Path] = [
                    image
                    for image in source_path.glob("*")
                    if image.is_file()
                       and image.suffix.lower() in VALID_IMAGE_EXTENSIONS
                ]

                for batches in create_batch(images, BATCH_SIZE):
                    futures: List[Future[bool]] = [
                        executor.submit(
                            self._process_single_image,
                            img,
                            out_folder / img.name,
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
                    self.lookfor,
                    copied_count,
                    removed_count,
                )
