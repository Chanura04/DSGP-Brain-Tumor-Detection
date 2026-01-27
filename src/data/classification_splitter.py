"""
ClassificationSplitter Module

This module provides a concrete classe for splitting classification datasets into
train, validation, and test sets.

Features:
    - Configurable train/val/test ratios.
    - Dry-run mode to simulate file operations without writing files.
    - Automatic creation of output directories.
    - Parallel batch copying of images to improve efficiency.
    - Logging of progress, skipped duplicates, and summary statistics.
    - Shuffling of images or image-mask pairs using a fixed random seed.

Typical usage:

    from classification_splitter import ClassificationSplitter

    splitter = ClassificationSplitter(
        interim_dataset_path="path/to/interim",
        processed_dataset_path="path/to/processed",
        lookfor="top_view",
        dry_run=False
    )
    splitter.split()

Dependencies:
    - pathlib, logging, shutil, numpy
    - concurrent.futures for parallel copying
    - utils: `get_time`, `log_action`, `RANDOM_SEED`
    - config: `MAX_WORKERS`, `BATCH_SIZE`

This module is useful for preparing datasets for training and evaluation of
machine learning models, ensuring a controlled, reproducible split of data.
"""

import shutil
import numpy as np
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from data.base_splitter import BaseSplitter

from utils.utils_config import RANDOM_SEED
from data.config import MAX_WORKERS, BATCH_SIZE

logger = logging.getLogger(__name__)


class ClassificationSplitter(BaseSplitter):
    """
    Splits an image dataset into train, validation, and test sets for classification tasks.

    Inherits from BaseSplitter and implements the `split` method for images only.

    Features:
        - Copies images to train/val/test folders.
        - Skips duplicates automatically.
        - Supports dry-run mode.
        - Uses parallel batch copying for efficiency.
    """

    def __init__(
        self,
        interim_dataset_path: str,
        processed_dataset_path: str,
        lookfor: str,
        dry_run: bool = False,
    ):
        super().__init__(interim_dataset_path, processed_dataset_path, lookfor, dry_run)

    def copy_image(self, folder: Path, image) -> bool:
        """
        Copy a single image to the specified folder, skipping duplicates.

        :param: folder: Destination folder.
        :param: image: Path to the source image.
        :return: True if image was copied, False if skipped or failed.
        """
        dest = folder / image.name
        if dest.exists():
            return False

        if self.dry_run:
            logger.info("Copying %s to %s", image, folder)
            return True
        else:
            try:
                shutil.copy2(image, folder)
                return True
            except Exception:
                logger.exception("Failed to copy %s: %s", image, folder)
                return False

    def split(
        self, train_ratio: float, val_ratio: float, seed: int = RANDOM_SEED
    ) -> None:
        """
        Split a dataset of images into training, validation, and test sets.

        This method performs the following steps:
            1. Validates that train_ratio + val_ratio <= 1.
            2. Creates train, validation, and test directories under the processed dataset path.
            3. Iterates over all source folders in the interim dataset.
            4. Collects all images under the specified source word subfolder.
            5. Shuffles the images using a reproducible random seed.
            6. Calculates the number of images for train and validation sets based on the specified ratios.
            The remaining images are assigned to the test set.
            7. Copies images to the corresponding output folders in parallel using `copy_images`.
            8. Logs progress and output folder paths.

        :param: train_ratio: Fraction of images to assign to the training set (0 < train_ratio <= 1).
        :param: val_ratio: Fraction of images to assign to the validation set (0 <= val_ratio <= 1).
        :param: seed: Random seed for reproducible shuffling. Defaults to `RANDOM_SEED`.

        :raises: ValueError: If `train_ratio + val_ratio > 1`.
        :raises: RuntimeError: If no images are found in a source subfolder.
        """

        if train_ratio + val_ratio > 1:
            logger.error("train_ratio + val_ratio must be <= 1")
            raise ValueError("train_ratio + val_ratio must be <= 1")

        rng: np.random.Generator = np.random.default_rng(seed)

        train_folder: Path = self.make_directory(
            self.processed_dataset_path, self.labels[0]
        )
        val_folder: Path = self.make_directory(
            self.processed_dataset_path, self.labels[1]
        )
        test_folder: Path = self.make_directory(
            self.processed_dataset_path, self.labels[2]
        )

        for source in self.source_folders:
            source_path: Path = source / self.source_word
            logger.info("Processing from: %s", str(source_path))

            base_name: str = source.name

            images = np.array(list(source_path.glob("*.*")))

            if len(images) == 0:
                logger.error("No images found in %s", source_path)
                raise RuntimeError(f"No images found in {source_path}")

            rng.shuffle(images)

            train_count: int = int(images.size * train_ratio)
            val_count: int = int(images.size * val_ratio)

            train_out_folder: Path = self.make_directory(train_folder, base_name)
            val_out_folder: Path = self.make_directory(val_folder, base_name)
            test_out_folder: Path = self.make_directory(test_folder, base_name)

            logger.info("Outputting to: %s", str(train_out_folder))
            logger.info("Outputting to: %s", str(val_out_folder))
            logger.info("Outputting to: %s", str(test_out_folder))

            self.copy_images(train_out_folder, images[:train_count])
            self.copy_images(
                val_out_folder, images[train_count : train_count + val_count]
            )
            self.copy_images(test_out_folder, images[train_count + val_count :])

    def copy_images(self, folder: Path, images) -> None:
        """
        Copy multiple images to the specified folder in parallel batches.

        Logs progress every 50 files and counts copied/skipped files.

        :param: folder: Destination folder.
        :param: images: List of image paths to copy.
        :return: None
        """
        copied_count: int = 0
        skipped_count: int = 0
        processed = 0

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for batches in BaseSplitter.batch(images, BATCH_SIZE):
                futures = [
                    executor.submit(self.copy_image, folder, img) for img in batches
                ]

                for future in as_completed(futures):
                    processed += 1
                    if future.result():
                        copied_count += 1
                    else:
                        skipped_count += 1

                    # Log progress every 50 files
                    if processed % 50 == 0:
                        logger.info(
                            "Processed %d files so far in folder %s",
                            processed,
                            folder,
                        )

        logger.info(
            "Class '%s': copied %d files, skipped %d duplicates",
            self.source_word,
            copied_count,
            skipped_count,
        )
