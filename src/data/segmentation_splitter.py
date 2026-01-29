"""
SegmentationSplitter Module

This module provides a concrete classe for splitting segmentation datasets into
train, validation, and test sets.

Features:
    - Configurable train/val/test ratios.
    - Dry-run mode to simulate file operations without writing files.
    - Automatic creation of output directories.
    - Parallel batch copying of images to improve efficiency.
    - Logging of progress, skipped duplicates, and summary statistics.
    - Shuffling of images or image-mask pairs using a fixed random seed.

Typical usage:

    from segmentation_splitter import SegmentationSplitter

    splitter = SegmentationSplitter(
        interim_dataset_path="path/to/interim",
        processed_dataset_path="path/to/processed",
        lookfor="no_black",
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

from src.data.base_splitter import BaseSplitter

from src.utils.utils_config import RANDOM_SEED
from src.data.config import MAX_WORKERS, BATCH_SIZE, DEFAULT_SPLITTING_LOOKFOR_DIR_NAME

logger = logging.getLogger(__name__)


class SegmentationSplitter(BaseSplitter):
    """
    Splits an image-mask paired dataset into train, validation, and test sets for segmentation tasks.

    Inherits from BaseSplitter and implements the `split` method for image-mask pairs.

    Features:
        - Automatically finds corresponding masks for each image.
        - Copies images and masks to their respective train/val/test folders.
        - Skips duplicates automatically.
        - Supports dry-run mode.
        - Uses parallel batch copying for efficiency.
    """

    def __init__(
        self,
        interim_dataset_path: str,
        processed_dataset_path: str,
        lookfor: str = DEFAULT_SPLITTING_LOOKFOR_DIR_NAME,
        dry_run: bool = False,
    ):
        super().__init__(interim_dataset_path, processed_dataset_path, lookfor, dry_run)
        self.base_name_image: str = "images"
        self.base_name_mask: str = "mask"

    def copy_image(self, folder_images: Path, folder_masks: Path, pairs) -> bool:
        """
        Copy a single image-mask pair to the specified folders, skipping duplicates.

        :param: folder_images: Destination folder for images.
        :param: folder_masks: Destination folder for masks.
        :param: Tuple of (image_path, mask_path).
        :return:True if both files were copied, False if skipped or failed.
        """
        image, mask = pairs

        dest_images = folder_images / image.name
        dest_masks = folder_masks / mask.name

        if dest_images.exists() and dest_masks.exists():
            return False

        if self.dry_run:
            logger.info(
                "Copying %s to %s and %s to %s",
                image,
                folder_images,
                mask,
                folder_masks,
            )
            return True
        else:
            try:
                shutil.copy2(image, folder_images)
                shutil.copy2(mask, folder_masks)
                return True
            except Exception:
                logger.exception(
                    "Failed to copy %s: %s and %s: %s",
                    image,
                    folder_images,
                    mask,
                    folder_masks,
                )
                return False

    def split(self, train_ratio: float, val_ratio: float, seed=RANDOM_SEED) -> None:
        """
        Split a paired image-mask dataset into training, validation, and test sets.

        This method performs the following steps:
            1. Validates that train_ratio + val_ratio <= 1.
            2. Identifies the source image and mask folders under the interim dataset.
            3. Collects all images and their corresponding masks that exist, forming (image, mask) pairs.
            4. Shuffles the pairs using a reproducible random seed.
            5. Calculates the number of pairs for train and validation sets; the remaining go to the test set.
            6. Creates output directories for images and masks for each split (train, val, test).
            7. Copies the pairs to their respective folders using `copy_images`.
            8. Logs progress and output folder paths.

        :param: train_ratio: Fraction of pairs to assign to the training set (0 < train_ratio <= 1).
        :param: val_ratio: Fraction of pairs to assign to the validation set (0 <= val_ratio <= 1).
        :param: seed: Random seed for reproducible shuffling. Defaults to `RANDOM_SEED`.

        :raises: ValueError: If `train_ratio + val_ratio > 1`.
        :raises: RuntimeError: If no images are found in the source image folder.
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

        image_folder: Path = [
            f for f in self.source_folders if str(f.name) == "images"
        ][0]
        mask_folder: Path = [f for f in self.source_folders if str(f.name) == "mask"][0]

        image_path: Path = image_folder / self.source_word
        mask_path: Path = mask_folder / self.source_word

        if len(list(image_path.glob("*.jpg"))) == 0:
            logger.error("No images found in %s", image_path)
            raise RuntimeError(f"No images found in {image_path}")

        logger.info("Processing from: %s", str(image_path))

        pairs = []

        for img in image_path.glob("*.*"):
            mask = mask_path / f"{img.stem}_m{img.suffix}"
            if mask.exists():
                pairs.append((img, mask))

        rng.shuffle(pairs)

        train_count: int = int(len(pairs) * train_ratio)
        val_count: int = int(len(pairs) * val_ratio)

        train_out_folder_images: Path = self.make_directory(
            train_folder, self.base_name_image
        )
        val_out_folder_images: Path = self.make_directory(
            val_folder, self.base_name_image
        )
        test_out_folder_images: Path = self.make_directory(
            test_folder, self.base_name_image
        )

        train_out_folder_masks: Path = self.make_directory(
            train_folder, self.base_name_mask
        )
        val_out_folder_masks: Path = self.make_directory(
            val_folder, self.base_name_mask
        )
        test_out_folder_masks: Path = self.make_directory(
            test_folder, self.base_name_mask
        )

        logger.info("Outputting to: %s", str(train_out_folder_images))
        logger.info("Outputting to: %s", str(val_out_folder_images))
        logger.info("Outputting to: %s", str(test_out_folder_images))

        self.copy_images(
            train_out_folder_images, train_out_folder_masks, pairs[:train_count]
        )
        self.copy_images(
            val_out_folder_images,
            val_out_folder_masks,
            pairs[train_count : train_count + val_count],
        )
        self.copy_images(
            test_out_folder_images,
            test_out_folder_masks,
            pairs[train_count + val_count :],
        )

    def copy_images(self, folder_images: Path, folder_masks: Path, images) -> None:
        """
        Copy multiple image-mask pairs to their respective folders in parallel batches.

        Logs progress every 50 files and counts copied/skipped pairs.

        :param: folder_images: Destination folder for images.
        :param: folder_masks: Destination folder for masks.
        :param: images: List of (image, mask) tuples to copy.
        :return: None
        """
        copied_count: int = 0
        skipped_count: int = 0
        processed = 0

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for batches in BaseSplitter.batch(images, BATCH_SIZE):
                futures = [
                    executor.submit(self.copy_image, folder_images, folder_masks, pair)
                    for pair in batches
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
                            "Processed %d files so far in folder %s and %s",
                            processed,
                            folder_images,
                            folder_masks,
                        )

        logger.info(
            "Class '%s': copied %d files, skipped %d duplicates",
            self.source_word,
            copied_count,
            skipped_count,
        )
