"""
MoveImages Module

This module provides the `MoveImages` class, which is designed to move or copy
image files from a raw dataset to an interim dataset. It supports:

- Filtering by folder names or classes.
- Copying only valid image extensions.
- Logging progress, duplicates, and summary information.
- Dry-run mode to simulate file operations without writing files.
- Measuring execution time for performance monitoring (via the `get_time` decorator).

Typical usage:

    from move_images import MoveImages

    mover = MoveImages(
        raw_dataset_path="path/to/raw",
        interim_dataset_path="path/to/interim",
        lookfor=["class1", "class2"],
        out="original",
        include=True,
        dry_run=False
    )
    mover.build_interim_dataset()

Dependencies:
- pathlib
- shutil
- logging
- typing
- decorators: `get_time`
- config: `DEFAULT_INCLUDE_MODE`, `DEFAULT_OUTPUT_DIR_NAME`
- utils: `VALID_IMAGE_EXTENSIONS`

This module is useful for preparing datasets for machine learning, ensuring that
only valid images are copied and that file operations are tracked.
"""

from pathlib import Path
import shutil
from typing import List, Generator
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.utils_config import VALID_IMAGE_EXTENSIONS
from utils.decorators import get_time, log_action, deprecated
from data.config import (
    DEFAULT_INCLUDE_MODE,
    DEFAULT_ORGANIZE_OUTPUT_DIR_NAME,
    MAX_WORKERS,
    BATCH_SIZE,
)

logger = logging.getLogger(__name__)


class MoveImages:
    """
    A class to move or copy image files from a raw dataset to an interim dataset.

    This class supports filtering by folder names, copying only valid image
    extensions, logging progress, and dry-run mode for testing.

    Attributes:
        raw_dataset_path (Path): Path to the raw dataset folder.
        interim_dataset_path (Path): Path to the target interim dataset folder.
        lookfor (List[str]): List of folder names or classes to process.
        out (str): Subdirectory name for the merged output.
        include (bool): If True, include subfolders matching folder names.
        dry_run (bool): If True, simulate copying without writing files.
    """

    def __init__(
        self,
        raw_dataset_path: str,
        interim_dataset_path: str,
        lookfor: List[str],
        out: str = DEFAULT_ORGANIZE_OUTPUT_DIR_NAME,
        include: bool = DEFAULT_INCLUDE_MODE,
        dry_run: bool = False,
    ) -> None:
        """
        Initialize the MoveImages instance.
        :param raw_dataset_path: Path to the raw dataset folder.
        :param interim_dataset_path: Path to the interim dataset folder.
        :param lookfor: List of folder names or classes to process.
        :param out: Name of output subdirectory. Defaults to DEFAULT_OUTPUT_DIR_NAME.
        :param include: If True, include subfolders matching folder names. Defaults to DEFAULT_INCLUDE_MODE.
        :param dry_run: If True, simulate copying without writing files. Defaults to False.
        :return: None
        """
        self.raw_dataset_path: Path = Path(raw_dataset_path)
        self.interim_dataset_path: Path = Path(interim_dataset_path)
        self.lookfor: List[str] = lookfor
        self.out: str = out
        self.include: bool = include
        self.dry_run = dry_run

    def __repr__(self) -> str:
        """
        __repr__ is meant to provide an unambiguous string representation of the object.
        It's often for debugging and should ideally return a string that could be used
        to recreate the object.
        :return: a developer friendly representation of the object
        """
        return (
            f"MoveImages(raw_dataset_path={self.raw_dataset_path}, interim_dataset_path={self.interim_dataset_path}, "
            f"lookfor={self.lookfor}, out={self.out}, include={self.include})"
        )

    def __str__(self) -> str:
        """
        __str__ is meant to provide a readable string representation of the object.
        It's what gets shown when you print the object or convert it to a string.
        :return: a user-friendly representation of the object
        """
        return f"Moving Images from {self.raw_dataset_path} to {self.interim_dataset_path} (dry_run={self.dry_run})"

    @log_action
    def get_specific_paths(self, word: str) -> Generator[Path, None, None]:
        """
        Yield paths to directories in the raw dataset based on the `include` flag.
        If `include=True`:
            Yields **all subdirectories (at any depth)** inside the folder
            named `word` in the raw dataset, ignoring folder names.
        - If `include=False`:
            Recursively searches the entire raw dataset and yields only
            directories whose names contain the specified `word`.


        :param word: The folder or class name to filter by.
        :return: Paths to matching directories.
        """
        word_lower: str = word.lower()
        base_path: Path = self.raw_dataset_path
        source_path: Path = base_path if not self.include else base_path / word
        for path in source_path.rglob("*"):
            if path.is_dir() and (self.include or word_lower in path.name.lower()):
                yield path

    @log_action
    def make_merged_directory(self, name: Path) -> Path:
        """
        Create the merged directory in the interim dataset.
        If the directory already exists, it does nothing.
        :param name: Subdirectory name to create inside interim dataset.
        :return: Full path to the merged directory.
        """
        merged_folder: Path = self.interim_dataset_path / name
        merged_folder.mkdir(parents=True, exist_ok=True)
        return merged_folder

    @deprecated("Use copy_files for faster concurent tasks")
    @log_action
    @get_time
    def copy_unique_files(self, src_folder: Path, dest_folder: Path, word: str) -> None:
        """
        Copy unique image files from a source folder to a destination folder.
        Logs progress every 50 files, handles duplicates, and optionally supports dry-run mode.
        :param src_folder: Path to the source folder containing images.
        :param dest_folder: Path to the destination folder.
        :param word: Name of the class or folder being processed.
        :return:
        """
        copied_count: int = 0
        skipped_count: int = 0
        for i, image in enumerate(src_folder.glob("*"), 1):
            if image.is_file() and image.suffix.lower() in VALID_IMAGE_EXTENSIONS:
                dest_file = dest_folder / image.name
                if not dest_file.exists():
                    if self.dry_run:
                        logger.info("Copying %s to %s", image, dest_file)
                    else:
                        try:
                            shutil.copy2(image, dest_file)
                        except Exception:
                            logger.exception("Failed to copy %s: %s", image, dest_file)
                    copied_count += 1
                else:
                    logger.debug("Duplicate skipped: %s", image)
                    skipped_count += 1

            # Log progress every 50 files
            if i % 50 == 0:
                logger.info("Processed %d files so far in folder %s", i, src_folder)

        logger.info(
            "Class '%s': copied %d files, skipped %d duplicates",
            word,
            copied_count,
            skipped_count,
        )

    def copy_file(self, src: Path, dest: Path) -> bool:
        """
        Copies an image from the source folder to destination folder,
        created to run concurrently.

        :param src: image file path to be copied
        :param dest: image file path to be pasted
        :return: if image successfully copied
        """
        if not dest.exists():
            if self.dry_run:
                logger.info("Copying %s to %s", src, dest)
                return True
            else:
                try:
                    shutil.copy2(src, dest)
                    return True
                except Exception:
                    logger.exception("Failed to copy %s: %s", src, dest)
                    return False
        else:
            logger.debug("Duplicate skipped: %s", src)
            return False

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

    def copy_files(self, src_folder: Path, dest_folder: Path, word: str) -> None:
        """
        Copy unique image files from a source folder to a destination folder concurrently.
        Logs progress every 50 files, handles duplicates, and optionally supports dry-run mode.
        :param src_folder: Path to the source folder containing images.
        :param dest_folder: Path to the destination folder.
        :param word: Name of the class or folder being processed.
        :return:
        """
        copied_count: int = 0
        skipped_count: int = 0
        processed = 0
        images: List[Path] = [
            image
            for image in src_folder.glob("*")
            if image.is_file() and image.suffix.lower() in VALID_IMAGE_EXTENSIONS
        ]

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for batches in MoveImages.batch(images, BATCH_SIZE):
                futures = [
                    executor.submit(self.copy_file, img, dest_folder / img.name)
                    for img in batches
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
                            src_folder,
                        )

        logger.info(
            "Class '%s': copied %d files, skipped %d duplicates",
            word,
            copied_count,
            skipped_count,
        )

    @log_action
    @get_time
    def build_interim_dataset(self) -> None:
        """
        Process all classes/folders specified in `lookfor`.
        For each class:
            1. Create merged directory in interim dataset.
            2. Find all source folders using `get_specific_paths`.
            3. Copy unique image files to merged folder using `copy_files`.
            4. Log summary information.
        :return:
        """
        for word in self.lookfor:
            logger.info("Processing class %s", word)
            out_folder_path: Path = Path(word) / self.out
            merged_folder: Path = self.make_merged_directory(out_folder_path)
            source_folders: Generator[Path] = self.get_specific_paths(word)

            for src in source_folders:
                self.copy_files(src, merged_folder, word)
                logger.info("Copied from %s", src)

            logger.info("Pasted to %s", self.interim_dataset_path / out_folder_path)
