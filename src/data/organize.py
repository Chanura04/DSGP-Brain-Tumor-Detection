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
    mover.do_all_processes()

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
from typing import Set, List, Generator, Optional
import logging

from utils.utils_config import VALID_IMAGE_EXTENSIONS
from utils.decorators import get_time
from data.config import DEFAULT_INCLUDE_MODE, DEFAULT_OUTPUT_DIR_NAME

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
        VALID_EXTENSIONS (Set[str]): Valid image file extensions.
    """

    VALID_EXTENSIONS: Set[str] = VALID_IMAGE_EXTENSIONS

    def __init__(
        self,
        raw_dataset_path: str,
        interim_dataset_path: str,
        lookfor: Optional[List[str]],
        out: str = DEFAULT_OUTPUT_DIR_NAME,
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
        :return:
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

    def get_specific_paths(self, word: str) -> Generator[Path, None, None]:
        """
        Yield paths in the raw dataset matching the specified word/class.
        Searches for `word` in path if `include` subdirectories is True
        else does not search
        :param word: Folder or class name to search for.
        :return: Matching subdirectory paths.
        """
        word_lower: str = word.lower()
        base_path: Path = Path(self.raw_dataset_path)
        source_path: Path = base_path if not self.include else base_path / word
        for path in source_path.rglob("*"):
            if path.is_dir() and (self.include or word_lower in path.name.lower()):
                yield path

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
        copied_count = 0
        skipped_count = 0
        for i, image in enumerate(Path(src_folder).glob("*"), 1):
            if image.is_file() and image.suffix.lower() in MoveImages.VALID_EXTENSIONS:
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

    @get_time
    def do_all_processes(self) -> None:
        """
        Process all classes/folders specified in `lookfor`.
        For each class:
            1. Create merged directory in interim dataset.
            2. Find all source folders using `get_specific_paths`.
            3. Copy unique image files to merged folder using `copy_unique_files`.
            4. Log summary information.
        :return:
        """
        for word in self.lookfor:
            logger.info("Processing class %s", word)
            out_folder_path: Path = Path(word) / self.out
            merged_folder: Path = self.make_merged_directory(out_folder_path)
            source_folders: Generator[Path] = self.get_specific_paths(word)

            for src in source_folders:
                self.copy_unique_files(src, merged_folder, word)
                logger.info("Copied from %s", src)

            logger.info("Pasted to %s", self.interim_dataset_path / out_folder_path)
