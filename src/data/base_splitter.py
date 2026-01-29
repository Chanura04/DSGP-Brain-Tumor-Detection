"""
Dataset Splitter Module

This module provides abstract and concrete classes for splitting datasets into
train, validation, and test sets. It supports both classification and segmentation
tasks.

Classes:
    - BaseSplitter: Abstract base class for dataset splitters.
    - ClassificationSplitter: Splits image datasets for classification tasks.
    - SegmentationSplitter: Splits image-mask pair datasets for segmentation tasks.

Features:
    - Configurable train/val/test ratios.
    - Dry-run mode to simulate file operations without writing files.
    - Automatic creation of output directories.
    - Parallel batch copying of images to improve efficiency.
    - Logging of progress, skipped duplicates, and summary statistics.
    - Shuffling of images or image-mask pairs using a fixed random seed.

Dependencies:
    - pathlib, logging, shutil, numpy
    - concurrent.futures for parallel copying
    - utils: `get_time`, `log_action`, `RANDOM_SEED`
    - config: `MAX_WORKERS`, `BATCH_SIZE`

This module is useful for preparing datasets for training and evaluation of
machine learning models, ensuring a controlled, reproducible split of data.
"""

from pathlib import Path
from abc import ABC, abstractmethod

from src.utils.utils_config import RANDOM_SEED
from src.utils.decorators import get_time, log_action
from src.data.config import DEFAULT_SPLITTING_LOOKFOR_DIR_NAME

from typing import List, Generator, TypeVar

T = TypeVar("T")  # can be Path or Tuple[Path, Path]


class BaseSplitter(ABC):
    """
    Abstract base class for dataset splitting.

    Handles basic setup of source folders, labels, and directory creation.

    Attributes:
        interim_dataset_path (Path): Path to the interim dataset containing raw data.
        processed_dataset_path (Path): Path to the folder where split datasets will be stored.
        source_folders (List[Path]): List of directories in the interim dataset.
        source_word (str): Folder name or category to process.
        labels (List[str]): Names of dataset splits ['train', 'val', 'test'].
        dry_run (bool): If True, simulate copying without writing files.
    """

    def __init__(
        self,
        interim_dataset_path: str,
        processed_dataset_path: str,
        lookfor: str = DEFAULT_SPLITTING_LOOKFOR_DIR_NAME,
        dry_run: bool = False,
    ):
        self.interim_dataset_path: Path = Path(interim_dataset_path)
        self.processed_dataset_path: Path = Path(processed_dataset_path)
        self.source_folders = [
            f for f in self.interim_dataset_path.iterdir() if f.is_dir()
        ]
        self.source_word: str = lookfor
        self.labels: List[str] = ["train", "val", "test"]
        self.dry_run = dry_run

    def __repr__(self):
        return f"{self.__class__.__name__}(source='{self.interim_dataset_path}')"

    @log_action
    def make_directory(self, base_path: Path, subfolder: str) -> Path:
        """
        Create the splitted directory in the interim dataset.
        If the directory already exists, it does nothing.
        :param base_path: Base folder where the subdirectory will be created.
        :param subfolder: Name of the subdirectory to create.
        :return: Full path to the splitted directory.
        """
        out_folder = base_path / subfolder
        out_folder.mkdir(parents=True, exist_ok=True)
        return out_folder

    @staticmethod
    def batch(iterable: List[T], n: int) -> Generator[List[T], None, None]:
        """
        Makes batches of the total images to reduce cpu overload,
        memory usage and have control over certain operation.
        e.g. to increase efficiency, reduce downtime, and improve consistency.

        :param iterable: image list to make batches
        :param n: number of images in a batch
        """
        batch_list: List[T] = []
        for item in iterable:
            batch_list.append(item)
            if len(batch_list) == n:
                yield batch_list
                batch_list = []
        if batch_list:
            yield batch_list

    @log_action
    @get_time
    @abstractmethod
    def split(
        self, train_ratio: float, val_ratio: float, seed: int = RANDOM_SEED
    ) -> None:
        """
        Abstract method to split dataset into train, validation, and test sets.

        :param: train_ratio: Fraction of data for the training set.
        :param: val_ratio: Fraction of data for the validation set.
        :param: seed: Random seed for reproducibility.

        :raises: NotImplementedError: Must be implemented by subclasses.
        """
        pass
