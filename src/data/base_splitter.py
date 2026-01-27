from pathlib import Path
from abc import ABC, abstractmethod

from utils.utils_config import RANDOM_SEED
from utils.decorators import get_time, log_action

from typing import List, Generator, TypeVar

T = TypeVar("T")  # can be Path or Tuple[Path, Path]


class BaseSplitter(ABC):
    def __init__(
        self,
        interim_dataset_path: str,
        processed_dataset_path: str,
        lookfor: str,
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
        pass
