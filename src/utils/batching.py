from typing import List, Generator
from pathlib import Path


def create_batch(iterable: List[Path], n: int) -> Generator[List[Path], None, None]:
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
