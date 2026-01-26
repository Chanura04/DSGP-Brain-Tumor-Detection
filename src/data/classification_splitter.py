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
    def __init__(
        self, interim_dataset_path: str, processed_dataset_path: str, lookfor: str
    ):
        super().__init__(interim_dataset_path, processed_dataset_path, lookfor)

    def copy_image(self, folder: Path, image) -> bool:
        dest = folder / image.name
        if dest.exists():
            return False

        try:
            shutil.copy2(image, folder)
            return True
        except Exception:
            logger.exception("Failed to copy %s: %s", image, folder)
            return False

    def split(
        self, train_ratio: float, val_ratio: float, seed: int = RANDOM_SEED
    ) -> None:
        if train_ratio + val_ratio > 1:
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

    def copy_images(self, folder: Path, images):
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
