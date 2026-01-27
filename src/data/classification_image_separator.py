from pathlib import Path
from typing import List
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
import shutil
import logging

from data.base_image_separator import ImageSeparator
from utils.utils_config import VALID_IMAGE_EXTENSIONS
from data.config import MAX_WORKERS, BATCH_SIZE

logger = logging.getLogger(__name__)


class ClassificationImageSeparator(ImageSeparator):
    def __init__(self, dataset_path: str, lookfor: str, out: str, dry_run: bool):
        super().__init__(dataset_path, lookfor, out, dry_run)

        self.source_folders: List[Path] = [
            f for f in self.dataset_path.iterdir() if f.is_dir()
        ]

    def process_images(self) -> None:
        for source in self.source_folders:
            source_path: Path = Path(source) / self.source_word

            if not source_path.exists():
                logger.debug(
                    "Skipping (no '%s' folder): %s", self.source_word, source_path
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
        try:
            if ImageSeparator.is_mostly_black(img):
                return True  # removed

            if self.dry_run:
                logger.info("Copying %s to %s", img, dest)
                return False
            else:
                shutil.copy2(img, dest)
                return False  # copied
        except Exception:
            logger.exception("File processing failed")
            return True

    def filter_low_intensity_images(self) -> None:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

            for source in self.source_folders:
                source_path: Path = Path(source) / self.source_word

                if not source_path.exists():
                    logger.debug(
                        "Skipping (no '%s' folder): %s", self.source_word, source_path
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

                for batches in ImageSeparator.batch(images, BATCH_SIZE):
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
                    self.source_word,
                    copied_count,
                    removed_count,
                )
