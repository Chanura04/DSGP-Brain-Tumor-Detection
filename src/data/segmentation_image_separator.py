from pathlib import Path
from typing import List
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
import shutil
import logging

from data.base_image_separator import ImageSeparator
from data.config import MAX_WORKERS, BATCH_SIZE

logger = logging.getLogger(__name__)


class SegmentationImageSeparator(ImageSeparator):
    def __init__(
        self, dataset_path: str, lookfor: str, out: str, source: str, apply_to: str
    ):
        super().__init__(dataset_path, lookfor, out)
        self.source = Path(source)
        self.apply_to = Path(apply_to)

    def process_images(self, source: str, apply_to: str) -> None:
        raise NotImplementedError("Use filter_low_intensity_images instead")

    @staticmethod
    def _process_pair_images(
        img: Path, img_mask: Path, dest: Path, dest_mask: Path
    ) -> bool:
        if img_mask.exists():
            try:
                if ImageSeparator.is_mostly_black(img):
                    return True  # removed

                shutil.copy2(img, dest)
                shutil.copy2(img_mask, dest_mask)
                return False  # copied

            except Exception:
                logger.exception("File processing failed")
                return True
        
        logger.warning("Missing mask for %s", img)
        return True

    def filter_low_intensity_images(self) -> None:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

            source_path: Path = self.dataset_path / self.source / self.source_word

            if not source_path.exists():
                logger.debug(
                    "Skipping (no '%s' folder): %s", self.source_word, source_path
                )

            out_folder: Path = self.make_directory(self.source)

            if source_path.resolve() == out_folder.resolve():
                logger.debug("Source and destination are the same, skipping")

            apply_path: Path = self.dataset_path / self.apply_to / self.source_word
            out_apply_to: Path = self.make_directory(self.apply_to)

            logger.info("Processing from: %s", source_path)
            logger.info("Applying to: %s", apply_path)
            logger.info("Outputting to: %s", out_folder)
            logger.info("Outputting to: %s", out_apply_to)

            # Process all images in source folder
            processed: int = 0
            copied_count: int = 0
            removed_count: int = 0
            images: List[Path] = [
                image for image in source_path.glob("*.jpg") if image.is_file()
            ]

            for batches in ImageSeparator.batch(images, BATCH_SIZE):
                futures: List[Future[bool]] = [
                    executor.submit(
                        SegmentationImageSeparator._process_pair_images,
                        img,
                        apply_path / img.name.replace(".jpg", "_m.jpg"),
                        out_folder / img.name,
                        out_apply_to / img.name.replace(".jpg", "_m.jpg"),
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
