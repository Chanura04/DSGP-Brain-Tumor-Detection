from pathlib import Path
import shutil

from data.base_image_separator import ImageSeperator
from utils.utils_config import VALID_IMAGE_EXTENSIONS


class SegmentationImageSeperator(ImageSeperator):
    def __init__(self, dataset_path, lookfor, out):
        super().__init__(dataset_path, lookfor, out)
        self.valid_extension = VALID_IMAGE_EXTENSIONS[1]

    def process_images(self, source, apply_to):
        source_path = Path(self.dataset_path) / source / self.source_word

        if not source_path.exists():
            print(f"Skipping (no '{self.source_word}' folder): {source_path}")

        out_folder = self.make_directory(Path(source))

        if source_path.resolve() == Path(out_folder).resolve():
            print("Source and destination are the same, skipping")

        apply_path = Path(self.dataset_path) / apply_to / self.source_word
        out_apply_to = self.make_directory(Path(apply_to))

        print(f"Processing from: {source_path}")
        print(f"Applying to: {apply_path}")
        print(f"Outputting to: {out_folder}")
        print(f"Outputting to: {out_apply_to}")

        # Process all images in source folder
        count_total = 0
        count_removed = 0
        for image in source_path.glob("*.jpg"):
            count_total += 1

            mask = apply_path / image.name.replace(".jpg", "_m.jpg")
            if not mask.exists():
                continue  # skip broken pairs

            if SegmentationImageSeperator.is_mostly_black(image):
                count_removed += 1
                continue

            shutil.copy2(image, out_folder / image.name)
            shutil.copy2(mask, out_apply_to / mask.name)

        print(
            f"Processed {count_total} images, removed {count_removed} mostly black images."
        )
