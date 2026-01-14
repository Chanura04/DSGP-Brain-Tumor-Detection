from pathlib import Path
import shutil

from data.base_image_separator import ImageSeperator
from utils.utils_config import VALID_IMAGE_EXTENSIONS


class ClassificationImageSeperator(ImageSeperator):
    def __init__(self, dataset_path, lookfor, out):
        super().__init__(dataset_path, lookfor, out)

        self.source_folders = [f for f in self.dataset_path.iterdir() if f.is_dir()]

    def process_images(self):
        for source in self.source_folders:
            source_path = Path(source) / self.source_word

            if not source_path.exists():
                print(f"Skipping (no '{self.source_word}' folder): {source_path}")
                continue

            out_folder = self.make_directory(source)

            print(f"Processing from: {source_path}")
            print(f"Outputting to: {out_folder}")

            if source_path.resolve() == Path(out_folder).resolve():
                print("Source and destination are the same, skipping")
                continue

            # Process all images in source folder
            count_total = 0
            count_removed = 0
            for image in source_path.glob("*.*"):
                if image.is_file() and image.suffix.lower() in VALID_IMAGE_EXTENSIONS:
                    count_total += 1
                    if ClassificationImageSeperator.is_mostly_black(image):
                        count_removed += 1
                        continue
                    shutil.copy2(image, out_folder)

            print(
                f"Processed {count_total} images, removed {count_removed} mostly black images."
            )
