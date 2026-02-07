# scripts/preprocess.py
import argparse
import logging

from src.data.organizer import MoveImages
from src.data.classification_image_separator import ClassificationImageSeparator
from src.data.top_view_image_filterer import TopViewImageSelector
from src.data.classification_splitter import ClassificationSplitter

from src.utils.file_utils import RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from src.utils.logging_utils import setup_logging

setup_logging()

logger = logging.getLogger(__name__)
logger.info("Starting preprocessing script...")


def organize_step():
    logger.info("Running organize_data (MoveImages)...")
    mover = MoveImages(raw_dataset_path=f"{RAW_DATA_DIR}/mri", interim_dataset_path=f"{INTERIM_DATA_DIR}/mri",
                       lookfor=["glioma", "meningioma", "pituitary"], out="original", include=False)
    mover.build_interim_dataset()
    logger.info("Organize step complete.")


def high_filter_step():
    logger.info("Running high_filter (ClassificationImageSeparator)...")
    seperator = ClassificationImageSeparator(dataset_path=f"{INTERIM_DATA_DIR}/mri", lookfor="original",
                                             out="no_black", dry_run=False)
    seperator.filter_low_intensity_images()
    logger.info("Removing intensity images step complete.")


def top_view_filter_step():
    logger.info("Running top-view (TopViewImageSelector)...")
    top_view_selector = TopViewImageSelector(path_to_trained_model="models/top_view.pth")
    top_view_selector.run_model()
    top_view_selector.model_predict(dataset_path=f"{INTERIM_DATA_DIR}/mri")
    logger.info("Filtering top-view images step complete.")


def split_dataset_step():
    logger.info("Running split_dataset (ClassificationSplitter)...")
    splitter = ClassificationSplitter(interim_dataset_path=f"{INTERIM_DATA_DIR}/mri",
                                      processed_dataset_path=f"{PROCESSED_DATA_DIR}/mri/raw", lookfor="top_view")
    splitter.split(train_ratio=0.7, val_ratio=0.2)
    logger.info("Splitting dataset step complete.")


def main(step):
    if step == "organize":
        organize_step()

    elif step == "high_filter":
        high_filter_step()

    elif step == "top_view_filter":
        top_view_filter_step()

    elif step == "split_dataset":
        split_dataset_step()

    elif step == "all":
        logger.info("Running full preprocessing pipeline...")
        organize_step()
        high_filter_step()
        top_view_filter_step()
        split_dataset_step()
    else:
        logger.error(f"Unknown step: {step}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run preprocessing steps.")
    parser.add_argument(
        "--step",
        type=str,
        required=True,
        choices=["organize", "high_filter", "top_view_filter", "split_dataset", "all"],
        help="Which preprocessing step to run",
    )
    args = parser.parse_args()
    main(args.step)
