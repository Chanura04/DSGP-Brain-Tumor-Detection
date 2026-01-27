"""
TopViewImageSelector Module

This module provides the `TopViewImageSelector` class, designed for semi-supervised selection
of top-view images from a labeled dataset and large unlabeled datasets. It supports:

- Training a ResNet18-based classifier on labeled images.
- Preprocessing images with resizing, color jitter, normalization, and tensor conversion.
- Filtering out low-confidence or non-top-view images during inference.
- Copying high-confidence top-view images to output directories.
- Generating CSV files with predictions and confidence scores.
- Logging training metrics and prediction confidence distributions via TensorBoard.
- Dry-run mode to simulate file operations without writing to disk.
- Parallel batch processing of images to improve efficiency.
- Configurable training parameters: batch size, number of epochs, learning rate.

Typical usage:

    from top_view_image_filterer import TopViewImageSelector

    top_view_filterer = TopViewImageSelector(
        log_dir="path/to/log",
        labeled_images_path="path/to/labeled_images",
        model=None,
        batch_size=8,
        num_epochs=5,
        learning_rate=1e-4,
        dry_run=False,
    )
    top_view_filterer.run_model()
    top_view_filterer.model_predict(...)


Dependencies:
- shutil
- pathlib
- torch
- torchvision
- PIL
- logging
- json
- csv
- typing
- decorators: `get_time`, `log_action`
- config: `LOG_DIR`, `LABELED_IMAGES_DATA_DIR`, `IMG_TRANSFORM_RESIZE_SIZE`, `IMG_TRANSFORM_BRIGHTNESS`,
          `IMG_TRANSFORM_CONTRAST`, `IMG_TRANSFORM_MEAN_VECTOR`, `IMG_TRANSFORM_STD_VECTOR`,
          `DEFAULT_TOPVIEW_LOOKFOR_DIR_NAME`, `DEFAULT_TOPVIEW_OUTPUT_DIR_NAME`,
          `DEFAULT_TOPVIEW_PREDICTIONS_FILE_NAME`, `CONFIDENCE_THRESHOLD`,
- utils: `VALID_IMAGE_EXTENSIONS`

This module is useful for preparing datasets for downstream machine learning tasks, ensuring
that only high-confidence top-view images are used for further processing or model training.
"""

import shutil
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from PIL import Image
import logging
import json
import csv
from torch.utils.tensorboard import SummaryWriter
from typing import List, Optional, Dict, Tuple, cast

from utils.utils_config import VALID_IMAGE_EXTENSIONS
from utils.decorators import get_time, log_action
from utils.file_utils import LOG_DIR, LABELED_IMAGES_DATA_DIR
from data.config import (
    IMG_TRANSFORM_RESIZE_SIZE,
    IMG_TRANSFORM_BRIGHTNESS,
    IMG_TRANSFORM_CONTRAST,
    IMG_TRANSFORM_MEAN_VECTOR,
    IMG_TRANSFORM_STD_VECTOR,
    DEFAULT_TOPVIEW_LOOKFOR_DIR_NAME,
    DEFAULT_TOPVIEW_OUTPUT_DIR_NAME,
    DEFAULT_TOPVIEW_PREDICTIONS_FILE_NAME,
    CONFIDENCE_THRESHOLD,
)


logger = logging.getLogger(__name__)


class TopViewImageSelector:
    """
    A semi-supervised image selector for top-view images using PyTorch.

    This class handles training a model on labeled top-view images, performing inference
    on unlabeled datasets, filtering low-intensity (mostly black) images, and copying
    high-confidence top-view images to output directories.

    Features:
        - Dataset handling and preprocessing with torchvision transforms.
        - Model definition, training, and evaluation using a ResNet18 architecture.
        - Top-view image prediction with confidence thresholding.
        - Optional dry-run mode for testing file operations.
        - TensorBoard logging for training metrics and prediction distributions.
        - File and directory management for dataset organization.

    Attributes:
        log_dir (Path): Directory to store logs and model checkpoints.
        labeled_images_path (Path): Path to labeled training images.
        batch_size (int): Batch size for training and inference.
        num_epochs (int): Number of epochs for model training.
        learning_rate (float): Learning rate for optimizer.
        dry_run (bool): If True, logs file operations without performing them.
        device (torch.device): Device used for training and inference (CPU or GPU).
        trained_dataset (Optional[datasets.ImageFolder]): PyTorch dataset for training.
        final_model (Optional[nn.Module]): Trained model for top-view image prediction.
        writer (SummaryWriter): TensorBoard writer for logging metrics.
        metrics (Dict[str, List[float]]): Dictionary to store training loss and other metrics.
    """

    def __init__(
        self,
        log_dir: Path = LOG_DIR,
        labeled_images_path: Path = LABELED_IMAGES_DATA_DIR,
        model=None,
        batch_size: int = 8,
        num_epochs: int = 5,
        learning_rate: float = 1e-4,
        dry_run: bool = False,
    ):
        self.log_dir: Path = log_dir

        # Labeled folders for training
        self.labeled_images_path: Path = labeled_images_path
        self.batch_size: int = batch_size
        self.num_epochs: int = num_epochs
        self.learning_rate: float = learning_rate
        self.dry_run = dry_run
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trained_dataset: Optional[datasets.ImageFolder] = None
        self.final_model: Optional[nn.Module] = model
        self.writer = SummaryWriter(self.log_dir / "tensorboard")
        self.metrics: Dict[str, List[float]] = {"epoch_loss": []}

    @log_action
    def make_directory(self, name: Path, out: Path) -> Path:
        """
        Create the filtered directory in the interim dataset.
        If the directory already exists, it does nothing.
        :param name: Subdirectory name to create inside interim dataset.
        :param out: Subdirectory name to create under the base path.
        :return: Full path to the filtered directory.
        """
        top_view_folder = name / out
        top_view_folder.mkdir(parents=True, exist_ok=True)
        return top_view_folder

    @log_action
    def _define_train(self) -> transforms.Compose:
        """
        Define the transformation pipeline for training images.

        :return: transforms.Compose: A composition of image transformations including:
                - Resize
                - Color jitter (brightness and contrast)
                - Conversion to tensor
                - Normalization using predefined mean and standard deviation
        """
        return transforms.Compose(
            [
                transforms.Resize(IMG_TRANSFORM_RESIZE_SIZE),
                transforms.ColorJitter(
                    brightness=IMG_TRANSFORM_BRIGHTNESS, contrast=IMG_TRANSFORM_CONTRAST
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMG_TRANSFORM_MEAN_VECTOR, std=IMG_TRANSFORM_STD_VECTOR
                ),
            ]
        )

    @log_action
    @get_time
    def _load_data(self, train_transform: transforms.Compose) -> DataLoader:
        """
        Load labeled images as a PyTorch DataLoader.

        :param: train_transform (transforms.Compose): Transformations to apply to images.
        :return: A DataLoader object for iterating over the labeled dataset.
        """
        self.trained_dataset = datasets.ImageFolder(
            self.labeled_images_path, transform=train_transform
        )
        dataloader = DataLoader(
            self.trained_dataset, batch_size=self.batch_size, shuffle=True
        )
        return dataloader

    @log_action
    def _define_model(self) -> nn.Module:
        """
        Define the ResNet18 model architecture for top-view classification.
        Modifies the final fully connected layer to output two classes: "top" vs "other".

        :return: Initialized and device-assigned ResNet18 model.
        """
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model: nn.Module = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 2)  # top vs other
        model = model.to(self.device)
        return model

    @log_action
    def _compile_model(
        self, model: nn.Module
    ) -> Tuple[nn.Module, torch.optim.Optimizer]:
        """
        Compile the model with loss function and optimizer for training.

        :param: model: Model to compile.
        :return: The loss criterion and optimizer.
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        return criterion, optimizer

    @log_action
    @get_time
    def run_model(self) -> None:
        """
        Train the top-view classifier model on labeled images.

        Workflow:
            - Prepares dataset and DataLoader.
            - Defines model, loss, and optimizer.
            - Runs training loop over specified number of epochs.
            - Logs epoch loss to TensorBoard.
            - Saves metrics to JSON and model checkpoint to disk.

        :return: None
        """
        if self.final_model is None and self.trained_dataset is not None:
            logger.info("===== TRAINING STARTED =====")
            logger.info(f"Dataset: {self.labeled_images_path}")
            logger.info(f"Epochs: {self.num_epochs}, Batch size: {self.batch_size}")
            train_transform = self._define_train()
            dataloader = self._load_data(train_transform)
            self.final_model = self._define_model()
            criterion, optimizer = self._compile_model(self.final_model)

            self.final_model.train()
            for epoch in range(self.num_epochs):
                running_loss: float = 0.0
                for imgs, labels in dataloader:
                    imgs, labels = imgs.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    outputs = self.final_model(imgs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

                epoch_loss = running_loss / len(dataloader)
                self.metrics["epoch_loss"].append(epoch_loss)
                self.writer.add_scalar("Loss/train", epoch_loss, epoch)
                logger.info(
                    f"Epoch {epoch+1}/{self.num_epochs}, Loss: {epoch_loss:.4f}"
                )

            with open(self.log_dir / "metrics.json", "w") as f:
                json.dump(self.metrics, f, indent=4)

            torch.save(
                {
                    "model_state_dict": self.final_model.state_dict(),
                    "class_to_idx": self.trained_dataset.class_to_idx,
                },
                self.log_dir / "model.pth",
            )
            logger.info(f"Model saved to {self.log_dir / 'model.pth'}")

            logger.info("===== TRAINING FINISHED =====")

        else:
            logger.info("Using given model!")

    @log_action
    @get_time
    def model_predict(
        self,
        dataset_path: str,
        lookfor: str = DEFAULT_TOPVIEW_LOOKFOR_DIR_NAME,
        out: str = DEFAULT_TOPVIEW_OUTPUT_DIR_NAME,
        predictions_out_file: str = DEFAULT_TOPVIEW_PREDICTIONS_FILE_NAME,
        confidence_thresh: float = CONFIDENCE_THRESHOLD,
    ) -> None:
        """
        Predict top-view images in a dataset using the trained model and copy high-confidence images.

        Workflow:
            1. Validates that the model is trained.
            2. Iterates over source folders to locate images in the `lookfor` subfolder.
            3. Applies preprocessing transforms and performs model inference.
            4. Filters images based on a confidence threshold for the "top" class.
            5. Copies high-confidence images to the output folder, optionally logging in dry-run mode.
            6. Saves predictions in CSV format and logs confidence distributions to TensorBoard.

        :param: dataset_path: Root path containing source folders of images.
        :param: lookfor: Subfolder name to look for images. Defaults to `DEFAULT_TOPVIEW_LOOKFOR_DIR_NAME`.
        :param: out: Output folder name for high-confidence images. Defaults to `DEFAULT_TOPVIEW_OUTPUT_DIR_NAME`.
        :param: predictions_out_file: CSV file to save predictions. Defaults to `DEFAULT_TOPVIEW_PREDICTIONS_FILE_NAME`.
        :param: confidence_thresh: Minimum probability threshold to consider an image as top-view.
        Defaults to `CONFIDENCE_THRESHOLD`.

        :raises: RuntimeError: If the model has not been trained.
        """
        if self.final_model is None or self.trained_dataset is None:
            logger.error("Model not trained. Call train_model() first.")
            raise RuntimeError("Model not trained. Call train_model() first.")

        # Source folders to inference / predict top-view images
        source_folders: List[Path] = [
            f for f in Path(dataset_path).iterdir() if f.is_dir()
        ]

        csv_path = self.log_dir / predictions_out_file
        with open(csv_path, "w", newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["image_path", "top_probability"])

        for source in source_folders:
            source_path = source / lookfor

            logger.info(f"Processing dataset: {source_path}")

            if not source_path.exists():
                logger.warning(f"Skipping (no '{lookfor}' folder): {source_path}")
                continue

            out_folder = self.make_directory(source, Path(out))

            if source_path.resolve() == Path(out_folder).resolve():
                logger.warning("Source and destination are the same, skipping")
                continue

            self.final_model.eval()
            preproc: transforms.Compose = transforms.Compose(
                [
                    transforms.Resize(IMG_TRANSFORM_RESIZE_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=IMG_TRANSFORM_MEAN_VECTOR, std=IMG_TRANSFORM_STD_VECTOR
                    ),
                ]
            )

            predictions: List[Tuple[Path, float]] = []

            with torch.inference_mode():
                top_idx = self.trained_dataset.class_to_idx.get("top", 0)
                for img_path in source_path.glob("*.*"):
                    if img_path.suffix.lower() in VALID_IMAGE_EXTENSIONS:
                        img: Image.Image = Image.open(img_path).convert("RGB")
                        img_tensor = cast(torch.Tensor, preproc(img))
                        x: torch.Tensor = img_tensor.unsqueeze(0).to(self.device)
                        logits: torch.Tensor = self.final_model(x)
                        probs: torch.Tensor = torch.softmax(logits, dim=1)
                        top_prob: float = probs[0][top_idx].item()
                        predictions.append((img_path, top_prob))

            predictions = [p for p in predictions if p[1] >= confidence_thresh]
            predictions.sort(key=lambda x: x[1], reverse=True)

            with open(csv_path, "a", newline="") as f:
                csv_writer = csv.writer(f)
                for img_path, prob in predictions:
                    dest = Path(out_folder) / img_path.name
                    if not dest.exists():
                        csv_writer.writerow([str(img_path), prob])
                        if self.dry_run:
                            logger.info("Copying %s to %s", img_path, dest)
                        else:
                            shutil.copy2(img_path, dest)

            if len(predictions) > 0:
                probs = torch.tensor([p[1] for p in predictions])
                self.writer.add_histogram(
                    "TopView_Confidence",
                    probs,
                    global_step=len(self.metrics["epoch_loss"]),
                )
                logger.info(
                    f"Top-view images copied to '{out_folder}'. Total: {len(predictions)}"
                )

        self.writer.close()
