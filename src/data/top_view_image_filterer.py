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

# Semi-supervised learning

# Train on: Small, clean, human-labeled data, Balanced (top vs other)
# Infer on: Large, noisy, unlabeled datasets
# Select: Only high-confidence top-view images


class TopViewImageSelector:
    def __init__(
        self,
        log_dir: Path = LOG_DIR,
        labeled_images_path: Path = LABELED_IMAGES_DATA_DIR,
        model=None,
        batch_size: int = 8,
        num_epochs: int = 5,
        learning_rate: float = 1e-4,
    ):
        self.log_dir: Path = log_dir

        # Labeled folders for training
        self.labeled_images_path: Path = labeled_images_path
        self.batch_size: int = batch_size
        self.num_epochs: int = num_epochs
        self.learning_rate: float = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trained_dataset: Optional[datasets.ImageFolder] = None
        self.final_model: Optional[nn.Module] = model
        self.writer = SummaryWriter(self.log_dir / "tensorboard")
        self.metrics: Dict[str, List[float]] = {"epoch_loss": []}

    @log_action
    def make_directory(self, name: Path, out: Path) -> Path:
        top_view_folder = name / out
        top_view_folder.mkdir(parents=True, exist_ok=True)
        return top_view_folder

    @log_action
    def define_train(self) -> transforms.Compose:
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
    def load_data(self, train_transform: transforms.Compose) -> DataLoader:
        self.trained_dataset = datasets.ImageFolder(
            self.labeled_images_path, transform=train_transform
        )
        dataloader = DataLoader(
            self.trained_dataset, batch_size=self.batch_size, shuffle=True
        )
        return dataloader

    @log_action
    def define_model(self) -> nn.Module:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model: nn.Module = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 2)  # top vs other
        model = model.to(self.device)
        return model

    @log_action
    def compile_model(
        self, model: nn.Module
    ) -> Tuple[nn.Module, torch.optim.Optimizer]:
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        return criterion, optimizer

    @log_action
    @get_time
    def train_model(self) -> None:
        if self.final_model is None and self.trained_dataset is not None:
            logger.info("===== TRAINING STARTED =====")
            logger.info(f"Dataset: {self.labeled_images_path}")
            logger.info(f"Epochs: {self.num_epochs}, Batch size: {self.batch_size}")
            train_transform = self.define_train()
            dataloader = self.load_data(train_transform)
            self.final_model = self.define_model()
            criterion, optimizer = self.compile_model(self.final_model)

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
        if self.final_model is None or self.trained_dataset is None:
            logger.warning("Model not trained. Call train_model() first.")
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
