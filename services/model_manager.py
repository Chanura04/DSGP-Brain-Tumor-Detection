import torch
import yaml
from keras import backend
from keras.layers import Conv2DTranspose
from keras.models import load_model
from torch import nn
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_head_detection_model():
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)  # top vs other
    model = model.to(device)

    checkpoint = torch.load("models/axial_view_detection_model.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint["class_to_idx"], device


def load_ct_tumor_detection_model():
    return load_model("models/ct_tumor_detection_model.keras")


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


class CustomCNN(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape, dropout, cfg):
        super().__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # Compute flatten size dynamically
        with torch.no_grad():
            x = torch.zeros(1, input_shape, *tuple(cfg["data"]["image_size"]))  # batch_size=1, input_shape channels
            x = self.conv_block_1(x)
            x = self.conv_block_2(x)
            n_features = x.numel() // x.shape[0]  # total features per sample

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=n_features, out_features=output_shape)
        )

    def forward(self, raw_img, processed_img):
        x = torch.cat([raw_img, processed_img], dim=1)  # Concatenate raw + processed channels -> [B, 2, H, W]
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x


def make_model(cfg, classes, device):
    model = CustomCNN(input_shape=cfg["model"]["input_dim"], hidden_units=cfg["model"]["hidden_units"],
                      output_shape=len(classes), dropout=cfg["model"]["dropout"], cfg=cfg).to(device)
    return model


def load_mri_tumor_classification_model():
    config = load_config("configs/config.yaml")
    # device = "cuda" if config["device"] == "cuda" and torch.cuda.is_available() else "cpu"
    classes = ['glioma', 'meningioma', 'pituitary']
    mri_tumor_classification_model = make_model(config, classes, device)
    checkpoint = torch.load("models/mri_tumor_classification_model.pth", map_location=device, weights_only=False)
    mri_tumor_classification_model.load_state_dict(checkpoint["model_state_dict"])

    return mri_tumor_classification_model, classes


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = backend.flatten(y_true)
    y_pred_f = backend.flatten(y_pred)
    intersection = backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (backend.sum(y_true_f) + backend.sum(y_pred_f) + smooth)


class Conv2DTransposeFixed(Conv2DTranspose):
    @classmethod
    def from_config(cls, config):
        config.pop("groups", None)
        return super().from_config(config)


def load_tumor_segmentation_model():
    return load_model("models/tumor_segmentation_model.h5",
                      custom_objects={"dice_coef": dice_coef, "Conv2DTranspose": Conv2DTransposeFixed},
                      compile=False)
