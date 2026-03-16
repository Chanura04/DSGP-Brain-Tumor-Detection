from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import io
from PIL import Image
import torch
from torch import nn
import yaml
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tensorflow.keras import backend
from tensorflow.keras.layers import Conv2DTranspose
import cv2

from src.utils.image_utils import HEAD_DETECTION_IMG_SIZE, CT_MRI_TUMOR_IMG_SIZE, SEGMENTATION_TUMOR_IMG_SIZE

axial_view_detection_ct_model = load_model("models/axial_view_detection_ct_model.keras")
axial_view_detection_mri_model = load_model("models/axial_view_detection_mri_model.keras")
ct_tumor_detection_model = load_model("models/ct_tumor_detection_model.keras")


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


config = load_config("configs/config.yaml")
device = "cuda" if config["device"] == "cuda" and torch.cuda.is_available() else "cpu"
classes = ['glioma', 'meningioma', 'pituitary']
mri_tumor_classification_model = make_model(config, classes, device)
checkpoint = torch.load("models/mri_tumor_classification_model.pth", map_location=device, weights_only=False)
mri_tumor_classification_model.load_state_dict(checkpoint["model_state_dict"])
transform = A.Compose([ToTensorV2()], additional_targets={'image0': 'image'})


def zscore_norm_tensor(x):
    x = x.float() / 255.0
    mean = x.mean(dim=[1, 2], keepdim=True)
    std = x.std(dim=[1, 2], keepdim=True, unbiased=False)
    return (x - mean) / (std + 1e-8)


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = backend.flatten(y_true)
    y_pred_f = backend.flatten(y_pred)
    intersection = backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (backend.sum(y_true_f) + backend.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    smooth = 1.0
    intersection = tf.reduce_sum(y_true * y_pred)
    return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)


class Conv2DTransposeFixed(Conv2DTranspose):
    @classmethod
    def from_config(cls, config):
        config.pop("groups", None)
        return super().from_config(config)


tumor_segmentation_model = load_model("models/tumor_segmentation_model.h5",
                                      custom_objects={"dice_coef": dice_coef, "Conv2DTranspose": Conv2DTransposeFixed},
                                      compile=False)


def ct_head_detection(ct_file_bytes):
    img = Image.open(io.BytesIO(ct_file_bytes)).convert("RGB")
    img = img.resize(HEAD_DETECTION_IMG_SIZE)

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = axial_view_detection_ct_model.predict(img_array).item()

    if prediction > 0.85:
        confidence = prediction
    else:
        confidence = 1 - prediction

    if confidence > 0.85:
        value = 1
    else:
        value = 0

    return value, confidence * 100


def mri_head_detection(mri_file_bytes):
    img = Image.open(io.BytesIO(mri_file_bytes)).convert("RGB")
    img = img.resize(HEAD_DETECTION_IMG_SIZE)

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = axial_view_detection_mri_model.predict(img_array).item()

    if prediction > 0.85:
        confidence = prediction
    else:
        confidence = 1 - prediction

    if confidence > 0.85:
        value = 1
    else:
        value = 0

    return value, confidence * 100


def ct_tumor_detection(ct_file_bytes):
    img = Image.open(io.BytesIO(ct_file_bytes)).convert("L")
    img = img.resize(CT_MRI_TUMOR_IMG_SIZE)

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = ct_tumor_detection_model.predict(img_array).item()

    if prediction > 0.5:
        label = "Tumor Detected"
    else:
        label = "No Tumor Detected"

    return label, prediction * 100


def mri_tumor_classification(mri_file_bytes):
    raw_image = cv2.imdecode(np.frombuffer(mri_file_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    proc_image = clahe.apply(raw_image)
    proc_image = cv2.resize(proc_image, CT_MRI_TUMOR_IMG_SIZE, interpolation=cv2.INTER_LINEAR)

    raw_img = cv2.resize(raw_image, CT_MRI_TUMOR_IMG_SIZE)
    proc_img = cv2.resize(proc_image, CT_MRI_TUMOR_IMG_SIZE)

    raw_img = raw_img[..., None]
    proc_img = proc_img[..., None]

    augmented = transform(image=raw_img, image0=proc_img)
    raw_tensor = augmented['image']
    proc_tensor = augmented['image0']

    # Optional: z-score normalization
    raw_tensor = zscore_norm_tensor(raw_tensor)
    proc_tensor = zscore_norm_tensor(proc_tensor)

    raw_tensor = raw_tensor.unsqueeze(0)
    proc_tensor = proc_tensor.unsqueeze(0)

    mri_tumor_classification_model.eval()
    with torch.inference_mode():
        raw_tensor, proc_tensor = raw_tensor.to(device, non_blocking=True), proc_tensor.to(device, non_blocking=True)
        pred = mri_tumor_classification_model(raw_tensor, proc_tensor)
        pred_probs = torch.softmax(pred, dim=1)
        predicted_idx = torch.argmax(pred_probs, dim=1).item()

    return classes[predicted_idx], pred_probs[0, predicted_idx].item() * 100


def tumor_segmentation(image_file_bytes):
    img = Image.open(io.BytesIO(image_file_bytes)).convert("RGB")
    img = img.resize(SEGMENTATION_TUMOR_IMG_SIZE)

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction_mask = tumor_segmentation_model.predict(img_array)[0]
    binary_mask = (prediction_mask > 0.5).astype(np.uint8)

    segmented_image = (binary_mask.squeeze() * 255).astype(np.uint8)
    return segmented_image


def overlay_mask(original_image_file_bytes, mask, alpha=0.5):
    original = np.array(Image.open(io.BytesIO(original_image_file_bytes)).convert("RGB"))

    mask = np.squeeze(mask)
    mask = cv2.resize(mask.astype(np.uint8), (original.shape[1], original.shape[0]))

    mask = (mask > 0.5).astype(np.uint8)

    overlay = original.copy()
    overlay[mask == 1] = (alpha * np.array([255, 0, 0]) + (1 - alpha) * overlay[mask == 1]).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)

    border = cv2.dilate(mask, kernel, iterations=1) - mask
    overlay[border == 1] = [255, 0, 0]
    return overlay
