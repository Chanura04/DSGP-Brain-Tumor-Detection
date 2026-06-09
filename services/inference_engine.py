import io

import albumentations as A
import cv2
import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from albumentations.pytorch import ToTensorV2
from keras import backend
from torchvision import transforms

from src.utils.image_utils import HEAD_DETECTION_IMG_SIZE, CT_MRI_TUMOR_IMG_SIZE, SEGMENTATION_TUMOR_IMG_SIZE


def predict_image(model, class_to_idx, device, img):
    preproc = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    x = preproc(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)

    top_idx = class_to_idx.get("top", 0)
    top_prob = probs[0][top_idx].item()
    return top_prob


def detect_head(model, file_bytes, class_to_idx, device):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(HEAD_DETECTION_IMG_SIZE)

    prediction = predict_image(model, class_to_idx, device, img)

    if prediction > 0.99:
        value = 1
    else:
        value = 0

    return value


def detect_tumor(model, file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("L")
    img = img.resize(CT_MRI_TUMOR_IMG_SIZE)

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array).item()

    if prediction > 0.5:
        label = "Tumor Detected"
    else:
        label = "No Tumor Detected"

    return label


transform = A.Compose([ToTensorV2()], additional_targets={'image0': 'image'})


def zscore_norm_tensor(x):
    x = x.float() / 255.0
    mean = x.mean(dim=[1, 2], keepdim=True)
    std = x.std(dim=[1, 2], keepdim=True, unbiased=False)
    return (x - mean) / (std + 1e-8)


def classify_tumor(model, file_bytes, classes, device):
    raw_image = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)

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

    model.eval()
    with torch.inference_mode():
        raw_tensor, proc_tensor = raw_tensor.to(device, non_blocking=True), proc_tensor.to(device, non_blocking=True)
        pred = model(raw_tensor, proc_tensor)
        pred_probs = torch.softmax(pred, dim=1)
        predicted_idx = torch.argmax(pred_probs, dim=1).item()

    return classes[predicted_idx], pred_probs[0, predicted_idx].item() * 100


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = backend.flatten(y_true)
    y_pred_f = backend.flatten(y_pred)
    intersection = backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (backend.sum(y_true_f) + backend.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    smooth = 1.0
    intersection = tf.reduce_sum(y_true * y_pred)
    return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)


def segment_tumor(model, image_file_bytes):
    img = Image.open(io.BytesIO(image_file_bytes)).convert("RGB")
    img = img.resize(SEGMENTATION_TUMOR_IMG_SIZE)

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction_mask = model.predict(img_array)[0]
    binary_mask = (prediction_mask > 0.5).astype(np.uint8)

    segmented_image = (binary_mask.squeeze() * 255).astype(np.uint8)
    return segmented_image


def mask_overlay(original_image_file_bytes, mask, alpha=0.5):
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
