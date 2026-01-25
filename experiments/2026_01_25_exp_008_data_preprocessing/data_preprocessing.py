import cv2
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

class Preprocessing:
  IMAGE_SIZE = (224, 224)

  @staticmethod
  def show_image(img):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img, cmap="grey")
    ax.axis("off")

  @staticmethod
  def denoise_img(img, **kwargs):
    # if img.dtype != np.uint8:
    #       img_min, img_max = img.min(), img.max()
    #       if img_max <= 1.0:
    #           img = (img * 255).astype(np.uint8)
    #       else:
    #           img = img.astype(np.uint8)

    # return cv2.fastNlMeansDenoising(img, h=10)

    return cv2.medianBlur(img, ksize=3)

  @staticmethod
  def zscore_norm(img, **kwargs):
    return (img - img.mean()) / (img.std() + 1e-8)

  @staticmethod
  def apply_augmentation(img):
    augment = A.Compose([
      A.Resize(height=Preprocessing.IMAGE_SIZE[0], width=Preprocessing.IMAGE_SIZE[1], interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST, area_for_downscale="image", p=1.0),

      A.Lambda(image=Preprocessing.denoise_img, p=1.0),

      A.Rotate(limit=10, border_mode=cv2.BORDER_REFLECT, p=0.5),

      A.HorizontalFlip(p=0.5),

      A.Lambda(image=Preprocessing.zscore_norm, p=1.0),

      ToTensorV2()
    ])

    return augment(image=img)["image"]
  