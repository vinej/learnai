"""
05 — Image augmentation

Augmentation creates new training samples by perturbing your originals
(flip, crop, rotate, color shift, ...). It's THE cheapest way to fight
overfitting in vision models.

Two libraries you'll see:

  torchvision.transforms.v2  — official, integrates with DataLoader.
  albumentations             — third-party, faster + bigger catalog.

This file demos both and visualizes what each transform does.

Run: python 05_augmentation.py
"""

from pathlib import Path

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2 as T


rng = np.random.default_rng(0)


# ───────────────────────────────────────────────────────────
# Build a tiny "image" we can visualize the augs on
# ───────────────────────────────────────────────────────────
size = 96
img = np.zeros((size, size, 3), dtype=np.uint8)
img[20:70, 30:80, 0] = 200            # red rectangle
img[40:60, 50:90, 1] = 200            # green stripe
pil_img = Image.fromarray(img)


# ───────────────────────────────────────────────────────────
# torchvision v2 transforms
# ───────────────────────────────────────────────────────────
# The standard pipeline you'd use in training:
torchvision_train = T.Compose([
    T.RandomResizedCrop(size=64, antialias=True),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    T.RandomRotation(degrees=15),
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# At eval time, NEVER augment — only resize / normalize.
torchvision_eval = T.Compose([
    T.Resize(64, antialias=True),
    T.CenterCrop(64),
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

aug_t = torchvision_train(pil_img)
print(f"torchvision augmented tensor shape: {tuple(aug_t.shape)}, "
      f"dtype: {aug_t.dtype}, mean ≈ {aug_t.mean():.3f}")


# ───────────────────────────────────────────────────────────
# albumentations
# ───────────────────────────────────────────────────────────
# albumentations operates on NumPy arrays (H, W, C). Often faster than
# torchvision and supports many vision tasks (boxes, masks).
albu_train = A.Compose([
    A.RandomResizedCrop(size=(64, 64), p=1.0),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.8),
    A.Rotate(limit=15, p=0.5),
    A.GaussNoise(p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
albu_out = albu_train(image=img)["image"]
print(f"albumentations output shape: {albu_out.shape}, dtype: {albu_out.dtype}")


# ───────────────────────────────────────────────────────────
# Visualize 8 random augmentations of the same image
# ───────────────────────────────────────────────────────────
viz_pipeline = T.Compose([
    T.RandomResizedCrop(size=64, antialias=True),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    T.RandomRotation(degrees=15),
])

fig, axes = plt.subplots(2, 4, figsize=(11, 6))
for i, ax in enumerate(axes.flat):
    if i == 0:
        ax.imshow(np.array(pil_img))
        ax.set_title("original")
    else:
        out = np.array(viz_pipeline(pil_img))
        ax.imshow(out)
        ax.set_title(f"aug {i}")
    ax.axis("off")

fig.tight_layout()
out = Path(__file__).parent / "figures"
out.mkdir(exist_ok=True)
fig.savefig(out / "05_augmentations.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"\nfigure saved to: {out / '05_augmentations.png'}")


# ───────────────────────────────────────────────────────────
# Mixup and CutMix — augmentations across IMAGES, not within
# ───────────────────────────────────────────────────────────
# MIXUP:    new_img = λ * a + (1-λ) * b;     new_label = λ * y_a + (1-λ) * y_b
# CUTMIX:   paste a random rectangle from b into a; mix labels by area ratio.
# These are batch-level augmentations and consistently improve top-tier results.
# torchvision v2 has them built in:  T.MixUp(num_classes=N)  and  T.CutMix(...).


# ───────────────────────────────────────────────────────────
# Augmentation rules of thumb
# ───────────────────────────────────────────────────────────
# • Use augmentations that respect what the model should be invariant to.
#   - Natural photos: flips, crops, color jitter, rotation, RandAugment.
#   - Medical / satellite: avoid horizontal flips if "left/right" is meaningful.
#   - OCR / numerals: NO flips/rotation — that breaks the labels.
# • Stronger augmentation helps WITH MORE TRAINING. With 5 epochs + heavy
#   augs, you can underfit. With 100 epochs, augmentation usually wins.
# • Test-time augmentation (TTA): apply N augmentations at inference, average
#   their predictions. Expensive but a free accuracy bump.
