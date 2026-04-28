"""
07 — Image segmentation

Segmentation = predict a class for EVERY PIXEL.

  SEMANTIC      — assign each pixel a class. "All road pixels say road."
                  Doesn't distinguish two cars from one big car shape.
  INSTANCE      — also separates individual instances of the same class.
                  Mask R-CNN, YOLOv8-seg.
  PANOPTIC      — semantic + instance. The full picture.

The classic architecture is U-Net: encoder downsamples, decoder upsamples,
with SKIP CONNECTIONS that let the decoder use high-resolution features
from the encoder. This file builds a tiny U-Net and demonstrates a forward
pass on a synthetic image.

Run: python 07_segmentation.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


torch.manual_seed(0)


# ───────────────────────────────────────────────────────────
# A tiny U-Net
# ───────────────────────────────────────────────────────────
def conv_block(in_c: int, out_c: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class TinyUNet(nn.Module):
    """4-level U-Net: 1 → 16 → 32 → 64 → 32 → 16 → 1 channels."""

    def __init__(self, in_c: int = 1, out_c: int = 1):
        super().__init__()
        self.down1 = conv_block(in_c, 16)
        self.down2 = conv_block(16,   32)
        self.down3 = conv_block(32,   64)

        self.pool = nn.MaxPool2d(2)

        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = conv_block(64, 32)        # 64 = 32 + skip(32)
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = conv_block(32, 16)        # 32 = 16 + skip(16)

        self.out = nn.Conv2d(16, out_c, 1)

    def forward(self, x):
        d1 = self.down1(x)                     # full res
        d2 = self.down2(self.pool(d1))         # /2
        d3 = self.down3(self.pool(d2))         # /4

        u2 = self.up2(d3)
        u2 = self.dec2(torch.cat([u2, d2], dim=1))   # SKIP from d2

        u1 = self.up1(u2)
        u1 = self.dec1(torch.cat([u1, d1], dim=1))   # SKIP from d1

        return self.out(u1)


model = TinyUNet()
print(f"TinyUNet params: {sum(p.numel() for p in model.parameters()):,}")
x = torch.randn(1, 1, 64, 64)
print(f"input: {tuple(x.shape)}, output: {tuple(model(x).shape)}")


# ───────────────────────────────────────────────────────────
# Demonstration: build a TINY synthetic dataset of "circles"
# ───────────────────────────────────────────────────────────
def synth_image(size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Return (image, mask). Mask is 1 inside a random circle, 0 outside."""
    rng = np.random.default_rng()
    img = rng.normal(0.5, 0.1, (size, size)).clip(0, 1)
    yy, xx = np.mgrid[:size, :size]
    cy, cx = rng.integers(15, size - 15, size=2)
    r = rng.integers(8, 18)
    mask = ((yy - cy) ** 2 + (xx - cx) ** 2) <= r ** 2
    img[mask] += rng.normal(0.4, 0.05)         # circle is brighter
    return img.astype(np.float32), mask.astype(np.float32)


img, mask = synth_image()


# ───────────────────────────────────────────────────────────
# Quick training: one epoch on 200 images
# ───────────────────────────────────────────────────────────
imgs = torch.stack([torch.from_numpy(synth_image()[0])[None] for _ in range(200)])
masks = torch.stack([torch.from_numpy(synth_image()[1])[None] for _ in range(200)])
# (Note: imgs & masks above are independently sampled — for a real run we'd
# pair them. Below we resample fresh aligned pairs at training time.)

opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = nn.BCEWithLogitsLoss()

print("\nbrief training run on synthetic circles ...")
for step in range(50):
    pairs = [synth_image() for _ in range(8)]
    bx = torch.stack([torch.from_numpy(i)[None] for i, _ in pairs])
    by = torch.stack([torch.from_numpy(m)[None] for _, m in pairs])
    opt.zero_grad()
    logits = model(bx)
    loss = loss_fn(logits, by)
    loss.backward()
    opt.step()
    if step % 10 == 0:
        print(f"  step {step:>2}: loss={loss.item():.4f}")


# ───────────────────────────────────────────────────────────
# Inference + visualization
# ───────────────────────────────────────────────────────────
model.eval()
test_img, test_mask = synth_image()
with torch.no_grad():
    pred = torch.sigmoid(model(torch.from_numpy(test_img)[None, None])).squeeze().numpy()

fig, axes = plt.subplots(1, 3, figsize=(10, 4))
axes[0].imshow(test_img, cmap="gray"); axes[0].set_title("input"); axes[0].axis("off")
axes[1].imshow(test_mask, cmap="gray"); axes[1].set_title("ground truth"); axes[1].axis("off")
axes[2].imshow(pred, cmap="gray"); axes[2].set_title("U-Net prediction"); axes[2].axis("off")

fig.tight_layout()
out = Path(__file__).parent / "figures"
out.mkdir(exist_ok=True)
fig.savefig(out / "07_unet_segmentation.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"\nfigure saved to: {out / '07_unet_segmentation.png'}")


# ───────────────────────────────────────────────────────────
# Loss functions for segmentation
# ───────────────────────────────────────────────────────────
# • BCEWithLogitsLoss — pixel-wise binary cross-entropy. Easy default.
# • Dice loss          — 1 − 2|A∩B| / (|A| + |B|). Better for class imbalance
#   (when foreground is tiny, e.g. a tumor on a CT scan).
# • Combo                — sum BCE + Dice — often the strongest baseline.
# • Focal loss         — down-weights easy pixels. Useful for hard imbalance.


# ───────────────────────────────────────────────────────────
# Pretrained segmentation models you'd use in production
# ───────────────────────────────────────────────────────────
# • torchvision.models.segmentation — DeepLabV3, FCN.
# • SAM (Segment Anything, Meta) — prompt-able universal segmenter.
# • Mask R-CNN (torchvision.models.detection.maskrcnn_*) — instance seg.
# • YOLOv8-seg — fast instance segmentation via ultralytics.
# • U-Net + medical transfer learning — still strong in medical imaging.
