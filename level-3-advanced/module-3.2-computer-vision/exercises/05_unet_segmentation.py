"""
Exercise 5 — Train a tiny U-Net for binary segmentation

Generate a synthetic dataset of grayscale images, each containing one
random circle of varying intensity. Train a small U-Net to segment the
circle from background.

The script asserts the validation Dice coefficient ≥ 0.85.

Run: python exercises/05_unet_segmentation.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


torch.manual_seed(0)
np.random.seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"


# ───────────────────────────────────────────────────────────
# Synthetic circle dataset
# ───────────────────────────────────────────────────────────
class CircleDataset(Dataset):
    def __init__(self, n: int, size: int = 64, seed: int = 0):
        self.n, self.size = n, size
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # Re-seed per index for determinism across epochs
        rng = np.random.default_rng(idx)
        size = self.size
        img = rng.normal(0.5, 0.1, (size, size)).clip(0, 1).astype(np.float32)
        yy, xx = np.mgrid[:size, :size]
        cy, cx = rng.integers(15, size - 15, size=2)
        r = rng.integers(8, 18)
        mask = ((yy - cy) ** 2 + (xx - cx) ** 2) <= r ** 2
        img[mask] += rng.normal(0.4, 0.05)         # circle is brighter
        img = img.clip(0, 1)
        return (
            torch.from_numpy(img)[None],            # (1, H, W)
            torch.from_numpy(mask.astype(np.float32))[None],
        )


train_set = CircleDataset(n=400, seed=0)
val_set   = CircleDataset(n=100, seed=99)
train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=32, shuffle=False)


# ───────────────────────────────────────────────────────────
# Tiny U-Net (same architecture as file 07)
# ───────────────────────────────────────────────────────────
def conv_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = conv_block(1,  16)
        self.d2 = conv_block(16, 32)
        self.d3 = conv_block(32, 64)
        self.pool = nn.MaxPool2d(2)
        self.up2  = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = conv_block(64, 32)
        self.up1  = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = conv_block(32, 16)
        self.out  = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(self.pool(d1))
        d3 = self.d3(self.pool(d2))
        u2 = self.up2(d3)
        u2 = self.dec2(torch.cat([u2, d2], dim=1))
        u1 = self.up1(u2)
        u1 = self.dec1(torch.cat([u1, d1], dim=1))
        return self.out(u1)


model = TinyUNet().to(device)
print(f"params: {sum(p.numel() for p in model.parameters()):,}")


# ───────────────────────────────────────────────────────────
# Training
# ───────────────────────────────────────────────────────────
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.BCEWithLogitsLoss()


def dice_score(probs: torch.Tensor, target: torch.Tensor, eps=1e-6) -> float:
    """Dice coefficient: 2|A ∩ B| / (|A| + |B|)."""
    pred = (probs > 0.5).float()
    inter = (pred * target).sum()
    return ((2 * inter + eps) / (pred.sum() + target.sum() + eps)).item()


def evaluate():
    model.eval()
    total_dice, n = 0.0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            probs = torch.sigmoid(model(x))
            total_dice += dice_score(probs, y) * y.size(0)
            n += y.size(0)
    model.train()
    return total_dice / n


EPOCHS = 8
for epoch in range(EPOCHS):
    model.train()
    running = 0.0
    n = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        running += loss.item() * y.size(0)
        n += y.size(0)
    val_dice = evaluate()
    print(f"epoch {epoch + 1}/{EPOCHS}  loss={running / n:.4f}  val_dice={val_dice:.4f}")


# ───────────────────────────────────────────────────────────
# Visualize a few predictions
# ───────────────────────────────────────────────────────────
model.eval()
fig, axes = plt.subplots(3, 3, figsize=(9, 9))
for i in range(3):
    img, mask = val_set[i]
    with torch.no_grad():
        pred = torch.sigmoid(model(img[None].to(device))).cpu().squeeze().numpy()
    axes[i, 0].imshow(img.squeeze(), cmap="gray"); axes[i, 0].set_title("input"); axes[i, 0].axis("off")
    axes[i, 1].imshow(mask.squeeze(), cmap="gray"); axes[i, 1].set_title("ground truth"); axes[i, 1].axis("off")
    axes[i, 2].imshow(pred, cmap="gray"); axes[i, 2].set_title("prediction"); axes[i, 2].axis("off")

fig.tight_layout()
out = Path(__file__).parent.parent / "figures"
out.mkdir(exist_ok=True)
fig.savefig(out / "exercise_05_unet.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"\nfigure saved to: {out / 'exercise_05_unet.png'}")

final_dice = evaluate()
print(f"\nfinal val dice: {final_dice:.4f}")
assert final_dice >= 0.85, f"target dice ≥ 0.85, got {final_dice:.4f}"
print("U-Net segments circles ✅")


# ── Stretch ─────────────────────────────────────────────────────────────────
# • Add Dice loss (1 - dice) as a second loss term — combine BCE + Dice.
# • Make the synthetic dataset harder (multiple shapes, occlusions, noise).
# • Replace TinyUNet with a pretrained backbone-encoder-style model from
#   `segmentation_models_pytorch` (smp.Unet("resnet34", encoder_weights="imagenet")).
# ─────────────────────────────────────────────────────────────────────────────
