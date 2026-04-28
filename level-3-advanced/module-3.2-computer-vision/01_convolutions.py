"""
01 — Convolutions

A 2-D convolution slides a small KERNEL (also called filter) over the image,
computing a weighted sum at each position. The output is a FEATURE MAP that
highlights wherever the kernel's pattern matches the input.

Key parameters:

  KERNEL SIZE — typically 3×3 or 5×5. Determines what local pattern it sees.
  STRIDE      — step size when sliding. Stride 2 halves the spatial size.
  PADDING     — pixels added around the input so output isn't smaller than
                input ('same' = pad enough to keep size).
  DILATION    — gaps between kernel taps (atrous conv). Used for larger
                receptive fields without downsampling.

In PyTorch: nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

Run: python 01_convolutions.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


rng = np.random.default_rng(0)


# ───────────────────────────────────────────────────────────
# Build a fake image: a square inside noise
# ───────────────────────────────────────────────────────────
img = rng.normal(0, 0.2, (64, 64))
img[20:44, 20:44] += 1.5                       # bright square
img = torch.from_numpy(img).float()[None, None]  # → (1, 1, 64, 64)


# ───────────────────────────────────────────────────────────
# Hand-crafted kernels
# ───────────────────────────────────────────────────────────
# Sobel-X — detects vertical edges (sharp horizontal change in intensity)
sobel_x = torch.tensor([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=torch.float32)[None, None]

# Sobel-Y — detects horizontal edges
sobel_y = torch.tensor([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]], dtype=torch.float32)[None, None]

# 3×3 box blur
box = torch.ones((3, 3), dtype=torch.float32)[None, None] / 9


# ───────────────────────────────────────────────────────────
# Apply each kernel
# ───────────────────────────────────────────────────────────
edge_x = F.conv2d(img, sobel_x, padding=1)
edge_y = F.conv2d(img, sobel_y, padding=1)
blurred = F.conv2d(img, box, padding=1)


# ───────────────────────────────────────────────────────────
# Stride & padding affect output size
# ───────────────────────────────────────────────────────────
# Output spatial size:  ((H + 2P - K) / S) + 1
# With H=64, K=3:
print("padding=0 stride=1:", F.conv2d(img, sobel_x, padding=0, stride=1).shape)  # 62
print("padding=1 stride=1:", F.conv2d(img, sobel_x, padding=1, stride=1).shape)  # 64 (same)
print("padding=1 stride=2:", F.conv2d(img, sobel_x, padding=1, stride=2).shape)  # 32

# Dilation = 2 with K=3 makes the effective kernel 5×5
print("dilation=2:        ", F.conv2d(img, sobel_x, padding=2, dilation=2).shape)


# ───────────────────────────────────────────────────────────
# Plot
# ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(13, 3.5))
for ax, t, title in zip(
    axes,
    [img, edge_x, edge_y, blurred],
    ["original", "Sobel-X (vertical edges)", "Sobel-Y (horizontal edges)", "box blur"],
):
    ax.imshow(t.squeeze().numpy(), cmap="gray")
    ax.set_title(title)
    ax.axis("off")

fig.tight_layout()
out = Path(__file__).parent / "figures"
out.mkdir(exist_ok=True)
fig.savefig(out / "01_convolutions.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"\nfigure saved to: {out / '01_convolutions.png'}")


# ───────────────────────────────────────────────────────────
# Why CNNs work
# ───────────────────────────────────────────────────────────
# • LEARNED kernels: in a CNN, the values inside each kernel are PARAMETERS.
#   Backprop updates them so the network learns useful detectors automatically.
# • PARAMETER SHARING: the same kernel slides everywhere. A "vertical edge"
#   detector is one tiny set of weights, applied across the whole image.
# • TRANSLATION EQUIVARIANCE: a feature in any location triggers the same
#   response. The model doesn't have to learn "left ear" and "right ear" as
#   different patterns.
# These three are why CNNs need ORDERS OF MAGNITUDE fewer parameters than a
# fully-connected net to do the same image task.


# ───────────────────────────────────────────────────────────
# nn.Conv2d basics
# ───────────────────────────────────────────────────────────
# A typical convolutional layer:
#
#   layer = nn.Conv2d(
#       in_channels=3,     # RGB → 3
#       out_channels=64,   # learn 64 different detectors
#       kernel_size=3,
#       stride=1,
#       padding=1,         # 'same' padding for K=3
#       bias=False,        # often False when followed by BatchNorm
#   )
#
# Output: (N, 64, H, W). 64 feature maps, one per learned detector.
