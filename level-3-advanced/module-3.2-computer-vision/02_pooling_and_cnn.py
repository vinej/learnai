"""
02 — Pooling, receptive fields, and a complete small CNN

POOLING reduces spatial size (downsamples).
  • MaxPool       — keep the max in each window. Most common; emphasizes
                    "did this feature occur anywhere in the patch?"
  • AvgPool       — average. Smoother.
  • GlobalAvgPool — average over the entire feature map → one number per
                    channel. Modern classifiers use this just before the
                    final Linear layer.

RECEPTIVE FIELD = the region of the input image that influences a given
output unit. Stacking conv + pool layers EXPANDS it. A 3×3 conv has receptive
field 3×3; two stacked 3×3 convs see 5×5; throw in a stride-2 pool and you
double the effective receptive field.

This file builds and explains a complete small CNN you can train on
CIFAR-10 (exercise 1).

Run: python 02_pooling_and_cnn.py
"""

import torch
import torch.nn as nn


# ───────────────────────────────────────────────────────────
# Pooling demo
# ───────────────────────────────────────────────────────────
x = torch.arange(1, 17).reshape(1, 1, 4, 4).float()
print("input (1×1×4×4):")
print(x.squeeze().tolist())

print("\nMaxPool2d k=2 s=2:")
print(nn.MaxPool2d(2)(x).squeeze().tolist())

print("\nAvgPool2d k=2 s=2:")
print(nn.AvgPool2d(2)(x).squeeze().tolist())

# Global average pool — collapse spatial dims to 1×1
print("\nAdaptiveAvgPool2d(1) on a (1, 3, 5, 5):")
y = torch.randn(1, 3, 5, 5)
print(nn.AdaptiveAvgPool2d(1)(y).shape)            # (1, 3, 1, 1)


# ───────────────────────────────────────────────────────────
# Receptive field intuition
# ───────────────────────────────────────────────────────────
# A unit deep in the network sees a wider patch of the input.
# Recipe (rough): RF_out = RF_in + (K - 1) * stride_so_far
# Two 3×3 convs (no stride) → RF 5×5.
# Add a pool stride 2 → RF 10×10 effectively (in input pixels).
# Three blocks of conv-conv-pool → RF ~28×28, plenty for CIFAR-style 32×32 images.


# ───────────────────────────────────────────────────────────
# A small CNN you can actually train (exercise 1)
# ───────────────────────────────────────────────────────────
class SmallCNN(nn.Module):
    """A modern-style small CNN for CIFAR-10 (32×32 RGB).

    Pattern: [Conv → BN → ReLU → Conv → BN → ReLU → Pool] × 3 → GlobalAvgPool → FC
    """

    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            self._block(3,   32),     # 32 × 32 → 16 × 16
            self._block(32,  64),     # 16 × 16 →  8 ×  8
            self._block(64, 128),     #  8 ×  8 →  4 ×  4
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # → (B, 128, 1, 1)
            nn.Flatten(),             # → (B, 128)
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    @staticmethod
    def _block(in_c: int, out_c: int) -> nn.Sequential:
        """Two convs + BN + ReLU, followed by a 2×2 max pool."""
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.head(self.features(x))


model = SmallCNN()
print(f"\nSmallCNN params: {sum(p.numel() for p in model.parameters()):,}")
print(f"output shape on a (1, 3, 32, 32) input: {tuple(model(torch.randn(1, 3, 32, 32)).shape)}")


# Walk through the spatial sizes
def trace_shapes(m: nn.Module, x: torch.Tensor):
    print(f"\nshape trace through SmallCNN:")
    print(f"  input          : {tuple(x.shape)}")
    for i, block in enumerate(m.features, start=1):
        x = block(x)
        print(f"  after block {i}  : {tuple(x.shape)}")
    x = m.head[0](x)
    print(f"  after global AP: {tuple(x.shape)}")
    x = m.head[1](x)
    print(f"  after flatten  : {tuple(x.shape)}")


trace_shapes(model, torch.randn(1, 3, 32, 32))


# ───────────────────────────────────────────────────────────
# Modern design rules of thumb
# ───────────────────────────────────────────────────────────
# • Two stacked 3×3 convs have the same receptive field as one 5×5 conv,
#   but fewer params and more non-linearity. Stack 3×3s.
# • Use BatchNorm right after Conv (with bias=False on the conv).
# • Use ReLU (or GELU/SiLU) AFTER batch norm.
# • Downsample with stride-2 conv or 2×2 max pool — both work; mix as needed.
# • End with adaptive avg pool to make the network input-size agnostic.
# • Final classifier = single Linear layer after global average pooling.
