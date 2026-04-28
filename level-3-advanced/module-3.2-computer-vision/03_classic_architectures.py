"""
03 — Classic image architectures

A whirlwind tour. Each one solved a problem the previous couldn't.

  LeNet-5 (1998)        — 5 layers. Read MNIST digits. Started it all.
  AlexNet (2012)        — 8 layers + GPU + ReLU + dropout. Won ImageNet,
                          launched the deep learning era.
  VGG (2014)            — 16-19 deep, all 3×3 convs. Simple, strong; heavy.
  GoogLeNet/Inception   — Multi-branch "inception" modules. Cheaper.
  ResNet (2015)         — RESIDUAL CONNECTIONS let you train 100s of layers.
                          The biggest single idea in image NNs.
  DenseNet              — Each layer sees ALL previous outputs concatenated.
  EfficientNet (2019)   — Carefully scaled depth/width/resolution.
                          Strong accuracy/cost ratio.
  Vision Transformer    — Apply transformers to image patches. Beats CNNs at
                          scale, often weaker on tiny datasets.
  ConvNeXt (2022)       — "What if we modernize a ResNet with ViT tricks?"
                          A strong CNN baseline that holds its own vs ViT.

Run: python 03_classic_architectures.py
"""

import torch
import torch.nn as nn
from torchvision import models


# ───────────────────────────────────────────────────────────
# The residual connection — ResNet's key idea, in 10 lines
# ───────────────────────────────────────────────────────────
# Without residuals, very deep nets train poorly: gradients vanish.
# A residual block computes:   y = F(x) + x
# so the gradient flows directly through the "+ x" path. Suddenly you can
# train 50, 101, 152, even 1000 layers.
class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)        # the SHORTCUT
        return out


# ───────────────────────────────────────────────────────────
# Use a pretrained ResNet from torchvision
# ───────────────────────────────────────────────────────────
# `weights="DEFAULT"` downloads the most recent ImageNet-trained weights
# (~45 MB on first run, cached after that).
resnet18 = models.resnet18(weights="DEFAULT")
resnet18.eval()
print("ResNet-18 head + final layer:")
print(resnet18.fc)
print(f"  params: {sum(p.numel() for p in resnet18.parameters()):,}")

x = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    logits = resnet18(x)
print(f"  output (1000 ImageNet classes): shape {tuple(logits.shape)}")


# ───────────────────────────────────────────────────────────
# Other torchvision models (one line each)
# ───────────────────────────────────────────────────────────
# All follow the same .eval()/forward()/.fc-or-.classifier pattern.
TORCHVISION_BUFFET = {
    "resnet50":      models.resnet50,
    "efficientnet_b0": models.efficientnet_b0,
    "vit_b_16":      models.vit_b_16,
    "convnext_tiny": models.convnext_tiny,
    "mobilenet_v3_small": models.mobilenet_v3_small,
}
print("\navailable in torchvision (sample):")
for name in TORCHVISION_BUFFET:
    print(f"  {name}")


# ───────────────────────────────────────────────────────────
# `timm` — bigger catalog (~1000 models)
# ───────────────────────────────────────────────────────────
# Hugging Face's image model zoo. SOTA architectures including ConvNeXt-V2,
# EVA, BEiT, MaxViT, plus all the classics.
import timm
print(f"\ntimm has {len(timm.list_models())} architecture variants total")
print("a few examples:", timm.list_models("convnext*")[:3])

# Loading is one line. Inference identical to torchvision.
tiny = timm.create_model("convnext_tiny", pretrained=False, num_classes=10)
print(f"\nconvnext_tiny adapted for 10 classes — params: "
      f"{sum(p.numel() for p in tiny.parameters()):,}")


# ───────────────────────────────────────────────────────────
# Picking an architecture
# ───────────────────────────────────────────────────────────
# DEFAULT for new image tasks (small/medium data):
#   timm.create_model("convnext_tiny", pretrained=True) → fine-tune.
#
# If you need TINY (mobile, edge):
#   mobilenet_v3_small, efficientnet_lite, mobilevit.
#
# If you have LOTS of data and compute:
#   ViT-B/L, ConvNeXt-V2-base/large.
#
# Don't agonize. The architecture rarely beats:
#   1. better data
#   2. more data
#   3. proper augmentation
#   4. a sane training schedule.
