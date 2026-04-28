"""
04 — Transfer learning

Training a vision model from scratch usually loses to fine-tuning a pretrained
one — especially when you don't have millions of labeled images. The
pretrained model already knows edges, textures, parts. You only have to
adapt the final layers to your task.

Two flavors:

  FEATURE EXTRACTION
    Freeze the backbone. Train ONLY a new classifier head on top.
    Fast, small risk of overfitting, suitable for tiny datasets.

  FINE-TUNING
    Train EVERYTHING, often with smaller learning rates for early layers
    (which already know low-level features) and bigger ones for the head.
    Stronger results when you have enough data.

This file shows both, on a fake "3-class flowers" dataset.

Run: python 04_transfer_learning.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models


torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"


# ───────────────────────────────────────────────────────────
# Pretrained ResNet18
# ───────────────────────────────────────────────────────────
backbone = models.resnet18(weights="DEFAULT")
print("ResNet18 final layer (originally 1000 ImageNet classes):")
print(backbone.fc)


# ───────────────────────────────────────────────────────────
# Replace the head — "transfer" = swap the classifier layer
# ───────────────────────────────────────────────────────────
n_classes = 3
backbone.fc = nn.Linear(backbone.fc.in_features, n_classes)
print(f"\nadapted head for {n_classes} classes:")
print(backbone.fc)


# ───────────────────────────────────────────────────────────
# Mode 1 — feature extraction (freeze backbone)
# ───────────────────────────────────────────────────────────
def freeze_backbone(model: nn.Module) -> None:
    for name, p in model.named_parameters():
        if not name.startswith("fc."):
            p.requires_grad = False


frozen = models.resnet18(weights="DEFAULT")
frozen.fc = nn.Linear(frozen.fc.in_features, n_classes)
freeze_backbone(frozen)

trainable = sum(p.numel() for p in frozen.parameters() if p.requires_grad)
total     = sum(p.numel() for p in frozen.parameters())
print(f"\n[feature extraction]")
print(f"  trainable params: {trainable:,} / {total:,} "
      f"(only {trainable / total:.1%})")


# ───────────────────────────────────────────────────────────
# Mode 2 — full fine-tune with DIFFERENTIAL learning rates
# ───────────────────────────────────────────────────────────
# The backbone is already smart. Tune it gently. Train the new head harder.
finetune = models.resnet18(weights="DEFAULT")
finetune.fc = nn.Linear(finetune.fc.in_features, n_classes)

# Two parameter groups, different lrs:
backbone_params = [p for n, p in finetune.named_parameters() if not n.startswith("fc.")]
head_params     = [p for n, p in finetune.named_parameters() if n.startswith("fc.")]

optimizer = torch.optim.AdamW([
    {"params": backbone_params, "lr": 1e-4},     # gentle
    {"params": head_params,     "lr": 1e-3},     # aggressive
], weight_decay=1e-2)

print(f"\n[fine-tune with differential lr]")
print(f"  backbone lr: 1e-4")
print(f"  head lr:     1e-3")


# ───────────────────────────────────────────────────────────
# Tiny synthetic dataset to actually train both
# ───────────────────────────────────────────────────────────
# Don't read this as a real benchmark — it's so small the model just
# memorizes. Real exercise comes in exercise 02.
def fake_data(n_per_class=20):
    Xs, ys = [], []
    for cls in range(n_classes):
        # A 64×64×3 image whose mean intensity encodes the class.
        x = torch.randn(n_per_class, 3, 64, 64) + (cls - 1) * 0.5
        Xs.append(x)
        ys.extend([cls] * n_per_class)
    return torch.cat(Xs), torch.tensor(ys)


X, y = fake_data()
loader = DataLoader(TensorDataset(X, y), batch_size=8, shuffle=True)
criterion = nn.CrossEntropyLoss()


def train_a_few_steps(model, opt, name, steps=20):
    model.train()
    losses = []
    for i, (bx, by) in enumerate(loader):
        if i >= steps:
            break
        opt.zero_grad()
        loss = criterion(model(bx), by)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    print(f"  {name:<22}  loss: {losses[0]:.3f} → {losses[-1]:.3f}")


print("\nrunning a few steps each ...")
train_a_few_steps(
    frozen,
    torch.optim.AdamW(filter(lambda p: p.requires_grad, frozen.parameters()), lr=1e-3),
    "feature extraction",
)
train_a_few_steps(finetune, optimizer, "fine-tuning")


# ───────────────────────────────────────────────────────────
# Decision tree
# ───────────────────────────────────────────────────────────
# • < 1000 labeled images, classes similar to ImageNet's:
#       feature extraction. You won't have enough signal to safely tune the
#       backbone.
# • 1000-100k images, similar domain:
#       fine-tune the WHOLE thing with differential LR (backbone slow, head fast).
# • Lots of data, very different domain (medical, satellite):
#       full fine-tune; consider freezing only the very first conv block, or
#       using `timm` to load a backbone pretrained on a closer domain.
# • Almost no data:
#       use the model as a FEATURE EXTRACTOR (run images through the backbone,
#       then train a sklearn classifier on the embeddings). It's surprisingly
#       hard to beat.
