"""
Exercise 2 — Fine-tune a pretrained ResNet on a small custom-style dataset

We pretend we have a small custom photo dataset by taking a 3-class subset
of CIFAR-10 (cat, dog, ship) and using just 600 training images. That's
the regime where transfer learning shines.

  Run A — feature extraction (backbone frozen): expect ≥ 70% val accuracy
  Run B — full fine-tune with differential LR: should match or beat A
        with these few epochs and few samples

Run: python exercises/02_fine_tune_resnet.py
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms


torch.manual_seed(0)
np.random.seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


# ───────────────────────────────────────────────────────────
# Build a tiny 3-class dataset from CIFAR-10
# ───────────────────────────────────────────────────────────
# CIFAR-10 classes: 0 airplane 1 automobile 2 bird 3 cat 4 deer 5 dog
#                   6 frog     7 horse      8 ship 9 truck
PICK_CLASSES = [3, 5, 8]      # cat, dog, ship
NEW_LABEL = {orig: i for i, orig in enumerate(PICK_CLASSES)}

# Pretrained ResNet expects 224×224 normalized like ImageNet
imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std  = (0.229, 0.224, 0.225)

train_tf = transforms.Compose([
    transforms.Resize(96),
    transforms.RandomCrop(96, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std),
])
val_tf = transforms.Compose([
    transforms.Resize(96),
    transforms.CenterCrop(96),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std),
])

full_train = datasets.CIFAR10(DATA_DIR, train=True,  download=True, transform=train_tf)
full_val   = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=val_tf)


def subset_and_relabel(ds, max_per_class=200):
    indices_by_class = {c: [] for c in PICK_CLASSES}
    for idx, (_, label) in enumerate(ds):
        if label in indices_by_class and len(indices_by_class[label]) < max_per_class:
            indices_by_class[label].append(idx)
        if all(len(v) >= max_per_class for v in indices_by_class.values()):
            break
    indices = sum(indices_by_class.values(), [])
    return Subset(ds, indices)


# Pretend we have ~600 train and ~300 val images
train_subset = subset_and_relabel(full_train, max_per_class=200)
val_subset   = subset_and_relabel(full_val,   max_per_class=100)
print(f"train: {len(train_subset)}, val: {len(val_subset)}")


def remap_targets(batch):
    images, labels = zip(*batch)
    images = torch.stack(images)
    labels = torch.tensor([NEW_LABEL[int(l)] for l in labels])
    return images, labels


train_loader = DataLoader(train_subset, batch_size=32, shuffle=True,
                          num_workers=0, collate_fn=remap_targets)
val_loader   = DataLoader(val_subset, batch_size=64, shuffle=False,
                          num_workers=0, collate_fn=remap_targets)


# ───────────────────────────────────────────────────────────
# Helper to train + evaluate a model
# ───────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()


def evaluate(model):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(dim=1) == y).sum().item()
            total   += y.numel()
    return correct / total


def train(model, optimizer, epochs=3, label=""):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        running, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * y.size(0)
            n += y.size(0)
        acc = evaluate(model)
        print(f"  [{label}] epoch {epoch+1}/{epochs}  loss={running/n:.4f}  val_acc={acc:.4f}")
    return acc


def make_resnet18(n_classes=3):
    m = models.resnet18(weights="DEFAULT")
    m.fc = nn.Linear(m.fc.in_features, n_classes)
    return m


# ───────────────────────────────────────────────────────────
# Run A — feature extraction (backbone frozen)
# ───────────────────────────────────────────────────────────
print("\n--- A: feature extraction (backbone frozen) ---")
model_a = make_resnet18()
for name, p in model_a.named_parameters():
    if not name.startswith("fc."):
        p.requires_grad = False

opt_a = torch.optim.AdamW(filter(lambda p: p.requires_grad, model_a.parameters()),
                          lr=1e-3, weight_decay=1e-2)
acc_a = train(model_a, opt_a, epochs=3, label="freeze")


# ───────────────────────────────────────────────────────────
# Run B — full fine-tune with differential LR
# ───────────────────────────────────────────────────────────
print("\n--- B: full fine-tune (backbone lr=1e-4, head lr=1e-3) ---")
model_b = make_resnet18()
backbone_params = [p for n, p in model_b.named_parameters() if not n.startswith("fc.")]
head_params     = [p for n, p in model_b.named_parameters() if n.startswith("fc.")]
opt_b = torch.optim.AdamW([
    {"params": backbone_params, "lr": 1e-4},
    {"params": head_params,     "lr": 1e-3},
], weight_decay=1e-2)
acc_b = train(model_b, opt_b, epochs=3, label="finetune")


print(f"\nfeature-extraction val accuracy: {acc_a:.4f}")
print(f"fine-tune          val accuracy: {acc_b:.4f}")

assert acc_a >= 0.70, f"feature-extraction should hit ≥ 70% val acc, got {acc_a:.4f}"
print("transfer learning works ✅")


# ── Stretch ─────────────────────────────────────────────────────────────────
# • Compare with training the same architecture FROM SCRATCH on the same
#   small dataset. Pretraining should win by a wide margin.
# • Try `timm.create_model("convnext_tiny", pretrained=True, num_classes=3)`
#   — should beat ResNet18 with similar effort.
# • Replace the 3 chosen classes with your own folder of photos
#   (use `torchvision.datasets.ImageFolder`).
# ─────────────────────────────────────────────────────────────────────────────
