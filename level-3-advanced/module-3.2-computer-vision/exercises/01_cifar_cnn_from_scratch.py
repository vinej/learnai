"""
Exercise 1 — Train a CNN on CIFAR-10 from scratch

Reach test accuracy ≥ 65% in 5 epochs on CPU. The README's stretch goal is
85% — that takes a deeper model (ResNet-style) and 30+ epochs. With a small
CNN, 65% is achievable in ~3-5 minutes on a modern laptop CPU.

The script:
  • Downloads CIFAR-10 (~170 MB) into ./data/
  • Trains the SmallCNN from file 02
  • Asserts test accuracy ≥ 0.65

Run: python exercises/01_cifar_cnn_from_scratch.py
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")


# ───────────────────────────────────────────────────────────
# Data with augmentation on training, plain on test
# ───────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)

train_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])
test_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])

train_set = datasets.CIFAR10(DATA_DIR, train=True,  download=True, transform=train_tf)
test_set  = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=test_tf)
print(f"train: {len(train_set)}, test: {len(test_set)}")

train_loader = DataLoader(train_set, batch_size=128, shuffle=True,
                          num_workers=2, pin_memory=device == "cuda")
test_loader  = DataLoader(test_set, batch_size=256, shuffle=False,
                          num_workers=2, pin_memory=device == "cuda")


# ───────────────────────────────────────────────────────────
# Same SmallCNN as file 02
# ───────────────────────────────────────────────────────────
class SmallCNN(nn.Module):
    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            self._block(3,   32),
            self._block(32,  64),
            self._block(64, 128),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    @staticmethod
    def _block(in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c,  out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.head(self.features(x))


model = SmallCNN().to(device)
print(f"params: {sum(p.numel() for p in model.parameters()):,}")


# ───────────────────────────────────────────────────────────
# Optimizer + cosine LR schedule
# ───────────────────────────────────────────────────────────
EPOCHS = 5
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.CrossEntropyLoss()


def evaluate() -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.numel()
    model.train()
    return correct / total


# ───────────────────────────────────────────────────────────
# Train
# ───────────────────────────────────────────────────────────
for epoch in range(EPOCHS):
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
    scheduler.step()
    test_acc = evaluate()
    print(f"epoch {epoch + 1}/{EPOCHS}  "
          f"train_loss={running / n:.4f}  test_acc={test_acc:.4f}  "
          f"lr={scheduler.get_last_lr()[0]:.5f}")

final = evaluate()
print(f"\nfinal test accuracy: {final:.4f}")
assert final >= 0.65, f"target ≥ 0.65, got {final:.4f}"
print("CIFAR-10 small-CNN passes ✅")


# ── Stretch ─────────────────────────────────────────────────────────────────
# • Replace SmallCNN with a torchvision ResNet18 (input is 32×32 — adapt
#   the first conv stride and the final fc). Train 30+ epochs to hit 85%+.
# • Add MixUp/CutMix (torchvision.transforms.v2.MixUp) at the dataloader.
# • Ablate the augmentations: how much does RandomCrop vs HFlip alone help?
# ─────────────────────────────────────────────────────────────────────────────
