"""
Exercise 3 — Train an MLP classifier on MNIST and reach >97% test accuracy

MNIST is the "hello world" of deep learning: 28×28 grayscale digits, 60k
training + 10k test. Easy to over-perform; useful for verifying your
training infrastructure works.

The script:
  • Downloads MNIST on first run (~10MB) into ./data/
  • Trains an MLP for ~5 epochs
  • Asserts test accuracy ≥ 97%

Run: python exercises/03_mnist_classifier.py
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
# Data
# ───────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# transforms: convert to tensor (0..1 float), normalize using MNIST stats
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1307], std=[0.3081]),
])

train_set = datasets.MNIST(root=DATA_DIR, train=True,  download=True, transform=transform)
test_set  = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)
print(f"train {len(train_set)}, test {len(test_set)}")

train_loader = DataLoader(train_set, batch_size=128, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_set,  batch_size=512, shuffle=False, num_workers=0)


# ───────────────────────────────────────────────────────────
# Model — a plain MLP. (CNNs are next module.)
# ───────────────────────────────────────────────────────────
class MNISTMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.net(x)


model = MNISTMLP().to(device)
print(f"params: {sum(p.numel() for p in model.parameters()):,}")


# ───────────────────────────────────────────────────────────
# Training
# ───────────────────────────────────────────────────────────
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()


def evaluate() -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total   += y.numel()
    model.train()
    return correct / total


EPOCHS = 5
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    n_seen = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * y.size(0)
        n_seen += y.size(0)

    test_acc = evaluate()
    print(f"epoch {epoch + 1}/{EPOCHS}  "
          f"train_loss={running_loss / n_seen:.4f}  test_acc={test_acc:.4f}")


final_acc = evaluate()
print(f"\nfinal test accuracy: {final_acc:.4f}")
assert final_acc >= 0.97, f"goal is ≥ 0.97, got {final_acc:.4f}"
print("MNIST classifier hits the bar ✅")


# ── Stretch ─────────────────────────────────────────────────────────────────
# • Replace the MLP with a small CNN (Conv2d + Pool + Linear). Aim for ≥99%.
# • Add a learning-rate scheduler (CosineAnnealingLR) and re-measure.
# • Add data augmentation: transforms.RandomRotation(10), RandomAffine.
#   Does it hurt or help on MNIST?
# ─────────────────────────────────────────────────────────────────────────────
