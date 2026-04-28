"""
10 — Datasets, DataLoaders, and checkpoints

In real training, data is too big to fit in memory and shuffling per epoch
matters. PyTorch ships two abstractions:

  Dataset    — knows how to fetch one sample by index.
  DataLoader — wraps a Dataset; produces batches, shuffles, parallelizes I/O.

You'll also need to save and restore models. Use `state_dict()` — never
pickle whole `nn.Module` objects (fragile across versions).

Run: python 10_dataloaders_and_checkpoints.py
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split


torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
SCRATCH = Path(__file__).parent / "scratch"
SCRATCH.mkdir(exist_ok=True)


# ───────────────────────────────────────────────────────────
# Dataset 1: in-memory tensors → use TensorDataset
# ───────────────────────────────────────────────────────────
N, D = 1000, 8
X = torch.randn(N, D)
y = (X.sum(dim=1) > 0).long()

dataset = TensorDataset(X, y)
print(f"len(dataset) = {len(dataset)}")
sample_x, sample_y = dataset[0]
print(f"first sample: x.shape={tuple(sample_x.shape)}, y={sample_y.item()}")


# ───────────────────────────────────────────────────────────
# Dataset 2: a custom one — for files on disk, JSON records, etc.
# ───────────────────────────────────────────────────────────
class TabularDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        # In a real dataset this might load a file, decode an image, etc.
        return self.X[idx], self.y[idx]


custom = TabularDataset(X, y)


# ───────────────────────────────────────────────────────────
# Splitting
# ───────────────────────────────────────────────────────────
train_set, val_set = random_split(dataset, [800, 200],
                                  generator=torch.Generator().manual_seed(0))
print(f"\nsplit: train {len(train_set)}, val {len(val_set)}")


# ───────────────────────────────────────────────────────────
# DataLoader
# ───────────────────────────────────────────────────────────
# Key arguments:
#   batch_size    — samples per batch
#   shuffle       — randomize order each epoch (TRAIN only)
#   num_workers   — background processes for I/O
#   pin_memory    — pinned host memory; faster transfer to GPU
#   drop_last     — discard the last incomplete batch (sometimes needed for BN)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True,
                          num_workers=0, pin_memory=False)
val_loader   = DataLoader(val_set,   batch_size=128, shuffle=False)

batch_x, batch_y = next(iter(train_loader))
print(f"\nbatch shapes: x {tuple(batch_x.shape)}, y {tuple(batch_y.shape)}")


# ───────────────────────────────────────────────────────────
# Custom collate_fn — for variable-length sequences, etc.
# ───────────────────────────────────────────────────────────
def pad_collate(batch):
    """Pad sequences to max length in the batch."""
    seqs, lengths = zip(*batch)
    max_len = max(s.size(0) for s in seqs)
    padded = torch.zeros(len(seqs), max_len)
    for i, s in enumerate(seqs):
        padded[i, : s.size(0)] = s
    return padded, torch.tensor(lengths)


# Demo with synthetic variable-length sequences
seq_dataset = [(torch.randn(torch.randint(2, 8, ()).item()), 0) for _ in range(20)]
seq_dataset = [(s, len(s)) for s, _ in seq_dataset]
seq_loader = DataLoader(seq_dataset, batch_size=4, collate_fn=pad_collate)
seqs, lens = next(iter(seq_loader))
print(f"\npadded seq batch shape: {tuple(seqs.shape)}, lengths: {lens.tolist()}")


# ───────────────────────────────────────────────────────────
# Train a tiny model on the dataset, then save / load
# ───────────────────────────────────────────────────────────
model = nn.Sequential(
    nn.Linear(D, 32), nn.ReLU(),
    nn.Linear(32, 2),
).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()


def evaluate():
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for bx, by in val_loader:
            bx, by = bx.to(device), by.to(device)
            preds = model(bx).argmax(dim=1)
            correct += (preds == by).sum().item()
            total   += by.numel()
    model.train()
    return correct / total


for epoch in range(5):
    for bx, by in train_loader:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        criterion(model(bx), by).backward()
        optimizer.step()
    print(f"epoch {epoch}: val_acc={evaluate():.3f}")


# ───────────────────────────────────────────────────────────
# Saving — state_dict, NOT the whole module
# ───────────────────────────────────────────────────────────
ckpt_path = SCRATCH / "checkpoint.pt"
torch.save({
    "model_state_dict":     model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "epoch":                5,
    "val_acc":              evaluate(),
}, ckpt_path)
print(f"\nsaved checkpoint to: {ckpt_path}  "
      f"({ckpt_path.stat().st_size / 1024:.1f} KB)")


# ───────────────────────────────────────────────────────────
# Loading
# ───────────────────────────────────────────────────────────
model_reloaded = nn.Sequential(
    nn.Linear(D, 32), nn.ReLU(),
    nn.Linear(32, 2),
).to(device)
ckpt = torch.load(ckpt_path, map_location=device)
model_reloaded.load_state_dict(ckpt["model_state_dict"])
print(f"reloaded model — val_acc: {evaluate():.3f}")
# Continuing training: also restore optimizer state and the epoch counter.


# ───────────────────────────────────────────────────────────
# TensorBoard (optional)
# ───────────────────────────────────────────────────────────
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter("runs/exp1")
# for step, ...:
#     writer.add_scalar("loss/train", loss.item(), step)
#     writer.add_scalar("loss/val",   val_loss, step)
# writer.close()
#
# Then: tensorboard --logdir runs


# ───────────────────────────────────────────────────────────
# Tips
# ───────────────────────────────────────────────────────────
# • Set num_workers > 0 for faster I/O (start with 4 on a CPU machine).
# • pin_memory=True helps when you're moving batches to GPU.
# • For images, look at torchvision.datasets.ImageFolder + transforms.
# • For HuggingFace text, look at the `datasets` library — drop-in compatible.
