"""
Exercise 4 — Build an image-similarity search

Embed a small batch of images with a pretrained ResNet, then for each query
image return the K most similar by cosine similarity. The script asserts
the top-1 retrieved neighbour belongs to the SAME color class as the query.

This is the smallest possible "reverse image search". Real systems use
CLIP or DINOv2 embeddings + a vector store like FAISS or pgvector.

Run: python exercises/04_image_search.py
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


torch.manual_seed(0)
np.random.seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"


# ───────────────────────────────────────────────────────────
# Embedder = ResNet18 minus the classifier head
# ───────────────────────────────────────────────────────────
backbone = models.resnet18(weights="DEFAULT")
embedder = nn.Sequential(*list(backbone.children())[:-1], nn.Flatten()).to(device).eval()
preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


# ───────────────────────────────────────────────────────────
# Synthetic "image gallery" — colored squares with random offsets
# ───────────────────────────────────────────────────────────
def make_image(color: tuple[int, int, int]) -> torch.Tensor:
    img = np.full((96, 96, 3), 240, dtype=np.uint8)
    cx = np.random.randint(8, 96 - 38)
    cy = np.random.randint(8, 96 - 38)
    img[cy:cy + 30, cx:cx + 30] = color
    return preprocess(Image.fromarray(img))


CLASSES = {
    "red":     (220, 60,  40),
    "green":   (30,  170, 60),
    "blue":    (50,  90,  220),
    "yellow":  (230, 220, 50),
    "magenta": (190, 50,  200),
}

images, labels = [], []
for name, color in CLASSES.items():
    for _ in range(6):
        images.append(make_image(color))
        labels.append(name)


# ───────────────────────────────────────────────────────────
# Embed in one batch
# ───────────────────────────────────────────────────────────
batch = torch.stack(images).to(device)
with torch.no_grad():
    emb = embedder(batch).cpu().numpy()
emb /= np.linalg.norm(emb, axis=1, keepdims=True)
print(f"embedded {len(images)} images, dim={emb.shape[1]}")


# ───────────────────────────────────────────────────────────
# Cosine similarity → retrieval
# ───────────────────────────────────────────────────────────
sim = emb @ emb.T


def search(query_idx: int, k: int = 5) -> list[int]:
    scores = sim[query_idx].copy()
    scores[query_idx] = -np.inf       # exclude self
    return scores.argsort()[::-1][:k].tolist()


print("\nquery → top-5 nearest (label):")
top1_correct = 0
for q in range(0, len(images), 6):    # one query per class
    nn_idx = search(q, k=5)
    nn_labels = [labels[j] for j in nn_idx]
    print(f"  query={labels[q]:<8}  →  {nn_labels}")
    if nn_labels[0] == labels[q]:
        top1_correct += 1

print(f"\ntop-1 same-class rate (across {len(CLASSES)} queries): "
      f"{top1_correct}/{len(CLASSES)}")
assert top1_correct >= len(CLASSES) - 1, \
    "top-1 retrieval should match the query class for at least 4/5 colors"
print("similarity search works ✅")


# ── Stretch ─────────────────────────────────────────────────────────────────
# • Replace the synthetic squares with your own folder of photos
#   (loaded with torchvision.datasets.ImageFolder).
# • Index the embeddings with FAISS for sub-millisecond search at scale:
#     import faiss
#     index = faiss.IndexFlatIP(emb.shape[1])
#     index.add(emb.astype("float32"))
#     D, I = index.search(query_emb.astype("float32"), k=5)
# • Use CLIP (`open_clip`) to enable TEXT queries: "find me sunset photos".
# ─────────────────────────────────────────────────────────────────────────────
