"""
08 — Image embeddings: similarity search on images

Take a pretrained backbone, chop off the classifier head, and the remaining
output for an image is an EMBEDDING — a fixed-size vector that captures
"what this image looks like" semantically. Similar images → nearby vectors.

Use cases:
  • Reverse image search ("find images that look like this")
  • Duplicate detection
  • Clustering / labeling at scale
  • RAG over image corpora

This file uses ResNet18 as the embedder. For TEXT-IMAGE alignment (CLIP),
see the stretch goal at the bottom.

Run: python 08_image_embeddings.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms


torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"


# ───────────────────────────────────────────────────────────
# Build an embedder from ResNet18
# ───────────────────────────────────────────────────────────
backbone = models.resnet18(weights="DEFAULT")
# Drop the final fc layer → output is the 512-D global-pooled feature vector.
embedder = nn.Sequential(*list(backbone.children())[:-1], nn.Flatten()).to(device).eval()
print(f"embedding dim: 512")
print(f"params: {sum(p.numel() for p in embedder.parameters()):,}")


# ───────────────────────────────────────────────────────────
# Standard ImageNet preprocessing (the same transforms ResNet was trained on)
# ───────────────────────────────────────────────────────────
preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])


# ───────────────────────────────────────────────────────────
# Embed a few synthetic images (real images in exercise 4)
# ───────────────────────────────────────────────────────────
# We make 5 "classes" of images, 4 per class — same coloured rectangle in
# random positions. Embeddings should cluster by colour.
def make_image(color: tuple[int, int, int], jitter: int = 12) -> torch.Tensor:
    img = np.full((96, 96, 3), 255, dtype=np.uint8)
    cx = np.random.randint(jitter, 96 - jitter - 30)
    cy = np.random.randint(jitter, 96 - jitter - 30)
    img[cy:cy + 30, cx:cx + 30] = color
    pil = transforms.functional.to_pil_image(img)
    return preprocess(pil)


classes = {
    "red":   (220, 60,  40),
    "green": (30,  170, 60),
    "blue":  (50,  90,  220),
    "yellow":(230, 220, 50),
    "magenta":(190, 50, 200),
}
imgs = []
labels = []
np.random.seed(0)
for name, color in classes.items():
    for _ in range(4):
        imgs.append(make_image(color))
        labels.append(name)

batch = torch.stack(imgs).to(device)
with torch.no_grad():
    embeddings = embedder(batch).cpu().numpy()
print(f"embedded {len(imgs)} images → matrix shape {embeddings.shape}")


# ───────────────────────────────────────────────────────────
# Cosine similarity for retrieval
# ───────────────────────────────────────────────────────────
def cosine_similarity_matrix(X):
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    return Xn @ Xn.T


sim = cosine_similarity_matrix(embeddings)
print(f"\nsim matrix shape: {sim.shape}, mean off-diagonal: {sim[~np.eye(20, dtype=bool)].mean():.3f}")


# Demonstrate: for each query image, the nearest neighbours should be the
# SAME COLOR class.
def top_k(query_idx: int, k: int = 3) -> list[int]:
    # Exclude self
    scores = sim[query_idx].copy()
    scores[query_idx] = -np.inf
    return scores.argsort()[::-1][:k].tolist()


print("\ntop-3 nearest neighbours for each image:")
correct = 0
for i, label in enumerate(labels):
    nn_indices = top_k(i, k=3)
    nn_labels = [labels[j] for j in nn_indices]
    same = nn_labels.count(label)
    correct += same / 3
    if i % 4 == 0:
        print(f"  {label:<8} → {nn_labels}")

print(f"\naverage correct-class rate among top-3: {correct / len(labels):.2%}")


# ───────────────────────────────────────────────────────────
# Visualize the embedding space (PCA → 2D)
# ───────────────────────────────────────────────────────────
from sklearn.decomposition import PCA

pca = PCA(n_components=2).fit_transform(embeddings)
fig, ax = plt.subplots(figsize=(7, 6))
for label in classes:
    mask = [l == label for l in labels]
    ax.scatter(pca[mask, 0], pca[mask, 1], label=label, s=80, alpha=0.7)
ax.set_title("ResNet embeddings of color squares (PCA → 2D)")
ax.legend()
ax.grid(True, alpha=0.3)

out = Path(__file__).parent / "figures"
out.mkdir(exist_ok=True)
fig.savefig(out / "08_image_embeddings.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"\nfigure saved to: {out / '08_image_embeddings.png'}")


# ───────────────────────────────────────────────────────────
# Stretch: CLIP for text↔image alignment
# ───────────────────────────────────────────────────────────
# `pip install open-clip-torch` then:
#
#   import open_clip
#   model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
#   tokenizer = open_clip.get_tokenizer("ViT-B-32")
#
#   img_tokens = preprocess(pil_image).unsqueeze(0)
#   text_tokens = tokenizer(["a photo of a dog", "a photo of a cat"])
#   with torch.no_grad():
#       img_feat  = model.encode_image(img_tokens)   # 512-D
#       text_feat = model.encode_text(text_tokens)   # 512-D
#   sim = (img_feat @ text_feat.T).softmax(dim=-1)
#
# CLIP puts images and text in the SAME embedding space, so you can search
# images with a text query — the foundation of modern multimodal search.
