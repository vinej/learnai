"""
05 — Recommenders

Two main flavors of recommender:

  COLLABORATIVE FILTERING — uses past user-item interactions only.
                            "Users who bought X also bought Y."
  CONTENT-BASED          — uses item / user features (text, image, profile).

Within collaborative:
  MATRIX FACTORIZATION — factor the user-item rating matrix R ≈ U @ V^T.
                          Learned U is each user's K-D taste vector;
                          learned V is each item's K-D feature vector.
                          predict(u, i) = U[u] · V[i] + bias.
  TWO-TOWER NN          — a user network and an item network produce
                          embeddings; their dot product = score.
  SEQUENTIAL            — model the SEQUENCE of past interactions
                          (SASRec, BERT4Rec, GRU4Rec). Often strongest now.

This file demos matrix factorization on a synthetic ratings matrix.

Run: python 05_recommenders.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


torch.manual_seed(0)
np.random.seed(0)


# ───────────────────────────────────────────────────────────
# Synthetic ratings matrix
# ───────────────────────────────────────────────────────────
# Imagine N_USERS users rating N_ITEMS items on a 1-5 scale. The TRUE
# rating depends on a small number of "taste factors". We hide most of the
# matrix and try to recover it.
N_USERS = 200
N_ITEMS = 100
TRUE_K  = 5

true_U = np.random.normal(0, 1, (N_USERS, TRUE_K))
true_V = np.random.normal(0, 1, (N_ITEMS, TRUE_K))
ratings = (true_U @ true_V.T) + np.random.normal(0, 0.1, (N_USERS, N_ITEMS))
ratings = np.clip(ratings, -3, 3)            # keep numbers reasonable

# Hide ~80% of entries (very sparse, like real systems)
mask = np.random.random((N_USERS, N_ITEMS)) < 0.2
print(f"observed ratings: {mask.sum()} / {ratings.size} ({mask.mean():.1%})")


# Convert observed entries to (user, item, rating) triples
users, items = np.where(mask)
labels = torch.tensor(ratings[users, items], dtype=torch.float32)
users  = torch.tensor(users,  dtype=torch.long)
items  = torch.tensor(items,  dtype=torch.long)


# ───────────────────────────────────────────────────────────
# Matrix factorization model
# ───────────────────────────────────────────────────────────
class MFModel(nn.Module):
    def __init__(self, n_users: int, n_items: int, k: int):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, k)
        self.item_emb = nn.Embedding(n_items, k)
        self.user_b   = nn.Embedding(n_users, 1)
        self.item_b   = nn.Embedding(n_items, 1)
        self.global_b = nn.Parameter(torch.zeros(1))
        # Small init helps avoid huge initial losses
        nn.init.normal_(self.user_emb.weight, std=0.05)
        nn.init.normal_(self.item_emb.weight, std=0.05)

    def forward(self, u, i):
        u_v = self.user_emb(u)
        i_v = self.item_emb(i)
        dot = (u_v * i_v).sum(dim=-1)
        return dot + self.user_b(u).squeeze() + self.item_b(i).squeeze() + self.global_b


model = MFModel(N_USERS, N_ITEMS, k=8)
opt = torch.optim.AdamW(model.parameters(), lr=5e-2, weight_decay=1e-4)
criterion = nn.MSELoss()


# ───────────────────────────────────────────────────────────
# Train
# ───────────────────────────────────────────────────────────
losses = []
for epoch in range(200):
    opt.zero_grad()
    pred = model(users, items)
    loss = criterion(pred, labels)
    loss.backward()
    opt.step()
    losses.append(loss.item())
    if epoch % 25 == 0:
        print(f"epoch {epoch:>3}  train MSE {loss.item():.4f}")


# ───────────────────────────────────────────────────────────
# How well do we predict the HELD-OUT entries?
# ───────────────────────────────────────────────────────────
held_users, held_items = np.where(~mask)
held_labels = torch.tensor(ratings[held_users, held_items], dtype=torch.float32)
held_u = torch.tensor(held_users, dtype=torch.long)
held_i = torch.tensor(held_items, dtype=torch.long)

with torch.no_grad():
    held_pred = model(held_u, held_i)
held_rmse = ((held_pred - held_labels) ** 2).mean().sqrt().item()
naive_rmse = ((labels.mean() - held_labels) ** 2).mean().sqrt().item()

print(f"\nheld-out RMSE: {held_rmse:.3f}    "
      f"naive (predict global mean) RMSE: {naive_rmse:.3f}")


# ───────────────────────────────────────────────────────────
# Visualize learned item factors via PCA
# ───────────────────────────────────────────────────────────
from sklearn.decomposition import PCA

item_factors = model.item_emb.weight.detach().numpy()
pca = PCA(n_components=2).fit_transform(item_factors)

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(pca[:, 0], pca[:, 1], s=15)
ax.set_title("Learned item factors (PCA → 2D)")
ax.grid(True, alpha=0.3)
out = Path(__file__).parent / "figures"
out.mkdir(exist_ok=True)
fig.savefig(out / "05_recommender.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"figure saved to: {out / '05_recommender.png'}")


# ───────────────────────────────────────────────────────────
# Two-tower neural recommender (sketch)
# ───────────────────────────────────────────────────────────
# When you have FEATURES (item title text, user profile), do this:
#
#   class TwoTower(nn.Module):
#       def __init__(self):
#           super().__init__()
#           self.user_tower = MLP(user_feature_dim → k)   # or BERT for text users
#           self.item_tower = MLP(item_feature_dim → k)   # or CNN/BERT for items
#       def forward(self, user_feat, item_feat):
#           u = self.user_tower(user_feat)
#           v = self.item_tower(item_feat)
#           return (u * v).sum(dim=-1)
#
# Train with PAIRS (positive interactions) + sampled negatives. At serving
# time, encode all items once, store in an ANN index (FAISS), and retrieve
# top-k nearest items for a user's vector. Used everywhere: YouTube, Spotify,
# Pinterest.


# ───────────────────────────────────────────────────────────
# Sequential recommenders
# ───────────────────────────────────────────────────────────
# Treat a user's interaction history like a SENTENCE of item IDs and apply
# a transformer:
#   • SASRec     — transformer-based sequential recommender.
#   • BERT4Rec   — BERT-style masked LM on item sequences.
#   • GRU4Rec    — RNN baseline.
# Often strongest in production today.
