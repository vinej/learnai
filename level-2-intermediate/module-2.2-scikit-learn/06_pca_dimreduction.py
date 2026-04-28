"""
06 — Dimensionality reduction: PCA, t-SNE, UMAP

Why reduce dimensions?
  1. VISUALIZATION   — humans see in 2D/3D, not 64D.
  2. SPEED           — fewer features → faster fitting and prediction.
  3. NOISE REDUCTION — drop low-variance components that are mostly noise.
  4. DECORRELATION   — PCA produces orthogonal components.

Methods (broadly):
  PCA   — linear, fast, exact; great for compression and lightweight viz.
  t-SNE — non-linear, slow, EMPHASIZES local structure; great for clusters.
  UMAP  — non-linear, faster than t-SNE; preserves more global structure.

Run: python 06_pca_dimreduction.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


# Digits: 1797 samples × 64 features (8×8 pixel images)
data = load_digits()
X, y = data.data, data.target
print(f"original shape: {X.shape}")

X_std = StandardScaler().fit_transform(X)


# ───────────────────────────────────────────────────────────
# PCA
# ───────────────────────────────────────────────────────────
pca = PCA(n_components=10).fit(X_std)
print(f"\nexplained variance ratio (per component):")
print(pca.explained_variance_ratio_.round(3))
print(f"cumulative: {np.cumsum(pca.explained_variance_ratio_).round(3)}")

# How many components do you need to retain, say, 90% of the variance?
pca_full = PCA().fit(X_std)
cumulative = np.cumsum(pca_full.explained_variance_ratio_)
n_for_90 = int(np.searchsorted(cumulative, 0.90)) + 1
print(f"\n{n_for_90} components retain 90% of variance "
      f"(out of {X.shape[1]} original features)")


# ───────────────────────────────────────────────────────────
# 2-D PCA for visualization
# ───────────────────────────────────────────────────────────
X_pca_2d = PCA(n_components=2).fit_transform(X_std)


# ───────────────────────────────────────────────────────────
# 2-D t-SNE — much slower but tighter clusters
# ───────────────────────────────────────────────────────────
print("\nrunning t-SNE (this takes a few seconds)...")
X_tsne_2d = TSNE(n_components=2, perplexity=30, random_state=0,
                 init="pca", learning_rate="auto").fit_transform(X_std)


# ───────────────────────────────────────────────────────────
# Plot
# ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Scree plot
axes[0].plot(np.arange(1, len(cumulative) + 1), cumulative, "o-")
axes[0].axhline(0.9, color="red", linestyle="--", alpha=0.5, label="90% variance")
axes[0].axvline(n_for_90, color="green", linestyle="--", alpha=0.5,
                label=f"{n_for_90} components")
axes[0].set_xlabel("number of components")
axes[0].set_ylabel("cumulative explained variance")
axes[0].set_title("PCA scree plot")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 2-D PCA scatter
axes[1].scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=y, cmap="tab10", s=8, alpha=0.7)
axes[1].set_title("PCA → 2D")
axes[1].set_xlabel("PC1")
axes[1].set_ylabel("PC2")

# 2-D t-SNE scatter
axes[2].scatter(X_tsne_2d[:, 0], X_tsne_2d[:, 1], c=y, cmap="tab10", s=8, alpha=0.7)
axes[2].set_title("t-SNE → 2D")
axes[2].set_xticks([])
axes[2].set_yticks([])

fig.tight_layout()
out = Path(__file__).parent / "figures"
out.mkdir(exist_ok=True)
fig.savefig(out / "06_pca_tsne.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"figure saved to: {out / '06_pca_tsne.png'}")


# ───────────────────────────────────────────────────────────
# When to use what
# ───────────────────────────────────────────────────────────
# • Compression / preprocessing for ML       → PCA.
# • 2D plot of clusters in a notebook        → t-SNE or UMAP.
# • Interpretable component "directions"     → PCA (each component is a
#                                              linear combination of features).
# • Need to embed NEW samples after fitting  → PCA, UMAP. (t-SNE doesn't
#                                              naturally do this.)
#
# UMAP isn't in the default install (`pip install umap-learn`). If you do
# install it, the API mirrors t-SNE: `from umap import UMAP; UMAP(...).fit_transform(X)`.
