"""
05 — Clustering: KMeans, hierarchical, DBSCAN

Clustering = unsupervised grouping. No labels — find structure.

  KMeans            — assigns points to K spherical clusters by minimizing
                      within-cluster sum of squares. Fast, easy, but you
                      must pick K up front and clusters must be roughly
                      equal-sized and spherical.
  Hierarchical      — builds a tree of merges. Doesn't need K — cut the
                      tree at any depth.
  DBSCAN            — density-based; finds clusters of arbitrary shape and
                      labels noise as -1. Doesn't need K.

Run: python 05_clustering.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


rng = np.random.default_rng(0)

# Two synthetic datasets to highlight strengths/weaknesses
X_blobs, _ = make_blobs(n_samples=400, centers=4, cluster_std=1.0, random_state=0)
X_moons, _ = make_moons(n_samples=400, noise=0.07, random_state=0)

X_blobs = StandardScaler().fit_transform(X_blobs)
X_moons = StandardScaler().fit_transform(X_moons)


# ───────────────────────────────────────────────────────────
# KMeans
# ───────────────────────────────────────────────────────────
km = KMeans(n_clusters=4, n_init="auto", random_state=0).fit(X_blobs)
print(f"KMeans inertia (within-cluster SS): {km.inertia_:.1f}")
print(f"silhouette score on blobs:           {silhouette_score(X_blobs, km.labels_):.3f}")


# ───────────────────────────────────────────────────────────
# Picking K — the elbow method
# ───────────────────────────────────────────────────────────
# Plot K vs inertia. The "elbow" — where the curve bends — is a heuristic K.
ks = range(1, 10)
inertias = [
    KMeans(n_clusters=k, n_init="auto", random_state=0).fit(X_blobs).inertia_
    for k in ks
]


# ───────────────────────────────────────────────────────────
# Compare on two datasets, three algorithms
# ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(13, 8))

datasets = [("blobs", X_blobs), ("moons", X_moons)]
algos = {
    "KMeans (k=4)": KMeans(n_clusters=4, n_init="auto", random_state=0),
    "Agglomerative": AgglomerativeClustering(n_clusters=4),
    "DBSCAN":        DBSCAN(eps=0.3, min_samples=10),
}

for row, (data_name, Xd) in enumerate(datasets):
    for col, (algo_name, algo) in enumerate(algos.items()):
        labels = algo.fit_predict(Xd)
        axes[row, col].scatter(Xd[:, 0], Xd[:, 1], c=labels, cmap="tab10", s=10)
        axes[row, col].set_title(f"{algo_name}  on  {data_name}", fontsize=10)
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])

# Bonus: elbow plot in a separate figure
elbow_fig, elbow_ax = plt.subplots(figsize=(6, 4))
elbow_ax.plot(list(ks), inertias, "o-")
elbow_ax.set_xlabel("k")
elbow_ax.set_ylabel("inertia (lower = tighter clusters)")
elbow_ax.set_title("KMeans elbow plot on blobs")
elbow_ax.grid(True, alpha=0.3)


out = Path(__file__).parent / "figures"
out.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(out / "05_clusters.png", dpi=120, bbox_inches="tight")
plt.close(fig)
elbow_fig.savefig(out / "05_elbow.png", dpi=120, bbox_inches="tight")
plt.close(elbow_fig)
print(f"figures saved to: {out}")


# ───────────────────────────────────────────────────────────
# What you should observe in the cluster comparison
# ───────────────────────────────────────────────────────────
# BLOBS (round, well-separated):
#   • All 3 work fine.
#
# MOONS (interleaved crescents):
#   • KMeans fails — it can only draw straight boundaries.
#   • Agglomerative may also fail with default linkage.
#   • DBSCAN nails it because it follows local density, not centroids.


# ───────────────────────────────────────────────────────────
# When to use what
# ───────────────────────────────────────────────────────────
# • Customer segmentation, image-quantization → KMeans is fast & good enough.
# • Anomaly detection, irregular shapes        → DBSCAN.
# • Exploratory dendrogram / unknown K         → Agglomerative.
# • For very high-dimensional data, reduce dim with PCA first (see file 06).
