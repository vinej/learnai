"""
Exercise 4 — Cluster customers with KMeans, visualize with PCA

Steps:
  1. Generate a synthetic high-dim "customer" dataset with 4 hidden segments.
  2. Pick K via the elbow method + silhouette score.
  3. Cluster with KMeans.
  4. Reduce to 2-D with PCA and plot the clusters.

Run: python exercises/04_kmeans_pca.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


rng = np.random.default_rng(0)


def main():
    # 4 hidden customer segments, each described by 12 features
    X, y_true = make_blobs(
        n_samples=2000, centers=4, n_features=12, cluster_std=2.0,
        random_state=0,
    )
    X = StandardScaler().fit_transform(X)

    # ─────────────────────────
    # Step 1: try several K values
    # ─────────────────────────
    ks = list(range(2, 9))
    inertias = []
    silhouettes = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init="auto", random_state=0).fit(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, km.labels_))

    best_k = ks[int(np.argmax(silhouettes))]
    print(f"silhouette per K: "
          f"{dict(zip(ks, [round(s, 3) for s in silhouettes]))}")
    print(f"best K by silhouette: {best_k}")

    # ─────────────────────────
    # Step 2: final cluster
    # ─────────────────────────
    km_final = KMeans(n_clusters=best_k, n_init="auto", random_state=0).fit(X)
    labels = km_final.labels_

    # ─────────────────────────
    # Step 3: PCA → 2-D
    # ─────────────────────────
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    centers_2d = pca.transform(km_final.cluster_centers_)
    var_pct = pca.explained_variance_ratio_ * 100

    # ─────────────────────────
    # Step 4: plot
    # ─────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Elbow / inertia
    axes[0].plot(ks, inertias, "o-")
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("inertia")
    axes[0].set_title("Elbow plot")
    axes[0].grid(True, alpha=0.3)

    # Silhouette
    axes[1].plot(ks, silhouettes, "o-", color="tab:orange")
    axes[1].axvline(best_k, color="red", linestyle="--",
                    alpha=0.5, label=f"best K = {best_k}")
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("silhouette")
    axes[1].set_title("Silhouette per K (higher is better)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 2-D scatter colored by cluster
    axes[2].scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10", s=8, alpha=0.7)
    axes[2].scatter(centers_2d[:, 0], centers_2d[:, 1],
                    marker="X", s=200, c="black", edgecolors="white", linewidths=2,
                    label="centroids")
    axes[2].set_xlabel(f"PC1 ({var_pct[0]:.1f}% var)")
    axes[2].set_ylabel(f"PC2 ({var_pct[1]:.1f}% var)")
    axes[2].set_title(f"KMeans (K={best_k}) projected via PCA")
    axes[2].legend()

    fig.tight_layout()
    out = Path(__file__).parent.parent / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "exercise_04_kmeans_pca.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"figure saved to: {out / 'exercise_04_kmeans_pca.png'}")

    assert best_k == 4, "silhouette should pick K=4 since we generated 4 blobs"
    print("clustering recovered the 4 segments ✅")


if __name__ == "__main__":
    main()


# ── Stretch ─────────────────────────────────────────────────────────────────
# • Replace KMeans with DBSCAN — does it work without specifying K?
# • Try non-spherical clusters (make_moons, make_circles) — observe failure.
# • Compute mean feature values per cluster to "name" each segment.
# ─────────────────────────────────────────────────────────────────────────────
