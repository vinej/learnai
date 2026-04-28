"""
04 — k-Nearest Neighbors and Support Vector Machines

KNN
  Predict by majority vote (or average) over the k closest training points.
  No real "training" — all the work is at predict time. SCALING MATTERS,
  because distances are everything.

SVM
  Find a hyperplane that maximally separates classes. Use the "kernel trick"
  (RBF / poly) to handle non-linear boundaries. Scaling matters here too.

Both are conceptually beautiful and rarely the best on tabular data — but
they're indispensable teaching tools for "what does a decision boundary
look like?" intuition.

Run: python 04_knn_svm.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# A non-linear toy dataset
X, y = make_moons(n_samples=400, noise=0.25, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)


models = {
    "KNN (k=1)":         make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=1)),
    "KNN (k=15)":        make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15)),
    "SVM (linear)":      make_pipeline(StandardScaler(), SVC(kernel="linear")),
    "SVM (rbf, C=1)":    make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale")),
    "SVM (rbf, C=100)":  make_pipeline(StandardScaler(), SVC(kernel="rbf", C=100, gamma="scale")),
}


print(f"{'model':<22}  test accuracy")
for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"{name:<22}  {accuracy_score(y_test, model.predict(X_test)):.3f}")


# ───────────────────────────────────────────────────────────
# Visualize the decision boundary
# ───────────────────────────────────────────────────────────
def plot_boundary(ax, model, title):
    pad = 0.5
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors="k",
               cmap="coolwarm", s=18)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


fig, axes = plt.subplots(2, 3, figsize=(13, 8))
items = list(models.items()) + [(None, None)]   # one filler for grid
for ax, (name, model) in zip(axes.flatten(), items):
    if model is None:
        ax.axis("off")
    else:
        plot_boundary(ax, model, name)

fig.suptitle("Decision boundaries on the 'moons' dataset", fontsize=12)
fig.tight_layout()
out = Path(__file__).parent / "figures"
out.mkdir(exist_ok=True)
fig.savefig(out / "04_knn_svm_boundaries.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"\nfigure saved to: {out / '04_knn_svm_boundaries.png'}")


# ───────────────────────────────────────────────────────────
# Practical knobs
# ───────────────────────────────────────────────────────────
# KNN:
#   n_neighbors  — small k = jagged boundary (high variance);
#                  large k = smoother (high bias).
#   weights      — 'uniform' or 'distance' (closer neighbors weigh more).
#   metric       — minkowski / euclidean / manhattan / cosine.
#
# SVM:
#   C            — regularization. Small C = wider margin (more slack);
#                  large C = harder margin (overfit risk).
#   kernel       — linear / rbf / poly / sigmoid.
#   gamma (rbf)  — controls kernel width. Larger = more wiggly fit.


# ───────────────────────────────────────────────────────────
# When to pick what (real-world)
# ───────────────────────────────────────────────────────────
# • Plain tabular, decent N      → tree ensembles. Almost always.
# • Very small N, smooth target  → KNN/SVM/Linear are competitive.
# • Need probability scores      → calibrated linear/GBM/RF; SVM needs CalibratedClassifierCV.
# • Highly non-linear, low-dim   → SVM with RBF can shine.
