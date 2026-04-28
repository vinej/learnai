"""
03 — Eigenvalues and eigenvectors (intuition only)

For a square matrix A, an *eigenvector* v is a vector whose direction is
unchanged when A is applied to it:

    A v = λ v

`v` is the eigenvector and the scalar `λ` is the matching *eigenvalue*.
The matrix scales `v` by λ but doesn't rotate it.

Why you'll care later:
  • PCA reduces high-dim data using top eigenvectors of the covariance matrix.
  • Stability of dynamical systems (recurrent nets) depends on eigenvalues.
  • The "spectral" view of graphs, attention, etc. all uses these.

Run: python 03_eigen.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# A simple 2×2 matrix
A = np.array([[2.0, 1.0],
              [1.0, 3.0]])

eigvals, eigvecs = np.linalg.eig(A)
print("eigenvalues:", eigvals)
print("eigenvectors (columns):")
print(eigvecs)

# Verify A v = λ v for each eigenpair
for i in range(len(eigvals)):
    v = eigvecs[:, i]
    lam = eigvals[i]
    print(f"\npair {i}:  λ = {lam:.4f}")
    print(f"  A @ v = {A @ v}")
    print(f"  λ * v = {lam * v}")
    print(f"  match? {np.allclose(A @ v, lam * v)}")


# --- Visualize ---
# Plot the unit circle, then plot what A does to it. Eigenvectors are the
# axes of the resulting ellipse.
theta = np.linspace(0, 2 * np.pi, 200)
unit_circle = np.stack([np.cos(theta), np.sin(theta)])  # 2 × 200
transformed = A @ unit_circle                            # 2 × 200

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(unit_circle[0], unit_circle[1], label="unit circle", linestyle="--")
ax.plot(transformed[0], transformed[1], label="A @ circle")

# Draw eigenvectors (scaled by their eigenvalue)
colors = ["tab:red", "tab:green"]
for i in range(2):
    v = eigvecs[:, i]
    lam = eigvals[i]
    ax.arrow(0, 0, v[0], v[1],
             head_width=0.08, color=colors[i], label=f"eigvec {i}")
    ax.arrow(0, 0, lam * v[0], lam * v[1],
             head_width=0.08, color=colors[i], alpha=0.4,
             label=f"λ * eigvec {i}")

ax.set_aspect("equal")
ax.grid(True)
ax.set_title(f"A = {A.tolist()}\nEigenvalues: {eigvals.round(2).tolist()}")
ax.legend(loc="upper left", fontsize=8)

out = Path(__file__).parent / "figures"
out.mkdir(exist_ok=True)
fig_path = out / "03_eigen.png"
fig.savefig(fig_path, dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"\nfigure saved to: {fig_path}")
