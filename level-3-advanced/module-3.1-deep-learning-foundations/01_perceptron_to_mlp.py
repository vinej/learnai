"""
01 — From perceptron to multi-layer perceptron (MLP)

A PERCEPTRON is the simplest "neuron": output = activation(w · x + b).
A single layer can only learn LINEARLY separable functions.

The XOR problem is the classic counter-example: there's no straight line
that separates {(0,0), (1,1)} from {(0,1), (1,0)}.

A 2-LAYER MLP — perceptrons stacked, with a non-linear activation between
layers — can. That's the foundation of every neural net you'll meet.

Run: python 01_perceptron_to_mlp.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


rng = np.random.default_rng(0)


# ───────────────────────────────────────────────────────────
# A single perceptron, hand-trained
# ───────────────────────────────────────────────────────────
def perceptron_predict(W, b, X):
    """Forward pass for a single perceptron. Returns 0/1."""
    return (X @ W + b > 0).astype(int)


def train_perceptron(X, y, lr=0.1, epochs=20):
    W = rng.normal(0, 0.1, X.shape[1])
    b = 0.0
    for _ in range(epochs):
        for xi, yi in zip(X, y):
            pred = int(xi @ W + b > 0)
            err = yi - pred
            W += lr * err * xi
            b += lr * err
    return W, b


# Linearly separable problem: AND
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y_and = np.array([0, 0, 0, 1])
W, b = train_perceptron(X_and, y_and)
print("AND learned:", perceptron_predict(W, b, X_and).tolist())  # [0, 0, 0, 1]

# XOR — NOT linearly separable
X_xor = X_and
y_xor = np.array([0, 1, 1, 0])
W_xor, b_xor = train_perceptron(X_xor, y_xor, epochs=200)
print("XOR by perceptron (will fail):",
      perceptron_predict(W_xor, b_xor, X_xor).tolist())


# ───────────────────────────────────────────────────────────
# A tiny 2-layer MLP, hand-trained on XOR
# ───────────────────────────────────────────────────────────
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def mlp_forward(X, W1, b1, W2, b2):
    h = sigmoid(X @ W1 + b1)
    return sigmoid(h @ W2 + b2), h


def train_mlp_xor():
    """Tiny MLP: 2 → 4 → 1, sigmoid activations, MSE loss."""
    W1 = rng.normal(0, 1, (2, 4))
    b1 = np.zeros(4)
    W2 = rng.normal(0, 1, (4, 1))
    b2 = np.zeros(1)

    lr = 1.0
    losses = []
    for step in range(5000):
        out, h = mlp_forward(X_xor, W1, b1, W2, b2)
        loss = ((out.ravel() - y_xor) ** 2).mean()
        losses.append(loss)

        # Backprop (worked out by hand for sigmoid + MSE)
        d_out  = 2 * (out.ravel() - y_xor) * out.ravel() * (1 - out.ravel())
        d_W2   = h.T @ d_out[:, None] / len(y_xor)
        d_b2   = d_out.mean()
        d_h    = (d_out[:, None] @ W2.T) * h * (1 - h)
        d_W1   = X_xor.T @ d_h / len(y_xor)
        d_b1   = d_h.mean(axis=0)

        W2 -= lr * d_W2
        b2 -= lr * d_b2
        W1 -= lr * d_W1
        b1 -= lr * d_b1

    return W1, b1, W2, b2, losses


W1, b1, W2, b2, losses = train_mlp_xor()
out, _ = mlp_forward(X_xor, W1, b1, W2, b2)
print("XOR by 2-layer MLP:", (out.ravel() > 0.5).astype(int).tolist())


# ───────────────────────────────────────────────────────────
# Visualize the decision boundary
# ───────────────────────────────────────────────────────────
def plot_boundary(ax, predict_fn, title):
    xx, yy = np.meshgrid(np.linspace(-0.2, 1.2, 200), np.linspace(-0.2, 1.2, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = predict_fn(grid).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    ax.scatter(X_xor[:, 0], X_xor[:, 1], c=y_xor, cmap="coolwarm",
               edgecolors="k", s=100)
    ax.set_title(title)


fig, axes = plt.subplots(1, 3, figsize=(13, 4))
plot_boundary(axes[0], lambda g: perceptron_predict(W_xor, b_xor, g), "Perceptron on XOR")
plot_boundary(axes[1], lambda g: (mlp_forward(g, W1, b1, W2, b2)[0] > 0.5).astype(int).ravel(),
              "MLP on XOR")
axes[2].plot(losses)
axes[2].set_title("MLP training loss")
axes[2].set_xlabel("step")
axes[2].set_yscale("log")
axes[2].grid(True, alpha=0.3)

fig.tight_layout()
out_dir = Path(__file__).parent / "figures"
out_dir.mkdir(exist_ok=True)
fig.savefig(out_dir / "01_perceptron_mlp.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"\nfigure saved to: {out_dir / '01_perceptron_mlp.png'}")


# ───────────────────────────────────────────────────────────
# Why depth?
# ───────────────────────────────────────────────────────────
# • A 1-layer perceptron is a linear classifier — same as logistic regression.
# • A 2-layer MLP with non-linear activations is a UNIVERSAL approximator
#   in theory: given enough hidden units, it can approximate any continuous
#   function. (In practice, "enough" can be huge.)
# • Deeper nets compose features hierarchically: low-level → mid-level →
#   high-level. That's the win — depth makes complex functions LEARNABLE
#   with reasonable parameter counts.
