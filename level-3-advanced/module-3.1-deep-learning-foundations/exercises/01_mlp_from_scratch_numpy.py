"""
Exercise 1 — Implement a 2-layer MLP from scratch in NumPy

This is the "show me you understand backprop" exercise. Build a tiny MLP
without any deep learning library:
  • forward pass
  • cross-entropy loss
  • manual backward pass
  • SGD update

Train on the moons dataset and reach ≥95% test accuracy.

Run: python exercises/01_mlp_from_scratch_numpy.py
"""

import numpy as np
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def relu(z):
    return np.maximum(0, z)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy(probs, y_true):
    return -np.log(probs[np.arange(len(y_true)), y_true] + 1e-12).mean()


def one_hot(y, num_classes):
    out = np.zeros((len(y), num_classes))
    out[np.arange(len(y)), y] = 1
    return out


class MLP:
    """A 2-layer MLP: input → hidden (ReLU) → output (softmax)."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        # Kaiming init for ReLU
        self.W1 = rng.normal(0, np.sqrt(2 / in_dim),  (in_dim, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, np.sqrt(2 / hidden), (hidden, out_dim))
        self.b2 = np.zeros(out_dim)

    def forward(self, X):
        # ▼▼▼ Your code: set self.X, self.z1, self.h1, self.z2, self.probs ▼▼▼
        self.X = X
        self.z1 = X @ self.W1 + self.b1
        self.h1 = relu(self.z1)
        self.z2 = self.h1 @ self.W2 + self.b2
        self.probs = softmax(self.z2)
        return self.probs
        # ▲▲▲

    def backward(self, y_true):
        N = len(y_true)
        y_oh = one_hot(y_true, self.probs.shape[1])

        # ▼▼▼ Your code: compute self.dW1, self.db1, self.dW2, self.db2 ▼▼▼
        # softmax + CE shortcut: d/dz2 = (probs - target) / N
        dz2 = (self.probs - y_oh) / N
        self.dW2 = self.h1.T @ dz2
        self.db2 = dz2.sum(axis=0)

        dh1 = dz2 @ self.W2.T
        dz1 = dh1 * (self.z1 > 0)             # ReLU' is 1 if z>0 else 0
        self.dW1 = self.X.T @ dz1
        self.db1 = dz1.sum(axis=0)
        # ▲▲▲

    def step(self, lr: float):
        self.W1 -= lr * self.dW1
        self.b1 -= lr * self.db1
        self.W2 -= lr * self.dW2
        self.b2 -= lr * self.db2


def main():
    X, y = make_moons(n_samples=2000, noise=0.2, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=0, stratify=y, test_size=0.25,
    )

    model = MLP(in_dim=2, hidden=32, out_dim=2)

    for epoch in range(2000):
        probs = model.forward(X_train)
        loss = cross_entropy(probs, y_train)
        model.backward(y_train)
        model.step(lr=0.5)

        if epoch % 200 == 0:
            train_acc = accuracy_score(y_train, model.forward(X_train).argmax(axis=1))
            test_acc  = accuracy_score(y_test,  model.forward(X_test).argmax(axis=1))
            print(f"epoch {epoch:>4}  loss={loss:.4f}  "
                  f"train_acc={train_acc:.3f}  test_acc={test_acc:.3f}")

    test_acc = accuracy_score(y_test, model.forward(X_test).argmax(axis=1))
    print(f"\nfinal test accuracy: {test_acc:.3f}")
    assert test_acc > 0.95, "you should reach > 95% on moons"
    print("from-scratch MLP works ✅")


if __name__ == "__main__":
    main()


# ── Stretch ─────────────────────────────────────────────────────────────────
# • Add a numerical gradient check on W1[0,0] like file 03 did. They should
#   match within 1e-5.
# • Try different hidden sizes — when does the model start overfitting?
# • Add L2 regularization (weight decay): add λ * w to each weight gradient.
# ─────────────────────────────────────────────────────────────────────────────
