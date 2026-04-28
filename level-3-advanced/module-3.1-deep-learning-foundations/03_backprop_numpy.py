"""
03 — Backpropagation, by hand, in NumPy

Backprop = the chain rule applied through every layer of a network.

Forward pass:
    z1 = X @ W1 + b1
    h1 = ReLU(z1)
    z2 = h1 @ W2 + b2
    y_hat = softmax(z2)
    L = cross-entropy(y_hat, y)

Backward pass (compute ∂L/∂param for each weight):
    dL/dz2 = y_hat - y_onehot                  # softmax + CE shortcut
    dL/dW2 = h1.T @ dL/dz2 / N
    dL/db2 = mean(dL/dz2, axis=0)
    dL/dh1 = dL/dz2 @ W2.T
    dL/dz1 = dL/dh1 * (z1 > 0)                 # ReLU derivative
    dL/dW1 = X.T  @ dL/dz1 / N
    dL/db1 = mean(dL/dz1, axis=0)

This is what `loss.backward()` does for you in PyTorch — done here by hand
on a 2-layer MLP for classification.

Run: python 03_backprop_numpy.py
"""

import numpy as np
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


rng = np.random.default_rng(0)


# ───────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────
def relu(z):
    return np.maximum(0, z)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)            # numerical stability
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy(probs, y):
    """Mean negative log-likelihood. y is integer class index."""
    return -np.log(probs[np.arange(len(y)), y] + 1e-12).mean()


def one_hot(y, num_classes):
    out = np.zeros((len(y), num_classes))
    out[np.arange(len(y)), y] = 1
    return out


# ───────────────────────────────────────────────────────────
# 2-layer MLP from scratch
# ───────────────────────────────────────────────────────────
class TinyMLP:
    def __init__(self, in_dim, hidden, out_dim, seed=0):
        # Kaiming-style init: sqrt(2/n_in) for ReLU
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2 / in_dim),  (in_dim, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, np.sqrt(2 / hidden), (hidden, out_dim))
        self.b2 = np.zeros(out_dim)

    def forward(self, X):
        self.X = X
        self.z1 = X @ self.W1 + self.b1
        self.h1 = relu(self.z1)
        self.z2 = self.h1 @ self.W2 + self.b2
        self.probs = softmax(self.z2)
        return self.probs

    def backward(self, y):
        N = len(y)
        y_oh = one_hot(y, self.probs.shape[1])

        # Softmax + CE shortcut: gradient at z2 is just (probs - target).
        dz2 = (self.probs - y_oh) / N
        self.dW2 = self.h1.T @ dz2
        self.db2 = dz2.sum(axis=0)

        dh1 = dz2 @ self.W2.T
        dz1 = dh1 * (self.z1 > 0)              # ReLU derivative is 1 if z>0, else 0
        self.dW1 = self.X.T @ dz1
        self.db1 = dz1.sum(axis=0)

    def step(self, lr):
        self.W1 -= lr * self.dW1
        self.b1 -= lr * self.db1
        self.W2 -= lr * self.dW2
        self.b2 -= lr * self.db2


# ───────────────────────────────────────────────────────────
# Train on a non-linear toy problem
# ───────────────────────────────────────────────────────────
X, y = make_moons(n_samples=1000, noise=0.2, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, stratify=y)

model = TinyMLP(in_dim=2, hidden=16, out_dim=2)

losses = []
for step in range(2000):
    probs = model.forward(X_train)
    loss = cross_entropy(probs, y_train)
    model.backward(y_train)
    model.step(lr=0.5)

    if step % 200 == 0:
        train_acc = accuracy_score(y_train, model.forward(X_train).argmax(axis=1))
        test_acc  = accuracy_score(y_test,  model.forward(X_test).argmax(axis=1))
        print(f"step {step:>4}  loss={loss:.4f}  "
              f"train_acc={train_acc:.3f}  test_acc={test_acc:.3f}")
    losses.append(loss)


# ───────────────────────────────────────────────────────────
# Sanity check the gradient with a numerical approximation
# ───────────────────────────────────────────────────────────
# For one weight, compute (L(w + h) - L(w - h)) / 2h and compare against
# the analytic gradient. They should be very close.
X_small = X_train[:32]
y_small = y_train[:32]

model.forward(X_small)
loss = cross_entropy(model.probs, y_small)
model.backward(y_small)
analytic = model.dW1[0, 0]

h = 1e-5
W1_orig = model.W1.copy()
model.W1[0, 0] = W1_orig[0, 0] + h
loss_plus = cross_entropy(model.forward(X_small), y_small)
model.W1[0, 0] = W1_orig[0, 0] - h
loss_minus = cross_entropy(model.forward(X_small), y_small)
numerical = (loss_plus - loss_minus) / (2 * h)
model.W1 = W1_orig

print(f"\ngradient check: analytic={analytic:.6e}  numerical={numerical:.6e}")
print(f"relative diff: {abs(analytic - numerical) / (abs(numerical) + 1e-9):.6f}")


# ───────────────────────────────────────────────────────────
# Why we don't do this by hand in real life
# ───────────────────────────────────────────────────────────
# • Each new layer = derive its derivative manually + maintain the code.
# • Easy to miscompute one term and silently train a broken model.
# • PyTorch's `autograd` records the forward pass as a graph and applies
#   the chain rule automatically. Same idea, no manual derivatives.
# Next file: how PyTorch tensors and devices work, then autograd in action.
