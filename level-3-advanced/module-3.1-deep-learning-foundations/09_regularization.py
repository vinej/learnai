"""
09 — Regularization for neural networks

Without regularization, big networks memorize the training data and
generalize poorly. Standard techniques:

  WEIGHT DECAY (L2)   — penalize large weights. Free in AdamW.
  DROPOUT             — randomly zero a fraction of activations during training.
                        Forces redundancy. Disabled at eval time.
  BATCH NORM          — normalize activations to mean 0 / var 1 per batch,
                        then learn an affine. Stabilizes training, sometimes
                        regularizes as a side effect.
  LAYER NORM          — like batch norm but per-sample (used in transformers).
  EARLY STOPPING      — halt when validation loss stops improving.
  DATA AUGMENTATION   — train on perturbed copies of the input. Module 3.2.

Run: python 09_regularization.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn


torch.manual_seed(0)


# ───────────────────────────────────────────────────────────
# A small but overfittable problem
# ───────────────────────────────────────────────────────────
N = 200
X = torch.randn(N, 50)
y = (X[:, 0] + 0.5 * X[:, 1] - 0.2 * X[:, 2] + 0.1 * torch.randn(N))

split = int(0.7 * N)
X_train, y_train = X[:split], y[:split]
X_val,   y_val   = X[split:], y[split:]


# ───────────────────────────────────────────────────────────
# Models
# ───────────────────────────────────────────────────────────
def base_model():
    """Big model, no regularization — guaranteed to overfit on 140 samples."""
    return nn.Sequential(
        nn.Linear(50, 256), nn.ReLU(),
        nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, 1),
    )


def dropout_model(p=0.3):
    return nn.Sequential(
        nn.Linear(50, 256), nn.ReLU(), nn.Dropout(p),
        nn.Linear(256, 256), nn.ReLU(), nn.Dropout(p),
        nn.Linear(256, 1),
    )


def batchnorm_model():
    return nn.Sequential(
        nn.Linear(50, 256), nn.BatchNorm1d(256), nn.ReLU(),
        nn.Linear(256, 256), nn.BatchNorm1d(256), nn.ReLU(),
        nn.Linear(256, 1),
    )


def train(model, weight_decay=0.0, label=""):
    """Train and return train/val loss curves."""
    torch.manual_seed(0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    train_curve, val_curve = [], []

    for _ in range(400):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train).squeeze(-1)
        loss = ((pred - y_train) ** 2).mean()
        loss.backward()
        optimizer.step()
        train_curve.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_loss = ((model(X_val).squeeze(-1) - y_val) ** 2).mean().item()
        val_curve.append(val_loss)

    print(f"  {label:<28}  final train={train_curve[-1]:.4f}  val={val_curve[-1]:.4f}")
    return train_curve, val_curve


# ───────────────────────────────────────────────────────────
# Compare
# ───────────────────────────────────────────────────────────
print("=== regularization comparison ===")
results = {
    "base (overfits)":        train(base_model(),       label="base"),
    "+ weight_decay 1e-2":    train(base_model(), 1e-2, "+ weight decay 1e-2"),
    "+ dropout 0.3":          train(dropout_model(0.3), label="+ dropout 0.3"),
    "+ batch norm":           train(batchnorm_model(),  label="+ batch norm"),
    "+ both (do + bn)":       train(nn.Sequential(
                                    nn.Linear(50, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
                                    nn.Linear(256, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
                                    nn.Linear(256, 1),
                                ), 1e-3, "+ all"),
}


# ───────────────────────────────────────────────────────────
# Plot
# ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for name, (tr, val) in results.items():
    axes[0].plot(tr, label=name)
    axes[1].plot(val, label=name)
for ax, title in zip(axes, ["TRAIN loss", "VALIDATION loss"]):
    ax.set_title(title)
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig.tight_layout()
out = Path(__file__).parent / "figures"
out.mkdir(exist_ok=True)
fig.savefig(out / "09_regularization.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"\nfigure saved to: {out / '09_regularization.png'}")


# ───────────────────────────────────────────────────────────
# Don't forget train() / eval() mode!
# ───────────────────────────────────────────────────────────
# Dropout and BatchNorm behave DIFFERENTLY at train vs eval time:
#   • Dropout zeros activations during train, is a no-op during eval.
#   • BatchNorm uses batch stats during train, running stats during eval.
# Forgetting to switch produces silent, mysterious bugs.
#
# Always:
#   model.train()  before the training loop body
#   model.eval()   before any validation or inference call
#   with torch.no_grad():   inside eval, to skip autograd bookkeeping


# ───────────────────────────────────────────────────────────
# Early stopping
# ───────────────────────────────────────────────────────────
# A simple version:
#   1. Track best val loss + the model's state_dict at that step.
#   2. If val loss hasn't improved for `patience` steps, stop.
#   3. Restore the best state before reporting final metrics.
# In real code, use Lightning / your trainer of choice — they ship a
# `EarlyStopping` callback so you don't have to roll your own.
