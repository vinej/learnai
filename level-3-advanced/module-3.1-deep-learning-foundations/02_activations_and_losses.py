"""
02 — Activation functions and loss functions

ACTIVATIONS introduce non-linearity. Without them, stacking layers would
collapse into a single linear transformation.

  ReLU(x)    = max(0, x)        cheap, sparse, the workhorse
  GELU(x)    ≈ x · Φ(x)         smooth ReLU, default in transformers
  Sigmoid(x) = 1/(1+e^-x)       outputs in (0,1) — for binary probabilities
  Tanh(x)                       outputs in (-1,1) — used in RNNs & some heads
  Softmax(x)                    turns a vector into a probability distribution

LOSSES tell the model how wrong it is. Common pairings:
  MSE                ↔ regression with linear output
  BCE (binary CE)    ↔ binary classification with sigmoid output
  CE  (categorical)  ↔ multi-class classification with softmax output

Run: python 02_activations_and_losses.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


# ───────────────────────────────────────────────────────────
# Plot the activation functions
# ───────────────────────────────────────────────────────────
x = np.linspace(-5, 5, 400)
funcs = {
    "ReLU":    np.maximum(0, x),
    "Leaky ReLU (α=0.1)": np.where(x > 0, x, 0.1 * x),
    "GELU":    F.gelu(torch.from_numpy(x)).numpy(),
    "Sigmoid": 1 / (1 + np.exp(-x)),
    "Tanh":    np.tanh(x),
    "SiLU/Swish": F.silu(torch.from_numpy(x)).numpy(),
}

fig, ax = plt.subplots(figsize=(8, 5))
for name, y in funcs.items():
    ax.plot(x, y, label=name)
ax.axhline(0, color="black", linewidth=0.5)
ax.axvline(0, color="black", linewidth=0.5)
ax.set_title("Common activation functions")
ax.legend()
ax.grid(True, alpha=0.3)

out = Path(__file__).parent / "figures"
out.mkdir(exist_ok=True)
fig.savefig(out / "02_activations.png", dpi=120, bbox_inches="tight")
plt.close(fig)


# ───────────────────────────────────────────────────────────
# Softmax — turns logits into a probability distribution
# ───────────────────────────────────────────────────────────
logits = torch.tensor([2.0, 1.0, 0.1])
probs = F.softmax(logits, dim=0)
print(f"logits: {logits.tolist()}")
print(f"probs:  {[f'{p:.3f}' for p in probs.tolist()]}  (sum = {probs.sum():.3f})")


# ───────────────────────────────────────────────────────────
# Loss functions — implemented two ways
# ───────────────────────────────────────────────────────────
# 1. Regression: MSE
y_true_reg = torch.tensor([1.0, 2.0, 3.0])
y_pred_reg = torch.tensor([1.1, 2.5, 3.0])
mse = F.mse_loss(y_pred_reg, y_true_reg)
print(f"\nMSE: {mse:.4f}    (mean of (pred - true)²)")
manual_mse = ((y_pred_reg - y_true_reg) ** 2).mean()
print(f"manual MSE: {manual_mse:.4f}")

# 2. Binary classification: BCE
# Inputs: PROBABILITIES, targets: 0/1.
y_true_bin = torch.tensor([1.0, 0.0, 1.0, 0.0])
y_pred_bin = torch.tensor([0.9, 0.1, 0.7, 0.3])
bce = F.binary_cross_entropy(y_pred_bin, y_true_bin)
print(f"\nBCE: {bce:.4f}")

# In practice, prefer `binary_cross_entropy_with_logits`:
# it takes RAW LOGITS (no sigmoid) and is numerically more stable.
logits_bin = torch.tensor([2.2, -2.2, 0.85, -0.85])
bce_logits = F.binary_cross_entropy_with_logits(logits_bin, y_true_bin)
print(f"BCE-with-logits: {bce_logits:.4f}")

# 3. Multi-class classification: cross-entropy
# `cross_entropy` expects RAW LOGITS + integer class indices.
logits_mc = torch.tensor([
    [2.0, 1.0, 0.1],     # sample 1, 3 classes
    [0.5, 2.5, 0.3],     # sample 2
    [0.1, 0.2, 3.0],     # sample 3
])
y_true_mc = torch.tensor([0, 1, 2])    # correct class indices
ce = F.cross_entropy(logits_mc, y_true_mc)
print(f"\ncross-entropy: {ce:.4f}")
print("(applies log-softmax + NLL; use this directly, NOT softmax + log + NLL)")


# ───────────────────────────────────────────────────────────
# Why the right pairing matters
# ───────────────────────────────────────────────────────────
# • Sigmoid + MSE on classification → flat gradients far from the boundary
#   → training stalls. BCE-with-logits has well-behaved gradients.
# • Softmax + log + NLL by hand → numerical issues. cross_entropy combines
#   them with the log-sum-exp trick.
# • Tanh + MSE for regression with bounded targets in (-1, 1) is fine.
#
# DEFAULTS YOU CAN'T REALLY GO WRONG WITH:
#   regression →            nn.Linear output  + nn.MSELoss
#   binary classification → nn.Linear output  + nn.BCEWithLogitsLoss
#   multi-class →           nn.Linear output  + nn.CrossEntropyLoss


# ───────────────────────────────────────────────────────────
# Why ReLU dominates hidden layers
# ───────────────────────────────────────────────────────────
# • Cheap to compute (one comparison).
# • Doesn't saturate for positive inputs → larger gradients → faster learning.
# • Sparse activations → easier to optimize.
# • The "dying ReLU" problem (units stuck at 0) is usually fixable with
#   better init, smaller learning rates, or Leaky/GELU/SiLU variants.
print(f"\nfigure saved to: {out / '02_activations.png'}")
