"""
08 — Optimizers

An optimizer turns gradients into parameter updates. The classics:

  SGD         — w ← w − lr * g                                  vanilla
  SGD+Momentum — w ← w − lr * v;   v ← βv + g                   smoother
  RMSProp     — divides g by a running RMS to normalize per-parameter scale
  Adam        — momentum + RMS-style adaptive scaling. The default for years.
  AdamW       — Adam with DECOUPLED weight decay (proper L2). Default today.

Higher-level adjustments:

  Learning rate schedulers    — anneal lr over training.
  Warmup                      — start small, ramp up, then anneal.
  Gradient clipping           — cap |grad| to avoid explosions (RNNs/transformers).

Run: python 08_optimizers.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn


torch.manual_seed(0)


# ───────────────────────────────────────────────────────────
# A small fixed regression problem we can solve many times
# ───────────────────────────────────────────────────────────
def make_problem(n=512, in_dim=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    true_w = torch.randn(in_dim, generator=g)
    X = torch.randn(n, in_dim, generator=g)
    y = X @ true_w + 0.1 * torch.randn(n, generator=g)
    return X, y


def make_model(in_dim=20):
    return nn.Sequential(
        nn.Linear(in_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )


def train(opt_factory, X, y, steps=500, label=""):
    torch.manual_seed(0)             # reset every run for fair comparison
    model = make_model()
    opt = opt_factory(model.parameters())
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        loss = ((model(X).squeeze(-1) - y) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    print(f"  {label:<24}  final loss {losses[-1]:.5f}")
    return losses


X, y = make_problem()


# ───────────────────────────────────────────────────────────
# Compare optimizers at a fixed lr
# ───────────────────────────────────────────────────────────
print("=== same LR (1e-2), 500 steps ===")
results = {
    "SGD":             train(lambda p: torch.optim.SGD(p, lr=1e-2),                  X, y, label="SGD"),
    "SGD+momentum":    train(lambda p: torch.optim.SGD(p, lr=1e-2, momentum=0.9),    X, y, label="SGD+momentum"),
    "RMSProp":         train(lambda p: torch.optim.RMSprop(p, lr=1e-2),              X, y, label="RMSprop"),
    "Adam":            train(lambda p: torch.optim.Adam(p, lr=1e-2),                 X, y, label="Adam"),
    "AdamW (wd=1e-2)": train(lambda p: torch.optim.AdamW(p, lr=1e-2, weight_decay=1e-2), X, y, label="AdamW"),
}


# ───────────────────────────────────────────────────────────
# Effect of learning rate
# ───────────────────────────────────────────────────────────
print("\n=== Adam at different learning rates ===")
lrs = {
    "lr=1e-1":  train(lambda p: torch.optim.Adam(p, lr=1e-1), X, y, label="Adam lr=1e-1"),
    "lr=1e-2":  train(lambda p: torch.optim.Adam(p, lr=1e-2), X, y, label="Adam lr=1e-2"),
    "lr=1e-3":  train(lambda p: torch.optim.Adam(p, lr=1e-3), X, y, label="Adam lr=1e-3"),
    "lr=1e-4":  train(lambda p: torch.optim.Adam(p, lr=1e-4), X, y, label="Adam lr=1e-4"),
}


# ───────────────────────────────────────────────────────────
# LR scheduler — cosine annealing
# ───────────────────────────────────────────────────────────
torch.manual_seed(0)
model = make_model()
opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=500)
sched_losses = []
sched_lrs = []
for _ in range(500):
    opt.zero_grad()
    loss = ((model(X).squeeze(-1) - y) ** 2).mean()
    loss.backward()
    opt.step()
    sched.step()
    sched_losses.append(loss.item())
    sched_lrs.append(opt.param_groups[0]["lr"])
print(f"  AdamW + cosine annealing  final loss {sched_losses[-1]:.5f}")


# ───────────────────────────────────────────────────────────
# Plots
# ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for name, losses in results.items():
    axes[0].plot(losses, label=name)
axes[0].set_yscale("log")
axes[0].set_title("Optimizers @ lr=1e-2")
axes[0].set_xlabel("step")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

for name, losses in lrs.items():
    axes[1].plot(losses, label=name)
axes[1].set_yscale("log")
axes[1].set_title("Adam at different LRs")
axes[1].set_xlabel("step")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(sched_losses, label="loss")
ax2 = axes[2].twinx()
ax2.plot(sched_lrs, "r--", alpha=0.5, label="lr")
axes[2].set_yscale("log")
axes[2].set_title("Cosine annealing LR")
axes[2].set_xlabel("step")
axes[2].legend(loc="upper left")
ax2.legend(loc="upper right")
axes[2].grid(True, alpha=0.3)

fig.tight_layout()
out = Path(__file__).parent / "figures"
out.mkdir(exist_ok=True)
fig.savefig(out / "08_optimizers.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"\nfigure saved to: {out / '08_optimizers.png'}")


# ───────────────────────────────────────────────────────────
# Picking what to use
# ───────────────────────────────────────────────────────────
# Default 90% of the time:                AdamW + cosine schedule
# Reaching peak performance on tabular:   AdamW or SGD+momentum, sweep LR
# Training a transformer / LLM:           AdamW + warmup + cosine, lr ~1e-4
# RNNs:                                   Adam + grad clipping (clip_grad_norm_)
# Massive batch sizes / SOTA work:        LARS / LAMB (specialized)
