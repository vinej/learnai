"""
Exercise 4 — Compare optimizers and learning rates on the same problem

Train the same MLP on a synthetic regression problem with:
  • SGD at lr ∈ {1e-3, 1e-2, 1e-1}
  • Adam at lr ∈ {1e-4, 1e-3, 1e-2}
  • AdamW at lr=1e-3 with weight_decay=1e-2

Plot all loss curves on one figure.

The script asserts that AdamW lr=1e-3 reaches a lower final loss than
SGD lr=1e-3 — Adam-style methods usually converge faster than vanilla SGD.

Run: python exercises/04_optimizer_comparison.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn


def make_problem(n=1024, in_dim=20):
    g = torch.Generator().manual_seed(0)
    true_w = torch.randn(in_dim, generator=g)
    X = torch.randn(n, in_dim, generator=g)
    y = X @ true_w + 0.1 * torch.randn(n, generator=g)
    return X, y


def make_model(in_dim=20):
    return nn.Sequential(
        nn.Linear(in_dim, 64), nn.ReLU(),
        nn.Linear(64, 64),     nn.ReLU(),
        nn.Linear(64, 1),
    )


def train(opt_factory, X, y, steps=400):
    torch.manual_seed(0)
    model = make_model()
    opt = opt_factory(model.parameters())
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        loss = ((model(X).squeeze(-1) - y) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def main():
    X, y = make_problem()

    runs = {
        "SGD lr=1e-3":  train(lambda p: torch.optim.SGD(p,  lr=1e-3), X, y),
        "SGD lr=1e-2":  train(lambda p: torch.optim.SGD(p,  lr=1e-2), X, y),
        "SGD lr=1e-1":  train(lambda p: torch.optim.SGD(p,  lr=1e-1), X, y),
        "Adam lr=1e-4": train(lambda p: torch.optim.Adam(p, lr=1e-4), X, y),
        "Adam lr=1e-3": train(lambda p: torch.optim.Adam(p, lr=1e-3), X, y),
        "Adam lr=1e-2": train(lambda p: torch.optim.Adam(p, lr=1e-2), X, y),
        "AdamW lr=1e-3 wd=1e-2": train(
            lambda p: torch.optim.AdamW(p, lr=1e-3, weight_decay=1e-2), X, y,
        ),
    }

    print("final loss per config:")
    for name, losses in runs.items():
        print(f"  {name:<30}  {losses[-1]:.5f}")

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, losses in runs.items():
        ax.plot(losses, label=name)
    ax.set_xlabel("step")
    ax.set_ylabel("training MSE")
    ax.set_yscale("log")
    ax.set_title("Optimizer + LR comparison (same model, same data)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    out = Path(__file__).parent.parent / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "exercise_04_optimizers.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nfigure saved to: {out / 'exercise_04_optimizers.png'}")

    assert runs["AdamW lr=1e-3 wd=1e-2"][-1] < runs["SGD lr=1e-3"][-1], \
        "AdamW should converge to a lower loss than SGD at the same lr"
    print("AdamW beats vanilla SGD here ✅")


if __name__ == "__main__":
    main()


# ── Reflection ──────────────────────────────────────────────────────────────
# Notes you might draw from the plot:
#   • SGD with too-small lr crawls; with too-large lr diverges or oscillates.
#   • Adam is more forgiving across LRs (per-parameter adaptive scaling).
#   • AdamW + weight decay generalizes a touch better than plain Adam.
#   • These are loss curves on TRAINING data — for full picture, also plot
#     validation loss and look for overfitting.
# ─────────────────────────────────────────────────────────────────────────────
