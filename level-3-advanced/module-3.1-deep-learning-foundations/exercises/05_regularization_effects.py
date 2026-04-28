"""
Exercise 5 — Add dropout / batch norm and observe overfitting reduction

Train the SAME big MLP four ways on a small dataset (so it can overfit):
  A. base (no regularization)
  B. + weight decay 1e-2
  C. + dropout 0.3
  D. + batch norm + dropout + weight decay (everything)

Record train and validation loss curves. Assert that D has a smaller
train/val gap than A.

Run: python exercises/05_regularization_effects.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn


torch.manual_seed(0)


def make_overfittable_problem(n=200, in_dim=50):
    g = torch.Generator().manual_seed(0)
    true_w = torch.randn(in_dim, generator=g)
    X = torch.randn(n, in_dim, generator=g)
    y = X @ true_w[:5] + 0.5 * torch.randn(n, generator=g)
    return X, y


X, y = make_overfittable_problem()
split = int(0.7 * len(X))
X_train, y_train = X[:split], y[:split]
X_val, y_val     = X[split:], y[split:]


def base_model():
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


def all_in_model(p=0.3):
    return nn.Sequential(
        nn.Linear(50, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(p),
        nn.Linear(256, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(p),
        nn.Linear(256, 1),
    )


def train_and_record(model, weight_decay=0.0, steps=400):
    torch.manual_seed(0)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    train_curve, val_curve = [], []
    for _ in range(steps):
        model.train()
        opt.zero_grad()
        loss = ((model(X_train).squeeze(-1) - y_train) ** 2).mean()
        loss.backward()
        opt.step()
        train_curve.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_loss = ((model(X_val).squeeze(-1) - y_val) ** 2).mean().item()
        val_curve.append(val_loss)
    return train_curve, val_curve


def main():
    runs = {
        "A. base":              train_and_record(base_model(),          0.0),
        "B. + weight decay":    train_and_record(base_model(),          1e-2),
        "C. + dropout":         train_and_record(dropout_model(0.3),    0.0),
        "D. all (bn+do+wd)":    train_and_record(all_in_model(0.3),     1e-3),
    }

    print(f"{'config':<22}  {'final train':>12}  {'final val':>12}  {'gap':>12}")
    gaps = {}
    for name, (tr, val) in runs.items():
        gap = val[-1] - tr[-1]
        gaps[name] = gap
        print(f"{name:<22}  {tr[-1]:>12.4f}  {val[-1]:>12.4f}  {gap:>12.4f}")

    # ── Plot ──
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for name, (tr, val) in runs.items():
        axes[0].plot(tr, label=name)
        axes[1].plot(val, label=name)
    for ax, title in zip(axes, ["TRAIN loss", "VAL loss"]):
        ax.set_title(title)
        ax.set_yscale("log")
        ax.set_xlabel("step")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = Path(__file__).parent.parent / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "exercise_05_regularization.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nfigure saved to: {out / 'exercise_05_regularization.png'}")

    assert gaps["D. all (bn+do+wd)"] < gaps["A. base"], \
        "regularization should narrow the train/val gap"
    print("regularization reduces overfitting ✅")


if __name__ == "__main__":
    main()


# ── Stretch ─────────────────────────────────────────────────────────────────
# • Try just batch norm (no dropout, no wd) — does it help or hurt here?
# • Plot the average weight magnitude per layer with vs without weight decay.
# • Add early stopping: track best val loss + stop after 30 steps no-improve.
# ─────────────────────────────────────────────────────────────────────────────
