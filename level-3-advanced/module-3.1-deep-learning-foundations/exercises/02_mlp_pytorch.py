"""
Exercise 2 — Reimplement Exercise 1 in PyTorch

Same problem (moons), same architecture (2 → 32 → 2), but now PyTorch
handles the gradients. Compare how much shorter the code becomes.

The script asserts the test accuracy ≥ 95%.

Run: python exercises/02_mlp_pytorch.py
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


torch.manual_seed(0)


def main():
    X, y = make_moons(n_samples=2000, noise=0.2, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=0, stratify=y, test_size=0.25,
    )

    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).long()
    X_test_t  = torch.from_numpy(X_test).float()
    y_test_t  = torch.from_numpy(y_test).long()

    # ▼▼▼ Your model & training loop ▼▼▼
    model = nn.Sequential(
        nn.Linear(2, 32),
        nn.ReLU(),
        nn.Linear(32, 2),
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    for epoch in range(2000):
        optimizer.zero_grad()
        logits = model(X_train_t)
        loss = criterion(logits, y_train_t)
        loss.backward()
        optimizer.step()

        if epoch % 200 == 0:
            with torch.no_grad():
                train_acc = (logits.argmax(dim=1) == y_train_t).float().mean().item()
                test_logits = model(X_test_t)
                test_acc = (test_logits.argmax(dim=1) == y_test_t).float().mean().item()
            print(f"epoch {epoch:>4}  loss={loss.item():.4f}  "
                  f"train_acc={train_acc:.3f}  test_acc={test_acc:.3f}")
    # ▲▲▲

    model.eval()
    with torch.no_grad():
        preds = model(X_test_t).argmax(dim=1).numpy()
    test_acc = accuracy_score(y_test, preds)
    print(f"\nfinal test accuracy: {test_acc:.3f}")
    assert test_acc > 0.95, "you should reach > 95% — same as exercise 1"
    print("PyTorch MLP matches the from-scratch version ✅")


if __name__ == "__main__":
    main()


# ── Reflection ──────────────────────────────────────────────────────────────
# Compare the line counts of exercises 1 vs 2:
#   - You wrote no derivatives in exercise 2.
#   - You can swap nn.ReLU for nn.GELU and it still works.
#   - Switching to AdamW with weight_decay=1e-4 is a one-line change.
# This is what frameworks BUY you. The reason exercise 1 exists is so you
# trust them: nothing is magic.
# ─────────────────────────────────────────────────────────────────────────────
