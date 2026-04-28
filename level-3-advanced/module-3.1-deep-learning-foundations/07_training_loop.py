"""
07 — The PyTorch training loop

The six-line dance you'll write a thousand times:

    optimizer.zero_grad()       # 1. clear stale grads
    output = model(batch_x)     # 2. forward pass
    loss = criterion(output, y) # 3. compute loss
    loss.backward()             # 4. compute grads via autograd
    optimizer.step()            # 5. update parameters
    # (6. log metrics)

Every PyTorch program is a variation on this. This file shows it on a small
synthetic regression problem so you can see every part.

Run: python 07_training_loop.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn


torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"


# ───────────────────────────────────────────────────────────
# Synthetic regression dataset
# ───────────────────────────────────────────────────────────
# y = sin(x) + small noise. Train an MLP to fit it.
N = 500
X = torch.linspace(-2 * torch.pi, 2 * torch.pi, N).reshape(-1, 1)
y_true = torch.sin(X).squeeze(-1)
y = y_true + 0.1 * torch.randn(N)

# 80/20 split
split = int(0.8 * N)
X_train, y_train = X[:split].to(device), y[:split].to(device)
X_test,  y_test  = X[split:].to(device), y[split:].to(device)


# ───────────────────────────────────────────────────────────
# Model
# ───────────────────────────────────────────────────────────
model = nn.Sequential(
    nn.Linear(1, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
).to(device)


# ───────────────────────────────────────────────────────────
# Loss + optimizer
# ───────────────────────────────────────────────────────────
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)


# ───────────────────────────────────────────────────────────
# THE LOOP
# ───────────────────────────────────────────────────────────
n_epochs = 500
train_losses, test_losses = [], []

for epoch in range(n_epochs):
    model.train()                                # training mode
    optimizer.zero_grad()
    pred = model(X_train).squeeze(-1)
    loss = criterion(pred, y_train)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    # Evaluate every 20 epochs
    if epoch % 20 == 0 or epoch == n_epochs - 1:
        model.eval()                             # turn off dropout/batchnorm noise
        with torch.no_grad():                    # don't waste memory tracking grads
            test_pred = model(X_test).squeeze(-1)
            test_loss = criterion(test_pred, y_test).item()
        test_losses.append((epoch, test_loss))
        if epoch % 100 == 0 or epoch == n_epochs - 1:
            print(f"epoch {epoch:>4}: train_loss={loss.item():.4f}  "
                  f"test_loss={test_loss:.4f}")


# ───────────────────────────────────────────────────────────
# Visualize
# ───────────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    y_pred = model(X.to(device)).cpu().squeeze(-1)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].scatter(X.cpu(), y, s=8, alpha=0.4, label="training data")
axes[0].plot(X.cpu(), y_pred, "r-", linewidth=2, label="MLP prediction")
axes[0].plot(X.cpu(), y_true, "g--", linewidth=1.5, label="true sin(x)")
axes[0].set_title("MLP fit to noisy sine")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(train_losses, label="train")
test_epochs, test_vals = zip(*test_losses)
axes[1].plot(test_epochs, test_vals, "o-", label="test")
axes[1].set_xlabel("epoch")
axes[1].set_ylabel("MSE")
axes[1].set_yscale("log")
axes[1].set_title("Loss curves")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

fig.tight_layout()
out = Path(__file__).parent / "figures"
out.mkdir(exist_ok=True)
fig.savefig(out / "07_training_loop.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"\nfigure saved to: {out / '07_training_loop.png'}")


# ───────────────────────────────────────────────────────────
# Anatomy of every PyTorch script you'll write
# ───────────────────────────────────────────────────────────
# 1. Pick a device (CPU / CUDA / MPS).
# 2. Build a Dataset and DataLoader (file 10).
# 3. Build the Model (subclass nn.Module).
# 4. Pick a loss (criterion) and an optimizer.
# 5. Loop over epochs and minibatches:
#       - move batch to device
#       - zero_grad → forward → loss → backward → step
#       - log metrics
# 6. Periodically evaluate on the validation set in eval() mode + no_grad.
# 7. Save checkpoints, plot losses, ship the model.
