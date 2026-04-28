"""
07 — Deep learning for time series

For forecasting and classification of time series, you have a CHOICE.

CLASSICAL methods (often the right answer):
  • ARIMA / ETS / Prophet — strong on small data, interpretable.
  • Linear regression on lag features — surprisingly hard to beat.
  • Gradient boosting on engineered features (Module 2.3).

DEEP methods (when you have lots of data and rich features):
  • LSTMs, GRUs (Module 3.3).
  • TCN — Temporal Convolutional Network — dilated 1-D convs with causal
    masking. Often beats LSTMs at long horizons.
  • N-BEATS, NHiTS — pure MLPs with structured "blocks".
  • Temporal Fusion Transformer (TFT) — transformer for multivariate ts
    with categorical, static, and time-varying inputs.

This file builds a small TCN and forecasts a synthetic seasonal series.

Run: python 07_time_series_dl.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


torch.manual_seed(0)
np.random.seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"


# ───────────────────────────────────────────────────────────
# Synthetic series: trend + weekly + yearly seasonality + noise
# ───────────────────────────────────────────────────────────
N = 1000
t = np.arange(N)
series = (
    0.01 * t                                  # trend
    + 5 * np.sin(2 * np.pi * t / 7)           # weekly
    + 3 * np.sin(2 * np.pi * t / 50)          # ~bi-monthly
    + np.random.normal(0, 0.5, N)             # noise
).astype(np.float32)

# Sliding windows for supervised forecasting:
#   given input window of length L, predict the NEXT value.
L = 24


def make_dataset(series, L):
    X = np.stack([series[i:i + L]   for i in range(len(series) - L - 1)])
    y = np.array([series[i + L]      for i in range(len(series) - L - 1)])
    return X, y


X, y = make_dataset(series, L)
split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_test,  y_test  = X[split:], y[split:]
print(f"train: {len(X_train)}, test: {len(X_test)}")


# ───────────────────────────────────────────────────────────
# A small TCN: dilated, causal 1-D convs
# ───────────────────────────────────────────────────────────
class TCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size,
                               padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size,
                               padding=pad, dilation=dilation)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Causal: chop the right side so output length matches input length
        out = self.conv1(x)[..., :x.size(-1)]
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)[..., :x.size(-1)]
        out = self.relu(out)
        return self.dropout(out + x)            # residual


class TCN(nn.Module):
    def __init__(self, channels: int = 16, n_blocks: int = 3):
        super().__init__()
        self.input = nn.Conv1d(1, channels, 1)
        self.blocks = nn.Sequential(*[
            TCNBlock(channels, kernel_size=3, dilation=2 ** i)
            for i in range(n_blocks)
        ])
        self.output = nn.Linear(channels, 1)

    def forward(self, x):
        # x: (B, L) — add channel dim → (B, 1, L)
        h = self.input(x.unsqueeze(1))           # (B, C, L)
        h = self.blocks(h)                        # (B, C, L)
        h = h[..., -1]                            # take LAST step
        return self.output(h).squeeze(-1)


model = TCN().to(device)
print(f"params: {sum(p.numel() for p in model.parameters()):,}")


# ───────────────────────────────────────────────────────────
# Train
# ───────────────────────────────────────────────────────────
opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
loss_fn = nn.MSELoss()

X_train_t = torch.from_numpy(X_train).to(device)
y_train_t = torch.from_numpy(y_train).to(device)
X_test_t  = torch.from_numpy(X_test).to(device)
y_test_t  = torch.from_numpy(y_test).to(device)

for epoch in range(60):
    perm = torch.randperm(X_train_t.size(0))
    bs = 64
    model.train()
    for i in range(0, len(perm), bs):
        idx = perm[i:i + bs]
        opt.zero_grad()
        pred = model(X_train_t[idx])
        loss = loss_fn(pred, y_train_t[idx])
        loss.backward()
        opt.step()
    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            train_loss = loss_fn(model(X_train_t), y_train_t).item()
            test_loss  = loss_fn(model(X_test_t),  y_test_t).item()
        print(f"epoch {epoch:>2}  train MSE {train_loss:.3f}  test MSE {test_loss:.3f}")


# ───────────────────────────────────────────────────────────
# Compare to a "yesterday" baseline
# ───────────────────────────────────────────────────────────
yesterday = X_test[:, -1]
baseline_mse = ((y_test - yesterday) ** 2).mean()
model.eval()
with torch.no_grad():
    pred = model(X_test_t).cpu().numpy()
tcn_mse = ((y_test - pred) ** 2).mean()

print(f"\nbaseline 'predict yesterday' MSE: {baseline_mse:.3f}")
print(f"TCN MSE:                          {tcn_mse:.3f}")
print(f"TCN beats baseline: {tcn_mse < baseline_mse}")


# ───────────────────────────────────────────────────────────
# Visualize forecasts
# ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 4))
window = 200
ax.plot(y_test[:window], label="actual", alpha=0.8)
ax.plot(pred[:window],   label="TCN forecast", alpha=0.8)
ax.plot(yesterday[:window], label="naive (last value)", alpha=0.5, linestyle="--")
ax.set_title("Test-set forecasts (first 200 points)")
ax.legend()
ax.grid(True, alpha=0.3)

out = Path(__file__).parent / "figures"
out.mkdir(exist_ok=True)
fig.savefig(out / "07_tcn.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"\nfigure saved to: {out / '07_tcn.png'}")


# ───────────────────────────────────────────────────────────
# When classical models still win
# ───────────────────────────────────────────────────────────
# • Few hundred training points.
# • Univariate, smooth, with clear seasonality and trend.
# • Need calibrated PROBABILISTIC forecasts (Prophet, ETS).
# • Strict latency / memory budget on edge devices.
#
# When deep wins:
#   • Multivariate inputs (many series + categorical exogenous variables).
#   • Long histories / many series sharing patterns (one model trained
#     across thousands of series — N-BEATS / TFT shine here).
#   • Anomaly detection at scale.


# ───────────────────────────────────────────────────────────
# Libraries to know
# ───────────────────────────────────────────────────────────
# • `darts` — unified API across ARIMA, Prophet, N-BEATS, TFT.
# • `gluonts` — AWS's time-series toolkit, great for probabilistic forecasts.
# • `prophet` — Facebook's classical model with built-in trend/seasonality.
# • `sktime` — sklearn-style API for time-series classification/regression.
