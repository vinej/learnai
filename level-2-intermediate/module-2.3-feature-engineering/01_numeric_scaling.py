"""
01 — Scaling numeric features

Most ML models care about feature SCALE:
  • Distance-based (KNN, K-Means, SVM with RBF) — totally dominated by scale.
  • Linear models — coefficients depend on scale, regularization is per-coef.
  • Neural networks — converge much faster with standardized inputs.
  • Tree-based (RF, GBM, XGBoost) — DON'T need scaling. Drop the scaler.

The four standard scalers in scikit-learn:

  StandardScaler   — (x - mean) / std            → mean 0, std 1
  MinMaxScaler     — (x - min) / (max - min)     → range [0, 1]
  MaxAbsScaler     — x / max(|x|)                → range [-1, 1], preserves sparsity
  RobustScaler     — (x - median) / IQR          → robust to outliers

Run: python 01_numeric_scaling.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)


rng = np.random.default_rng(0)


# ───────────────────────────────────────────────────────────
# Build a feature with very different scale + outliers
# ───────────────────────────────────────────────────────────
# 1000 normally-distributed values around 100, plus 10 huge outliers.
core    = rng.normal(loc=100, scale=15, size=1000)
outliers = rng.normal(loc=500, scale=20, size=10)
x = np.concatenate([core, outliers]).reshape(-1, 1)

print(f"raw stats: mean={x.mean():.1f}, std={x.std():.1f}, "
      f"min={x.min():.1f}, max={x.max():.1f}")


# ───────────────────────────────────────────────────────────
# Apply each scaler
# ───────────────────────────────────────────────────────────
scalers = {
    "StandardScaler": StandardScaler(),
    "MinMaxScaler":   MinMaxScaler(),
    "MaxAbsScaler":   MaxAbsScaler(),
    "RobustScaler":   RobustScaler(),
}

stats = {}
for name, scaler in scalers.items():
    x_scaled = scaler.fit_transform(x)
    stats[name] = pd.Series({
        "mean":   x_scaled.mean(),
        "std":    x_scaled.std(),
        "min":    x_scaled.min(),
        "max":    x_scaled.max(),
        "median": np.median(x_scaled),
    })

print("\nstats after scaling:")
print(pd.DataFrame(stats).round(3))


# ───────────────────────────────────────────────────────────
# Visualize the effect on the same data
# ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 5, figsize=(16, 3.5), sharey=False)
axes[0].hist(x, bins=40, color="grey")
axes[0].set_title("raw")
for ax, (name, scaler) in zip(axes[1:], scalers.items()):
    ax.hist(scaler.fit_transform(x), bins=40)
    ax.set_title(name)

fig.suptitle("Effect of each scaler on the same feature (with outliers)")
fig.tight_layout()
out = Path(__file__).parent / "figures"
out.mkdir(exist_ok=True)
fig.savefig(out / "01_scaling.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"\nfigure saved to: {out / '01_scaling.png'}")


# ───────────────────────────────────────────────────────────
# Why RobustScaler exists — outlier sensitivity
# ───────────────────────────────────────────────────────────
# StandardScaler uses mean/std → outliers shift them dramatically.
# MinMaxScaler uses min/max → a single huge outlier squishes everything else.
# RobustScaler uses median/IQR → unaffected by a few extreme values.
print(f"\noutlier impact on each scaler:")
for name, scaler in scalers.items():
    no_outliers = scaler.fit_transform(core.reshape(-1, 1)).ravel()
    with_outliers = scaler.fit_transform(x).ravel()[:1000]   # same core points
    diff = np.abs(with_outliers - no_outliers).mean()
    print(f"  {name:<16}  avg shift on core points: {diff:.4f}")


# ───────────────────────────────────────────────────────────
# Rules of thumb
# ───────────────────────────────────────────────────────────
# • Default: StandardScaler.
# • Lots of outliers: RobustScaler.
# • Sparse data (many zeros, e.g. text features): MaxAbsScaler — preserves sparsity.
# • Need outputs in [0, 1] (e.g., for some neural nets): MinMaxScaler.
# • Tree-based models: skip scaling entirely.
#
# CRITICAL: fit on the TRAINING data only, then transform train AND test.
# Use a Pipeline (see Module 2.2.07_pipelines) and you don't have to think
# about it.
