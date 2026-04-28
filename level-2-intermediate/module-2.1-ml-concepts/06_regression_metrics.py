"""
06 — Regression metrics

  MAE  (Mean Absolute Error):     mean(|y − ŷ|)
  MSE  (Mean Squared Error):      mean((y − ŷ)²)
  RMSE (Root MSE):                √MSE
  R²   (coefficient of determination): 1 − SSE/SST  (1.0 = perfect, 0 = baseline, negative = worse than the mean)

WHEN TO USE WHAT
  • MAE  — robust to outliers. Same units as y. Most interpretable.
  • RMSE — penalizes big errors more (squares them first). Common default.
  • MSE  — used in optimization; reading it is harder because of squared units.
  • R²   — relative metric; "how much of the variance does my model explain?"

Run: python 06_regression_metrics.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


# California housing — predict median house value (in $100k units)
data = fetch_california_housing()
X, y = data.data, data.target
print(f"features: {data.feature_names}")
print(f"target stats: mean={y.mean():.2f}, std={y.std():.2f}, range=[{y.min():.2f}, {y.max():.2f}]")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = make_pipeline(StandardScaler(), LinearRegression())
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# ───────────────────────────────────────────────────────────
# Compute the metrics
# ───────────────────────────────────────────────────────────
mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2   = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

print(f"\nMAE  = {mae:.3f}     (typical error in target units)")
print(f"MSE  = {mse:.3f}")
print(f"RMSE = {rmse:.3f}    (typical error, but penalizes large errors)")
print(f"R²   = {r2:.3f}      (1.0 = perfect; 0 = predicting the mean)")
print(f"MAPE = {mape:.3f}    (mean absolute percentage error)")


# ───────────────────────────────────────────────────────────
# Sanity check vs a baseline
# ───────────────────────────────────────────────────────────
# A model that always predicts the training-mean is the trivial baseline.
baseline_pred = np.full_like(y_test, y_train.mean())
baseline_rmse = mean_squared_error(y_test, baseline_pred) ** 0.5
print(f"\nbaseline (always predict mean) RMSE: {baseline_rmse:.3f}")
print(f"our model is {(1 - rmse / baseline_rmse) * 100:.1f}% better than baseline")


# ───────────────────────────────────────────────────────────
# Diagnostics: predicted-vs-actual and residuals
# ───────────────────────────────────────────────────────────
# A good model:
#   • Predicted vs actual lies along the diagonal y = x.
#   • Residuals (y − ŷ) look like random noise centered on 0, with
#     no pattern vs the prediction.
residuals = y_test - y_pred

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].scatter(y_test, y_pred, alpha=0.3, s=10)
limits = [y_test.min(), y_test.max()]
axes[0].plot(limits, limits, "r--", linewidth=1)
axes[0].set_xlabel("actual")
axes[0].set_ylabel("predicted")
axes[0].set_title("Predicted vs actual")
axes[0].grid(True, alpha=0.3)

axes[1].scatter(y_pred, residuals, alpha=0.3, s=10)
axes[1].axhline(0, color="red", linestyle="--")
axes[1].set_xlabel("predicted")
axes[1].set_ylabel("residual (y − ŷ)")
axes[1].set_title("Residuals — should be patternless")
axes[1].grid(True, alpha=0.3)

fig.tight_layout()
out = Path(__file__).parent / "figures"
out.mkdir(exist_ok=True)
fig.savefig(out / "06_regression_metrics.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"\nfigure saved to: {out / '06_regression_metrics.png'}")


# ───────────────────────────────────────────────────────────
# Negative R²?
# ───────────────────────────────────────────────────────────
# R² < 0 means your model is WORSE than predicting the mean. Almost always
# a sign of data leakage in the wrong direction, train/test distribution
# shift, or a bug — not "the model is just bad".
