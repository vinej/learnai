"""
03 — Bias-variance tradeoff, overfitting and underfitting

Two ways a model can be wrong:

  HIGH BIAS (UNDERFIT)
    The model is too simple to capture the pattern.
    Symptom: high training error AND high validation error.

  HIGH VARIANCE (OVERFIT)
    The model is too flexible — it memorized the training set.
    Symptom: very low training error but high validation error.

The sweet spot balances both. We'll demonstrate by fitting polynomials of
increasing degree to a noisy sine curve.

Run: python 03_bias_variance.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


rng = np.random.default_rng(0)

# Synthetic data: a noisy sine wave
def true_function(x):
    return np.sin(2 * np.pi * x)

n = 30
X_train = rng.uniform(0, 1, size=(n, 1))
y_train = true_function(X_train.ravel()) + rng.normal(0, 0.2, n)

X_test = rng.uniform(0, 1, size=(200, 1))
y_test = true_function(X_test.ravel()) + rng.normal(0, 0.2, 200)


# ───────────────────────────────────────────────────────────
# Fit polynomials of degree 1, 4, and 15
# ───────────────────────────────────────────────────────────
degrees = [1, 4, 15]
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

x_plot = np.linspace(0, 1, 200).reshape(-1, 1)

print(f"{'degree':>6}  {'train MSE':>10}  {'test MSE':>10}  diagnosis")
for ax, degree in zip(axes, degrees):
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X_train, y_train)

    train_mse = mean_squared_error(y_train, model.predict(X_train))
    test_mse  = mean_squared_error(y_test,  model.predict(X_test))

    if train_mse > 0.1:
        diag = "HIGH BIAS — underfit"
    elif test_mse > 5 * train_mse:
        diag = "HIGH VARIANCE — overfit"
    else:
        diag = "good fit"
    print(f"{degree:>6}  {train_mse:>10.4f}  {test_mse:>10.4f}  {diag}")

    ax.scatter(X_train, y_train, s=25, alpha=0.7, label="training data")
    ax.plot(x_plot, true_function(x_plot.ravel()), "g--", label="true function", linewidth=2)
    ax.plot(x_plot, model.predict(x_plot), "r-", label=f"degree {degree} fit", linewidth=2)
    ax.set_title(f"degree = {degree}\ntrain MSE={train_mse:.3f}, test MSE={test_mse:.3f}")
    ax.set_ylim(-1.8, 1.8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig.tight_layout()
out = Path(__file__).parent / "figures"
out.mkdir(exist_ok=True)
fig.savefig(out / "03_bias_variance.png", dpi=120, bbox_inches="tight")
plt.close(fig)


# ───────────────────────────────────────────────────────────
# Learning curve: error vs. training-set size
# ───────────────────────────────────────────────────────────
# A second diagnostic. As you add more data:
#   • Underfit model → both errors stay high and CONVERGE close together.
#   • Overfit model → train error stays low, test error stays high (a GAP).
#   • Right-sized model → both errors converge to a low value.
from sklearn.model_selection import learning_curve

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, degree in zip(axes, degrees):
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train, cv=5,
        train_sizes=np.linspace(0.2, 1.0, 6),
        scoring="neg_mean_squared_error",
    )
    train_mse = -train_scores.mean(axis=1)
    val_mse   = -val_scores.mean(axis=1)

    ax.plot(sizes, train_mse, "o-", label="train")
    ax.plot(sizes, val_mse,   "o-", label="validation")
    ax.set_title(f"learning curve, degree {degree}")
    ax.set_xlabel("training-set size")
    ax.set_ylabel("MSE")
    ax.set_ylim(0, max(val_mse.max(), train_mse.max()) * 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(out / "03_learning_curves.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"\nfigures saved to: {out}")


# ───────────────────────────────────────────────────────────
# What to do about each
# ───────────────────────────────────────────────────────────
# UNDERFIT  → use a more flexible model, add features, train longer
# OVERFIT   → simpler model, more data, regularization, dropout, early stopping
