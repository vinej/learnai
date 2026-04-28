"""
03 — Binning, discretization, and polynomial / interaction features

Sometimes the relationship between a feature and the target isn't linear, but
your model is linear. Two cheap fixes:

  BINNING (KBinsDiscretizer) — slice a numeric feature into ranges and treat
                               each range as its own category. Lets a linear
                               model fit a piecewise-constant function.

  POLYNOMIAL FEATURES — add x², x³, x*y. Lets a linear model fit curves and
                        interactions, at the cost of dimensionality.

Run: python 03_binning_polynomials.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer, PolynomialFeatures, StandardScaler


rng = np.random.default_rng(0)


# ───────────────────────────────────────────────────────────
# Synthetic non-linear data
# ───────────────────────────────────────────────────────────
def true_function(x):
    return np.sin(2 * np.pi * x)

n = 200
X = rng.uniform(0, 1, size=(n, 1))
y = true_function(X.ravel()) + rng.normal(0, 0.1, size=n)

x_grid = np.linspace(0, 1, 200).reshape(-1, 1)


# ───────────────────────────────────────────────────────────
# Plain linear regression — no chance on a sine curve
# ───────────────────────────────────────────────────────────
linear = LinearRegression().fit(X, y)


# ───────────────────────────────────────────────────────────
# Binning: split x into 10 quantile bins, one-hot, then linear regression
# ───────────────────────────────────────────────────────────
binned = make_pipeline(
    KBinsDiscretizer(n_bins=10, encode="onehot-dense", strategy="quantile"),
    LinearRegression(),
)
binned.fit(X, y)


# ───────────────────────────────────────────────────────────
# Polynomial features: degree 5
# ───────────────────────────────────────────────────────────
poly = make_pipeline(
    PolynomialFeatures(degree=5, include_bias=False),
    StandardScaler(),
    LinearRegression(),
)
poly.fit(X, y)


# ───────────────────────────────────────────────────────────
# Compare on the test grid
# ───────────────────────────────────────────────────────────
y_true_grid = true_function(x_grid.ravel())
for name, model in [("linear", linear), ("binned linear", binned), ("poly-5", poly)]:
    pred = model.predict(x_grid)
    err = mean_squared_error(y_true_grid, pred)
    print(f"{name:<14}  MSE vs truth: {err:.4f}")


# ───────────────────────────────────────────────────────────
# Plot
# ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(X, y, alpha=0.4, s=15, label="data")
ax.plot(x_grid, y_true_grid, "g--", linewidth=2, label="true function")
ax.plot(x_grid, linear.predict(x_grid), label="linear")
ax.plot(x_grid, binned.predict(x_grid), label="binned linear")
ax.plot(x_grid, poly.predict(x_grid),   label="polynomial degree 5")
ax.set_title("Same linear model, different feature engineering")
ax.legend()
ax.grid(True, alpha=0.3)

out = Path(__file__).parent / "figures"
out.mkdir(exist_ok=True)
fig.savefig(out / "03_binning_polynomial.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"\nfigure saved to: {out / '03_binning_polynomial.png'}")


# ───────────────────────────────────────────────────────────
# Interaction features (no squares — only products of distinct pairs)
# ───────────────────────────────────────────────────────────
# Useful when y depends on COMBINATIONS of inputs.
poly_interact = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
demo = np.array([[2.0, 3.0, 5.0]])
print("\noriginal:    ", demo.ravel())
print("interact deg2:", poly_interact.fit_transform(demo).ravel())
print("              (x1, x2, x3, x1*x2, x1*x3, x2*x3)")


# ───────────────────────────────────────────────────────────
# When NOT to bin or polynomialize
# ───────────────────────────────────────────────────────────
# • Tree-based models already handle non-linearity and interactions natively.
#   Adding binned or polynomial features mostly adds noise + overfitting risk.
# • High-degree polynomial features explode the dimensionality (combinatorial)
#   and amplify outliers. Stick to degrees 2-3.
# • Quantile binning vs uniform binning — quantile is usually safer because
#   each bin holds the same number of samples.
