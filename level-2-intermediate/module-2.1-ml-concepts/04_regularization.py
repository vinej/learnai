"""
04 — Regularization: keeping models honest

Regularization adds a penalty on model complexity to the loss function.
The model is then minimizing  loss + α * complexity, so it can't grow huge
weights without paying for them.

Two classic forms (linear regression flavors):

  RIDGE  (L2):  penalty = α * Σ wᵢ²
                pulls weights TOWARD ZERO but doesn't usually hit zero.
  LASSO  (L1):  penalty = α * Σ |wᵢ|
                can SET WEIGHTS EXACTLY TO ZERO → automatic feature selection.

Same idea applies to neural networks (weight decay = L2) and is essential
for high-dimensional data.

Run: python 04_regularization.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


rng = np.random.default_rng(0)


# ───────────────────────────────────────────────────────────
# Build a dataset with many features but only a few that matter
# ───────────────────────────────────────────────────────────
X, y, true_coef = make_regression(
    n_samples=200, n_features=20,
    n_informative=5,           # only 5 of the 20 features actually relate to y
    noise=10.0,
    coef=True,
    random_state=0,
)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# ───────────────────────────────────────────────────────────
# Compare three models
# ───────────────────────────────────────────────────────────
models = {
    "Linear (no reg)": LinearRegression(),
    "Ridge α=1.0":     Ridge(alpha=1.0),
    "Ridge α=10":      Ridge(alpha=10),
    "Lasso α=1.0":     Lasso(alpha=1.0),
}

print(f"{'model':<20} {'train MSE':>10} {'test MSE':>10} {'# nonzero':>10}")
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    train_mse = mean_squared_error(y_train, model.predict(X_train))
    test_mse  = mean_squared_error(y_test,  model.predict(X_test))
    nonzero = int(np.sum(np.abs(model.coef_) > 1e-3))
    print(f"{name:<20} {train_mse:>10.2f} {test_mse:>10.2f} {nonzero:>10}")
    results[name] = model.coef_


# ───────────────────────────────────────────────────────────
# Plot coefficient values
# ───────────────────────────────────────────────────────────
# Notice:
#   • Linear has 20 nonzero coefficients (some matching noise).
#   • Ridge shrinks them all toward zero — but rarely AT zero.
#   • Lasso zeros most of them out, keeping only the truly informative ones.
fig, ax = plt.subplots(figsize=(11, 5))
indices = np.arange(20)
width = 0.2

for i, (name, coef) in enumerate(results.items()):
    ax.bar(indices + i * width - 1.5 * width, coef, width=width, label=name)

ax.bar(indices + 2.5 * width, true_coef, width=width, color="black",
       alpha=0.4, label="true coef")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlabel("feature index")
ax.set_ylabel("coefficient")
ax.set_title("Coefficient comparison: how regularization shrinks weights")
ax.legend()
ax.grid(True, alpha=0.3)

out = Path(__file__).parent / "figures"
out.mkdir(exist_ok=True)
fig.savefig(out / "04_regularization.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"\nfigure saved to: {out / '04_regularization.png'}")


# ───────────────────────────────────────────────────────────
# Picking α
# ───────────────────────────────────────────────────────────
# α is a hyperparameter — sweep it with cross-validation. scikit-learn
# has built-in helpers: RidgeCV, LassoCV, ElasticNetCV. They try a grid
# of α values and return the best one in a single fit() call.
from sklearn.linear_model import LassoCV
lasso_cv = LassoCV(alphas=np.logspace(-3, 1, 20), cv=5).fit(X_train, y_train)
print(f"\nLassoCV best α: {lasso_cv.alpha_:.4f}")
print(f"  test MSE: {mean_squared_error(y_test, lasso_cv.predict(X_test)):.2f}")
print(f"  # nonzero coefs: {int(np.sum(np.abs(lasso_cv.coef_) > 1e-3))}")


# ───────────────────────────────────────────────────────────
# Other forms of regularization you'll meet later
# ───────────────────────────────────────────────────────────
# • Elastic Net — combines L1 and L2.
# • Dropout (deep learning) — randomly zeros activations during training.
# • Weight decay (deep learning) — equivalent to L2 in optimizers like AdamW.
# • Early stopping — halt training when validation loss stops improving.
# • Data augmentation — implicit regularization via more training variety.
