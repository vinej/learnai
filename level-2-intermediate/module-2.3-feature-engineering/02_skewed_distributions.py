"""
02 — Taming skewed distributions

Many real-world numeric features are heavily skewed (income, prices, response
times, file sizes...). Linear models and many distance-based methods perform
better when features are roughly NORMAL.

Tools:
  log / log1p           — simple, only for positive values, fast
  Box-Cox               — parametric, only for positive values
  Yeo-Johnson           — like Box-Cox but works for any real values
  QuantileTransformer   — maps to a uniform OR normal distribution by ranks

Run: python 02_skewed_distributions.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats
from sklearn.preprocessing import PowerTransformer, QuantileTransformer


rng = np.random.default_rng(0)


# ───────────────────────────────────────────────────────────
# Generate a heavily right-skewed feature (think: income)
# ───────────────────────────────────────────────────────────
n = 5000
income = rng.lognormal(mean=10, sigma=1, size=n)        # > 0, very skewed
print(f"raw income: mean={income.mean():.0f}, "
      f"median={np.median(income):.0f}, "
      f"skew={scipy_stats.skew(income):.2f}")


# ───────────────────────────────────────────────────────────
# Apply each transform
# ───────────────────────────────────────────────────────────
transforms = {
    "log1p":          np.log1p(income),
    "Box-Cox":        PowerTransformer(method="box-cox").fit_transform(income.reshape(-1, 1)).ravel(),
    "Yeo-Johnson":    PowerTransformer(method="yeo-johnson").fit_transform(income.reshape(-1, 1)).ravel(),
    "Quantile→Normal": QuantileTransformer(output_distribution="normal", random_state=0)
                          .fit_transform(income.reshape(-1, 1)).ravel(),
}

print("\nskew after each transform (closer to 0 is more 'normal'):")
print(f"  {'raw':<18} {scipy_stats.skew(income):+.3f}")
for name, vals in transforms.items():
    print(f"  {name:<18} {scipy_stats.skew(vals):+.3f}")


# ───────────────────────────────────────────────────────────
# Plot histograms before / after
# ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(13, 7))

axes[0, 0].hist(income, bins=60, color="grey")
axes[0, 0].set_title(f"raw  (skew={scipy_stats.skew(income):.2f})")

for ax, (name, vals) in zip(axes.flatten()[1:], transforms.items()):
    ax.hist(vals, bins=60)
    ax.set_title(f"{name}  (skew={scipy_stats.skew(vals):.2f})")

fig.tight_layout()
out = Path(__file__).parent / "figures"
out.mkdir(exist_ok=True)
fig.savefig(out / "02_skewed.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"\nfigure saved to: {out / '02_skewed.png'}")


# ───────────────────────────────────────────────────────────
# Yeo-Johnson on values that include negatives / zeros
# ───────────────────────────────────────────────────────────
# Box-Cox fails on x ≤ 0. Yeo-Johnson is the safer default.
mixed = rng.normal(0, 5, 1000)
yj = PowerTransformer(method="yeo-johnson").fit_transform(mixed.reshape(-1, 1))
print(f"\nYeo-Johnson on values with negatives: skew {scipy_stats.skew(mixed):.3f} → "
      f"{scipy_stats.skew(yj.ravel()):.3f}")


# ───────────────────────────────────────────────────────────
# When to use what
# ───────────────────────────────────────────────────────────
# • Strictly positive skewed data → log1p (cheap and good).
# • Skewed with possible zeros/negatives → Yeo-Johnson.
# • Want guaranteed near-normal → QuantileTransformer (output_distribution='normal').
# • Heavy outliers, want roughly uniform values → QuantileTransformer (default).
#
# IMPORTANT: when you predict, you'll often want to invert the transform on
# predictions so they're back in the original units. PowerTransformer and
# QuantileTransformer both support .inverse_transform().
