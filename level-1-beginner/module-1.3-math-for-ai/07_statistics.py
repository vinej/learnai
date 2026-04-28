"""
07 — Statistics: mean, variance, covariance, correlation

  • Mean (average): the "center" of the data.
  • Variance: how spread out the data is around the mean. Larger = noisier.
  • Standard deviation: sqrt(variance) — same units as the data.
  • Covariance: do two variables move together? Sign matters; magnitude is hard
    to interpret because units differ.
  • Correlation: covariance scaled to [-1, 1]. Unitless and comparable.

Run: python 07_statistics.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


rng = np.random.default_rng(seed=42)


# --- One variable ---
data = rng.normal(loc=100, scale=15, size=1000)

print(f"mean:    {data.mean():.2f}     (true 100)")
print(f"var:     {data.var():.2f}      (true 225)")
print(f"std:     {data.std():.2f}      (true 15)")
print(f"min/max: {data.min():.2f} / {data.max():.2f}")
print(f"median:  {np.median(data):.2f}")


# --- Two variables: covariance and correlation ---
# Build a clearly correlated pair: y = 2x + noise
n = 500
x = rng.normal(0, 1, n)
y = 2 * x + rng.normal(0, 0.5, n)

# np.cov returns the 2×2 covariance matrix:
#   [[var(x),    cov(x,y)],
#    [cov(x,y),  var(y) ]]
cov_matrix = np.cov(x, y)
corr_matrix = np.corrcoef(x, y)

print("\ncovariance matrix:")
print(cov_matrix)
print("correlation matrix:")
print(corr_matrix)
print(f"\ncorrelation between x and y: {corr_matrix[0, 1]:.3f}  "
      "(should be near +1)")


# --- Three contrasting cases for the plot ---
n = 300
xs = rng.normal(0, 1, n)
y_pos = 2 * xs + rng.normal(0, 0.5, n)         # strongly positive
y_neg = -2 * xs + rng.normal(0, 0.5, n)        # strongly negative
y_none = rng.normal(0, 1, n)                   # uncorrelated

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, ys, title in zip(axes,
                         [y_pos, y_neg, y_none],
                         ["positive", "negative", "no correlation"]):
    r = np.corrcoef(xs, ys)[0, 1]
    ax.scatter(xs, ys, alpha=0.5, s=15)
    ax.set_title(f"{title}\nr = {r:.2f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)

fig.tight_layout()
out = Path(__file__).parent / "figures"
out.mkdir(exist_ok=True)
path = out / "07_statistics.png"
fig.savefig(path, dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"\nfigure saved to: {path}")


# --- Important caveat: correlation ≠ causation ---
# Two variables can be highly correlated by coincidence or because both
# depend on a third (lurking) variable. Always look at the data and the
# domain before claiming one drives the other.
