"""
06 — Probability: distributions and sampling

A *probability distribution* assigns probabilities to possible outcomes.
The ones you'll see most:

  • Bernoulli(p)   — single coin flip, success prob p     (discrete)
  • Binomial(n,p)  — number of successes in n flips       (discrete)
  • Uniform(a,b)   — every value in [a,b] equally likely  (continuous)
  • Normal(μ,σ)    — bell curve, mean μ, std σ            (continuous)

NumPy's modern API: `rng = np.random.default_rng(seed)`.

Run: python 06_probability.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


rng = np.random.default_rng(seed=42)   # seeded for reproducibility


# --- Sample from each distribution ---
n = 10_000

bernoulli = rng.binomial(n=1, p=0.3, size=n)        # 1 trial, p=0.3
binomial  = rng.binomial(n=10, p=0.5, size=n)       # 10 trials, p=0.5
uniform   = rng.uniform(low=-1, high=1, size=n)
normal    = rng.normal(loc=0, scale=1, size=n)      # mean 0, std 1

print("Bernoulli(0.3) sample mean:", bernoulli.mean())   # should be ~0.3
print("Binomial(10, 0.5) sample mean:", binomial.mean()) # should be ~5
print("Uniform(-1, 1)    sample mean:", uniform.mean())  # should be ~0
print("Normal(0, 1)      sample mean:", normal.mean())   # should be ~0
print("Normal(0, 1)      sample std :", normal.std())    # should be ~1


# --- Plot histograms ---
fig, axes = plt.subplots(2, 2, figsize=(10, 7))

axes[0, 0].hist(bernoulli, bins=[-0.5, 0.5, 1.5], rwidth=0.7)
axes[0, 0].set_title("Bernoulli(p=0.3)")
axes[0, 0].set_xticks([0, 1])

axes[0, 1].hist(binomial, bins=range(12), rwidth=0.8, align="left")
axes[0, 1].set_title("Binomial(n=10, p=0.5)")

axes[1, 0].hist(uniform, bins=40)
axes[1, 0].set_title("Uniform(-1, 1)")

axes[1, 1].hist(normal, bins=40, density=True, alpha=0.7)
# Overlay the analytical PDF for the normal distribution.
xs = np.linspace(-4, 4, 200)
pdf = (1 / np.sqrt(2 * np.pi)) * np.exp(-xs ** 2 / 2)
axes[1, 1].plot(xs, pdf, "r-", linewidth=2, label="theoretical PDF")
axes[1, 1].set_title("Normal(0, 1)")
axes[1, 1].legend()

fig.suptitle(f"{n:,} samples per distribution")
fig.tight_layout()

out = Path(__file__).parent / "figures"
out.mkdir(exist_ok=True)
path = out / "06_probability.png"
fig.savefig(path, dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"\nfigure saved to: {path}")


# --- Tip: always seed your RNG in research code ---
# Reproducibility means you (and reviewers) can rerun and get the same
# numbers. Without seeds, every run produces different random samples.
