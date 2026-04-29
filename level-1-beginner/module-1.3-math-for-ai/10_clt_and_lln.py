"""
09 — Two pillars of statistics: LLN and CLT

LAW OF LARGE NUMBERS (LLN)
    The sample mean of independent draws converges to the true mean
    as you collect more data. "More data → better estimates."

CENTRAL LIMIT THEOREM (CLT)
    The DISTRIBUTION of the sample mean approaches a NORMAL distribution
    as n grows — *regardless* of the original distribution's shape.
    This is why the bell curve shows up everywhere.

Run: python 09_clt_and_lln.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


rng = np.random.default_rng(seed=42)


# ───────────────────────────────────────────────────────────
# Law of Large Numbers
# ───────────────────────────────────────────────────────────
# Roll a fair 6-sided die many times; track the running average.
# True mean = (1+2+3+4+5+6) / 6 = 3.5
n = 5_000
rolls = rng.integers(1, 7, size=n)        # uniform in {1, 2, 3, 4, 5, 6}
running_mean = np.cumsum(rolls) / np.arange(1, n + 1)

print(f"after  10 rolls: mean = {running_mean[9]:.3f}")
print(f"after 100 rolls: mean = {running_mean[99]:.3f}")
print(f"after {n} rolls: mean = {running_mean[-1]:.3f}   (true: 3.5)")


# ───────────────────────────────────────────────────────────
# Central Limit Theorem
# ───────────────────────────────────────────────────────────
# Take samples from a heavily-skewed distribution (exponential).
# Compute the mean of each sample of size n.
# The DISTRIBUTION of those means becomes normal as n grows.
def sample_means(sample_size, num_experiments=10_000):
    means = np.empty(num_experiments)
    for i in range(num_experiments):
        sample = rng.exponential(scale=1.0, size=sample_size)  # NOT a normal!
        means[i] = sample.mean()
    return means

means_n1   = sample_means(1)              # one draw → looks like the original (skewed)
means_n5   = sample_means(5)              # already starting to look bell-shaped
means_n30  = sample_means(30)             # quite normal
means_n100 = sample_means(100)            # very normal


# ───────────────────────────────────────────────────────────
# Visualize
# ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(11, 7))

# LLN plot
ax = axes[0, 0]
ax.plot(running_mean, label="running mean")
ax.axhline(3.5, color="red", linestyle="--", label="true mean = 3.5")
ax.set_title("LLN: sample mean of dice rolls converges")
ax.set_xlabel("rolls")
ax.set_ylabel("running mean")
ax.legend()
ax.grid(True, alpha=0.3)

# CLT plots
for ax, data, n_size in zip(
    [axes[0, 1], axes[1, 0], axes[1, 1]],
    [means_n1, means_n5, means_n30],
    [1, 5, 30],
):
    ax.hist(data, bins=50, density=True, alpha=0.7)
    ax.set_title(f"CLT: distribution of sample mean (n={n_size})")
    ax.set_xlabel("sample mean")

fig.tight_layout()
out = Path(__file__).parent / "figures"
out.mkdir(exist_ok=True)
path = out / "09_clt_lln.png"
fig.savefig(path, dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"\nfigure saved to: {path}")


# ───────────────────────────────────────────────────────────
# Why this matters for ML
# ───────────────────────────────────────────────────────────
# • LLN justifies "more data is better" — empirical loss → true loss.
# • CLT explains why training loss & metrics often look bell-shaped
#   when averaged across batches/runs, and why confidence intervals
#   on means use the normal distribution.
