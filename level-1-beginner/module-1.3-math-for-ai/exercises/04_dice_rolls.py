"""
Exercise 4 — Dice rolls and the Law of Large Numbers

Simulate 10,000 rolls of a fair 6-sided die.

Verify:
  • The empirical mean approaches the theoretical mean of 3.5.
  • The empirical variance approaches 35/12 ≈ 2.9167.
  • The face-frequencies are roughly uniform.

Run: python exercises/04_dice_rolls.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main(n_rolls: int = 10_000, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    rolls = rng.integers(low=1, high=7, size=n_rolls)   # high is exclusive

    # ── Stats ──
    empirical_mean = rolls.mean()
    empirical_var  = rolls.var()
    theoretical_mean = 3.5
    theoretical_var  = ((6 ** 2 - 1) / 12)              # = 35/12 ≈ 2.917

    print(f"rolls: {n_rolls}")
    print(f"empirical mean    = {empirical_mean:.4f}   (theoretical {theoretical_mean})")
    print(f"empirical variance= {empirical_var:.4f}   (theoretical {theoretical_var:.4f})")

    counts = np.bincount(rolls)[1:]                     # drop bucket for 0
    expected = n_rolls / 6
    print("\nface counts (expected ~", int(expected), "each):")
    for face, c in enumerate(counts, start=1):
        deviation = (c - expected) / expected * 100
        print(f"  {face}: {c:>5}   ({deviation:+.1f}% off expected)")

    # ── Plot ──
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].bar(range(1, 7), counts, color="steelblue", edgecolor="black")
    axes[0].axhline(expected, color="red", linestyle="--",
                    label=f"expected = {int(expected)}")
    axes[0].set_xlabel("face")
    axes[0].set_ylabel("count")
    axes[0].set_title(f"Histogram over {n_rolls:,} rolls")
    axes[0].legend()

    running = np.cumsum(rolls) / np.arange(1, n_rolls + 1)
    axes[1].plot(running)
    axes[1].axhline(theoretical_mean, color="red", linestyle="--",
                    label="true mean = 3.5")
    axes[1].set_xlabel("rolls")
    axes[1].set_ylabel("running mean")
    axes[1].set_title("Convergence to the true mean (LLN)")
    axes[1].legend()

    fig.tight_layout()
    out = Path(__file__).parent.parent / "figures"
    out.mkdir(exist_ok=True)
    path = out / "exercise_04_dice.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nfigure saved to: {path}")

    # ── Final assertions ──
    assert abs(empirical_mean - theoretical_mean) < 0.1, "mean far from 3.5"
    assert abs(empirical_var - theoretical_var) < 0.2, "variance far from theoretical"
    print("\nLLN sanity checks pass ✅")


if __name__ == "__main__":
    main()
