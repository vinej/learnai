"""
02 — CRPS and pinball loss.

CRPS (Continuous Ranked Probability Score) generalizes MAE to
distributional forecasts. For samples {x_1..x_n}:

    CRPS = mean_i |x_i - y| - 0.5 * mean_{i,j} |x_i - x_j|

Lower is better. CRPS reduces to MAE when the forecast is a point.

Pinball loss at level q (see 5.4 file 07):
    L_q = max(q * (y - q_hat), (q - 1) * (y - q_hat))

Coverage diagram: for many quantile pairs (alpha), plot empirical
coverage vs nominal alpha. A perfectly calibrated forecast lies on
the diagonal.

Run: python 02_crps_pinball.py
"""
from __future__ import annotations

import numpy as np


def crps_sample(samples: np.ndarray, y_true: float) -> float:
    """Sample-based CRPS for one observation."""
    s = np.asarray(samples)
    term1 = np.mean(np.abs(s - y_true))
    term2 = 0.5 * np.mean(np.abs(s[:, None] - s[None, :]))
    return float(term1 - term2)


def crps_ensemble(samples: np.ndarray, y_true: np.ndarray) -> float:
    """samples: (n_obs, n_samples), y_true: (n_obs,)"""
    return float(np.mean([crps_sample(samples[i], y_true[i]) for i in range(len(y_true))]))


def pinball_loss(y_true, q_pred, q: float) -> float:
    err = np.asarray(y_true) - np.asarray(q_pred)
    return float(np.mean(np.maximum(q * err, (q - 1) * err)))


def coverage(y_true, lo, hi) -> float:
    return float(np.mean((np.asarray(y_true) >= np.asarray(lo)) &
                          (np.asarray(y_true) <= np.asarray(hi))))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 500
    y = rng.normal(0, 1, n)
    # Perfect probabilistic forecast: each obs has a Gaussian centered on truth.
    # Test: degrade the variance.
    samples_good = y[:, None] + rng.normal(0, 1, (n, 200))
    samples_overconf = y[:, None] + rng.normal(0, 0.5, (n, 200))

    print(f"CRPS, well-calibrated:   {crps_ensemble(samples_good, y):.3f}")
    print(f"CRPS, overconfident:     {crps_ensemble(samples_overconf, y):.3f}")

    q_levels = (0.1, 0.5, 0.9)
    for q in q_levels:
        q_pred = np.quantile(samples_good, q, axis=1)
        print(f"pinball@{q}: {pinball_loss(y, q_pred, q):.4f}")

    lo = np.quantile(samples_good, 0.1, axis=1)
    hi = np.quantile(samples_good, 0.9, axis=1)
    print(f"\n80% empirical coverage: {coverage(y, lo, hi):.3f}    (target 0.80)")
