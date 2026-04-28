"""
Exercise 3 — Replace a Python loop with NumPy

Compute the z-score normalization of an array:

    z[i] = (x[i] - mean(x)) / std(x)

Two implementations:
  1. A naive Python loop.
  2. A vectorized NumPy version.

Run on a 5,000,000-element array and compare the times.

Run: python exercises/03_vectorize_loop.py
"""

import time

import numpy as np


def zscore_loop(x: np.ndarray) -> np.ndarray:
    n = len(x)

    # mean
    s = 0.0
    for v in x:
        s += v
    mean = s / n

    # std
    sq = 0.0
    for v in x:
        sq += (v - mean) ** 2
    std = (sq / n) ** 0.5

    # apply
    out = np.empty_like(x)
    for i in range(n):
        out[i] = (x[i] - mean) / std
    return out


def zscore_vectorized(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / x.std()


def time_it(label: str, fn, *args) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    result = fn(*args)
    elapsed = time.perf_counter() - t0
    print(f"  {label:<22} {elapsed:.3f}s")
    return result, elapsed


def main():
    rng = np.random.default_rng(0)

    # Use a smaller size for the loop version (it's slow!) and a bigger one
    # for the vectorized version to highlight scalability.
    small = rng.normal(size=200_000)
    big   = rng.normal(size=5_000_000)

    print(f"loop version on {len(small):,} elements:")
    z_loop, t_loop = time_it("python loop:", zscore_loop, small)
    z_vec_small, _ = time_it("vectorized:", zscore_vectorized, small)
    assert np.allclose(z_loop, z_vec_small)

    print(f"\nvectorized on {len(big):,} elements:")
    _, t_vec = time_it("vectorized:", zscore_vectorized, big)

    print(f"\nspeedup at 200k elements: ~{t_loop / time_it('  warmup', zscore_vectorized, small)[1]:.0f}×")
    print(f"vectorized on 5M elements is still well under a second; "
          f"the loop on 5M elements would be very painful.")


if __name__ == "__main__":
    main()


# ── Lessons ──────────────────────────────────────────────────────────────────
# • Python `for` loops in numerics are usually 50-200× slower than NumPy.
# • Vectorize whenever you can. If you can't, look at numba / Cython.
# • In Pandas, the same lesson applies: prefer vectorized column ops over
#   .apply() with a Python function, which loops under the hood.
# ─────────────────────────────────────────────────────────────────────────────
