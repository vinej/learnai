"""
Exercise 1 — Dot product and matrix multiplication from scratch

Implement these without using numpy's @, np.dot, or np.matmul. The point is
to internalize the index gymnastics. After your implementations work,
the script verifies them against numpy.

Run: python exercises/01_dot_and_matmul_from_scratch.py
"""

import numpy as np


def dot_product(v: list[float], w: list[float]) -> float:
    """Compute v · w = sum(v[i] * w[i]). Raises if shapes don't match."""
    if len(v) != len(w):
        raise ValueError(f"length mismatch: {len(v)} vs {len(w)}")
    # ▼▼▼ Your code here ▼▼▼
    total = 0.0
    for a, b in zip(v, w):
        total += a * b
    return total


def matmul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """
    Compute C = A @ B.
      A is m × k
      B is k × n
      C is m × n,  where C[i][j] = sum_k A[i][k] * B[k][j]
    """
    m, k = len(A), len(A[0])
    k2, n = len(B), len(B[0])
    if k != k2:
        raise ValueError(f"inner dims must match: A is {m}x{k}, B is {k2}x{n}")

    # ▼▼▼ Your code here ▼▼▼
    C = [[0.0 for _ in range(n)] for _ in range(m)]
    for i in range(m):
        for j in range(n):
            s = 0.0
            for kk in range(k):
                s += A[i][kk] * B[kk][j]
            C[i][j] = s
    return C


# ─────────────────────────────────────────────
# Tests against numpy (do not edit below)
# ─────────────────────────────────────────────
def test():
    rng = np.random.default_rng(0)

    # Dot product
    for _ in range(10):
        n = rng.integers(1, 20)
        a = rng.standard_normal(n)
        b = rng.standard_normal(n)
        expected = float(a @ b)
        got = dot_product(a.tolist(), b.tolist())
        assert abs(got - expected) < 1e-9, f"dot mismatch: got {got}, expected {expected}"

    # Matrix multiply
    for _ in range(10):
        m = int(rng.integers(1, 6))
        k = int(rng.integers(1, 6))
        n = int(rng.integers(1, 6))
        A = rng.standard_normal((m, k))
        B = rng.standard_normal((k, n))
        expected = A @ B
        got = np.array(matmul(A.tolist(), B.tolist()))
        assert np.allclose(got, expected), "matmul mismatch"

    print("all tests pass ✅")


if __name__ == "__main__":
    test()


# Bonus thought: matmul is O(m·k·n). For two 1000×1000 matrices that's
# 10^9 multiplies. NumPy's BLAS-backed matmul does it in ~10 ms. Your
# pure-Python version above would take many seconds. THAT is why we use
# vectorized libraries.
