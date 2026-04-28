"""
02 — Broadcasting

Broadcasting lets NumPy operate on arrays of *different* shapes by virtually
stretching the smaller one to match the larger. No memory is actually copied.

THE RULES (applied right-to-left, dim by dim):
  1. The two dimensions are equal, OR
  2. One of them is 1 (it gets stretched), OR
  3. One is missing (treated as 1).

If neither condition holds for any pair of dims → ValueError.

Run: python 02_broadcasting.py
"""

import numpy as np


# ───────────────────────────────────────────────────────────
# Scalar + array
# ───────────────────────────────────────────────────────────
a = np.array([1, 2, 3])
print("a + 10 =", a + 10)         # 10 broadcast to [10, 10, 10]


# ───────────────────────────────────────────────────────────
# 2-D + 1-D row
# ───────────────────────────────────────────────────────────
M = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])           # shape (3, 3)
row = np.array([10, 20, 30])        # shape (3,)
print("M + row =\n", M + row)       # `row` is added to every row of M


# ───────────────────────────────────────────────────────────
# 2-D + 1-D column
# ───────────────────────────────────────────────────────────
# To broadcast as a COLUMN, add a length-1 axis with [:, None] or np.newaxis.
col = np.array([100, 200, 300])         # shape (3,)
col_2d = col[:, np.newaxis]             # shape (3, 1)
print("col_2d shape:", col_2d.shape)
print("M + col =\n", M + col_2d)        # added to each COLUMN of M


# ───────────────────────────────────────────────────────────
# Outer product via broadcasting
# ───────────────────────────────────────────────────────────
# (3, 1) * (1, 4) → (3, 4)
v1 = np.arange(1, 4)[:, None]       # column
v2 = np.arange(1, 5)[None, :]       # row
print("\nouter product:\n", v1 * v2)


# ───────────────────────────────────────────────────────────
# Real-world pattern: standardize each column
# ───────────────────────────────────────────────────────────
# (rows are samples, columns are features)
data = np.array([[1.0, 100.0],
                 [2.0, 200.0],
                 [3.0, 300.0],
                 [4.0, 400.0]])

mean = data.mean(axis=0)            # shape (2,) → one mean per column
std  = data.std(axis=0)             # shape (2,)
standardized = (data - mean) / std  # broadcasts to shape (4, 2)
print("standardized:\n", standardized)
print("each column has mean ≈ 0 and std ≈ 1:")
print("  means:", standardized.mean(axis=0).round(6))
print("  stds: ", standardized.std(axis=0).round(6))


# ───────────────────────────────────────────────────────────
# When broadcasting FAILS
# ───────────────────────────────────────────────────────────
A = np.zeros((3, 4))
B = np.zeros((3,))         # last dim 3 vs A's last dim 4 → can't align
try:
    A + B
except ValueError as e:
    print("\nas expected:", e)

# Fix by reshaping to (3, 1) — now broadcastable along the column axis
print("with reshape:", (A + B[:, None]).shape)
