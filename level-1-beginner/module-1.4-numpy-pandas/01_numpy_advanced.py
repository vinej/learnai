"""
01 — NumPy: indexing, axes, reductions, useful functions

Module 1.3 covered the basics. This file goes deeper into the moves you'll
make every day with real data.

Run: python 01_numpy_advanced.py
"""

import numpy as np


rng = np.random.default_rng(0)
a = rng.integers(0, 100, size=(4, 5))
print("a =\n", a)


# ───────────────────────────────────────────────────────────
# Fancy indexing
# ───────────────────────────────────────────────────────────
# Pass arrays of indices (or booleans) to pull out arbitrary subsets.

# Integer-array indexing — returns a NEW array (not a view!)
rows_to_pick = [0, 2]
print("rows 0 and 2:\n", a[rows_to_pick])

# Pick specific (row, col) pairs
rows = np.array([0, 1, 2, 3])
cols = np.array([4, 3, 2, 1])
print("diagonal-ish pick:", a[rows, cols])

# Boolean mask
mask = a > 50
print("mask:\n", mask)
print("values > 50:", a[mask])     # always returns 1-D


# ───────────────────────────────────────────────────────────
# Reductions along axes
# ───────────────────────────────────────────────────────────
# axis=0 → reduce ROWS away (one value per column)
# axis=1 → reduce COLS away (one value per row)
# axis=None (default) → reduce everything to a scalar

print("\nshape:", a.shape)
print("a.sum()        :", a.sum())            # all elements
print("a.sum(axis=0)  :", a.sum(axis=0))      # column sums  → shape (5,)
print("a.sum(axis=1)  :", a.sum(axis=1))      # row sums     → shape (4,)
print("a.mean(axis=0) :", a.mean(axis=0))
print("a.std(axis=1)  :", a.std(axis=1))


# ───────────────────────────────────────────────────────────
# argmax / argmin / argsort
# ───────────────────────────────────────────────────────────
# These return INDICES, not values. Crucial for "which is the best class?"
# kinds of questions.
scores = np.array([[0.1, 0.7, 0.2],
                   [0.4, 0.3, 0.3],
                   [0.2, 0.2, 0.6]])
predicted_class = scores.argmax(axis=1)
print("\npredicted classes:", predicted_class)


# ───────────────────────────────────────────────────────────
# np.where  — vectorized if/else
# ───────────────────────────────────────────────────────────
x = np.arange(-3, 4)
print("\nx:", x)
print("clip negatives to 0:", np.where(x < 0, 0, x))
print("piecewise:", np.where(x > 0, x ** 2, -x))


# ───────────────────────────────────────────────────────────
# Concatenation & stacking
# ───────────────────────────────────────────────────────────
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("\nvstack:\n", np.vstack([A, B]))     # stack rows
print("hstack:\n", np.hstack([A, B]))       # stack columns
print("axis=0:\n", np.concatenate([A, B], axis=0))   # same as vstack
print("axis=1:\n", np.concatenate([A, B], axis=1))   # same as hstack

# stack adds a NEW axis (useful for batching)
print("stack →", np.stack([A, B]).shape)            # (2, 2, 2)


# ───────────────────────────────────────────────────────────
# Reshape, ravel, transpose
# ───────────────────────────────────────────────────────────
m = np.arange(12)
print("\nreshape (3, 4):\n", m.reshape(3, 4))
print("reshape (-1, 2):\n", m.reshape(-1, 2))   # -1 means "infer this dim"
print("ravel:", m.reshape(3, 4).ravel())        # back to 1-D
print("T:\n",  m.reshape(3, 4).T)


# ───────────────────────────────────────────────────────────
# Useful one-liners
# ───────────────────────────────────────────────────────────
print("\nunique values:    ", np.unique(np.array([1, 2, 2, 3, 3, 3])))
print("clip to [0, 10]:  ", np.clip(np.array([-5, 3, 15, 8]), 0, 10))
print("element-wise max: ", np.maximum([1, 5, 2], [3, 4, 6]))
print("cumulative sum:   ", np.cumsum([1, 2, 3, 4]))
