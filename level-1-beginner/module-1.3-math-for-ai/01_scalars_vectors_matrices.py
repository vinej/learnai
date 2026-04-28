"""
01 — Scalars, vectors, matrices, tensors

The four building blocks of every ML library:
    scalar  -- a single number              shape ()
    vector  -- a 1-D array of n numbers     shape (n,)
    matrix  -- a 2-D table                  shape (m, n)
    tensor  -- 3+ dimensional               shape (a, b, c, ...)

NumPy represents all of them as one type: `ndarray`.

Run: python 01_scalars_vectors_matrices.py
"""

import numpy as np


# --- Scalar ---
s = np.float64(3.14)
print("scalar:", s, "shape:", s.shape, "ndim:", s.ndim)


# --- Vector ---
v = np.array([1.0, 2.0, 3.0])
print("vector:", v, "shape:", v.shape, "ndim:", v.ndim, "dtype:", v.dtype)


# --- Matrix ---
M = np.array([[1, 2, 3],
              [4, 5, 6]])
print("matrix:")
print(M, "shape:", M.shape)


# --- Tensor (e.g. a small batch of 4 RGB 3x3 'images') ---
T = np.zeros((4, 3, 3, 3))      # (batch, channels, height, width) — common in DL
print("tensor shape:", T.shape, "ndim:", T.ndim, "elements:", T.size)


# --- Creating arrays ---
print(np.zeros(5))
print(np.ones((2, 3)))
print(np.arange(0, 10, 2))         # like range, but as an array
print(np.linspace(0, 1, 5))        # 5 evenly spaced points in [0, 1]
print(np.eye(4))                   # identity matrix


# --- Indexing & slicing ---
a = np.arange(12).reshape(3, 4)
print("a =\n", a)
print("first row:    ", a[0])
print("second col:   ", a[:, 1])
print("element [1,2]:", a[1, 2])
print("a > 5:        ", a[a > 5])  # boolean mask — pulls out matching values


# --- Reshape & transpose ---
print("flat:        ", a.flatten())
print("transpose:\n", a.T)
print("reshape 4x3:\n", a.reshape(4, 3))


# --- Broadcasting (NumPy's killer feature) ---
# Arithmetic between arrays of compatible shapes auto-extends the smaller one.
row = np.array([10, 20, 30, 40])
result = a + row              # adds `row` to EACH row of a (no loop!)
print("a + row =\n", result)


# --- Vectorized ops vs Python loops ---
# NumPy is fast because operations run as C code under the hood.
import time

n = 1_000_000
py_list = list(range(n))
np_arr  = np.arange(n)

t0 = time.perf_counter()
sum_py = sum(x * 2 for x in py_list)
t1 = time.perf_counter()
sum_np = (np_arr * 2).sum()
t2 = time.perf_counter()

print(f"python loop: {t1 - t0:.3f}s   numpy: {t2 - t1:.3f}s   "
      f"speedup: {(t1 - t0) / (t2 - t1):.0f}×")

assert sum_py == sum_np
