"""
02 — Linear algebra: dot products, matmul, norms, distances, cosine similarity

These operations show up *everywhere* in ML:
  • Dot product   → similarity, attention scores, projections
  • Matmul        → every neural-network layer
  • Norms         → regularization, gradient clipping
  • Cosine sim    → embeddings, retrieval, RAG

Run: python 02_linear_algebra.py
"""

import numpy as np


# --- Dot product ---
# v · w = sum(v[i] * w[i])      (a scalar)
v = np.array([1, 2, 3])
w = np.array([4, 5, 6])

print("v · w =", v @ w)            # @ is the matmul / dot operator
print("equivalent:", np.dot(v, w))
print("manual:    ", sum(a * b for a, b in zip(v, w)))


# --- Matrix multiplication ---
# (m × k) @ (k × n) → (m × n)
# The k dimensions must match. The result drops the inner dim.
A = np.array([[1, 2],
              [3, 4],
              [5, 6]])           # 3 × 2
B = np.array([[7, 8, 9],
              [10, 11, 12]])     # 2 × 3

print("A @ B =\n", A @ B)        # → 3 × 3

# Element-wise multiplication is DIFFERENT — uses *
C = np.array([[1, 2], [3, 4]])
D = np.array([[5, 6], [7, 8]])
print("element-wise C * D:\n", C * D)
print("matrix prod  C @ D:\n", C @ D)


# --- Transpose ---
# Swap axes. `M.T` is the same as `np.transpose(M)`.
print("A.T =\n", A.T)


# --- Identity & inverse ---
# I @ M = M @ I = M
I = np.eye(3)
M = np.array([[2.0, 1.0],
              [1.0, 3.0]])
M_inv = np.linalg.inv(M)
print("M @ M_inv ≈ I:\n", M @ M_inv)


# --- Solving linear systems ---
# A x = b  →  x = A^-1 b   (but use np.linalg.solve, it's more numerically stable)
A_sys = np.array([[3.0, 1.0], [1.0, 2.0]])
b     = np.array([9.0, 8.0])
x = np.linalg.solve(A_sys, b)
print("solution x:", x)
print("verify  A @ x:", A_sys @ x)


# --- Norms ---
# A norm measures the "length" of a vector.
u = np.array([3.0, -4.0])

print("L2 (Euclidean):", np.linalg.norm(u))         # sqrt(3² + 4²) = 5
print("L1 (Manhattan):", np.linalg.norm(u, ord=1))  # |3| + |-4|   = 7
print("L∞ (max):     ", np.linalg.norm(u, ord=np.inf))


# --- Distances between two vectors ---
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 6.0, 8.0])

euclidean = np.linalg.norm(a - b)
manhattan = np.sum(np.abs(a - b))
print("euclidean d:", euclidean)
print("manhattan d:", manhattan)


# --- Cosine similarity ---
# Measures the angle between two vectors. Range: [-1, 1].
#   1  → same direction
#   0  → orthogonal
#  -1  → opposite direction
# Crucial for embeddings & semantic search.
def cosine_similarity(a, b):
    return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))

x1 = np.array([1.0, 0.0])
x2 = np.array([0.5, 0.5])
x3 = np.array([-1.0, 0.0])
x4 = np.array([2.0, 0.0])    # same direction as x1, just longer

print("cos(x1, x2):", cosine_similarity(x1, x2))   # 45° → ~0.707
print("cos(x1, x3):", cosine_similarity(x1, x3))   # opposite → -1
print("cos(x1, x4):", cosine_similarity(x1, x4))   # parallel → 1


# --- Why cosine and not Euclidean for embeddings? ---
# Embeddings often have arbitrary magnitudes. Cosine cares only about
# DIRECTION — so two embeddings pointing the same way are "similar"
# even if one is twice as long.
