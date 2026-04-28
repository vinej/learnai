"""
04 — PyTorch tensors

A PyTorch `Tensor` is like a NumPy `ndarray` plus two superpowers:
  1. It can live on a GPU (or Apple's MPS) for massive speedups.
  2. It can RECORD operations for automatic differentiation (next file).

Run: python 04_pytorch_tensors.py
"""

import time

import numpy as np
import torch


# ───────────────────────────────────────────────────────────
# Creating tensors
# ───────────────────────────────────────────────────────────
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.zeros(2, 3)
c = torch.ones(2, 3, dtype=torch.float64)
d = torch.arange(0, 10, 2)
e = torch.linspace(0, 1, 5)
f = torch.randn(3, 3)              # standard normal
g = torch.eye(4)

print(a, a.dtype, a.shape)
print(b)
print(c.dtype)
print(d.tolist())
print(f"random:\n{f}")


# ───────────────────────────────────────────────────────────
# dtype and reshaping
# ───────────────────────────────────────────────────────────
x = torch.tensor([1, 2, 3, 4, 5, 6])
print("\nshape:", x.shape, "dtype:", x.dtype)

# View vs reshape: view shares memory; reshape may copy. Use reshape unless
# you specifically need a view-without-copy guarantee.
y = x.reshape(2, 3)
print("y:\n", y)

# Cast a tensor's dtype
print("float64:", x.to(torch.float64).dtype)
print("float32:", x.float().dtype)


# ───────────────────────────────────────────────────────────
# NumPy ↔ Tensor (zero copy when possible)
# ───────────────────────────────────────────────────────────
arr = np.array([[1.0, 2.0], [3.0, 4.0]])
t = torch.from_numpy(arr)            # shares memory
arr[0, 0] = 99
print("\nfrom_numpy is a view:", t)   # changed!

t2 = torch.tensor(arr)               # copies
arr[0, 0] = 0
print("torch.tensor copies:", t2)    # untouched

# Back to numpy
print(t.numpy())


# ───────────────────────────────────────────────────────────
# Devices: CPU, CUDA, MPS
# ───────────────────────────────────────────────────────────
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"\nusing device: {device}")

x = torch.randn(2, 3)
x_on_dev = x.to(device)
print("x device:", x_on_dev.device)

# YOU MUST HAVE TENSORS ON THE SAME DEVICE TO COMPUTE WITH THEM.
# Mixing CPU and GPU tensors raises RuntimeError.


# ───────────────────────────────────────────────────────────
# Operations — same as NumPy, mostly
# ───────────────────────────────────────────────────────────
A = torch.randn(3, 4)
B = torch.randn(4, 5)

print("matmul A @ B:", (A @ B).shape)
print("element-wise:", (torch.tensor([1, 2, 3]) * torch.tensor([10, 10, 10])).tolist())
print("sum:", A.sum(), "mean:", A.mean(), "argmax:", A.argmax(dim=1))


# ───────────────────────────────────────────────────────────
# Broadcasting (same rules as NumPy)
# ───────────────────────────────────────────────────────────
M = torch.arange(12).reshape(3, 4).float()
row = torch.tensor([10.0, 20.0, 30.0, 40.0])
print("\nbroadcast row addition:")
print(M + row)


# ───────────────────────────────────────────────────────────
# Why GPUs matter (CPU-only demo here, but the principle is the same)
# ───────────────────────────────────────────────────────────
# A 4096×4096 matmul on a modern GPU is ~100-1000× faster than on CPU.
# Even on CPU, PyTorch's BLAS-backed matmul is fast.
n = 1024
a = torch.randn(n, n)
b = torch.randn(n, n)

t0 = time.perf_counter()
for _ in range(10):
    c = a @ b
torch_time = time.perf_counter() - t0

a_np = a.numpy()
b_np = b.numpy()
t0 = time.perf_counter()
for _ in range(10):
    c_np = a_np @ b_np
np_time = time.perf_counter() - t0

print(f"\n{n}×{n} matmul × 10:  torch={torch_time:.3f}s  numpy={np_time:.3f}s")


# ───────────────────────────────────────────────────────────
# Helpful little APIs
# ───────────────────────────────────────────────────────────
x = torch.tensor([3, 1, 4, 1, 5, 9, 2, 6])
print("\ntop-3:", torch.topk(x, k=3))
print("unique:", torch.unique(x).tolist())
print("clamp [2,5]:", torch.clamp(x, 2, 5).tolist())
print("cumsum:", torch.cumsum(x, dim=0).tolist())


# ───────────────────────────────────────────────────────────
# Tip
# ───────────────────────────────────────────────────────────
# • Default float dtype is float32. ML usually wants float32 (or bfloat16/fp16
#   for speed on GPUs). Avoid float64 unless you need the precision.
# • For images, the convention is (N, C, H, W) — batch, channels, height, width.
# • For sequences/text, (N, T, D) — batch, time, features. Sometimes (T, N, D).
