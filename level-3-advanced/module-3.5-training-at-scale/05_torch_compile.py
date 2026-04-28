"""
05 — torch.compile (PyTorch 2.x)

`torch.compile(model)` JIT-compiles your model into optimized GPU kernels.
It can give 1.5-3× speedups on training and inference, with NO code change
beyond the wrapping line.

Modes:
  default       — balance speed / compile time. Good first move.
  reduce-overhead — better for small/medium models with many small ops.
  max-autotune  — slowest to compile, fastest at runtime.

Run: python 05_torch_compile.py
"""

import time

import torch
import torch.nn as nn


torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"


model = nn.Sequential(
    nn.Linear(2048, 4096), nn.ReLU(),
    nn.Linear(4096, 4096), nn.ReLU(),
    nn.Linear(4096, 1024),
).to(device)


# Wrap. That's the entire change.
compiled = torch.compile(model)
# Optional: choose a mode:
#   compiled = torch.compile(model, mode="reduce-overhead")
#   compiled = torch.compile(model, mode="max-autotune")


# ───────────────────────────────────────────────────────────
# Benchmark forward
# ───────────────────────────────────────────────────────────
def time_forward(net, n=50):
    x = torch.randn(64, 2048, device=device)
    # Warm up — compilation happens on the FIRST forward, so don't time it.
    for _ in range(5):
        _ = net(x)
    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n):
        _ = net(x)
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n


eager = time_forward(model)
print(f"eager:    {eager * 1000:.2f} ms / iter")

# torch.compile on CPU works but the speedups are usually modest;
# real wins are on GPU.
print("compiling ...")
compiled_t = time_forward(compiled)
print(f"compiled: {compiled_t * 1000:.2f} ms / iter")
print(f"speedup: {eager / compiled_t:.2f}×")


# ───────────────────────────────────────────────────────────
# What torch.compile does under the hood (sketch)
# ───────────────────────────────────────────────────────────
# • TorchDynamo intercepts Python bytecode and captures the model's compute
#   graph (a TorchFX graph).
# • AOTAutograd computes the backward graph from the same trace.
# • TorchInductor lowers the graph to optimized kernels (Triton on GPU,
#   C++ on CPU).
# Result: fewer Python overhead and better fused kernels.


# ───────────────────────────────────────────────────────────
# When compile WON'T help (or might HURT)
# ───────────────────────────────────────────────────────────
# • Very small models / batch sizes — kernel-launch overhead dominates.
# • Models with lots of dynamic Python control flow (if/else on tensor
#   values). torch.compile handles these but pays a cost ("graph break").
# • Models with custom CUDA ops or autograd functions. These usually still
#   work but need careful testing.
#
# Rule of thumb: test with one line of compile, measure on a representative
# batch, decide.


# ───────────────────────────────────────────────────────────
# Compile + amp + checkpointing combine fine
# ───────────────────────────────────────────────────────────
# Modern stack for big models:
#   model = torch.compile(model)
#   for batch in loader:
#       with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
#           out = model(batch)
#           loss = criterion(out, target)
#       loss.backward()
#       optimizer.step()
#       optimizer.zero_grad()
