"""
Exercise 1 — Measure mixed-precision speedup

Train a moderately-sized MLP for a few iterations under three precision
modes and report wall-clock time and final loss.

  A. fp32 baseline
  B. autocast bf16
  C. autocast fp16 + GradScaler  (CUDA only)

The script asserts that bf16 and fp32 produce SIMILAR final losses (within
30%) — i.e. mixed precision doesn't break training. Real speedups only show
up on a CUDA GPU.

Run: python exercises/01_mixed_precision_speedup.py
"""

import time

import torch
import torch.nn as nn


torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")


class WideMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2048, 4096), nn.ReLU(),
            nn.Linear(4096, 4096), nn.ReLU(),
            nn.Linear(4096, 4096), nn.ReLU(),
            nn.Linear(4096, 10),
        )

    def forward(self, x):
        return self.layers(x)


def make_batch():
    return torch.randn(64, 2048, device=device), torch.randint(0, 10, (64,), device=device)


def synchronize():
    if device == "cuda":
        torch.cuda.synchronize()


# ───────────────────────────────────────────────────────────
# Run
# ───────────────────────────────────────────────────────────
ITERS = 30


def run_fp32():
    torch.manual_seed(0)
    model = WideMLP().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    for _ in range(5):
        x, y = make_batch()
        opt.zero_grad()
        crit(model(x), y).backward()
        opt.step()

    synchronize()
    t0 = time.perf_counter()
    losses = []
    for _ in range(ITERS):
        x, y = make_batch()
        opt.zero_grad()
        loss = crit(model(x), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    synchronize()
    return time.perf_counter() - t0, losses[-1]


def run_bf16():
    torch.manual_seed(0)
    model = WideMLP().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    for _ in range(5):
        x, y = make_batch()
        opt.zero_grad()
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            crit(model(x), y).backward()
        opt.step()

    synchronize()
    t0 = time.perf_counter()
    losses = []
    for _ in range(ITERS):
        x, y = make_batch()
        opt.zero_grad()
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            loss = crit(model(x), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    synchronize()
    return time.perf_counter() - t0, losses[-1]


def run_fp16():
    torch.manual_seed(0)
    model = WideMLP().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda")

    for _ in range(5):
        x, y = make_batch()
        opt.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = crit(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    losses = []
    for _ in range(ITERS):
        x, y = make_batch()
        opt.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = crit(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        losses.append(loss.item())
    torch.cuda.synchronize()
    return time.perf_counter() - t0, losses[-1]


# ───────────────────────────────────────────────────────────
# Compare
# ───────────────────────────────────────────────────────────
print(f"\n{'mode':<26}  {'time (s)':>10}  {'iter/s':>8}  {'final loss':>11}")
t_fp32, l_fp32 = run_fp32()
print(f"{'fp32 baseline':<26}  {t_fp32:>10.3f}  {ITERS / t_fp32:>8.1f}  {l_fp32:>11.4f}")

t_bf16, l_bf16 = run_bf16()
print(f"{'autocast bf16':<26}  {t_bf16:>10.3f}  {ITERS / t_bf16:>8.1f}  {l_bf16:>11.4f}")

if device == "cuda":
    t_fp16, l_fp16 = run_fp16()
    print(f"{'autocast fp16 + GradScaler':<26}  {t_fp16:>10.3f}  {ITERS / t_fp16:>8.1f}  {l_fp16:>11.4f}")

print(f"\nbf16 speedup vs fp32: {t_fp32 / t_bf16:.2f}×")
print("(speedup is small/zero on CPU; on Ampere GPU expect ~2×.)")

assert abs(l_bf16 - l_fp32) < max(0.5, abs(l_fp32) * 0.3), \
    "bf16 final loss should be close to fp32 final loss"
print("\nmixed precision works without breaking training ✅")
