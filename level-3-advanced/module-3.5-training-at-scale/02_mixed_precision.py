"""
02 — Mixed-precision training (torch.amp)

The idea: do most computation in low-precision (bf16 or fp16) but keep a
master copy of weights and optimizer state in fp32. You get ~2× speedup
and ~2× less memory on modern GPUs, with no accuracy loss for almost
every model.

PyTorch's torch.amp does this automatically with two pieces:

  autocast(...)   — wraps a forward pass. Ops run in low precision when safe.
  GradScaler      — only needed for fp16 (not bf16). Scales the loss before
                    backward to keep small gradients from underflowing.

Modern hardware:
  bf16 (brain float)   — same dynamic range as fp32 (large exponent), less
                         precision (mantissa). Default on Ampere+ / TPUs.
                         No GradScaler needed.
  fp16                  — narrower exponent → can underflow. Works fine
                         WITH GradScaler. Standard before bf16 on V100/T4.
  fp8                   — newest, on H100 / Blackwell. Even faster.

Run: python 02_mixed_precision.py
"""

import time

import torch
import torch.nn as nn


torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")


# ───────────────────────────────────────────────────────────
# A model and synthetic batches large enough to feel the difference
# ───────────────────────────────────────────────────────────
class BigMLP(nn.Module):
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


def make_batch(B=64):
    return torch.randn(B, 2048, device=device), torch.randint(0, 10, (B,), device=device)


# ───────────────────────────────────────────────────────────
# Standard fp32 training step
# ───────────────────────────────────────────────────────────
def step_fp32(model, opt, x, y, criterion):
    opt.zero_grad()
    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    opt.step()
    return loss


# ───────────────────────────────────────────────────────────
# Mixed-precision training step (bf16, no scaler needed)
# ───────────────────────────────────────────────────────────
def step_mixed_bf16(model, opt, x, y, criterion):
    opt.zero_grad()
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        out = model(x)
        loss = criterion(out, y)
    loss.backward()
    opt.step()
    return loss


# ───────────────────────────────────────────────────────────
# Mixed-precision training step (fp16 with GradScaler)
# ───────────────────────────────────────────────────────────
def step_mixed_fp16(model, opt, scaler, x, y, criterion):
    opt.zero_grad()
    with torch.autocast(device_type=device, dtype=torch.float16):
        out = model(x)
        loss = criterion(out, y)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    return loss


# ───────────────────────────────────────────────────────────
# Run a benchmark: 50 iterations each
# ───────────────────────────────────────────────────────────
ITERS = 50


def benchmark(name: str, step_fn):
    model = BigMLP().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Warm up (especially important on CUDA where the first call compiles kernels)
    for _ in range(5):
        x, y = make_batch()
        step_fn(model, opt, x, y, criterion)

    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        x, y = make_batch()
        step_fn(model, opt, x, y, criterion)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    print(f"  {name:<20}  {ITERS} iters in {elapsed:.2f}s  ({ITERS / elapsed:.1f} iter/s)")
    return elapsed


t_fp32 = benchmark("fp32 baseline", step_fp32)
t_bf16 = benchmark("autocast bf16", step_mixed_bf16)

if device == "cuda":
    scaler = torch.amp.GradScaler("cuda")
    t_fp16 = benchmark("autocast fp16 + GradScaler",
                       lambda m, o, x, y, c: step_mixed_fp16(m, o, scaler, x, y, c))
else:
    print("\n(fp16 + GradScaler is GPU-only; skipped on CPU.)")
    t_fp16 = None


# ───────────────────────────────────────────────────────────
# Verdict
# ───────────────────────────────────────────────────────────
if t_bf16:
    print(f"\nbf16 speedup vs fp32: {t_fp32 / t_bf16:.2f}×")
print(
    "On CPU the speedup is small or negative (no fp16 GEMM kernels). "
    "On a modern NVIDIA GPU (Ampere+), expect ~2× faster, ~2× less memory."
)


# ───────────────────────────────────────────────────────────
# Practical guidance
# ───────────────────────────────────────────────────────────
# • Default on Ampere/H100: bf16 — no GradScaler needed.
# • Default on V100/T4 (Volta/Turing): fp16 + GradScaler.
# • For inference: same idea, no GradScaler needed (no backward).
# • Some ops can NaN-out in fp16. autocast already excludes the worst
#   offenders, but if you see NaN losses, try bf16 first.
# • For HuggingFace Trainer:
#       TrainingArguments(..., bf16=True)   # or fp16=True
# • For accelerate:
#       accelerate launch --mixed_precision=bf16 train.py
