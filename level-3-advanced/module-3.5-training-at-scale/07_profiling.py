"""
07 — Profiling: torch.profiler

Before optimizing, MEASURE. The PyTorch profiler reports per-op CPU/GPU time,
memory allocations, and shape statistics, exportable to TensorBoard or
Chrome's tracing UI.

This file profiles a small training step and prints a per-op summary.

Run: python 07_profiling.py
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile, record_function


device = "cuda" if torch.cuda.is_available() else "cpu"


# ───────────────────────────────────────────────────────────
# Tiny model + step
# ───────────────────────────────────────────────────────────
model = nn.Sequential(
    nn.Linear(1024, 2048), nn.ReLU(),
    nn.Linear(2048, 2048), nn.ReLU(),
    nn.Linear(2048, 100),
).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)


def make_batch():
    return torch.randn(64, 1024, device=device), torch.randint(0, 100, (64,), device=device)


# Warm up first
for _ in range(3):
    x, y = make_batch()
    opt.zero_grad()
    loss = nn.functional.cross_entropy(model(x), y)
    loss.backward()
    opt.step()


# ───────────────────────────────────────────────────────────
# Profile a few steps
# ───────────────────────────────────────────────────────────
activities = [ProfilerActivity.CPU]
if device == "cuda":
    activities.append(ProfilerActivity.CUDA)

with profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
    for _ in range(5):
        x, y = make_batch()
        with record_function("step"):
            with record_function("forward"):
                logits = model(x)
                loss = nn.functional.cross_entropy(logits, y)
            with record_function("backward"):
                loss.backward()
            with record_function("optimizer_step"):
                opt.step()
                opt.zero_grad()


# ───────────────────────────────────────────────────────────
# Print: top ops by time
# ───────────────────────────────────────────────────────────
print("=== top 15 ops by CPU time ===")
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))


# ───────────────────────────────────────────────────────────
# Export Chrome trace (open in chrome://tracing or perfetto.dev)
# ───────────────────────────────────────────────────────────
trace_path = Path(__file__).parent / "scratch" / "trace.json"
trace_path.parent.mkdir(exist_ok=True)
prof.export_chrome_trace(str(trace_path))
print(f"\nChrome trace exported: {trace_path}")
print("Open at https://ui.perfetto.dev or chrome://tracing")


# ───────────────────────────────────────────────────────────
# How to integrate the profiler in real training
# ───────────────────────────────────────────────────────────
# A typical pattern is to profile a few warm steps with a SCHEDULE so the
# trace stays small:
#
#   from torch.profiler import schedule, tensorboard_trace_handler
#
#   prof = profile(
#       activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
#       schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
#       on_trace_ready=tensorboard_trace_handler("./tb_logs"),
#   )
#   prof.start()
#   for step, batch in enumerate(loader):
#       ...
#       prof.step()      # advance the schedule
#   prof.stop()


# ───────────────────────────────────────────────────────────
# Reading a trace effectively
# ───────────────────────────────────────────────────────────
# In Perfetto / chrome://tracing, look for:
#   • Big gaps on the GPU lane → starvation (dataloader, sync ops).
#   • Lots of small kernel launches → fuse with torch.compile.
#   • aten::copy_ everywhere → too many .cpu() / .item() in the inner loop.
#   • ProcessGroupNCCL allreduce times → distributed comm overhead.


# ───────────────────────────────────────────────────────────
# Lighter-weight alternatives
# ───────────────────────────────────────────────────────────
# • cProfile / line_profiler — Python-level hotspots, no GPU info.
# • nvtop / nvitop — terminal GPU dashboards.
# • nsys profile (NVIDIA Nsight) — system-level, the gold standard.
# • Just `time.perf_counter()` around the suspect block. Often enough.
