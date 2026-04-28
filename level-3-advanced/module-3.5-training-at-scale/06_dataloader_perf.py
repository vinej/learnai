"""
06 — DataLoader performance

A common surprise: your GPU is at 30% utilization. Why? Because the
DataLoader is starving it.

Three knobs that almost always help:

  num_workers > 0   — spawn worker processes that prefetch batches in
                      parallel. Without them, your main process blocks on I/O.
  pin_memory=True   — allocate batches in pinned host memory; CUDA copies
                      from pinned memory are async + faster. GPU only.
  persistent_workers=True
                    — keep workers alive across epochs (saves the warmup
                      cost on every epoch).

Plus:
  prefetch_factor   — how many batches each worker prefetches (default 2).
  drop_last=True    — discard last incomplete batch; some BN ops need uniform shapes.

This file benchmarks num_workers on a CPU-bound synthetic dataset.

Run: python 06_dataloader_perf.py
"""

import time

import torch
from torch.utils.data import DataLoader, Dataset


# ───────────────────────────────────────────────────────────
# A deliberately CPU-bound dataset
# ───────────────────────────────────────────────────────────
# We add a small sleep to simulate file I/O / decoding cost.
class SlowDataset(Dataset):
    def __init__(self, n: int, work_ms: float = 1.0):
        self.n = n
        self.work_ms = work_ms

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # Simulate disk I/O / image decode work
        time.sleep(self.work_ms / 1000.0)
        return torch.randn(64), torch.tensor(idx % 10)


# ───────────────────────────────────────────────────────────
# Benchmark different worker counts
# ───────────────────────────────────────────────────────────
N = 1000          # samples
ms = 1.5          # per-sample fake "work"
ds = SlowDataset(N, work_ms=ms)


def benchmark(num_workers: int) -> float:
    loader = DataLoader(
        ds,
        batch_size=32,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,            # pinned memory only matters for GPU
        persistent_workers=num_workers > 0,
    )
    t0 = time.perf_counter()
    for _ in loader:
        pass
    return time.perf_counter() - t0


print(f"dataset: {N} samples × ~{ms} ms each → ideal serial time ~{N * ms / 1000:.2f}s")
print(f"{'workers':>8}  {'time (s)':>10}  {'speedup vs 0':>14}")
times = {}
for nw in [0, 2, 4, 8]:
    t = benchmark(nw)
    times[nw] = t
    speedup = times[0] / t if 0 in times else 1
    print(f"{nw:>8}  {t:>10.2f}  {speedup:>14.2f}×")


# ───────────────────────────────────────────────────────────
# What about pin_memory?
# ───────────────────────────────────────────────────────────
# On CUDA, pin_memory=True allocates batches in PINNED (page-locked) host
# memory. The GPU can DMA from pinned memory asynchronously, overlapping
# the copy with compute. Without pinned memory, the copy is synchronous.
#
# Combined with non_blocking=True on .to():
#
#   x = batch.to(device, non_blocking=True)
#
# you get fully asynchronous host→device transfer.


# ───────────────────────────────────────────────────────────
# Common mistakes
# ───────────────────────────────────────────────────────────
# • num_workers=0 in production. Default is 0 — change it.
# • num_workers > num_cpus. More processes than CPUs causes contention.
#   Start with min(8, cpu_count - 1).
# • CUDA tensors created INSIDE workers — workers run as child processes
#   and can't share CUDA memory. Always return CPU tensors and move them
#   in the main loop.
# • Reading huge files in __getitem__ without caching. Use lmdb, webdataset,
#   or pre-tokenized parquet.


# ───────────────────────────────────────────────────────────
# When the dataset is the bottleneck
# ───────────────────────────────────────────────────────────
# Profile (file 07) to confirm. Then:
#   1. Increase num_workers.
#   2. Pre-process / cache to a faster format (parquet, npz, webdataset).
#   3. Use streaming datasets (HF datasets with streaming=True).
#   4. As a last resort, train on a SMALLER dataset that fits in RAM.
