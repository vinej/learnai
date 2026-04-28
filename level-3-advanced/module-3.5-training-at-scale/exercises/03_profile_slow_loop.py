"""
Exercise 3 — Profile a deliberately slow training loop and find the bug

Two versions of the same training loop:

  slow_loop()   — has TWO bottlenecks: a Python loop in __getitem__ that
                  could be vectorized, and a `.cpu().item()` call inside
                  the inner loop that forces a host-device sync.

  fast_loop()   — same logic, both bottlenecks fixed.

Profile both, confirm fast is significantly faster, and verify with the
profiler that the slow version's CPU time is dominated by the offending ops.

Run: python exercises/03_profile_slow_loop.py
"""

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile
from torch.utils.data import DataLoader, Dataset


device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)


# ───────────────────────────────────────────────────────────
# A dataset that's secretly slow
# ───────────────────────────────────────────────────────────
class SecretlySlowDataset(Dataset):
    """`__getitem__` does silly per-element work in a Python loop."""

    def __init__(self, n=512, dim=128):
        rng = torch.Generator().manual_seed(0)
        self.data = torch.randn(n, dim, generator=rng)
        self.labels = torch.randint(0, 10, (n,), generator=rng)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, i):
        x = self.data[i]
        # 🐞 Silly per-element loop — could be a single vectorized op.
        squared = torch.zeros_like(x)
        for j in range(x.size(0)):
            squared[j] = x[j] * x[j]
        return squared, self.labels[i]


class FastDataset(Dataset):
    def __init__(self, n=512, dim=128):
        rng = torch.Generator().manual_seed(0)
        self.data = torch.randn(n, dim, generator=rng)
        self.labels = torch.randint(0, 10, (n,), generator=rng)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, i):
        x = self.data[i]
        return x * x, self.labels[i]            # vectorized


# ───────────────────────────────────────────────────────────
# Two training loops
# ───────────────────────────────────────────────────────────
def make_model():
    return nn.Sequential(
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, 10),
    ).to(device)


def slow_loop():
    model = make_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    loader = DataLoader(SecretlySlowDataset(), batch_size=32, num_workers=0)

    losses = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = crit(model(x), y)
        loss.backward()
        opt.step()
        # 🐞 host-device sync inside the loop
        losses.append(loss.cpu().item())
    return losses


def fast_loop():
    model = make_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    loader = DataLoader(FastDataset(), batch_size=32, num_workers=0)

    losses_t = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = crit(model(x), y)
        loss.backward()
        opt.step()
        # Keep loss on-device; sync only at the end
        losses_t.append(loss.detach())
    return [l.cpu().item() for l in losses_t]


# ───────────────────────────────────────────────────────────
# Time both
# ───────────────────────────────────────────────────────────
def time_loop(fn, label):
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    print(f"  {label:<14}  {elapsed:.3f}s")
    return elapsed


print("first run is the slow loop, second is the fast one:")
t_slow = time_loop(slow_loop, "slow")
t_fast = time_loop(fast_loop, "fast")
print(f"\nspeedup: {t_slow / t_fast:.1f}×")


# ───────────────────────────────────────────────────────────
# Profile the slow loop to confirm WHERE the time goes
# ───────────────────────────────────────────────────────────
print("\nprofiling slow_loop ...")
with profile(activities=[ProfilerActivity.CPU], record_shapes=False) as prof:
    slow_loop()

# Show the hottest ops by total CPU time. You should see lots of small
# tensor ops (from the per-element loop in __getitem__) and aten::copy_
# from the .cpu() syncs.
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))


assert t_slow > t_fast * 1.5, "the slow version really should be slower"
print("\nbottlenecks identified and fixed ✅")


# ── Lessons ──────────────────────────────────────────────────────────────────
# 1. NEVER write per-element Python loops over tensor elements. Vectorize.
# 2. Avoid .cpu() / .item() / .tolist() inside the inner training loop.
#    They force the GPU to flush and the CPU to wait.
# 3. Profile FIRST. Don't guess.
# ─────────────────────────────────────────────────────────────────────────────
