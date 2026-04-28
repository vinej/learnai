"""
03 — Gradient checkpointing

During training, PyTorch saves all intermediate activations from the forward
pass so they're available for backprop. For deep networks, those activations
can dominate VRAM.

GRADIENT CHECKPOINTING trades MEMORY for COMPUTE:
  - Drop most activations during forward.
  - During backward, RECOMPUTE them on the fly from saved checkpoints.
  - You save memory (~√L instead of L), at the cost of an extra forward pass
    for each segment (~30% slower).

For very deep models (transformers with 32+ layers, huge image models),
this is the difference between fitting your batch and OOM'ing.

Run: python 03_gradient_checkpointing.py
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint, checkpoint_sequential


device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)


# ───────────────────────────────────────────────────────────
# A deep, memory-hungry model
# ───────────────────────────────────────────────────────────
class DeepBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.fc2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        return torch.relu(self.fc2(torch.relu(self.fc1(x))))


class DeepMLP(nn.Module):
    def __init__(self, dim=1024, depth=24, ckpt: bool = False):
        super().__init__()
        self.blocks = nn.ModuleList([DeepBlock(dim) for _ in range(depth)])
        self.head = nn.Linear(dim, 10)
        self.ckpt = ckpt

    def forward(self, x):
        for blk in self.blocks:
            if self.ckpt and self.training:
                # The lambda makes `checkpoint` call blk(x) without saving
                # its activations.
                x = checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        return self.head(x)


# ───────────────────────────────────────────────────────────
# Train one step each, comparing memory + speed
# ───────────────────────────────────────────────────────────
import time


def measure(label, model):
    model.train()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    x = torch.randn(32, 1024, device=device, requires_grad=True)
    y = torch.randint(0, 10, (32,), device=device)

    t0 = time.perf_counter()
    out = model(x)
    loss = nn.functional.cross_entropy(out, y)
    loss.backward()
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    if device == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1e6
        print(f"  {label:<24}  {elapsed:.3f}s   peak GPU memory: {peak:.0f} MB")
    else:
        print(f"  {label:<24}  {elapsed:.3f}s   (no GPU memory stats on CPU)")


print("Forward + backward, batch=32, depth=24 layers, dim=1024")
plain = DeepMLP(ckpt=False).to(device)
measure("no checkpointing", plain)

ckpt_model = DeepMLP(ckpt=True).to(device)
measure("with checkpointing", ckpt_model)


# ───────────────────────────────────────────────────────────
# checkpoint vs checkpoint_sequential
# ───────────────────────────────────────────────────────────
# checkpoint(fn, *args)
#   Wraps a single function (e.g. one transformer block).
#
# checkpoint_sequential(model_sequential, n_segments, x)
#   Splits an nn.Sequential into n contiguous chunks and checkpoints each.
#   Easier when your model is one big Sequential.


# ───────────────────────────────────────────────────────────
# How to use it in HuggingFace
# ───────────────────────────────────────────────────────────
# Most HF transformer models support it natively:
#
#   model.gradient_checkpointing_enable()
#
# In TrainingArguments:
#
#   TrainingArguments(..., gradient_checkpointing=True)
#
# Often paired with bf16 for max VRAM savings.


# ───────────────────────────────────────────────────────────
# Trade-off rule of thumb
# ───────────────────────────────────────────────────────────
# Memory drop: ~ √(depth) factor, sometimes more.
# Speed cost:  ~ 25-35% slower (one extra forward per segment).
# Worth it whenever batch size or sequence length is the bottleneck.
