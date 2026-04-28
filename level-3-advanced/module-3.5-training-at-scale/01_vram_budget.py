"""
01 — VRAM budget: where does the memory go?

When training a model, GPU memory has 4 main consumers:

  PARAMS         — the model's weights.            P bytes
  GRADIENTS      — same shape as params.           P bytes
  OPTIMIZER STATE — Adam stores 2× params (momentum + variance).  ~2P bytes
  ACTIVATIONS    — saved during forward for backward use. Depends on batch.

Plus some constant overhead for the framework / CUDA context (~1-2 GB).

Bytes per parameter depend on dtype:
  fp32 (float32) → 4 bytes
  bf16 / fp16    → 2 bytes
  int8           → 1 byte
  4-bit (NF4)    → 0.5 bytes (for QLoRA-style fine-tuning)

This file gives you the math you'll need to plan ANY training run.

Run: python 01_vram_budget.py
"""

import torch
import torch.nn as nn


# ───────────────────────────────────────────────────────────
# Count parameters — Module 3.1 sized
# ───────────────────────────────────────────────────────────
def count_params(model):
    return sum(p.numel() for p in model.parameters())


small = nn.Sequential(
    nn.Linear(784, 256), nn.ReLU(),
    nn.Linear(256, 256), nn.ReLU(),
    nn.Linear(256, 10),
)
print(f"small MLP: {count_params(small):,} params")


# ───────────────────────────────────────────────────────────
# A model-size-vs-memory table
# ───────────────────────────────────────────────────────────
def vram_estimate_bytes(num_params: int, dtype_bytes: int,
                       optimizer="adamw") -> dict[str, int]:
    """Memory in bytes for params + grads + optimizer state.

    Excludes activations — those depend on batch & sequence length.
    """
    p = num_params * dtype_bytes
    g = num_params * dtype_bytes
    if optimizer == "adamw":
        o = 2 * num_params * 4         # m, v in fp32 even if model is bf16
    elif optimizer == "sgd":
        o = num_params * dtype_bytes   # momentum, same dtype
    elif optimizer == "8bit_adamw":
        o = 2 * num_params * 1         # 8-bit Adam (bnb)
    else:
        o = 0
    return {"params": p, "grads": g, "opt": o, "total": p + g + o}


def fmt(n_bytes: int) -> str:
    units = [("PB", 1e15), ("TB", 1e12), ("GB", 1e9), ("MB", 1e6), ("KB", 1e3)]
    for label, scale in units:
        if n_bytes >= scale:
            return f"{n_bytes / scale:.2f} {label}"
    return f"{n_bytes} B"


print("\n=== Model-size vs memory (params + grads + AdamW state, NO activations) ===")
print(f"{'model':<10}  {'fp32':>10}  {'bf16+fp32 opt':>18}  {'8-bit opt':>14}  {'4-bit + LoRA':>16}")

models = [
    ("125M",   125_000_000),
    ("350M",   350_000_000),
    ("1B",     1_000_000_000),
    ("7B",     7_000_000_000),
    ("13B",    13_000_000_000),
    ("70B",    70_000_000_000),
]

for name, n in models:
    fp32 = vram_estimate_bytes(n, 4)["total"]
    bf16 = vram_estimate_bytes(n, 2)["total"]   # bf16 model + fp32 Adam state
    bit8 = vram_estimate_bytes(n, 2, "8bit_adamw")["total"]
    # LoRA: only ~0.5% of params trainable. Frozen base in 4-bit.
    lora_trainable = int(n * 0.005)
    lora_total = (n * 0.5)                    # 4-bit base, no grad/opt state for it
    lora_total += vram_estimate_bytes(lora_trainable, 2)["total"]
    print(f"{name:<10}  {fmt(fp32):>10}  {fmt(bf16):>18}  {fmt(bit8):>14}  {fmt(int(lora_total)):>16}")


# ───────────────────────────────────────────────────────────
# Activations — the BATCH-DEPENDENT part
# ───────────────────────────────────────────────────────────
# A rough rule for a transformer:
#   activations ≈ B × T × L × H × 4 bytes × ~10
# where:
#   B = batch size
#   T = sequence length
#   L = number of layers
#   H = hidden dim
# The "× 10" is a fudge factor for intermediate tensors saved for backward.
def transformer_activations_bytes(B, T, L, H):
    return B * T * L * H * 4 * 10


print("\n=== Activations for a 7B transformer (L=32, H=4096) at different batch/seq ===")
for batch, seq in [(1, 512), (1, 2048), (4, 2048), (8, 2048)]:
    a = transformer_activations_bytes(batch, seq, L=32, H=4096)
    print(f"  B={batch}, T={seq}: {fmt(a)}")


# ───────────────────────────────────────────────────────────
# How to actually fit a 7B on a 24GB GPU
# ───────────────────────────────────────────────────────────
# Tricks (each saves activations or state):
#   1. bf16 weights + Adam in 8-bit (bnb)        -> ~22 GB total state
#   2. + GRADIENT CHECKPOINTING                  -> activations × ~2 instead of × 10
#   3. + GRADIENT ACCUMULATION (small batches)
#   4. + LoRA / QLoRA (4-bit base, only adapter trains)  -> single 16-24 GB GPU
#   5. + FSDP or DeepSpeed Zero-3 across GPUs    -> shard everything
#
# Files 02-08 cover each of these in detail. This file just gave you the math
# to PLAN a run before you queue it up and waste hours.


# ───────────────────────────────────────────────────────────
# Real-time memory inspection (PyTorch on CUDA)
# ───────────────────────────────────────────────────────────
# torch.cuda.memory_allocated()  — current bytes in use
# torch.cuda.max_memory_allocated() — peak
# torch.cuda.memory_summary()    — detailed breakdown
# nvidia-smi / nvitop            — system-level
#
# On CPU these all return 0; rely on python's resource module or psutil instead.
