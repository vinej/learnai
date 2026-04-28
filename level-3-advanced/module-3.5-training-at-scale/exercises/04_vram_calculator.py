"""
Exercise 4 — VRAM calculator

Given a model size and training configuration, estimate the peak GPU
memory required. The script asserts a few sanity-check numbers from
well-known recipes (7B fp32 needs ~112 GB; 7B QLoRA fits in ~16 GB).

Run: python exercises/04_vram_calculator.py
"""

from dataclasses import dataclass


@dataclass
class TrainConfig:
    n_params: int                # total model parameters
    dtype_bytes: int             # 4 fp32, 2 bf16/fp16, 1 int8, 0.5 4-bit
    optimizer: str               # 'adamw', '8bit_adamw', 'sgd', 'adafactor', 'none'
    trainable_params: int = None # for LoRA: only the adapter trains
    activation_factor: float = 10.0   # rough bytes-per-param for activations
    batch_seq_factor: float = 1.0     # multiplied into activations


def memory_estimate(cfg: TrainConfig) -> dict[str, float]:
    """All numbers in bytes."""
    p = cfg.n_params * cfg.dtype_bytes              # weights
    trainable = cfg.trainable_params or cfg.n_params
    g = trainable * cfg.dtype_bytes                  # gradients (only for trainable)

    # Optimizer state — usually fp32 even with bf16 weights.
    if cfg.optimizer == "adamw":
        o = 2 * trainable * 4
    elif cfg.optimizer == "8bit_adamw":
        o = 2 * trainable * 1
    elif cfg.optimizer == "sgd":
        o = trainable * cfg.dtype_bytes
    elif cfg.optimizer == "adafactor":
        # Adafactor stores per-row & per-col stats — roughly 0.1× params
        o = int(0.1 * trainable * 4)
    elif cfg.optimizer == "none":
        o = 0
    else:
        raise ValueError(f"unknown optimizer {cfg.optimizer}")

    # Activations — VERY rough estimate: factor × params × dtype_bytes
    # × batch_seq_factor (you'd refine for a real model).
    a = int(cfg.activation_factor * cfg.dtype_bytes * cfg.batch_seq_factor * cfg.n_params)

    return {
        "weights":      p,
        "grads":        g,
        "optimizer":    o,
        "activations":  a,
        "total":        p + g + o + a,
    }


def fmt_gb(n_bytes: float) -> str:
    return f"{n_bytes / 1e9:.2f} GB"


def print_breakdown(label: str, mem: dict):
    total = mem["total"]
    print(f"\n=== {label} ===")
    for k in ["weights", "grads", "optimizer", "activations"]:
        share = mem[k] / max(total, 1) * 100
        print(f"  {k:<12}  {fmt_gb(mem[k]):>10}  ({share:>4.1f}%)")
    print(f"  {'TOTAL':<12}  {fmt_gb(total):>10}")


# ───────────────────────────────────────────────────────────
# Reference scenarios
# ───────────────────────────────────────────────────────────
N_7B  = 7_000_000_000
N_13B = 13_000_000_000

# A. 7B full fine-tune in fp32 + AdamW. The "what you'd hope was small".
fp32_full = memory_estimate(TrainConfig(
    n_params=N_7B, dtype_bytes=4, optimizer="adamw",
))
print_breakdown("7B full fine-tune, fp32 + AdamW", fp32_full)

# B. 7B full fine-tune in bf16 + 8-bit Adam — common modern recipe.
bf16_8bit = memory_estimate(TrainConfig(
    n_params=N_7B, dtype_bytes=2, optimizer="8bit_adamw",
))
print_breakdown("7B full fine-tune, bf16 + 8-bit AdamW", bf16_8bit)

# C. 7B QLoRA: 4-bit base, only LoRA adapter trains.
# Trainable ~ 0.5% of full.
qlora = memory_estimate(TrainConfig(
    n_params=N_7B, dtype_bytes=1,            # base in 4-bit (we lie that 1 byte ≈ 4-bit since LoRA itself is bf16)
    optimizer="adamw",
    trainable_params=int(N_7B * 0.005),
    activation_factor=4.0,                    # checkpointing typically on
    batch_seq_factor=1.0,
))
# Override the weights term to use 0.5 byte for the 4-bit frozen base
qlora["weights"] = int(N_7B * 0.5)
qlora["total"] = sum(qlora[k] for k in ["weights", "grads", "optimizer", "activations"])
print_breakdown("7B QLoRA (4-bit base + bf16 LoRA + AdamW)", qlora)

# D. 13B inference only: just weights (no gradients, no optimizer state).
inference = memory_estimate(TrainConfig(
    n_params=N_13B, dtype_bytes=2, optimizer="none",
    trainable_params=0, activation_factor=2.0,
))
print_breakdown("13B INFERENCE in bf16", inference)


# ───────────────────────────────────────────────────────────
# Sanity-check assertions
# ───────────────────────────────────────────────────────────
# fp32 full fine-tune of 7B should be > 100 GB (fits on 8×A100-80GB)
assert fp32_full["total"] > 100e9
# bf16 + 8-bit Adam should drop it well below 100 GB
assert bf16_8bit["total"] < fp32_full["total"] * 0.5
# QLoRA should fit in a single A100 (or 24GB consumer GPU with care)
assert qlora["total"] < 24e9, f"QLoRA estimate too big: {qlora['total'] / 1e9:.1f} GB"
# Inference is just weights for big models
assert inference["weights"] / inference["total"] > 0.5
print("\nVRAM estimates pass sanity checks ✅")


# ───────────────────────────────────────────────────────────
# Caveats
# ───────────────────────────────────────────────────────────
# • CUDA / framework constants (allocator overhead) add 1-2 GB.
# • Activation memory is very SHAPE-DEPENDENT. Long sequences explode it.
# • Gradient checkpointing reduces activations 5-10× at ~30% extra time.
# • In FSDP / Zero-3, each GPU only holds 1/N of params + grads + opt state,
#   so divide accordingly when sizing a multi-GPU run.
