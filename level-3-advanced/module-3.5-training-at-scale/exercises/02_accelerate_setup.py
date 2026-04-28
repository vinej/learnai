"""
Exercise 2 — Convert a single-process training script to use Accelerate

Accelerate is Hugging Face's lightweight wrapper around `torch.distributed`
that lets you write the SAME training script for:
  - 1 GPU
  - multiple GPUs on one machine
  - TPUs
  - mixed precision

The same `train()` function below works for all of them. Launch with
`accelerate launch --multi_gpu --num_processes=N train.py` (or set up via
`accelerate config`).

This script demonstrates the conversion. It runs a single-process step and
asserts it produces a valid training loss. To go multi-process, see the
launch commands at the bottom.

Run: python exercises/02_accelerate_setup.py
"""

import torch
import torch.nn as nn
from accelerate import Accelerator
from torch.utils.data import DataLoader, TensorDataset


# ───────────────────────────────────────────────────────────
# The Accelerate-flavored training function
# ───────────────────────────────────────────────────────────
def train(epochs: int = 1):
    # 1. Single object that hides the distributed details.
    accelerator = Accelerator(mixed_precision="bf16" if torch.cuda.is_available() else "no")

    # 2. Build model, optimizer, dataloader as usual.
    model = nn.Sequential(
        nn.Linear(64, 128), nn.ReLU(),
        nn.Linear(128, 10),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    X = torch.randn(512, 64)
    y = torch.randint(0, 10, (512,))
    loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)

    # 3. Hand them to the accelerator. It moves them to the right device,
    #    wraps the model in DDP if multi-GPU, and shards the dataloader.
    model, opt, loader = accelerator.prepare(model, opt, loader)

    final_loss = None
    for epoch in range(epochs):
        for x, y in loader:
            opt.zero_grad()
            loss = criterion(model(x), y)
            # 4. accelerator.backward replaces loss.backward(). It handles
            #    GradScaler / sync points / etc. correctly across configs.
            accelerator.backward(loss)
            opt.step()
            final_loss = loss.item()
        # accelerator.print only prints on RANK 0 — no console flood
        accelerator.print(f"epoch {epoch + 1} done, last loss={final_loss:.4f}")

    return final_loss


def main():
    final = train()
    print(f"\nfinal training loss: {final:.4f}")
    assert final < 5.0, "loss should be a finite, sensible number"
    print("Accelerate-wrapped training works ✅")


if __name__ == "__main__":
    main()


# ───────────────────────────────────────────────────────────
# How to actually go multi-GPU
# ───────────────────────────────────────────────────────────
# Step 1: configure once (interactive).
#
#   accelerate config
#
# This writes ~/.cache/huggingface/accelerate/default_config.yaml
# capturing how many GPUs, mixed precision, TPU, etc.
#
# Step 2: launch your script.
#
#   accelerate launch exercises/02_accelerate_setup.py
#
# Or override config from the command line:
#
#   accelerate launch \
#       --multi_gpu --num_processes=4 \
#       --mixed_precision=bf16 \
#       exercises/02_accelerate_setup.py


# ───────────────────────────────────────────────────────────
# What `prepare()` does
# ───────────────────────────────────────────────────────────
# • Moves model to the device assigned to the current process.
# • Wraps the model in DDP / FSDP / DeepSpeed (depending on config).
# • Splits the dataloader so each process sees a disjoint shard.
# • Hooks the optimizer for gradient accumulation if you opt in.
# • Sets up GradScaler for fp16 or bf16 autocast.
#
# All in one call, idempotent across configs.


# ───────────────────────────────────────────────────────────
# Other useful Accelerate APIs
# ───────────────────────────────────────────────────────────
# accelerator.gather(tensor)         — collect tensors from all ranks
# accelerator.is_main_process        — usually used to gate logging/saving
# accelerator.save_state(directory)  — saves model + opt + RNG, distributed-aware
# accelerator.load_state(directory)  — reload it
# accelerator.set_seed(42)            — reproducible across ranks
