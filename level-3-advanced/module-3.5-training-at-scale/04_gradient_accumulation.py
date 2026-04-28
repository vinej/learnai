"""
04 — Gradient accumulation

Sometimes the BATCH SIZE that matters for accuracy doesn't fit in GPU memory.
Gradient accumulation simulates a big effective batch by accumulating
gradients across smaller micro-batches before stepping the optimizer.

Effective batch = micro_batch_size × accumulation_steps × num_gpus

It's a free trick. The only catch: divide the loss by accumulation_steps so
the gradient magnitudes stay correct.

Run: python 04_gradient_accumulation.py
"""

import torch
import torch.nn as nn


torch.manual_seed(0)
device = "cpu"     # works the same on GPU


model = nn.Sequential(
    nn.Linear(64, 128), nn.ReLU(),
    nn.Linear(128, 10),
).to(device)
opt = torch.optim.SGD(model.parameters(), lr=1e-2)
criterion = nn.CrossEntropyLoss()


# ───────────────────────────────────────────────────────────
# Reference: a single big batch (size=32)
# ───────────────────────────────────────────────────────────
torch.manual_seed(0)
x = torch.randn(32, 64, device=device)
y = torch.randint(0, 10, (32,), device=device)

opt.zero_grad()
loss = criterion(model(x), y)
loss.backward()
big_grad = model[0].weight.grad.clone()


# ───────────────────────────────────────────────────────────
# Accumulated: 4 micro-batches of 8 → same gradient
# ───────────────────────────────────────────────────────────
torch.manual_seed(0)
model_acc = nn.Sequential(
    nn.Linear(64, 128), nn.ReLU(),
    nn.Linear(128, 10),
).to(device)
# Same init via the seed.

opt_acc = torch.optim.SGD(model_acc.parameters(), lr=1e-2)
opt_acc.zero_grad()

ACCUM = 4
MICRO = 8
xs = x.split(MICRO)
ys = y.split(MICRO)

for xb, yb in zip(xs, ys):
    loss = criterion(model_acc(xb), yb) / ACCUM    # CRITICAL: divide!
    loss.backward()                                 # gradients accumulate

# Don't call opt.step() yet — collect the grad first
acc_grad = model_acc[0].weight.grad.clone()


# ───────────────────────────────────────────────────────────
# Compare
# ───────────────────────────────────────────────────────────
diff = (big_grad - acc_grad).abs().max().item()
print(f"max grad difference (single batch=32 vs 4×8 accumulated): {diff:.2e}")
# Should be tiny (just numerical noise from float ops).


# ───────────────────────────────────────────────────────────
# Pattern in a real training loop
# ───────────────────────────────────────────────────────────
# from itertools import islice
#
# ACCUM = 4
# for step, batch in enumerate(loader):
#     loss = criterion(model(batch.x), batch.y) / ACCUM
#     loss.backward()
#
#     if (step + 1) % ACCUM == 0:
#         optimizer.step()
#         optimizer.zero_grad()


# ───────────────────────────────────────────────────────────
# In HuggingFace Trainer
# ───────────────────────────────────────────────────────────
#   TrainingArguments(
#       per_device_train_batch_size=4,      # micro
#       gradient_accumulation_steps=8,      # → effective batch 32 (per device)
#   )


# ───────────────────────────────────────────────────────────
# Things to watch
# ───────────────────────────────────────────────────────────
# • BatchNorm: gradient accumulation MAY hurt because BN stats are computed
#   on each micro-batch, not the full effective batch. Use SyncBN or LayerNorm
#   for distributed/accumulated training.
# • Logging: when you log "loss per step", keep in mind it's loss for ONE
#   micro-batch. Log loss * ACCUM for an "effective batch" view.
# • If you accumulate too aggressively, you reduce wall-clock training speed
#   even if accuracy is the same — a fewer-step bigger-batch run beats a
#   billion-step one.
