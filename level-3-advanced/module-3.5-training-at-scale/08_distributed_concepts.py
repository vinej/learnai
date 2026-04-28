"""
08 — Distributed training: concepts and frameworks

The naming is dense. Here's the cheat sheet.

DATA PARALLEL (DP)
  Same model copied to each GPU. Each GPU processes a different mini-batch.
  Gradients are averaged across GPUs (allreduce) and applied. Old `nn.DataParallel`
  uses Python threads (slow, deprecated). DDP (next) is the right choice.

DISTRIBUTED DATA PARALLEL (DDP)
  Same as DP but using one PROCESS per GPU + NCCL all-reduce. Fast, the
  default for multi-GPU training.

MODEL PARALLEL
  Different LAYERS of the model live on different GPUs. Used when the model
  itself can't fit on one GPU. Variants:
  - PIPELINE PARALLEL: layers split into stages, micro-batches flow through
    in pipeline (GPipe, PipeDream).
  - TENSOR PARALLEL: a SINGLE matrix multiply is split across GPUs
    (Megatron-LM). Most useful within a single node with NVLink.

FULLY-SHARDED DATA PARALLEL (FSDP) / DEEPSPEED ZERO
  Each GPU holds only a SHARD of the model parameters, gradients, and
  optimizer state. Communication is more complex but you can train
  models far bigger than any single GPU.

FRAMEWORKS
  torch.distributed (DDP, FSDP)  — built-in PyTorch.
  Hugging Face Accelerate         — thin wrapper, single config.
  DeepSpeed                       — Zero stages 1/2/3, offloading, MoE.
  Megatron-LM                     — for very large language models, tensor
                                    + pipeline parallel.
  PyTorch Lightning / Fabric      — high-level trainer abstractions.

This file is conceptual — it doesn't run multi-process. The next file's
exercise (Accelerate) shows the real one-line conversion.

Run: python 08_distributed_concepts.py
"""

import os


# ───────────────────────────────────────────────────────────
# Minimal DDP launch script (you would NOT run this here)
# ───────────────────────────────────────────────────────────
# Usually saved as `train_ddp.py` and launched with `torchrun`:
#
#   torchrun --nproc_per_node=4 train_ddp.py
#
# The launcher sets RANK / LOCAL_RANK / WORLD_SIZE env vars per process.
DDP_TEMPLATE = """
import os, torch, torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 1. Init the process group
dist.init_process_group("nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
device = f"cuda:{local_rank}"

# 2. Build the model and wrap in DDP
model = MyModel().to(device)
model = DDP(model, device_ids=[local_rank])

# 3. Build the dataloader with a DistributedSampler so each rank sees
#    a different shard.
from torch.utils.data.distributed import DistributedSampler
sampler = DistributedSampler(dataset)
loader = DataLoader(dataset, batch_size=BATCH, sampler=sampler)

# 4. Standard training loop. DDP automatically all-reduces gradients
#    on backward.
for epoch in range(EPOCHS):
    sampler.set_epoch(epoch)            # so shuffling differs per epoch
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

dist.destroy_process_group()
"""

print("DDP template:")
print(DDP_TEMPLATE)


# ───────────────────────────────────────────────────────────
# When to pick what
# ───────────────────────────────────────────────────────────
DECISION_TREE = """
Single GPU, model fits comfortably?
    → just train normally.

Multiple GPUs on ONE machine, model fits per GPU?
    → DDP (or accelerate). Linear scaling.

Multiple GPUs, model BARELY fits?
    → DDP + gradient checkpointing + bf16.
    → if still too big: FSDP / DeepSpeed Zero-3.

Model TOO BIG for one GPU even with the above?
    → FSDP / DeepSpeed Zero-3 (shard params + grads + optimizer state).
    → For very large LLMs (>30B params), tensor + pipeline parallel
      (Megatron-LM, deepspeed-megatron, NeMo).

Multi-NODE training?
    → all the above + a launcher: torchrun, slurm, kubeflow, ray.
    → make sure your network (NVLink/InfiniBand) is fast enough to not
      bottleneck communication.
"""
print(DECISION_TREE)


# ───────────────────────────────────────────────────────────
# Communication primitives you'll see
# ───────────────────────────────────────────────────────────
# all_reduce      — every rank gets the SUM (or mean) of all ranks' tensors.
#                   The DDP backward pass uses all_reduce on gradients.
# all_gather      — every rank gets the CONCATENATION of all ranks' tensors.
# reduce_scatter  — sum across ranks, each rank gets one slice. Used in FSDP.
# broadcast       — one rank shares its tensor with everyone else.
#
# These all happen on a "process group", and the choice of BACKEND matters:
#   - NCCL    — for CUDA tensors, fastest by far. Default in DDP.
#   - GLOO    — CPU tensors, more flexible but slower.
#   - MPI     — HPC clusters with MPI installed.


# ───────────────────────────────────────────────────────────
# A friendly reality check
# ───────────────────────────────────────────────────────────
# Most engineers will train on 1 GPU for years. Distributed training is
# tricky to set up and debug — so DON'T rush into it.
#
# Common mistakes:
#   • Wrong learning-rate scaling. Linear-scale rule: LR ∝ effective batch size.
#   • Forgetting DistributedSampler.set_epoch(epoch) → identical shuffling
#     across epochs, model overfits to data order.
#   • model = DDP(model.to("cuda"))     — wrap AFTER moving to device.
#   • Spawning workers = world_size × num_workers, blowing up memory.
#   • `print()` from every rank — flood. Print only from rank 0.
print(f"\n[demo] is this multi-process? RANK={os.environ.get('RANK', '<unset>')}")
