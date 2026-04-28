# Setup — Module 3.5 (Training at Scale)

This module is about **understanding** scale, not necessarily *running at* scale. Many techniques (DDP, FSDP, Flash Attention, fp16/bf16 GEMMs) only show real benefits on a GPU. The lessons run on CPU and explain what a GPU would change; the exercises measure effects you can observe even on CPU.

## 1. Python ≥ 3.11 + PyTorch

If you completed Module 3.1, you're set.

## 2. Create / activate the venv

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1     # Windows PS
# or
source .venv/bin/activate       # macOS/Linux
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

| Package        | Purpose                                          |
|----------------|--------------------------------------------------|
| `torch`        | Mixed precision, profiler, compile, checkpointing |
| `accelerate`   | Hugging Face's distributed-training abstraction  |
| `transformers` | Models for the examples                          |
| `numpy`, `matplotlib` | Computation + plots                       |

## 4. CUDA / MPS / CPU?

Check what you have:

```bash
python -c "import torch; print('cuda', torch.cuda.is_available()); print('mps', torch.backends.mps.is_available())"
```

- **No GPU**: most lessons still teach. Mixed-precision speed gains will be small or nonexistent on CPU; the math (memory, throughput) generalizes.
- **NVIDIA GPU with CUDA**: everything works fully (autocast, fused kernels, DDP).
- **Apple Silicon (MPS)**: most things work, but mixed precision is partial.

## 5. Run the lessons

```bash
python 01_vram_budget.py
python 02_mixed_precision.py
python 03_gradient_checkpointing.py
python 04_gradient_accumulation.py
python 05_torch_compile.py
python 06_dataloader_perf.py
python 07_profiling.py
python 08_distributed_concepts.py
```

## 6. Run the exercises

```bash
python exercises/01_mixed_precision_speedup.py
python exercises/02_accelerate_setup.py            # not multi-process; explains the diff
python exercises/03_profile_slow_loop.py
python exercises/04_vram_calculator.py
```

## Tip

When something is slow, RESIST the urge to throw a bigger GPU at it. First profile (file 07). Most "training is slow" issues turn out to be a too-small `num_workers`, an un-pinned dataloader, or a stupidly placed `.cpu()` in the inner loop.
