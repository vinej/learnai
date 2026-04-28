# Module 3.5 — Training at Scale

**Level:** 3 — Advanced
**Estimated time:** 2 weeks

## Goal
Train bigger models, faster, on more data — without melting your GPU budget.

## Topics
### Hardware basics
- GPU memory hierarchy (HBM, SRAM)
- VRAM budget: parameters + gradients + optimizer state + activations
- Tensor cores, fp32 vs fp16 vs bf16 vs fp8

### Speed & memory tricks
- Mixed precision training (`torch.amp`)
- Gradient checkpointing
- Gradient accumulation
- Efficient attention: Flash Attention
- `torch.compile` (PyTorch 2.x)

### Distributed training
- Data parallel (DP) vs Distributed Data Parallel (DDP)
- Model parallel, pipeline parallel, tensor parallel
- Frameworks: **Hugging Face Accelerate**, **DeepSpeed**, **FSDP**, **Megatron-LM**
- Cluster orchestration: Slurm, Ray, Kubernetes

### Profiling & optimization
- PyTorch profiler, `nvidia-smi`, `nvtop`
- Identifying bottlenecks (data loading vs compute vs comms)
- `num_workers`, pinned memory, prefetching

### Cloud training
- Renting GPUs (Lambda, RunPod, vast.ai, AWS/GCP)
- Spot instances, checkpointing for resilience
- Cost tracking

## Exercises
1. Take a Module 3.3 fine-tune and add mixed precision; measure speedup and memory savings.
2. Train the same model with DDP on 2 GPUs (or simulate with 2 processes) using Accelerate.
3. Profile a slow training loop and identify the bottleneck.
4. Estimate the VRAM needed to fully fine-tune a 7B-parameter model — and compare to LoRA.

## Capstone (Level 3)
Pick a domain (vision, text, audio). Fine-tune a strong pretrained model on a custom dataset, with mixed precision and proper experiment tracking. Ship a Streamlit or Gradio demo.

## Resources
- Hugging Face Accelerate docs
- DeepSpeed docs
- Stas Bekman's "Machine Learning Engineering" book (free)

## Checkpoint
You can plan the hardware required to train a given model, choose the right parallelism strategy, and diagnose why a training run is slow.
