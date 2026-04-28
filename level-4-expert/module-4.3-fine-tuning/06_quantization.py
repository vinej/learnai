"""
06 — Quantization: making big models fit

Quantization stores weights at lower precision (4-bit, 8-bit) to reduce
memory and speed up inference. Three families you'll meet:

  POST-TRAINING QUANTIZATION (PTQ) — quantize a TRAINED model after the fact.
    AWQ      — activation-aware; small calibration set; near-lossless 4-bit.
    GPTQ     — sequential layer-wise quantization; well-supported.
    GGUF     — llama.cpp's mmap-friendly format; runs on CPU and Apple Silicon.
    bitsandbytes nf4 — what QLoRA uses for training; easy 4-bit loading.

  QUANTIZATION-AWARE TRAINING (QAT) — train with simulated quantization.
    Higher quality at very low precision (int4, int2) but more involved.
    Used in production: Llama 3 / 4 publish QAT-flavored variants.

  KV CACHE QUANTIZATION — quantize the attention cache during inference.
    Long-context generation becomes feasible on smaller GPUs.

This file compares effective MODEL SIZE at different precisions and
discusses the trade-offs. Actual quantized inference needs GPU + bitsandbytes.

Run: python 06_quantization.py
"""

import torch
from transformers import AutoModelForCausalLM


# ───────────────────────────────────────────────────────────
# Compute the on-disk size of a model at different precisions
# ───────────────────────────────────────────────────────────
def size_in_gb(num_params: int, bits_per_weight: float) -> float:
    return num_params * bits_per_weight / 8 / 1e9


sizes = {
    "fp32 (4 bytes)":              32,
    "bf16 / fp16 (2 bytes)":       16,
    "int8 (1 byte)":                8,
    "4-bit (NF4 / GPTQ / AWQ)":     4,
    "3-bit (rare; quality drop)":   3,
    "2-bit (extreme, lossy)":       2,
}

print(f"=== model SIZE in GB at different precisions ===")
print(f"{'precision':<28}  {'7B':>8}  {'13B':>8}  {'70B':>8}")
for label, bits in sizes.items():
    s = size_in_gb(7e9, bits)
    m = size_in_gb(13e9, bits)
    l = size_in_gb(70e9, bits)
    print(f"{label:<28}  {s:>8.2f}  {m:>8.2f}  {l:>8.2f}")


# ───────────────────────────────────────────────────────────
# What you LOSE going to 4-bit
# ───────────────────────────────────────────────────────────
QUALITY_NOTES = """\
4-bit AWQ / GPTQ on a well-pretrained model:
  • MMLU drop:         < 1 percentage point (often 0)
  • Real-world chat:   imperceptible
  • Edge cases:        sometimes more "drift" on long generations

3-bit / 2-bit:
  • Visible quality regression. Reserve for ultra-constrained edge devices
  • Or use SPECIALIZED methods like AQLM and QuIP#.

Conclusion: the 4-bit tier is essentially free. Quantize.
"""
print(QUALITY_NOTES)


# ───────────────────────────────────────────────────────────
# How to actually load a quantized model
# ───────────────────────────────────────────────────────────
SNIPPETS = {
    "bitsandbytes 4-bit (training & inference)": """\
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    quantization_config=config,
    device_map="auto",
)
""",

    "AWQ (autoawq)": """\
# pip install autoawq
from awq import AutoAWQForCausalLM

# Pre-quantized models on HF: TheBloke/Llama-3-8B-AWQ, etc.
model = AutoAWQForCausalLM.from_quantized(
    "TheBloke/Llama-3-8B-AWQ",
    fuse_layers=True,
    device_map="auto",
)
""",

    "GPTQ (auto-gptq)": """\
# pip install auto-gptq
from auto_gptq import AutoGPTQForCausalLM
model = AutoGPTQForCausalLM.from_quantized(
    "TheBloke/Llama-3-8B-GPTQ",
    device="cuda:0",
)
""",

    "GGUF + llama-cpp-python (CPU & Apple Silicon)": """\
# pip install llama-cpp-python
from llama_cpp import Llama
llm = Llama(
    model_path="path/to/llama-3-8b-instruct.Q4_K_M.gguf",
    n_ctx=4096,
    n_gpu_layers=0,                # > 0 to offload to GPU on macOS
)
print(llm("Hello", max_tokens=64)["choices"][0]["text"])
""",

    "Ollama (CLI server, easiest)": """\
# After installing Ollama (https://ollama.com):
#   ollama pull llama3.1:8b       # downloads quantized GGUF for you
#   ollama run llama3.1:8b
# Or use as an OpenAI-compatible API:
from openai import OpenAI
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
resp = client.chat.completions.create(
    model="llama3.1:8b",
    messages=[{"role": "user", "content": "hi"}],
)
""",
}

for label, code in SNIPPETS.items():
    print(f"\n=== {label} ===")
    print(code)


# ───────────────────────────────────────────────────────────
# Picking a format
# ───────────────────────────────────────────────────────────
DECISION = """\
MAC / CPU inference     → GGUF (run via Ollama or llama-cpp-python).
Single GPU inference    → AWQ or GPTQ. Both are fast; AWQ slightly higher quality.
GPU training (QLoRA)    → bitsandbytes NF4 (4-bit base + LoRA adapters).
Production fleet of GPUs → vLLM with AWQ or GPTQ; high throughput, paged attention.
Edge / mobile            → AQLM, QuIP# (2-3-bit), specialized framework.
"""
print(DECISION)


# ───────────────────────────────────────────────────────────
# Distilgpt2 actual size (sanity demo, not really quantized)
# ───────────────────────────────────────────────────────────
m = AutoModelForCausalLM.from_pretrained("distilgpt2")
n = sum(p.numel() for p in m.parameters())
print(f"\ndistilgpt2: {n:,} params")
for label, bits in sizes.items():
    print(f"  {label:<28}  {size_in_gb(n, bits) * 1000:>7.1f} MB")
