"""
Exercise 3 — Compare model sizes and inference time at different precisions

Cast the same model to fp32 / bf16 / fp16 and compare:
  - on-disk size estimate
  - inference latency on a fixed prompt
  - generated text quality (roughly the same; eyeball it)

True 4-bit / 8-bit quantization needs `bitsandbytes` and a CUDA GPU.
On CPU we can still demonstrate fp32 vs bf16 vs fp16 size and speed.

The script asserts that bf16/fp16 model sizes (in memory) are ~half of fp32.

Run: python exercises/03_quantization_compare.py
"""

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "distilgpt2"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
prompt = "The capital of France is"
input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)


# ───────────────────────────────────────────────────────────
# Helper: load model at a given dtype, measure size and speed
# ───────────────────────────────────────────────────────────
def load_and_measure(dtype, label):
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype).to(device)

    # Approximate in-memory size from parameters
    bytes_per = {torch.float32: 4, torch.float16: 2, torch.bfloat16: 2}[dtype]
    param_count = sum(p.numel() for p in model.parameters())
    size_mb = param_count * bytes_per / 1e6

    # Inference latency: 5 forward passes, take avg
    model.eval()
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(5):
            _ = model(input_ids).logits
    if device == "cuda":
        torch.cuda.synchronize()
    fwd_ms = (time.perf_counter() - t0) / 5 * 1000

    # Quick generation
    out = model.generate(input_ids, max_new_tokens=20, do_sample=False,
                         pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(out[0], skip_special_tokens=True)

    print(f"  {label:<10}  size {size_mb:>6.1f} MB   fwd {fwd_ms:>6.1f} ms   text='{text[:60]}'")
    return size_mb, fwd_ms


print(f"\n=== same model, three precisions ===")
sz_fp32, _ = load_and_measure(torch.float32, "fp32")
sz_bf16, _ = load_and_measure(torch.bfloat16, "bf16")

# fp16 is GPU-only; on CPU it falls back to slower kernels but still works.
sz_fp16, _ = load_and_measure(torch.float16, "fp16")


# ───────────────────────────────────────────────────────────
# Sanity
# ───────────────────────────────────────────────────────────
assert abs(sz_bf16 / sz_fp32 - 0.5) < 0.05, \
    "bf16 should be ~half the size of fp32"
assert abs(sz_fp16 / sz_fp32 - 0.5) < 0.05, \
    "fp16 should be ~half the size of fp32"
print("\nsize halving as expected ✅")


# ───────────────────────────────────────────────────────────
# What about TRUE 4-bit?
# ───────────────────────────────────────────────────────────
HOW_TO_4BIT = """\
TRUE 4-bit (NF4 via bitsandbytes) — GPU only:

  pip install bitsandbytes
  ```
  from transformers import BitsAndBytesConfig
  config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_compute_dtype=torch.bfloat16,
  )
  model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=config)
  ```
  Storage: ~0.5 byte/param. A 7B model fits in ~3.5 GB of VRAM.

GGUF Q4_K_M for CPU/Apple Silicon:
  Convert with llama.cpp's `convert_hf_to_gguf.py`, then quantize:
    ./quantize model.gguf model.Q4_K_M.gguf Q4_K_M
  Run via `llama-cpp-python` or Ollama.

A common workflow:
  - Train on GPU with QLoRA (4-bit base + LoRA adapter).
  - For deployment: merge adapter + base, quantize again to Q4_K_M GGUF
    for distribution.
"""
print(HOW_TO_4BIT)


# ── Stretch ─────────────────────────────────────────────────────────────────
# • If you have a GPU, install bitsandbytes and add a 4-bit row to the table.
# • Generate 200 tokens at each precision; compare quality with LLM-as-judge.
# • Compare against an actual GGUF Q4_K_M model loaded via llama-cpp-python.
# ─────────────────────────────────────────────────────────────────────────────
