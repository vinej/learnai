"""
07 — Serving fine-tuned models

You've trained a LoRA. Now what? Production inference for an open-weight
LLM needs different tools depending on scale and hardware.

THE LANDSCAPE
  vLLM             — single-machine, GPU-first, paged attention, very fast.
                     Continuous batching → 10-30× throughput vs naive HF.
  TGI              — Hugging Face's production server. Similar to vLLM.
  TensorRT-LLM     — NVIDIA's; squeezes more performance on H100/L40.
  llama.cpp / Ollama — CPU and Apple Silicon, GGUF format.
  SGLang           — newer; best-in-class for structured outputs.

This file is conceptual — running these requires the corresponding daemon.
The key point: pretty much all of them expose an OpenAI-compatible HTTP API,
so your application code is portable.

Run: python 07_serving.py
"""


VLLM_LAUNCH = """\
# Single-line vLLM serve. One daemon, one model.
#   pip install vllm
#   vllm serve mistralai/Mistral-7B-Instruct-v0.3 \\
#       --port 8000 --tensor-parallel-size 1 --max-model-len 8192

# To LOAD A LoRA ADAPTER on top of a base model:
#   vllm serve meta-llama/Llama-3.1-8B \\
#       --enable-lora \\
#       --lora-modules my_adapter=/path/to/lora_dir

# Then use the OpenAI SDK against it:
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
resp = client.chat.completions.create(
    model="my_adapter",                          # the adapter name
    messages=[{"role": "user", "content": "Hi"}],
)
print(resp.choices[0].message.content)
"""


TGI_LAUNCH = """\
# Text Generation Inference (Hugging Face)
#   docker run --gpus all --shm-size 1g -p 8080:80 \\
#     ghcr.io/huggingface/text-generation-inference:latest \\
#     --model-id meta-llama/Llama-3.1-8B-Instruct
"""


OLLAMA_FLOW = """\
# Ollama — easiest local serving (Mac/Linux/Windows).
#   ollama pull llama3.1:8b
#   ollama run llama3.1:8b "hello"
#
# To serve YOUR fine-tuned model:
#   1. Convert to GGUF with `llama.cpp/convert_hf_to_gguf.py`.
#      (Bake your LoRA into the base first, or merge during conversion.)
#   2. Create a Modelfile:
#         FROM ./my-llama3-8b-q4.gguf
#         PARAMETER temperature 0.7
#         PARAMETER stop "<|eot_id|>"
#   3. Build and run it:
#         ollama create my-tuned -f Modelfile
#         ollama run my-tuned
#
# Ollama's HTTP API mirrors OpenAI's at http://localhost:11434/v1.
"""


print("=== vLLM ===")
print(VLLM_LAUNCH)
print("\n=== TGI ===")
print(TGI_LAUNCH)
print("\n=== Ollama ===")
print(OLLAMA_FLOW)


# ───────────────────────────────────────────────────────────
# Throughput tricks the modern serving stack uses
# ───────────────────────────────────────────────────────────
TRICKS = """\
WHY VLLM / TGI ARE SO FAST
  • Continuous batching — start NEW sequences mid-batch as old ones finish.
                          Naive batching pads to the longest sequence; you waste
                          most of the GPU on padding.
  • Paged attention — store KV cache in non-contiguous "pages" instead of one
                      big block. Up to 4× more requests in flight.
  • Speculative decoding — a small "draft" model proposes tokens; the big
                           model verifies in parallel. ~2-3× decoding speed.
  • Prefix caching — multiple requests with the same system prompt share
                      a cached KV. Big win for chat assistants.
  • Quantization-aware kernels — int8 / fp8 GEMMs.
"""
print(TRICKS)


# ───────────────────────────────────────────────────────────
# Picking a stack
# ───────────────────────────────────────────────────────────
DECISION = """\
LOCAL DEVELOPMENT, ANY OS         → Ollama. One command per model.
Single GPU server, Linux           → vLLM. Best throughput / quality / community.
GPU fleet, k8s                     → vLLM or TGI behind a load balancer.
Edge / mobile / consumer laptop    → llama.cpp + GGUF.
NVIDIA-only, max performance       → TensorRT-LLM (build from sources).
Heavy structured output / agents   → SGLang (controlled decoding shines).
"""
print(DECISION)


# ───────────────────────────────────────────────────────────
# Putting it together: the "right" deployment
# ───────────────────────────────────────────────────────────
# 1. Train LoRA with QLoRA on a beefy GPU instance (rent it).
# 2. MERGE the LoRA into the base (peft.merge_and_unload) → single weights file.
# 3. Quantize the merged model to AWQ (for vLLM) or GGUF (for Ollama).
# 4. Serve with vLLM / Ollama behind your usual API gateway.
# 5. Monitor: latency, tokens/sec, error rate, hallucination rate.
# 6. Compare to a frontier API (Claude / GPT-4o) on cost AND quality.
#    If the gap is small enough, the savings from self-hosting pay back fast.
