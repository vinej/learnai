# Module 4.3 — Fine-tuning & Adapters

**Level:** 4 — Expert
**Estimated time:** 2-3 weeks

## Goal
Specialize open-source LLMs to your domain when prompting alone isn't enough.

## Topics
### When to fine-tune (vs RAG vs prompting)
- Decision tree: format compliance, domain knowledge, latency/cost, data sensitivity
- Cost/benefit reality check

### Data preparation
- Instruction format (Alpaca, ChatML, Llama chat template, system prompts)
- Dataset cleaning, deduplication, decontamination
- Synthetic data generation with stronger models
- Data quality > data quantity

### Fine-tuning techniques
- **Full fine-tuning** — when you can afford it
- **LoRA** — low-rank adapters, the workhorse
- **QLoRA** — 4-bit quantization + LoRA
- **PEFT** library from Hugging Face
- Adapters, prefix tuning, prompt tuning (briefly)

### Alignment techniques (conceptual)
- Supervised fine-tuning (SFT)
- RLHF (PPO)
- **DPO** (Direct Preference Optimization) — simpler & popular
- Constitutional AI / RLAIF

### Quantization & inference
- Post-training quantization: GPTQ, AWQ, GGUF
- `bitsandbytes`, `auto-gptq`
- Serving quantized models: **vLLM**, **TGI**, **Ollama**, **llama.cpp**

### Evaluation
- Standard benchmarks (MMLU, HellaSwag, GSM8K) — and their limits
- Domain-specific eval sets (build your own!)
- LLM-as-judge with strong reference models
- Pairwise human evaluation

## Exercises
1. QLoRA-fine-tune a 7B model (Llama 3, Mistral, Qwen) on an instruction dataset; serve it with vLLM.
2. Build a small SFT dataset from your own domain; fine-tune and measure quality.
3. Quantize the result to 4-bit GGUF and run it locally with Ollama.
4. Run a DPO training pass on top of an SFT model using a preference dataset.
5. Build a domain-specific eval set and compare base vs fine-tuned model.

## Resources
- Hugging Face PEFT docs
- "Fine-tuning Llama" guides on Hugging Face
- TRL library docs
- Sebastian Raschka's blog (excellent fine-tuning posts)

## Checkpoint
You can decide when to fine-tune, prepare a dataset properly, run a QLoRA job on a 7B+ model, and serve the result behind an OpenAI-compatible API.
