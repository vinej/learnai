# Setup — Module 4.3 (Fine-tuning & Adapters)

This module covers WHEN and HOW to fine-tune. Real fine-tuning of 7B+ models needs a GPU; the exercises here use TINY models (GPT-2 / distilgpt2 / TinyLlama) so they run on CPU in minutes — at the cost of unimpressive output quality. The TECHNIQUE is the lesson; quality scales with hardware.

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

| Package         | Purpose                                          |
|-----------------|--------------------------------------------------|
| `transformers`  | Tokenizers + models                              |
| `peft`          | LoRA / QLoRA / prefix tuning                     |
| `trl`           | SFTTrainer, DPOTrainer (alignment training)      |
| `datasets`      | Dataset loading                                  |
| `accelerate`    | Distributed / mixed precision                    |
| `anthropic`     | LLM-as-judge for the eval exercise               |
| `numpy`, `matplotlib` | Math + plots                                |

> **GPU-only optional packages** (skip if CPU-only):
> ```bash
> pip install bitsandbytes auto-gptq
> ```

## 4. ANTHROPIC_API_KEY (for the eval exercise)

Set as in [Module 4.1](../module-4.1-llms-prompt-engineering/INSTALL.md). The eval exercise uses Claude as a grader.

## 5. Disk + memory

- distilgpt2:        ~330 MB
- gpt2:              ~500 MB
- TinyLlama-1.1B:    ~2.2 GB
- Qwen2-0.5B:        ~1 GB

LoRA adapters add ~5-50 MB on top. CPU LoRA training of a 125M model on 1000 samples runs in a few minutes.

## 6. Run the lessons

```bash
python 01_when_to_finetune.py        # offline
python 02_data_preparation.py        # offline
python 03_lora.py                    # CPU (~1-2 min download + ~2 min train)
python 04_qlora.py                   # walk-through (CPU code paths)
python 05_alignment.py               # offline + small DPO toy
python 06_quantization.py            # offline (compares model sizes)
python 07_serving.py                 # offline
python 08_evaluation.py              # API
```

## 7. Run the exercises

```bash
python exercises/01_lora_finetune.py
python exercises/02_build_sft_dataset.py
python exercises/03_quantization_compare.py
python exercises/04_dpo_demo.py
python exercises/05_domain_eval.py
```

## Tip

For real fine-tuning of useful-size LLMs (7B+), you'll want a GPU with ≥ 16 GB VRAM and tools like Unsloth or Axolotl. This module's job is to make the recipes in those tools READABLE — not to train production models.
