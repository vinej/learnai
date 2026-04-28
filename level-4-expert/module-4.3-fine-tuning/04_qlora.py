"""
04 — QLoRA: 4-bit quantization + LoRA

LoRA freezes the base model in fp16/bf16. QLoRA goes further:

  1. Load the base model in 4-BIT precision (NF4 dtype, bitsandbytes).
  2. Train LoRA adapters in bf16 on top, gradients flowing only through
     the adapters.
  3. Activations are dequantized on the fly during forward; the 4-bit
     storage stays put.

Why it matters:
  - A 7B model in fp32 = 28 GB; in bf16 = 14 GB; in 4-bit = ~3.5 GB.
  - Fits a 7B fine-tune on a SINGLE 16 GB consumer GPU. Or 13B on 24 GB.
  - Uses paged optimizer states to spill to host memory if needed.

QLoRA needs `bitsandbytes` and a CUDA GPU. The code below is the canonical
recipe; it will skip actual training if you don't have a GPU. The lesson is
the SHAPE of the script — every modern QLoRA fine-tune looks like this.

Run: python 04_qlora.py
"""

import sys

import torch


HAS_CUDA = torch.cuda.is_available()


# ───────────────────────────────────────────────────────────
# The canonical QLoRA setup
# ───────────────────────────────────────────────────────────
QLORA_RECIPE = """\
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, TrainingArguments)
from trl import SFTTrainer, SFTConfig

MODEL = "meta-llama/Llama-3.1-8B"

# 1. 4-bit quant config — NF4 is the QLoRA paper's recommendation.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# 2. Load the base in 4-bit.
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    quantization_config=bnb_config,
    device_map="auto",          # spread across available GPUs
    torch_dtype=torch.bfloat16,
)

# 3. Prepare for k-bit training (cast layer norms to fp32, enable grad checkpointing).
model = prepare_model_for_kbit_training(model)

# 4. Attach LoRA.
lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, lora_config)

# 5. SFTTrainer (TRL) handles the boring parts: chat template, packing, LR sched.
ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:5000]")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=ds,
    args=SFTConfig(
        output_dir="qlora-llama3-8b",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,        # effective batch 16
        gradient_checkpointing=True,
        bf16=True,
        learning_rate=2e-4,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        max_seq_length=2048,
        packing=True,                          # pack multiple examples per seq
        logging_steps=10,
    ),
)

trainer.train()
trainer.save_model()
"""

print("=== QLoRA recipe (Llama-3.1-8B) ===")
print(QLORA_RECIPE)


# ───────────────────────────────────────────────────────────
# What changes vs vanilla LoRA?
# ───────────────────────────────────────────────────────────
# • BitsAndBytesConfig → load base in 4-bit.
# • prepare_model_for_kbit_training → fix dtypes / enable grad checkpointing.
# • LoRA targets ALL projection layers (boost quality at minor cost).
# • bf16 mixed precision (or fp16 on older GPUs).
# • Gradient accumulation lets you simulate bigger batches.
# • SFTTrainer / SFTConfig from TRL — opinionated wrapper around HF Trainer.


# ───────────────────────────────────────────────────────────
# Resource expectations (rough)
# ───────────────────────────────────────────────────────────
RESOURCE_TABLE = """\
=== Realistic VRAM and time estimates ===

7B QLoRA, 16-bit master, bs=4, seq=2048:
  GPU                  VRAM peak     time/1k samples
  RTX 4090 (24GB)         ~14 GB     ~5 min
  A100 40GB                ~14 GB     ~2 min
  T4 (Colab free, 16GB)   ~13 GB     ~15 min

13B QLoRA, same config:
  RTX 4090                ~22 GB (just fits)   ~10 min
  A100 80GB               ~22 GB             ~4 min

70B QLoRA:
  Single A100/H100 80GB     no — needs FSDP/Zero-3 across multi-GPU.
  Multi-GPU FSDP            ~80 GB / GPU       ~30 min/1k samples

Useful tools:
  • Unsloth — patches HF for ~2× faster QLoRA + lower memory.
  • Axolotl — config-driven QLoRA pipeline.
  • Together AI / Modal / Replicate — rent GPUs by the second.
"""
print(RESOURCE_TABLE)


# ───────────────────────────────────────────────────────────
# Sanity check: do you have a GPU?
# ───────────────────────────────────────────────────────────
if HAS_CUDA:
    name = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"\nGPU detected: {name} with {mem:.1f} GB")
    print("→ you can run actual QLoRA training. Use the recipe above.")
else:
    print("\nNo CUDA GPU detected.")
    print("→ QLoRA needs a GPU. Use Modal / Colab / RunPod, or rent an instance.")
