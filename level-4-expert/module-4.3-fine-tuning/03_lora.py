"""
03 — LoRA: Low-Rank Adaptation

Full fine-tuning a 7B model means updating ~7B parameters. LoRA replaces
that with two SMALL trainable matrices A (r×k) and B (d×r) per chosen
layer, where r is a tiny rank (typically 8 or 16). The base weights are
FROZEN. The model computes:

  output = W·x + α/r · (B·A·x)

Where W is the frozen base weight, B·A is the learned low-rank update,
and α/r is a scaling factor.

Why it works: most useful "fine-tunes" are LOW-RANK perturbations to the
base. Rank 8-32 captures the useful change at <1% of full FT parameters.

Result: trainable params drop ~100×, memory drops 4-10×, quality is
within ~1-2% of full fine-tune on most tasks. The default for any 7B+
fine-tune today.

Run: python 03_lora.py    # downloads ~330MB distilgpt2; trains in ~2 min on CPU
"""

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


MODEL_NAME = "distilgpt2"      # tiny, CPU-friendly
torch.manual_seed(0)


# ───────────────────────────────────────────────────────────
# Tokenizer + base model
# ───────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

base = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
total = sum(p.numel() for p in base.parameters())
print(f"base model:    {total:,} params (≈ {total / 1e6:.0f} M)")


# ───────────────────────────────────────────────────────────
# Wrap in a LoRA adapter
# ───────────────────────────────────────────────────────────
# `target_modules` — which layers to attach LoRA to. For GPT-2 the attention
# projection is `c_attn`. For Llama-style models it's `q_proj`, `v_proj`, etc.
# Most fine-tune scripts target the attention projection layers; a few add the
# FFN ones too.
lora_config = LoraConfig(
    r=8,                        # rank
    lora_alpha=16,              # scaling factor (alpha/r)
    target_modules=["c_attn"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(base, lora_config)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"LoRA trainable: {trainable:,} params  "
      f"({trainable / total * 100:.2f}% of base)")
model.print_trainable_parameters()


# ───────────────────────────────────────────────────────────
# Tiny SFT dataset — teach the model a fixed Q&A style
# ───────────────────────────────────────────────────────────
SAMPLES = [
    "Question: What is the capital of France?\nAnswer: Paris is the capital of France.\n",
    "Question: What is the capital of Germany?\nAnswer: Berlin is the capital of Germany.\n",
    "Question: What is the capital of Spain?\nAnswer: Madrid is the capital of Spain.\n",
    "Question: What is the capital of Italy?\nAnswer: Rome is the capital of Italy.\n",
    "Question: What is the capital of Japan?\nAnswer: Tokyo is the capital of Japan.\n",
    "Question: What is the capital of Egypt?\nAnswer: Cairo is the capital of Egypt.\n",
    "Question: What is the capital of Australia?\nAnswer: Canberra is the capital of Australia.\n",
    "Question: What is the capital of Canada?\nAnswer: Ottawa is the capital of Canada.\n",
    "Question: What is the capital of Brazil?\nAnswer: Brasilia is the capital of Brazil.\n",
    "Question: What is the capital of India?\nAnswer: New Delhi is the capital of India.\n",
] * 8        # repeat to give the tiny model enough steps to converge
dataset = Dataset.from_dict({"text": SAMPLES})


def tokenize(batch):
    enc = tokenizer(batch["text"], truncation=True, max_length=128, padding="max_length")
    enc["labels"] = enc["input_ids"].copy()
    return enc


dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])


# ───────────────────────────────────────────────────────────
# Training
# ───────────────────────────────────────────────────────────
args = TrainingArguments(
    output_dir="scratch/lora-distilgpt2",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=2e-4,           # higher LR is fine for LoRA
    weight_decay=0.0,
    logging_steps=10,
    save_strategy="no",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)


print("\nstarting LoRA fine-tune ...")
trainer.train()
print("done")


# ───────────────────────────────────────────────────────────
# Compare BEFORE vs AFTER on a held-out prompt
# ───────────────────────────────────────────────────────────
prompt = "Question: What is the capital of Portugal?\nAnswer:"

base_only = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
def generate(m, p, max_new=20):
    inputs = tokenizer(p, return_tensors="pt")
    out = m.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                     pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0], skip_special_tokens=True)

print("\n=== before fine-tune ===")
print(generate(base_only, prompt))
print("\n=== after LoRA ===")
print(generate(model, prompt))


# ───────────────────────────────────────────────────────────
# Saving and reloading
# ───────────────────────────────────────────────────────────
# LoRA saves only the ADAPTER weights (~5-50 MB), not the base model.
# Reload pattern:
#
#   from peft import PeftModel
#   base = AutoModelForCausalLM.from_pretrained("distilgpt2")
#   model = PeftModel.from_pretrained(base, "scratch/my-adapter")
#
# Multiple adapters can share one base model in memory at the same time.


# ───────────────────────────────────────────────────────────
# Tuning rank, alpha, target_modules
# ───────────────────────────────────────────────────────────
# • r (rank): 4-16 typical. Higher = more capacity, more risk of overfit.
# • alpha:    usually 2 × r. Effective scale = alpha / r.
# • target_modules:
#       - Llama / Mistral / Qwen: ["q_proj", "k_proj", "v_proj", "o_proj"].
#         Adding ["gate_proj", "up_proj", "down_proj"] for the FFN often
#         boosts quality at a modest extra cost.
#       - GPT-2: ["c_attn"].
#       - BERT-family: ["query", "value"].
# • lora_dropout: 0.05-0.1 helps when data is small.
