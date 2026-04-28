"""
Exercise 1 — LoRA-fine-tune a tiny model on an instruction-style dataset

We use distilgpt2 (~330 MB, runs on CPU) and a small synthetic instruction
dataset. The goal is the WORKFLOW, not the output quality.

The script asserts:
  - Trainable params drop to < 5% of total params (LoRA worked).
  - Loss decreases over training.

Run: python exercises/01_lora_finetune.py
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


MODEL_NAME = "distilgpt2"
torch.manual_seed(0)


# ───────────────────────────────────────────────────────────
# Dataset — small synthetic instruction set
# ───────────────────────────────────────────────────────────
INSTRUCTIONS = [
    ("Translate to French", "Hello, how are you?",          "Bonjour, comment allez-vous?"),
    ("Translate to French", "Good morning",                  "Bonjour"),
    ("Translate to French", "Thank you",                     "Merci"),
    ("Translate to French", "I love coffee",                 "J'aime le café"),
    ("Translate to French", "Where is the train station?",   "Où est la gare?"),
    ("Translate to French", "I am a student",                "Je suis étudiant"),
    ("Translate to Spanish", "Hello, how are you?",          "Hola, ¿cómo estás?"),
    ("Translate to Spanish", "Good morning",                 "Buenos días"),
    ("Translate to Spanish", "Thank you",                    "Gracias"),
    ("Translate to Spanish", "I love coffee",                "Me encanta el café"),
    ("Translate to Spanish", "Where is the train station?",  "¿Dónde está la estación?"),
    ("Translate to Spanish", "I am a student",               "Soy estudiante"),
] * 6      # repeat so the small model has enough steps to fit


def fmt(instr, inp, out):
    return f"### Instruction:\n{instr}\n\n### Input:\n{inp}\n\n### Response:\n{out}"


texts = [fmt(*row) for row in INSTRUCTIONS]
dataset = Dataset.from_dict({"text": texts})


# ───────────────────────────────────────────────────────────
# Tokenizer + base
# ───────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

base = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
base_total = sum(p.numel() for p in base.parameters())


# ───────────────────────────────────────────────────────────
# LoRA wrap
# ───────────────────────────────────────────────────────────
config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(base, config)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
ratio = trainable / base_total

print(f"base params:      {base_total:,}")
print(f"trainable params: {trainable:,}")
print(f"trainable %:      {ratio * 100:.2f}%")


# ───────────────────────────────────────────────────────────
# Tokenize
# ───────────────────────────────────────────────────────────
def tokenize(batch):
    enc = tokenizer(batch["text"], truncation=True, max_length=128,
                    padding="max_length")
    enc["labels"] = enc["input_ids"].copy()
    return enc


tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])


# ───────────────────────────────────────────────────────────
# Train
# ───────────────────────────────────────────────────────────
args = TrainingArguments(
    output_dir="scratch/lora-translation",
    num_train_epochs=4,
    per_device_train_batch_size=4,
    learning_rate=3e-4,
    weight_decay=0.0,
    logging_steps=10,
    save_strategy="no",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

print("\nstarting LoRA training ...")
result = trainer.train()
losses = [
    log["loss"] for log in trainer.state.log_history
    if "loss" in log and "eval_loss" not in log
]
print(f"losses: first={losses[0]:.3f}  last={losses[-1]:.3f}  "
      f"drop={losses[0] - losses[-1]:.3f}")


# ───────────────────────────────────────────────────────────
# Quick sample
# ───────────────────────────────────────────────────────────
def gen(m, prompt, max_new=20):
    inp = tokenizer(prompt, return_tensors="pt")
    out = m.generate(**inp, max_new_tokens=max_new, do_sample=False,
                     pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0], skip_special_tokens=True)


prompt = fmt("Translate to French", "Good night", "")
print("\n=== sample completion (LoRA) ===")
print(gen(model, prompt))


# ───────────────────────────────────────────────────────────
# Assertions
# ───────────────────────────────────────────────────────────
assert ratio < 0.05, f"LoRA should keep <5% trainable; got {ratio*100:.2f}%"
assert losses[-1] < losses[0], "loss should decrease during training"
print("\nLoRA fine-tune works ✅")
