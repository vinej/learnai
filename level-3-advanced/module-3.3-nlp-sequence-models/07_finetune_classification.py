"""
07 — Fine-tuning a transformer for classification

This file walks through the full Hugging Face fine-tuning workflow on a TINY
subset of SST-2 (Stanford Sentiment Treebank) so it runs in ~1-2 minutes on CPU.

Pieces:
  1. Load a dataset.
  2. Tokenize it (padding/truncation).
  3. Replace the model's classification head with a fresh one.
  4. Wire up the `Trainer`.
  5. Compute metrics.

The exercise (`exercises/03_finetune_imdb.py`) does this on a larger set.

Run: python 07_finetune_classification.py
"""

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


MODEL_NAME = "distilbert-base-uncased"


# ───────────────────────────────────────────────────────────
# 1. Data — small SST-2 subset
# ───────────────────────────────────────────────────────────
ds = load_dataset("glue", "sst2")
train_set = ds["train"].shuffle(seed=0).select(range(800))
val_set   = ds["validation"].shuffle(seed=0).select(range(200))
print(f"train: {len(train_set)}, val: {len(val_set)}")
print(f"first example: {train_set[0]}")


# ───────────────────────────────────────────────────────────
# 2. Tokenizer + tokenization
# ───────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize(example):
    return tokenizer(example["sentence"], truncation=True, max_length=128)


train_set = train_set.map(tokenize, batched=True)
val_set   = val_set.map(tokenize, batched=True)


# ───────────────────────────────────────────────────────────
# 3. Model — fresh classification head
# ───────────────────────────────────────────────────────────
# `num_labels=2` makes Hugging Face initialize a (hidden_dim → 2) head on
# top of the pretrained backbone. The pretrained weights warm-start the
# encoder; only the new head starts random.
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)


# ───────────────────────────────────────────────────────────
# 4. Trainer
# ───────────────────────────────────────────────────────────
args = TrainingArguments(
    output_dir="scratch/sst2-distilbert",
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    logging_steps=20,
    save_strategy="no",
    report_to="none",            # disable wandb / tensorboard auto-logging
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": float((preds == labels).mean())}


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_set,
    eval_dataset=val_set,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
)

# ───────────────────────────────────────────────────────────
# 5. Train + evaluate
# ───────────────────────────────────────────────────────────
print("\nstarting fine-tune ...")
trainer.train()
print("\nfinal evaluation:")
print(trainer.evaluate())


# ───────────────────────────────────────────────────────────
# Inference on new sentences
# ───────────────────────────────────────────────────────────
model.eval()
test_sentences = [
    "What an absolutely wonderful experience!",
    "I really regret seeing this — total disappointment.",
    "It was alright, neither great nor terrible.",
]
inputs = tokenizer(test_sentences, padding=True, truncation=True, return_tensors="pt")
with torch.no_grad():
    probs = model(**inputs).logits.softmax(dim=-1)
print("\npredictions:")
for s, p in zip(test_sentences, probs):
    label = ["NEG", "POS"][int(p.argmax())]
    print(f"  ({label} {p.max():.2f}) {s}")


# ───────────────────────────────────────────────────────────
# Picking sane defaults (so you don't waste compute)
# ───────────────────────────────────────────────────────────
# learning_rate     : 2e-5 → 5e-5 for BERT-family fine-tuning.
# weight_decay      : 0.01 (default in AdamW).
# num_train_epochs  : 2-5 is usually plenty; more risks overfitting.
# warmup_ratio      : 0.06 (warmup_ratio in TrainingArguments).
# batch_size        : as big as fits. Use gradient_accumulation_steps for big
#                     effective batches.
# fp16/bf16         : on GPU, set fp16=True or bf16=True for 2-4× speedup.
