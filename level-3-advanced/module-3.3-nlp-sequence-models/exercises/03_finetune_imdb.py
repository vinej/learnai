"""
Exercise 3 — Fine-tune DistilBERT for sentiment classification on IMDb

We use:
  • model: distilbert-base-uncased  (smaller, faster than BERT)
  • dataset: IMDb subset (2000 train, 500 val) — keeps CPU runtime ~5-10 min

The script asserts validation accuracy ≥ 0.85.

Run: python exercises/03_finetune_imdb.py
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
torch.manual_seed(0)


# ───────────────────────────────────────────────────────────
# Data — subsets to keep CPU feasible
# ───────────────────────────────────────────────────────────
print("loading IMDb subsets ...")
ds = load_dataset("imdb")
train_set = ds["train"].shuffle(seed=0).select(range(2000))
val_set   = ds["test"].shuffle(seed=0).select(range(500))
print(f"train: {len(train_set)}, val: {len(val_set)}")


# ───────────────────────────────────────────────────────────
# Tokenize
# ───────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize(example):
    return tokenizer(example["text"], truncation=True, max_length=256)


train_set = train_set.map(tokenize, batched=True)
val_set   = val_set.map(tokenize, batched=True)


# ───────────────────────────────────────────────────────────
# Model + Trainer
# ───────────────────────────────────────────────────────────
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2,
)

args = TrainingArguments(
    output_dir="scratch/imdb-distilbert",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.06,
    eval_strategy="epoch",
    save_strategy="no",
    logging_steps=20,
    report_to="none",
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


print("\nfine-tuning ...")
trainer.train()

print("\nfinal evaluation:")
metrics = trainer.evaluate()
print(metrics)

acc = float(metrics["eval_accuracy"])
print(f"\nval accuracy: {acc:.4f}")
assert acc >= 0.85, f"target ≥ 0.85, got {acc:.4f}"
print("DistilBERT IMDb fine-tune passed ✅")


# ───────────────────────────────────────────────────────────
# Try it on a couple of new reviews
# ───────────────────────────────────────────────────────────
model.eval()
samples = [
    "This movie was a beautifully crafted masterpiece. The acting was sublime.",
    "An incoherent mess. I want my two hours back.",
]
inputs = tokenizer(samples, padding=True, truncation=True, return_tensors="pt")
with torch.no_grad():
    probs = model(**inputs).logits.softmax(dim=-1)

print("\npredictions:")
for s, p in zip(samples, probs):
    label = ["NEG", "POS"][int(p.argmax())]
    print(f"  ({label} {p.max():.2f}) {s}")


# ── Stretch ─────────────────────────────────────────────────────────────────
# • Bump the train subset to 10000+ samples and 3 epochs — accuracy → 90%+.
# • Try `roberta-base` instead — usually a bit stronger than DistilBERT.
# • Save the fine-tuned model: `trainer.save_model("imdb-distilbert/")`.
#   Reload with AutoModelForSequenceClassification.from_pretrained(...).
# ─────────────────────────────────────────────────────────────────────────────
