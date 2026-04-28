"""
Exercise 4 — Fine-tune T5-small for summarization, measure ROUGE

T5 (Text-To-Text Transfer Transformer) frames every NLP task as
"text in → text out". For summarization, you prefix the input with
"summarize: " and the model generates a summary.

We use:
  • model: t5-small (~250MB, ~60M params)
  • dataset: tiny subset of `xsum` (300 train, 50 val) — CPU-friendly

Goal: ROUGE-L > the trivial "first sentence" baseline by 3+ points.

Run: python exercises/04_finetune_t5_summarization.py
"""

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5ForConditionalGeneration,
)


MODEL_NAME = "t5-small"
torch.manual_seed(0)


# ───────────────────────────────────────────────────────────
# Data — tiny xsum subset
# ───────────────────────────────────────────────────────────
print("loading xsum subset ...")
ds = load_dataset("EdinburghNLP/xsum", trust_remote_code=True)
train_set = ds["train"].shuffle(seed=0).select(range(300))
val_set   = ds["validation"].shuffle(seed=0).select(range(50))
print(f"train: {len(train_set)}, val: {len(val_set)}")


# ───────────────────────────────────────────────────────────
# Tokenize: input has 'summarize:' prefix, target is the summary
# ───────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
PREFIX = "summarize: "


def preprocess(examples):
    inputs = [PREFIX + doc for doc in examples["document"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True)
    labels = tokenizer(text_target=examples["summary"],
                       max_length=64, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


train_set = train_set.map(preprocess, batched=True, remove_columns=train_set.column_names)
val_set   = val_set.map(preprocess,   batched=True, remove_columns=val_set.column_names)


# ───────────────────────────────────────────────────────────
# Model + Trainer
# ───────────────────────────────────────────────────────────
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

args = Seq2SeqTrainingArguments(
    output_dir="scratch/t5-xsum",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    learning_rate=3e-4,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="no",
    predict_with_generate=True,
    generation_max_length=64,
    logging_steps=20,
    report_to="none",
)


rouge_metric = evaluate.load("rouge")


def compute_metrics(eval_pred):
    preds, labels = eval_pred
    if isinstance(preds, tuple):
        preds = preds[0]
    preds  = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 (label padding) with pad_token before decoding
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = rouge_metric.compute(predictions=preds, references=labels,
                                  use_stemmer=True)
    return {k: round(v * 100, 2) for k, v in result.items()}


trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=train_set,
    eval_dataset=val_set,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    compute_metrics=compute_metrics,
)


# ───────────────────────────────────────────────────────────
# Evaluate BEFORE training, to see the lift
# ───────────────────────────────────────────────────────────
print("\npre-fine-tune ROUGE:")
print(trainer.evaluate())


print("\nfine-tuning t5-small ...")
trainer.train()

print("\npost-fine-tune ROUGE:")
final = trainer.evaluate()
print(final)


# ───────────────────────────────────────────────────────────
# Try a sample summary
# ───────────────────────────────────────────────────────────
sample = (
    "summarize: The new policy will reduce greenhouse gas emissions by 40 percent "
    "by 2030, according to government scientists. Critics argue that the timeline "
    "is too aggressive and could disrupt industries that depend on fossil fuels. "
    "Supporters say it is the minimum needed to avoid dangerous climate impacts."
)
inputs = tokenizer(sample, return_tensors="pt", truncation=True, max_length=512)
ids = model.generate(**inputs, max_length=40, num_beams=4)
print(f"\nsample summary:\n  {tokenizer.decode(ids[0], skip_special_tokens=True)}")


# ───────────────────────────────────────────────────────────
# A loose assertion — fine-tuning should improve over the base model
# ───────────────────────────────────────────────────────────
# (We don't track pre vs post in `final` directly above, but we expect
# rougeL to be > 10 after even tiny fine-tuning. Without fine-tuning, t5-small
# typically scores under 8 on xsum.)
assert final["eval_rougeL"] > 10, f"rougeL should exceed 10, got {final['eval_rougeL']}"
print("T5 summarization fine-tune passed ✅")


# ── Stretch ─────────────────────────────────────────────────────────────────
# • Increase to 5000 train, 5 epochs — rougeL will climb to ~20+ on a GPU.
# • Try `t5-base` (~220M params) — better summaries, slower training.
# • Use `bart-large-cnn` (encoder-decoder pretrained on CNN/DailyMail) for
#   strong out-of-the-box summarization without fine-tuning.
# ─────────────────────────────────────────────────────────────────────────────
