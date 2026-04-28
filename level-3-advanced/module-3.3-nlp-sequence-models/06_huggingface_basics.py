"""
06 — Hugging Face: AutoTokenizer, AutoModel, pipelines, datasets

The `transformers` library is the lingua franca for pretrained NLP models.
The `Auto*` classes pick the right tokenizer/model class for any model on
the Hub by inspecting its config.

This file shows three levels of API:
  1. PIPELINE  — one-liner for inference (sentiment, summarization, NER, ...)
  2. AutoModel — manual control over inputs/outputs.
  3. datasets  — load and process standard datasets in one line.

Run: python 06_huggingface_basics.py
"""

import torch
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)


# ───────────────────────────────────────────────────────────
# Level 1: pipelines — fastest path to a working result
# ───────────────────────────────────────────────────────────
print("=== sentiment pipeline ===")
sentiment = pipeline("sentiment-analysis",
                     model="distilbert-base-uncased-finetuned-sst-2-english")
results = sentiment([
    "I loved this movie — it was fantastic!",
    "This was the worst experience of my life.",
    "The film was okay, nothing special.",
])
for r, text in zip(results, ["positive movie", "negative", "neutral-ish"]):
    print(f"  {text:<18} → {r}")


# ───────────────────────────────────────────────────────────
# Level 2: AutoTokenizer + AutoModel — manual control
# ───────────────────────────────────────────────────────────
print("\n=== AutoModel forward pass ===")
tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")
model.eval()

inputs = tok("Hello, world!", return_tensors="pt")
print(f"input_ids shape: {tuple(inputs['input_ids'].shape)}")

with torch.no_grad():
    out = model(**inputs)
# `out.last_hidden_state` is shape (B, T, hidden_dim) — one vector per token.
print(f"last_hidden_state shape: {tuple(out.last_hidden_state.shape)}")

# A SENTENCE EMBEDDING is often the average of token embeddings (or the [CLS]).
sentence_emb = out.last_hidden_state.mean(dim=1)
print(f"mean-pooled sentence embedding shape: {tuple(sentence_emb.shape)}")


# ───────────────────────────────────────────────────────────
# Task-specific heads — AutoModelForSequenceClassification, etc.
# ───────────────────────────────────────────────────────────
# For classification, use `AutoModelForSequenceClassification`. Internally
# it adds a linear layer on top of the [CLS] (or pooled) embedding.
clf = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)
clf.eval()
inputs = tok(["I'm thrilled.", "I'm furious."], padding=True, return_tensors="pt")
with torch.no_grad():
    logits = clf(**inputs).logits
probs = logits.softmax(dim=-1)
print(f"\nclassifier probs:\n{probs}")
print(f"label map: {clf.config.id2label}")


# ───────────────────────────────────────────────────────────
# Level 3: datasets library
# ───────────────────────────────────────────────────────────
print("\n=== datasets ===")
from datasets import load_dataset

# IMDb is ~80MB on first download. Use a small subset for the demo.
ds = load_dataset("imdb", split="train[:200]")
print(f"loaded {len(ds)} samples")
print(f"columns: {ds.column_names}")
print(f"first example: {ds[0]['text'][:80]}...  label={ds[0]['label']}")

# Tokenize the whole dataset in one map() call — parallelized automatically.
def tokenize(example):
    return tok(example["text"], truncation=True, max_length=128)

ds_tok = ds.map(tokenize, batched=True)
print(f"after tokenization, columns: {ds_tok.column_names}")
print(f"first ids length: {len(ds_tok[0]['input_ids'])}")


# ───────────────────────────────────────────────────────────
# Other useful pipelines (uncomment to try)
# ───────────────────────────────────────────────────────────
# pipeline("ner")                         — named entities
# pipeline("question-answering")          — extractive QA
# pipeline("summarization")               — abstractive summarization
# pipeline("translation_en_to_fr")        — translation
# pipeline("text-generation", model="gpt2") — generation
# pipeline("zero-shot-classification")    — classify without fine-tuning
#
# All return Python objects you can iterate over. Same logic, different model.


# ───────────────────────────────────────────────────────────
# When to reach for what
# ───────────────────────────────────────────────────────────
# • Quick demo / prototype       → pipeline.
# • Full control, custom logic   → AutoTokenizer + AutoModelFor<Task>.
# • Train your own / fine-tune   → AutoModelFor<Task> + Trainer / accelerate
#   (see 07).
# • Multi-task or multi-modal    → check the model card; some are special.
