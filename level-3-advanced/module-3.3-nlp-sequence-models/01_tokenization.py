"""
01 — Tokenization

Models eat numbers; text is characters. The bridge is TOKENIZATION.

Three strategies, in order of "modern":

  CHARACTER-LEVEL  — every char is a token. Tiny vocab, very long sequences.
  WORD-LEVEL       — split on whitespace/punctuation. Big vocab, fragile to
                     OOV (out-of-vocabulary) words.
  SUBWORD-LEVEL    — best of both. Common words stay whole; rare words break
                     into reusable pieces. Powers BPE, WordPiece, SentencePiece.
                     The de facto standard for transformers.

Special tokens:
  [CLS]/[SEP]   — BERT-style separator/sentinel tokens.
  <s>/</s>      — start/end of sequence in many decoder models.
  [PAD]         — padding to make batches rectangular.
  [UNK]         — unknown (rare in subword tokenizers — they decompose instead).

Run: python 01_tokenization.py
"""

from collections import Counter

from transformers import AutoTokenizer


# ───────────────────────────────────────────────────────────
# Character-level
# ───────────────────────────────────────────────────────────
text = "Hello, world!"
chars = list(text)
print(f"chars ({len(chars)}): {chars}")

# Build a tiny vocab: char → int
vocab = {ch: i for i, ch in enumerate(sorted(set(text)))}
ids = [vocab[ch] for ch in text]
print(f"char vocab size: {len(vocab)}, ids: {ids}")


# ───────────────────────────────────────────────────────────
# Word-level (naive)
# ───────────────────────────────────────────────────────────
import re

def word_tokenize(s):
    return re.findall(r"\w+|[^\w\s]", s.lower())


corpus = [
    "The cat sat on the mat.",
    "Cats and dogs are pets.",
    "Pet the cat and the dog.",
]
all_tokens = [tok for s in corpus for tok in word_tokenize(s)]
counter = Counter(all_tokens)
print(f"\nword vocab: {sorted(counter)}")
# Problem: "tokenization" or any unseen word becomes [UNK]. Subword fixes this.


# ───────────────────────────────────────────────────────────
# Subword: BPE (Byte-Pair Encoding) — the algorithm
# ───────────────────────────────────────────────────────────
# Pseudocode:
#   1. Start with a vocab = all characters in the corpus.
#   2. Count adjacent pairs of tokens across all words.
#   3. Merge the most-common pair into a new token. Repeat N times.
#   4. The resulting vocab has frequent words intact, rare words broken.
#
# Example after a few merges starting from chars:
#   "lower"   → ["l", "o", "w", "e", "r"]
#   merge ('e','r') → "er"        → ["l", "o", "w", "er"]
#   merge ('o','w') → "ow"        → ["l", "ow", "er"]
#   merge ('l','ow') → "low"      → ["low", "er"]
# Common variants: BPE (GPT), WordPiece (BERT), SentencePiece (T5/Llama).


# ───────────────────────────────────────────────────────────
# Subword in practice — a Hugging Face tokenizer
# ───────────────────────────────────────────────────────────
# This downloads ~5MB of vocab files on first run.
tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")

print(f"\nDistilBERT vocab size: {tok.vocab_size}")
sample = "Tokenization handles unknown words like supercalifragilistic gracefully."
ids = tok.encode(sample)
tokens = tok.convert_ids_to_tokens(ids)
print(f"\ninput: {sample}")
print(f"tokens ({len(tokens)}):")
for t in tokens:
    print(f"  {t}")


# ───────────────────────────────────────────────────────────
# Roundtrip
# ───────────────────────────────────────────────────────────
decoded = tok.decode(ids, skip_special_tokens=True)
print(f"\ndecoded: {decoded}")


# ───────────────────────────────────────────────────────────
# Padding and attention masks for batching
# ───────────────────────────────────────────────────────────
batch = ["short text", "a much longer piece of text for the batch"]
encoded = tok(batch, padding=True, truncation=True, return_tensors="pt")
print(f"\nbatch input_ids shape:     {encoded['input_ids'].shape}")
print(f"batch attention_mask shape: {encoded['attention_mask'].shape}")
print(f"first row attention_mask:   {encoded['attention_mask'][0].tolist()}")
# 1 = real token, 0 = padding. Models ignore padding via the attention mask.


# ───────────────────────────────────────────────────────────
# Special tokens
# ───────────────────────────────────────────────────────────
print(f"\ntokenizer's special tokens:")
print(f"  CLS:  {tok.cls_token!r}  → id {tok.cls_token_id}")
print(f"  SEP:  {tok.sep_token!r}  → id {tok.sep_token_id}")
print(f"  PAD:  {tok.pad_token!r}  → id {tok.pad_token_id}")
print(f"  UNK:  {tok.unk_token!r}  → id {tok.unk_token_id}")


# ───────────────────────────────────────────────────────────
# Choosing a tokenizer
# ───────────────────────────────────────────────────────────
# • Don't pick a tokenizer separately from the model — you MUST use the
#   tokenizer that the model was trained with. AutoTokenizer.from_pretrained
#   handles this for you.
# • Vocab size affects: model size (embedding table), sequence length, speed.
#   Modern LLM vocabs: 32k (Llama 1) → 100k+ (GPT-4o, Llama 3).
# • For a custom corpus (medical, legal, code), training your own tokenizer
#   often outperforms a generic one. The `tokenizers` library makes this easy.
