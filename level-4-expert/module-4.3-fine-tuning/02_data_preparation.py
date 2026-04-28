"""
02 — Preparing fine-tuning data

A fine-tune is only as good as its dataset. Three things matter:

  FORMAT      — the exact template the base model expects.
  QUALITY     — clean, diverse, faithful labels.
  VOLUME      — fewer GREAT examples beat more mediocre ones.

Common conversation formats:

  ALPACA            — {"instruction", "input", "output"}. Old but common.
  CHATML            — <|im_start|>role\n...<|im_end|>. OpenAI-ish.
  LLAMA CHAT        — [INST] ... [/INST] ... </s>. Llama-specific.
  TOKENIZER CHAT TEMPLATE — modern: tokenizer.apply_chat_template applies the
                            model's official template for you.

Run: python 02_data_preparation.py
"""

from transformers import AutoTokenizer


# ───────────────────────────────────────────────────────────
# Example raw conversation
# ───────────────────────────────────────────────────────────
example = {
    "messages": [
        {"role": "system",    "content": "You are a polite assistant."},
        {"role": "user",      "content": "Recommend a good Italian restaurant in Paris."},
        {"role": "assistant", "content": "I'd suggest Big Mamma — vibrant, authentic, busy. Book ahead."},
    ],
}


# ───────────────────────────────────────────────────────────
# 1. Alpaca-style (legacy but you'll see it everywhere)
# ───────────────────────────────────────────────────────────
ALPACA = """\
Below is an instruction that describes a task. Write a response that
appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output}
"""


def to_alpaca(messages):
    instr = next(m["content"] for m in messages if m["role"] == "user")
    out   = next(m["content"] for m in messages if m["role"] == "assistant")
    return ALPACA.format(instruction=instr, output=out)


print("=== Alpaca format ===")
print(to_alpaca(example["messages"]))


# ───────────────────────────────────────────────────────────
# 2. ChatML (TinyLlama, Qwen, etc.)
# ───────────────────────────────────────────────────────────
def to_chatml(messages):
    pieces = []
    for m in messages:
        pieces.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
    return "\n".join(pieces)


print("\n=== ChatML format ===")
print(to_chatml(example["messages"]))


# ───────────────────────────────────────────────────────────
# 3. Apply the model's OFFICIAL chat template
# ───────────────────────────────────────────────────────────
# Tokenizers ship a Jinja-style template; this is what the model ACTUALLY
# saw during instruction-tuning. Using anything else can hurt quality.
tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
templated = tok.apply_chat_template(example["messages"], tokenize=False)
print("\n=== TinyLlama-Chat template (the right way) ===")
print(templated)


# ───────────────────────────────────────────────────────────
# Building a SFT dataset record
# ───────────────────────────────────────────────────────────
# At training time, the typical pattern:
#   - Apply the chat template.
#   - Tokenize.
#   - LOSS-MASK the prompt portion (only train on completions, not on the
#     user message you fed in).
#
# `trl.SFTTrainer` does this for you when you set
# `dataset_kwargs={"add_special_tokens": False}` and pass `formatting_func`.


# ───────────────────────────────────────────────────────────
# Data quality checklist
# ───────────────────────────────────────────────────────────
QUALITY_CHECKLIST = """\
DATA QUALITY > DATA QUANTITY

Before training, manually inspect 20-50 random examples. You will find:
  • Duplicates / near-duplicates                      → dedupe
  • Examples that contradict each other               → resolve or remove
  • Output that mixes formats (sometimes JSON, sometimes prose)
                                                       → standardize
  • Truncated outputs (too-long inputs cut off)       → bump max_length
  • Hallucinations from synthetic-data generation     → re-check with a
                                                         strong reference model
  • PII / secrets that shouldn't be in training data  → redact
  • Responses with refusal text where the task is fine → filter
  • Tokenized counts way longer than your max_seq_len  → drop or summarize

If you don't have an eval set yet, write 50 prompts BEFORE training.
"""
print(QUALITY_CHECKLIST)


# ───────────────────────────────────────────────────────────
# Synthetic data generation (a.k.a. "self-instruct")
# ───────────────────────────────────────────────────────────
SYNTHETIC = """\
Generate a small synthetic SFT dataset by:
  1. Hand-write 5-10 SEED examples covering the diversity you want.
  2. Show them to a strong model (Claude Opus / GPT-4o):
       "Generate 100 more examples in this style and format."
  3. Filter aggressively — remove duplicates, refusals, off-task outputs.
  4. Hand-review at least 10% of generated examples.
  5. Mix synthetic with REAL examples — pure synthetic data drifts.

Tooling that does this well:
  - Self-Instruct (the original paper)
  - Magpie (latest distillation pipeline)
  - distilabel (HuggingFace's framework)
  - Distillery (OpenAI's variant)

Beware: training on a model's own outputs in a tight loop creates "model
collapse" — diversity collapses, quality drops. Always use a STRONGER
model than the one you're training.
"""
print(SYNTHETIC)
