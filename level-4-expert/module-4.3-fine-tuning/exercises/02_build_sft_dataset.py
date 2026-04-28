"""
Exercise 2 — Build a small SFT dataset and inspect its quality

A dataset is just a JSONL file where each line is a {prompt, response} (or
{messages: [...]}) record. This exercise:
  1. Generates 30 synthetic instruction-following examples.
  2. Cleans them: dedupes, removes too-short outputs, length-checks.
  3. Saves to JSONL.
  4. Reports stats: count, average lengths, vocabulary diversity.

The script asserts the cleaned set is non-empty and meets quality bars.

Run: python exercises/02_build_sft_dataset.py
"""

import hashlib
import json
import re
from collections import Counter
from pathlib import Path


HERE = Path(__file__).parent
OUT_PATH = HERE / "scratch" / "sft_dataset.jsonl"
OUT_PATH.parent.mkdir(exist_ok=True)


# ───────────────────────────────────────────────────────────
# 1. Synthetic raw examples (some intentionally bad to filter)
# ───────────────────────────────────────────────────────────
RAW = [
    # good
    {"instruction": "Convert to title case",     "input": "the quick brown fox",      "output": "The Quick Brown Fox"},
    {"instruction": "Convert to title case",     "input": "hello world",                "output": "Hello World"},
    {"instruction": "Convert to upper case",     "input": "be brave",                    "output": "BE BRAVE"},
    {"instruction": "Convert to upper case",     "input": "morning",                     "output": "MORNING"},
    {"instruction": "Reverse the string",        "input": "racecar",                     "output": "racecar"},
    {"instruction": "Reverse the string",        "input": "abcdef",                      "output": "fedcba"},
    {"instruction": "Count vowels",               "input": "education",                  "output": "5"},
    {"instruction": "Count vowels",               "input": "rhythm",                      "output": "0"},
    {"instruction": "Pluralize the word",         "input": "cat",                         "output": "cats"},
    {"instruction": "Pluralize the word",         "input": "wolf",                        "output": "wolves"},
    {"instruction": "Translate to French",        "input": "good morning",               "output": "bonjour"},
    {"instruction": "Translate to French",        "input": "thank you",                   "output": "merci"},
    {"instruction": "Find the antonym",           "input": "happy",                        "output": "sad"},
    {"instruction": "Find the antonym",           "input": "fast",                          "output": "slow"},
    {"instruction": "Round to two decimals",      "input": "3.14159",                     "output": "3.14"},
    {"instruction": "Round to two decimals",      "input": "2.71828",                      "output": "2.72"},

    # 5 duplicates
    {"instruction": "Convert to title case",      "input": "the quick brown fox",         "output": "The Quick Brown Fox"},
    {"instruction": "Convert to upper case",     "input": "morning",                       "output": "MORNING"},

    # 3 too-short / empty outputs (bad)
    {"instruction": "Reverse the string",         "input": "test",                          "output": ""},
    {"instruction": "Pluralize the word",         "input": "ox",                            "output": " "},
    {"instruction": "Count vowels",               "input": "sky",                            "output": "x"},

    # 2 with refusals (bad)
    {"instruction": "Translate to French",         "input": "I cannot do this",             "output": "I can't translate that, sorry."},
    {"instruction": "Find the antonym",             "input": "happy",                          "output": "I'm not sure I can help with that."},

    # 2 way-too-long outputs (probably hallucinated)
    {"instruction": "Round to two decimals",      "input": "3",                              "output": "3.00 — note that rounding floating-point numbers can introduce subtle errors due to binary representation, so for financial calculations you should use Decimal or rounding-half-to-even strategies that avoid bias..."},
    {"instruction": "Pluralize the word",         "input": "fish",                            "output": "Fish is one of those English words that has the same form whether singular or plural in most contexts; you'd say 'one fish' and 'many fish' and only rarely 'fishes' when referring to multiple species..."},
]
print(f"raw examples: {len(RAW)}")


# ───────────────────────────────────────────────────────────
# 2. Clean
# ───────────────────────────────────────────────────────────
def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def fingerprint(rec: dict) -> str:
    h = hashlib.md5()
    h.update((normalize(rec["instruction"]) + "|" + normalize(rec["input"])).encode())
    return h.hexdigest()


def is_refusal(text: str) -> bool:
    bad = ["i cannot", "i can't", "i'm not sure", "as an ai", "sorry, i"]
    t = text.lower()
    return any(p in t for p in bad)


def clean(records: list[dict]) -> list[dict]:
    seen = set()
    out = []
    rejected = []
    for r in records:
        # Skip duplicates
        fp = fingerprint(r)
        if fp in seen:
            rejected.append((r, "duplicate"))
            continue
        # Skip too short outputs
        if len(r["output"].strip()) < 1 or normalize(r["output"]) in {"", "x"}:
            rejected.append((r, "empty/too-short"))
            continue
        # Skip refusals
        if is_refusal(r["output"]):
            rejected.append((r, "refusal"))
            continue
        # Skip way-too-long outputs (more than 5x the input length, capped at 100 chars)
        if len(r["output"]) > max(5 * max(len(r["input"]), 1), 100):
            rejected.append((r, "too long"))
            continue
        seen.add(fp)
        out.append(r)

    print(f"\nrejections:")
    for rec, reason in rejected:
        print(f"  [{reason}] {rec['instruction']!r:<30}  input={rec['input']!r}")

    return out


cleaned = clean(RAW)
print(f"\ncleaned: {len(cleaned)} good examples")


# ───────────────────────────────────────────────────────────
# 3. Save as JSONL
# ───────────────────────────────────────────────────────────
with OUT_PATH.open("w", encoding="utf-8") as f:
    for rec in cleaned:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
print(f"saved → {OUT_PATH}")


# ───────────────────────────────────────────────────────────
# 4. Stats
# ───────────────────────────────────────────────────────────
input_lens  = [len(r["input"])  for r in cleaned]
output_lens = [len(r["output"]) for r in cleaned]
instructions = Counter(r["instruction"] for r in cleaned)

print(f"\n=== dataset stats ===")
print(f"  examples:                {len(cleaned)}")
print(f"  unique instructions:     {len(instructions)}")
print(f"  avg input length (chars): {sum(input_lens) / len(input_lens):.1f}")
print(f"  avg output length:        {sum(output_lens) / len(output_lens):.1f}")
print(f"  examples per instruction:")
for instr, n in instructions.most_common():
    print(f"    {n}× {instr}")


# ───────────────────────────────────────────────────────────
# 5. Quality assertions
# ───────────────────────────────────────────────────────────
assert len(cleaned) >= 12,         "dataset got too small after cleaning"
assert min(output_lens) >= 1,        "an empty output slipped through"
assert len(instructions) >= 5,       "want diverse instructions"
print("\ndataset passes quality bar ✅")


# ── Stretch ─────────────────────────────────────────────────────────────────
# • Plug into trl.SFTTrainer:
#     dataset = load_dataset("json", data_files=str(OUT_PATH))["train"]
#     trainer = SFTTrainer(model=model, train_dataset=dataset,
#                          formatting_func=lambda r: alpaca_template(r))
# • Add a `quality_score` field; manually score each example 1-5; only keep ≥4.
# • Generate an embedding for each (instruction, input) and de-dup with cosine.
# ─────────────────────────────────────────────────────────────────────────────
