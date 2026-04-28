"""
Exercise 5 — Few-shot LLM classifier vs a fine-tuned BERT (Module 3.3)

You'll often face the question: "do I FINE-TUNE a small model or
PROMPT a big LLM?" Both work; the right answer depends on data, latency
budget, and ops cost.

This exercise:
  1. Builds an LLM-based sentiment classifier using 6 few-shot examples.
  2. Evaluates it on 30 short held-out test sentences.
  3. Reports accuracy and per-call cost.

Run with both:  python exercises/05_few_shot_classifier.py
The script asserts the few-shot LLM hits ≥ 90% accuracy on the test set.

Compare manually: what did Module 3.3.exercise 3 (DistilBERT IMDb) require
in time, data, and ops? You'll see why few-shot LLMs are often a smart
first step before committing to a fine-tune.

Run: python exercises/05_few_shot_classifier.py     # requires ANTHROPIC_API_KEY
"""

import sys
from pathlib import Path

import anthropic

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _common import DEFAULT_MODEL, cost_of_usage, require_api_key


SYSTEM = (
    "You classify movie reviews as 'positive' or 'negative'. "
    "Output a single word: positive or negative. No punctuation."
)

# 6 few-shot examples (the entire 'training set'!)
FEW_SHOT = [
    {"role": "user", "content": "Beautifully shot, with stunning performances."},
    {"role": "assistant", "content": "positive"},
    {"role": "user", "content": "I wanted my two hours back."},
    {"role": "assistant", "content": "negative"},
    {"role": "user", "content": "A masterpiece. Will recommend to everyone."},
    {"role": "assistant", "content": "positive"},
    {"role": "user", "content": "Tedious script, wooden acting, awful pacing."},
    {"role": "assistant", "content": "negative"},
    {"role": "user", "content": "Genuinely heartwarming. I cried."},
    {"role": "assistant", "content": "positive"},
    {"role": "user", "content": "Avoid. The plot makes no sense."},
    {"role": "assistant", "content": "negative"},
]


# 30 test reviews with hand-labeled ground truth
TEST_SET = [
    ("Loved every minute of it.",                                      "positive"),
    ("Painfully boring from start to finish.",                          "negative"),
    ("The cinematography alone is worth the ticket.",                   "positive"),
    ("Terrible CGI and a forgettable script.",                          "negative"),
    ("Brilliantly acted and emotionally resonant.",                     "positive"),
    ("Confusing, dull, and far too long.",                              "negative"),
    ("Charming, witty, and original.",                                  "positive"),
    ("I left the theater regretting my ticket.",                        "negative"),
    ("A surprisingly delightful little film.",                           "positive"),
    ("Lazy writing and zero chemistry between the leads.",              "negative"),
    ("Visually stunning and emotionally true.",                          "positive"),
    ("A complete waste of two hours of my life.",                        "negative"),
    ("Funny, heartfelt, and exquisitely paced.",                         "positive"),
    ("Poorly directed and badly edited.",                                "negative"),
    ("The performances elevate a thin premise.",                         "positive"),
    ("It tries to be clever and ends up being annoying.",                "negative"),
    ("The best film I have seen all year.",                              "positive"),
    ("Cliché-ridden and predictable.",                                   "negative"),
    ("Captivating from beginning to end.",                                "positive"),
    ("Stale, derivative, and joyless.",                                   "negative"),
    ("A small gem with big heart.",                                       "positive"),
    ("Unbearable. The dialogue is wooden.",                               "negative"),
    ("Powerful, urgent, and beautifully photographed.",                   "positive"),
    ("Dreadful. Even the soundtrack feels phoned in.",                    "negative"),
    ("Wonderful — it stayed with me all week.",                           "positive"),
    ("Half-baked, tonally muddled, frustrating.",                         "negative"),
    ("A genuine surprise — funnier than I expected.",                     "positive"),
    ("It's not even bad-good. It's just bad.",                            "negative"),
    ("Tender, smart, and full of life.",                                  "positive"),
    ("The director should have stopped at the first draft.",              "negative"),
]


def classify(client: anthropic.Anthropic, text: str) -> tuple[str, float]:
    response = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=4,
        temperature=0,
        system=SYSTEM,
        messages=FEW_SHOT + [{"role": "user", "content": text}],
    )
    label = response.content[0].text.strip().lower()
    label = "positive" if "pos" in label else ("negative" if "neg" in label else "unknown")
    return label, cost_of_usage(response.usage)


def main():
    require_api_key()
    client = anthropic.Anthropic()

    correct = 0
    total_cost = 0.0
    wrong = []
    for text, gold in TEST_SET:
        pred, cost = classify(client, text)
        total_cost += cost
        if pred == gold:
            correct += 1
        else:
            wrong.append((text, gold, pred))

    acc = correct / len(TEST_SET)
    print(f"\n=== few-shot Claude-Haiku sentiment ===")
    print(f"accuracy: {correct}/{len(TEST_SET)} = {acc:.3f}")
    print(f"total cost on 30 calls: ~${total_cost:.4f}")

    if wrong:
        print("\nwrong predictions:")
        for text, gold, pred in wrong:
            print(f"  expected {gold:<8}  got {pred:<8}  → {text}")

    assert acc >= 0.9, f"few-shot accuracy should be ≥ 90%, got {acc:.3f}"
    print("\nfew-shot classifier hits the bar ✅")


if __name__ == "__main__":
    main()


# ── Reflection ──────────────────────────────────────────────────────────────
# When few-shot WINS:
#   • Tiny labeled data — you don't have 1k rows for fine-tuning.
#   • Ad hoc / shifting requirements — change a sentence in the prompt.
#   • Cross-language / multi-task — Claude handles many tasks out of the box.
#
# When fine-tuning WINS:
#   • Heavy traffic — per-token cost dominates; a small fine-tuned model
#     is much cheaper at scale.
#   • Latency-sensitive — a local 80M-param classifier replies in 5ms.
#   • Strict, narrow domain — fine-tuning bakes domain knowledge in.
#
# Hybrid: prototype with few-shot, ship that, then fine-tune if the
# product takes off.
# ─────────────────────────────────────────────────────────────────────────────
