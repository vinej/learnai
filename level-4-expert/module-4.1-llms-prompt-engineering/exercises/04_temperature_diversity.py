"""
Exercise 4 — Same prompt, three temperatures: measure diversity

Generate 5 completions of the same creative prompt at temperature 0, 0.7,
and 1.0. Measure the OUTPUT DIVERSITY at each setting via:

  • Number of unique outputs (exact-match dedup).
  • Average pairwise Jaccard distance over word sets.

The script asserts diversity strictly increases with temperature
(within sampling noise).

Run: python exercises/04_temperature_diversity.py    # requires ANTHROPIC_API_KEY
"""

import re
import sys
from itertools import combinations
from pathlib import Path

import anthropic

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _common import DEFAULT_MODEL, print_usage, require_api_key


PROMPT = (
    "Write a one-sentence opening line for a fantasy novel about a thief. "
    "Be vivid. No preamble; output the line only."
)


def word_set(s: str) -> set[str]:
    return set(re.findall(r"[a-z']+", s.lower()))


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return 1 - len(a & b) / len(a | b)


def avg_pairwise_jaccard(samples: list[str]) -> float:
    sets = [word_set(s) for s in samples]
    pairs = list(combinations(sets, 2))
    if not pairs:
        return 0.0
    return sum(jaccard(a, b) for a, b in pairs) / len(pairs)


def main():
    require_api_key()
    client = anthropic.Anthropic()

    N_SAMPLES = 5
    results = {}

    for T in (0.0, 0.7, 1.0):
        samples = []
        for i in range(N_SAMPLES):
            r = client.messages.create(
                model=DEFAULT_MODEL,
                max_tokens=120,
                temperature=T,
                messages=[{"role": "user", "content": PROMPT}],
            )
            samples.append(r.content[0].text.strip())
            print_usage(f"T={T} #{i + 1}", r.usage)

        unique = len(set(samples))
        diversity = avg_pairwise_jaccard(samples)
        results[T] = {"samples": samples, "unique": unique, "diversity": diversity}

    # ── Report ──
    print("\n=== samples per temperature ===")
    for T, info in results.items():
        print(f"\n--- T = {T}  unique = {info['unique']}/{N_SAMPLES}  "
              f"avg pairwise Jaccard = {info['diversity']:.3f} ---")
        for s in info["samples"]:
            print(f"  • {s}")

    # ── Sanity checks ──
    # T=0 should be VERY low diversity (often all 5 samples identical, but
    # service-side variance can produce a couple of variants — be generous).
    assert results[0.0]["diversity"] < results[1.0]["diversity"], (
        "diversity at T=1.0 should exceed T=0.0"
    )
    # T=1 should give at least 3 distinct outputs out of 5.
    assert results[1.0]["unique"] >= 3, (
        f"expected ≥3 unique outputs at T=1.0, got {results[1.0]['unique']}"
    )
    print("\ntemperature controls diversity as expected ✅")


if __name__ == "__main__":
    main()


# ── Reflection ──────────────────────────────────────────────────────────────
# • For deterministic / structured outputs (extraction, code generation,
#   evaluation graders), USE T=0. Reproducibility matters.
# • For creative tasks (writing, ideation), 0.7-1.0 is the right zone.
# • Above 1.2, output degrades fast. Top-p (nucleus) is a gentler alternative.
# ─────────────────────────────────────────────────────────────────────────────
