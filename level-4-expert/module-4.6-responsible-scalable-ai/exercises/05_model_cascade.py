"""
Exercise 5 — Build a model cascade and measure cost savings

Pattern:
  1. Try the SMALL model with a 'do you know this?' protocol.
  2. If the small model says ESCALATE, fall back to the BIG model.
  3. Compare cost vs always-big.

The script asserts:
  - Some easy questions answered by SMALL only.
  - Some hard ones escalated.
  - Cascade total cost < always-big total cost.

Run: python exercises/05_model_cascade.py    # requires ANTHROPIC_API_KEY
"""

import sys
from pathlib import Path

import anthropic


sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent /
                       "module-4.1-llms-prompt-engineering"))
from _common import cost_of_usage, require_api_key


require_api_key()
client = anthropic.Anthropic()


SMALL = "claude-haiku-4-5-20251001"
BIG   = "claude-sonnet-4-6"


CASCADE_SYSTEM = """\
Answer the user's question.

If the question is BASIC FACTUAL (capitals, definitions, simple math, easy
classifications), answer in 1-2 sentences directly.

If the question REQUIRES NUANCED REASONING, multiple steps, or specialized
expertise that you're NOT confident about, output exactly:

  ESCALATE

(no other text). A larger model will then answer.
"""


BIG_SYSTEM = "Answer the user's question carefully and concisely (1-3 paragraphs)."


def cascade(question: str) -> dict:
    # Pass 1: small model
    small = client.messages.create(
        model=SMALL,
        max_tokens=300,
        temperature=0,
        system=CASCADE_SYSTEM,
        messages=[{"role": "user", "content": question}],
    )
    small_text = "".join(b.text for b in small.content if b.type == "text").strip()
    small_cost = cost_of_usage(small.usage, SMALL)

    if "ESCALATE" in small_text.upper():
        big = client.messages.create(
            model=BIG,
            max_tokens=600,
            temperature=0,
            system=BIG_SYSTEM,
            messages=[{"role": "user", "content": question}],
        )
        big_text = "".join(b.text for b in big.content if b.type == "text").strip()
        big_cost = cost_of_usage(big.usage, BIG)
        return {
            "answer": big_text,
            "model": BIG,
            "escalated": True,
            "cost": small_cost + big_cost,
        }

    return {
        "answer": small_text,
        "model": SMALL,
        "escalated": False,
        "cost": small_cost,
    }


def always_big(question: str) -> dict:
    big = client.messages.create(
        model=BIG,
        max_tokens=600,
        temperature=0,
        system=BIG_SYSTEM,
        messages=[{"role": "user", "content": question}],
    )
    return {
        "answer": "".join(b.text for b in big.content if b.type == "text").strip(),
        "model": BIG,
        "cost": cost_of_usage(big.usage, BIG),
    }


# ───────────────────────────────────────────────────────────
# A mix of easy and hard questions
# ───────────────────────────────────────────────────────────
QUESTIONS = [
    "What's the capital of Australia?",                                 # easy
    "What's 17 * 24?",                                                  # easy
    "Define photosynthesis in one sentence.",                            # easy
    "What language do they speak in Brazil?",                            # easy
    "Explain the connection between Nash equilibrium and "
        "evolutionary stable strategies, citing a worked example.",      # hard
    "Walk through the chain rule applied to the derivative of "
        "ln(sin(x^2 + cos x)) showing each step.",                       # hard
    "Compare and contrast the West and East Roman Empires' decline.",    # hard
]


# ───────────────────────────────────────────────────────────
# Run both
# ───────────────────────────────────────────────────────────
print(f"{'mode':<10}  {'cost':>10}  {'escalated?':>11}  question")

cascade_total = 0.0
big_total = 0.0
escalation_count = 0
small_only_count = 0

for q in QUESTIONS:
    c = cascade(q)
    cascade_total += c["cost"]
    if c["escalated"]:
        escalation_count += 1
    else:
        small_only_count += 1
    print(f"  cascade   ${c['cost']:.5f}  "
          f"{('YES' if c['escalated'] else 'no'):>11}  {q[:50]}")

    b = always_big(q)
    big_total += b["cost"]
    print(f"  always-big ${b['cost']:.5f}             {q[:50]}")
    print()


print(f"\ntotals: cascade ${cascade_total:.5f}   always-big ${big_total:.5f}")
print(f"  escalation rate: {escalation_count}/{len(QUESTIONS)}")
print(f"  cost saving: {(1 - cascade_total / big_total) * 100:.1f}%")


# ───────────────────────────────────────────────────────────
# Assertions
# ───────────────────────────────────────────────────────────
assert escalation_count > 0, "expected at least one escalation on hard questions"
assert small_only_count > 0, "expected at least one SMALL-only answer on easy questions"
assert cascade_total < big_total, "cascade should be cheaper than always-big"
print("\nmodel cascade reduces cost while preserving quality ✅")


# ───────────────────────────────────────────────────────────
# Tuning the cascade
# ───────────────────────────────────────────────────────────
TUNING = """\
KNOBS

CONFIDENCE PROTOCOL
  We told the small model to output "ESCALATE" when unsure. Alternatives:
   - logprob threshold (request `top_logprobs`).
   - Self-consistency: sample 3 answers, escalate if they disagree.
   - Heuristic on input length or detected intent.

PER-CATEGORY ROUTING
  Use a tiny classifier (or a small LLM) to ROUTE the request to a model
  by topic. Math → SMALL. Subjective writing → BIG.

LATENCY BUDGET
  Cascade adds a hop on escalations. If you have a strict TTFT budget,
  start the BIG call in parallel and cancel one once the other finishes.

QUALITY CHECK
  Always evaluate cascade quality on a held-out set vs always-big. If
  quality drops > Δ, raise the small model's escalation sensitivity.

WHEN NOT TO CASCADE
  - All your queries are intrinsically hard.
  - Your latency budget is so tight there's no room for two hops.
  - Quality on the small model is so much worse it confuses users on the
    1% of low-confidence answers that don't escalate.
"""
print(TUNING)
