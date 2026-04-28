"""
08 — Evaluating fine-tuned models

Three layers, increasingly relevant to YOUR product:

  1. STANDARD BENCHMARKS    — MMLU, HellaSwag, TriviaQA, GSM8K, HumanEval.
                              Useful for sanity checks; weak signal for your
                              specific task. Often gamed.

  2. DOMAIN-SPECIFIC EVAL   — 50-500 hand-written questions covering YOUR
                              user's actual flows. The most predictive of
                              real-world quality.

  3. HUMAN / LLM-AS-JUDGE   — pairwise comparisons (A vs B), absolute scores
                              against a rubric. Subjective qualities (helpful,
                              concise, safe).

Pick #2 + #3. Skip #1 unless the benchmark literally matches your task.

This file shows the LLM-as-judge pattern, which is what you'll use most.

Run: python 08_evaluation.py    # requires ANTHROPIC_API_KEY
"""

import json
import sys
from pathlib import Path

import anthropic


sys.path.insert(0, str(Path(__file__).resolve().parent.parent /
                       "module-4.1-llms-prompt-engineering"))
from _common import DEFAULT_MODEL, require_api_key


require_api_key()
client = anthropic.Anthropic()


# ───────────────────────────────────────────────────────────
# A "domain eval set" — mock answers from two model versions
# ───────────────────────────────────────────────────────────
EVAL = [
    {
        "question": "What's the refund policy?",
        "version_a": "Refunds are issued within 30 days, pro-rated.",
        "version_b": "Acme refunds purchases within 30 days on a pro-rata basis. After 30 days, no refunds are issued.",
        "rubric": "Specifies BOTH the 30-day window AND that refunds are pro-rata. After 30 days, no refunds."
    },
    {
        "question": "How long is the free trial?",
        "version_a": "14 days.",
        "version_b": "I think it's about a month.",
        "rubric": "States 14 days clearly."
    },
    {
        "question": "Which plans support SSO?",
        "version_a": "SSO is available on all plans.",
        "version_b": "SAML SSO is available on Team and Enterprise plans only.",
        "rubric": "Names Team AND Enterprise; correctly excludes other plans."
    },
]


# ───────────────────────────────────────────────────────────
# Pairwise judge
# ───────────────────────────────────────────────────────────
PAIRWISE_PROMPT = """\
You compare two answers to a question, given a RUBRIC of what makes a good
answer. Output ONLY a compact JSON object:

  {"winner": "A" | "B" | "tie", "reason": "one short sentence"}

Tie ONLY if both answers are equally good.
"""


def pairwise(question: str, a: str, b: str, rubric: str) -> dict:
    user = (
        f"QUESTION: {question}\n\n"
        f"RUBRIC: {rubric}\n\n"
        f"ANSWER A: {a}\n\n"
        f"ANSWER B: {b}"
    )
    resp = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=120,
        temperature=0,
        system=PAIRWISE_PROMPT,
        messages=[{"role": "user", "content": user}],
    )
    raw = resp.content[0].text.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


# ───────────────────────────────────────────────────────────
# Run pairwise across the eval set
# ───────────────────────────────────────────────────────────
# IMPORTANT: also run the SAME pair with A and B swapped — judge models
# have position bias. Average the two runs.
print("=== pairwise judge: version A vs version B ===\n")
wins = {"A": 0, "B": 0, "tie": 0}

for case in EVAL:
    # Forward pass
    g1 = pairwise(case["question"], case["version_a"], case["version_b"], case["rubric"])
    # Reverse — swap A and B
    g2 = pairwise(case["question"], case["version_b"], case["version_a"], case["rubric"])
    # If consistent: agree. Otherwise call it a tie (position bias).
    if g1["winner"] == "A" and g2["winner"] == "B":
        verdict = "A"
    elif g1["winner"] == "B" and g2["winner"] == "A":
        verdict = "B"
    else:
        verdict = "tie"
    wins[verdict] += 1
    print(f"Q: {case['question']}")
    print(f"  fwd: {g1['winner']:<3} ({g1['reason']})")
    print(f"  rev: {g2['winner']:<3} ({g2['reason']})")
    print(f"  → consensus: {verdict}\n")


total = len(EVAL)
print(f"final: A={wins['A']}/{total}  B={wins['B']}/{total}  tie={wins['tie']}/{total}")


# ───────────────────────────────────────────────────────────
# Absolute (score-based) judging
# ───────────────────────────────────────────────────────────
# Useful when you need a regression-style metric. Ask for a 1-5 score
# against the rubric. Average across the eval set.
ABSOLUTE_PROMPT = """\
Score this answer on a 1-5 scale against the rubric. Output JSON:
  {"score": 1|2|3|4|5, "reason": "one short sentence"}

5 = nails the rubric, perfectly clear.
3 = partly correct, missing key detail.
1 = wrong or off-topic.
"""


def score(question: str, ans: str, rubric: str) -> dict:
    resp = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=80,
        temperature=0,
        system=ABSOLUTE_PROMPT,
        messages=[{"role": "user", "content":
                   f"QUESTION: {question}\nRUBRIC: {rubric}\nANSWER: {ans}"}],
    )
    raw = resp.content[0].text.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


print("\n=== absolute scores ===")
for case in EVAL:
    a = score(case["question"], case["version_a"], case["rubric"])
    b = score(case["question"], case["version_b"], case["rubric"])
    print(f"Q: {case['question']}")
    print(f"  A: score {a['score']}/5  ({a['reason']})")
    print(f"  B: score {b['score']}/5  ({b['reason']})")


# ───────────────────────────────────────────────────────────
# Best practices
# ───────────────────────────────────────────────────────────
BEST = """\
JUDGE-MODEL HYGIENE
  • Use a STRONGER model than the one you're judging. Sonnet/Opus to judge
    a Haiku app, or GPT-4o to judge a Llama 8B fine-tune.
  • Run BOTH ORDERS to mitigate position bias; average results.
  • Use a CRISP RUBRIC: vague rubrics give vague scores.
  • Calibrate against ~20 human-labeled examples to make sure the judge
    agrees with you on the easy cases.
  • Log every (question, A, B, judge_output, verdict). The "tie" cases are
    where rubric ambiguity hides — fix the rubric.

DATASETS YOU CAN STEAL
  • mt-bench (LMSYS) — multi-turn judging.
  • HELM (Stanford) — broad benchmark suite.
  • RAGAS / TruLens / DeepEval — pre-baked eval rubrics.
  • Your own product logs (with consent / redaction) — best signal.
"""
print(BEST)
