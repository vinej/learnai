"""
04 — Prompt engineering patterns that actually work

The real art is making LLMs RELIABLE. A few patterns get you most of the way.

  ROLES                  — system (instructions), user (input), assistant (output).
  CLEAR INSTRUCTIONS     — task, format, constraints, tone. Bullet lists help.
  FEW-SHOT EXAMPLES      — show 2-5 ideal input→output pairs.
  CHAIN OF THOUGHT (CoT) — ask the model to "think step by step" before
                           the final answer.
  XML / DELIMITERS       — wrap structured input in tags Claude is well-tuned to.
  REFUSAL HANDLING       — explicit fallback when input is ambiguous or unsafe.

Run: python 04_prompt_patterns.py     # requires ANTHROPIC_API_KEY
"""

import anthropic

from _common import DEFAULT_MODEL, print_usage, require_api_key


require_api_key()
client = anthropic.Anthropic()


# ───────────────────────────────────────────────────────────
# 1. Crisp instructions in a system prompt
# ───────────────────────────────────────────────────────────
print("=== clear instructions ===")
SYSTEM_REVIEW = """\
You classify product reviews. Output ONLY one of: positive, negative, neutral.
No explanation, no quotes. If the input is empty or unclear, output: unclear.
"""

reviews = [
    "Loved it. Will buy again.",
    "Stopped working after a week.",
    "It does what it says.",
    "",
]
for r in reviews:
    resp = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=10,
        system=SYSTEM_REVIEW,
        messages=[{"role": "user", "content": r or "(empty input)"}],
    )
    print(f"  {repr(r):<35} → {resp.content[0].text.strip()}")


# ───────────────────────────────────────────────────────────
# 2. Few-shot examples — when classes overlap or labels are subtle
# ───────────────────────────────────────────────────────────
print("\n=== few-shot ===")
SYSTEM_INTENT = "You classify customer support intents."
FEW_SHOT = [
    {"role": "user", "content": "How do I reset my password?"},
    {"role": "assistant", "content": "account_help"},
    {"role": "user", "content": "I was charged twice for my last order."},
    {"role": "assistant", "content": "billing"},
    {"role": "user", "content": "My package never arrived."},
    {"role": "assistant", "content": "shipping"},
]

new_query = "I need to update the credit card on file."
resp = client.messages.create(
    model=DEFAULT_MODEL,
    max_tokens=10,
    system=SYSTEM_INTENT,
    messages=FEW_SHOT + [{"role": "user", "content": new_query}],
)
print(f"  query: {new_query}")
print(f"  intent: {resp.content[0].text.strip()}")


# ───────────────────────────────────────────────────────────
# 3. Chain of thought (CoT) — for reasoning tasks
# ───────────────────────────────────────────────────────────
# Telling the model to reason step-by-step before answering improves accuracy
# on arithmetic, logic, and code questions, often by 10-20+ pp.
print("\n=== chain of thought ===")
SYSTEM_COT = """\
For each problem, first reason step by step inside <thinking>...</thinking>
tags. Then put the FINAL ANSWER inside <answer>...</answer> tags.
"""
PROBLEM = """\
A train leaves Paris at 9:00 going 80 km/h. A second train leaves Paris at
10:30 going 120 km/h. At what time will the second train catch up?
"""

resp = client.messages.create(
    model=DEFAULT_MODEL,
    max_tokens=600,
    system=SYSTEM_COT,
    messages=[{"role": "user", "content": PROBLEM}],
)
print(resp.content[0].text)


# ───────────────────────────────────────────────────────────
# 4. XML delimiters for structured input
# ───────────────────────────────────────────────────────────
# Claude is trained to handle XML-style tags well. Use them to
# separate context from instructions.
print("\n=== XML delimiters ===")
DOCUMENTS = """\
<documents>
  <doc id="1">Paris was founded around the 3rd century BC by a Celtic tribe.</doc>
  <doc id="2">London's name comes from the Roman 'Londinium', founded ~AD 47.</doc>
  <doc id="3">Tokyo was originally a fishing village called Edo.</doc>
</documents>
"""

resp = client.messages.create(
    model=DEFAULT_MODEL,
    max_tokens=200,
    system=(
        "Answer the user's question using ONLY the <documents>. "
        "Cite the relevant doc ids in square brackets. If no doc applies, "
        "say 'I don't know.'"
    ),
    messages=[{"role": "user", "content":
               f"{DOCUMENTS}\n\nQuestion: When was Tokyo founded?"}],
)
print(resp.content[0].text)


# ───────────────────────────────────────────────────────────
# 5. Persona / tone control
# ───────────────────────────────────────────────────────────
# Tone is set by the system prompt. Be SPECIFIC ("warm but concise") rather
# than vague ("be friendly").
SYSTEM_PERSONA = """\
You are a senior support engineer for a developer tool.
Tone: precise, calm, no apologies, no filler.
Length: ≤ 3 sentences. Always end with a concrete next step.
"""
resp = client.messages.create(
    model=DEFAULT_MODEL,
    max_tokens=200,
    system=SYSTEM_PERSONA,
    messages=[{"role": "user", "content": "My deployment is failing with exit code 137."}],
)
print("\n=== persona ===")
print(resp.content[0].text)


# ───────────────────────────────────────────────────────────
# 6. Common failure modes and how to fight them
# ───────────────────────────────────────────────────────────
# HALLUCINATION       — model invents facts. Fix: ground in retrieved docs
#                       (RAG, Module 4.2). Tell it to say "I don't know."
# FORMAT DRIFT        — output stops matching your spec mid-conversation.
#                       Fix: re-state the format constraints in user turn.
# REFUSAL ON BENIGN   — over-cautious. Fix: clarify the legitimate use case
#                       in the system prompt; keep prompts professional.
# OVER-VERBOSITY      — model adds preamble/disclaimers. Fix: explicitly say
#                       "no preamble", "no apologies", "answer directly".
# WRONG REASONING     — model jumps to conclusion. Fix: CoT (pattern 3).
print_usage("CoT", resp.usage)
