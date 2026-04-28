"""
04 — Multi-agent: when MORE agents help (and when they're a trap)

Multi-agent systems decompose a problem across specialized agents:

  SUPERVISOR / WORKER
    A planner agent breaks the task into subtasks. Each subtask is handed
    to a worker agent (often with a different system prompt).

  PLANNER + EXECUTOR
    A high-level planner produces a numbered plan. An executor walks it.

  CRITIC / RESEARCHER / WRITER
    Specialized roles collaborate. A "critic" iteration usually adds
    quality on long-form generation.

  DEBATE
    Two or more agents argue. The "judge" agent picks the winner. Useful
    for verification on hard reasoning tasks.

Run: python 04_multi_agent.py    # requires ANTHROPIC_API_KEY
"""

import sys
from pathlib import Path

import anthropic


sys.path.insert(0, str(Path(__file__).resolve().parent.parent /
                       "module-4.1-llms-prompt-engineering"))
from _common import DEFAULT_MODEL, print_usage, require_api_key


require_api_key()
client = anthropic.Anthropic()


def call(system: str, user: str, max_tokens: int = 500) -> str:
    resp = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    print_usage(system.split('.')[0][:30], resp.usage)
    return resp.content[0].text.strip()


# ───────────────────────────────────────────────────────────
# Pattern: WRITER → CRITIC → REVISER
# ───────────────────────────────────────────────────────────
WRITER_SYS = (
    "You write concise, vivid one-paragraph product descriptions. "
    "No marketing fluff."
)
CRITIC_SYS = (
    "You review product descriptions. Output a numbered list of specific, "
    "actionable improvements. No general praise."
)
REVISER_SYS = (
    "You take a draft and a critique and produce an improved version. "
    "Output ONLY the revised paragraph."
)

product_brief = (
    "A pour-over coffee maker for 1-2 cups. Borosilicate glass. "
    "Reusable stainless steel filter. No paper waste. Hand-blown."
)

draft = call(WRITER_SYS, product_brief, max_tokens=300)
print("\n=== draft ===\n", draft)

critique = call(CRITIC_SYS, f"DRAFT:\n{draft}", max_tokens=300)
print("\n=== critique ===\n", critique)

revised = call(REVISER_SYS, f"DRAFT:\n{draft}\n\nCRITIQUE:\n{critique}", max_tokens=300)
print("\n=== revised ===\n", revised)


# ───────────────────────────────────────────────────────────
# Pattern: SUPERVISOR / WORKERS
# ───────────────────────────────────────────────────────────
# A planner produces subtasks. Workers handle each subtask. The supervisor
# synthesizes their outputs.
SUPERVISOR_SYS = (
    "You decompose a research question into 3 sub-questions, each on its "
    "own line. Output ONLY the sub-questions, no preamble or numbers."
)
RESEARCHER_SYS = (
    "Answer the question with one short paragraph. Be precise and concrete."
)
SYNTHESIZER_SYS = (
    "Combine the research notes into a single short cohesive answer to the "
    "original question. Cite each note's claim by paraphrasing."
)


def multi_agent(question: str) -> str:
    sub_questions = [
        line for line in call(SUPERVISOR_SYS, question, max_tokens=300).splitlines()
        if line.strip()
    ]
    print(f"\nsub-questions:")
    for sq in sub_questions:
        print(f"  • {sq}")

    notes = []
    for sq in sub_questions:
        notes.append(call(RESEARCHER_SYS, sq, max_tokens=300))

    synthesis_prompt = (
        f"ORIGINAL QUESTION: {question}\n\n"
        + "\n\n".join(f"NOTE {i+1}: {n}" for i, n in enumerate(notes))
    )
    return call(SYNTHESIZER_SYS, synthesis_prompt, max_tokens=400)


print("\n\n=== supervisor / workers / synthesizer ===")
final = multi_agent("Why did the transformer architecture replace LSTMs in NLP?")
print("\n--- final answer ---")
print(final)


# ───────────────────────────────────────────────────────────
# When multi-agent is a TRAP
# ───────────────────────────────────────────────────────────
TRAP = """\
WHEN ONE AGENT IS BETTER
  • A single, capable model with a good system prompt usually wins.
  • Fewer LLM calls = faster, cheaper, easier to debug.
  • Multi-agent multiplies the failure surface (each agent can drift).

SYMPTOMS THAT YOU OVER-DECOMPOSED
  • Sub-agents start re-asking each other for context they should already share.
  • The "supervisor" looks like it's pretending to delegate; it actually does the work.
  • Total cost dwarfs a single Opus call to do the whole thing.

WHEN MULTI-AGENT GENUINELY HELPS
  • Tasks where roles have CLEARLY DIFFERENT system prompts and tools
    (e.g. coder vs reviewer vs runner).
  • Long-horizon work that benefits from explicit "plan, then execute".
  • Verification — having a separate critic does help on subjective output.
  • Parallelism — independent subtasks that can be done concurrently.

START SIMPLE. Add agents only when adding one is the next thing
that obviously fixes a real problem.
"""
print(TRAP)
