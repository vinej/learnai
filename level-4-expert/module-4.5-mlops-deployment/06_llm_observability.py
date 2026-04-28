"""
06 — LLM observability

Generic web observability (file 05) tells you about latency and errors.
LLM-specific observability also captures:

  • Full prompts and responses (with PII redaction).
  • Token usage (input, output, cache, total) per call.
  • Per-call cost in $ (rolled up to user / feature / day).
  • Quality metrics from background graders (file 08).
  • Tool-call traces for agents (file 4.4.07).

Tools:
  Langfuse        — open source, local Docker option, strong UI.
  LangSmith       — paid, deep LangChain integration.
  Phoenix (Arize) — open source, RAG-aware.
  Helicone        — proxy-based, drop-in.
  Or roll your own with OpenTelemetry + a JSON log.

This file shows the home-rolled approach so you understand what tools
are doing under the hood; then how to use Langfuse for "real."

Run: python 06_llm_observability.py    # requires ANTHROPIC_API_KEY
"""

import json
import os
import sys
import time
import uuid
from contextvars import ContextVar
from pathlib import Path

import anthropic


sys.path.insert(0, str(Path(__file__).resolve().parent.parent /
                       "module-4.1-llms-prompt-engineering"))
from _common import DEFAULT_MODEL, cost_of_usage, require_api_key


require_api_key()

LOG_DIR = Path(__file__).parent / "scratch"
LOG_DIR.mkdir(exist_ok=True)
LOG_PATH = LOG_DIR / "llm_traces.jsonl"


# ───────────────────────────────────────────────────────────
# A request-scoped trace ID
# ───────────────────────────────────────────────────────────
trace_id_var: ContextVar[str | None] = ContextVar("trace_id", default=None)


# ───────────────────────────────────────────────────────────
# Wrap the SDK call
# ───────────────────────────────────────────────────────────
client = anthropic.Anthropic()


def traced_complete(*, system: str | None = None, messages: list[dict],
                    user_id: str | None = None, feature: str | None = None,
                    model: str = DEFAULT_MODEL, max_tokens: int = 400) -> str:
    trace_id = trace_id_var.get() or uuid.uuid4().hex[:12]
    started = time.time()
    response = client.messages.create(
        model=model, max_tokens=max_tokens,
        system=system, messages=messages,
    )
    latency = time.time() - started
    cost = cost_of_usage(response.usage, model)

    # Redact: only HASH long content; store the rest as metadata.
    import hashlib
    def hash_msg(text: str) -> str:
        return "sha1:" + hashlib.sha1(text.encode()).hexdigest()[:12]

    record = {
        "trace_id": trace_id,
        "ts": round(time.time(), 3),
        "model": model,
        "user_id": user_id,
        "feature": feature,
        "input_hash": hash_msg(json.dumps(messages, ensure_ascii=False)),
        "input_chars": sum(len(m.get("content", "")) if isinstance(m.get("content"), str) else 0
                           for m in messages),
        "system_hash": hash_msg(system) if system else None,
        "stop_reason": response.stop_reason,
        "usage": {
            "input": response.usage.input_tokens,
            "output": response.usage.output_tokens,
            "cache_creation": response.usage.cache_creation_input_tokens or 0,
            "cache_read":     response.usage.cache_read_input_tokens or 0,
        },
        "latency_s": round(latency, 3),
        "cost_usd": round(cost, 6),
    }
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    return "".join(b.text for b in response.content if b.type == "text")


# ───────────────────────────────────────────────────────────
# A small "user-facing feature" calling the wrapped client
# ───────────────────────────────────────────────────────────
def summarize_for(user_id: str, text: str) -> str:
    trace_id_var.set(uuid.uuid4().hex[:12])
    return traced_complete(
        user_id=user_id,
        feature="summarize_v1",
        system="Summarize the input text in one sentence. No preamble.",
        messages=[{"role": "user", "content": text}],
    )


# ───────────────────────────────────────────────────────────
# Run a few requests, then inspect the log
# ───────────────────────────────────────────────────────────
print("running a few summarization calls...")
for u, text in [
    ("u_1", "Photosynthesis converts sunlight into chemical energy in plants."),
    ("u_2", "The first manned moon landing was Apollo 11 in 1969."),
    ("u_1", "Neural networks learn parameters through gradient descent."),
]:
    out = summarize_for(u, text)
    print(f"  [{u}]  {out.strip()}")


# Aggregate the log: per-user total cost, p95 latency, etc.
print(f"\n=== aggregating {LOG_PATH} ===")
records = [json.loads(line) for line in LOG_PATH.read_text().splitlines() if line.strip()]
by_user: dict[str, dict] = {}
for r in records:
    u = r["user_id"]
    by_user.setdefault(u, {"calls": 0, "cost": 0.0, "tokens": 0})
    by_user[u]["calls"]  += 1
    by_user[u]["cost"]   += r["cost_usd"]
    by_user[u]["tokens"] += r["usage"]["input"] + r["usage"]["output"]

for u, agg in by_user.items():
    print(f"  {u}: {agg['calls']} calls   ${agg['cost']:.5f}   {agg['tokens']} tokens")


# ───────────────────────────────────────────────────────────
# Same idea via Langfuse (if configured)
# ───────────────────────────────────────────────────────────
LANGFUSE_RECIPE = """\
# pip install langfuse
import os
from langfuse import Langfuse
from langfuse.anthropic import Anthropic     # drop-in wrapper

# Set env vars: LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST.
# (Docker-compose'd Langfuse for local dev:
#    https://langfuse.com/self-hosting/docker-compose)

client = Anthropic()        # auto-traces every messages.create

# Per-trace metadata:
@observe(name="summarize_for")
def summarize_for(user_id: str, text: str) -> str:
    langfuse_context.update_current_trace(user_id=user_id)
    return client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        messages=[{"role": "user", "content": text}],
    ).content[0].text
"""
print("\n=== Langfuse drop-in ===")
print(LANGFUSE_RECIPE)


# ───────────────────────────────────────────────────────────
# What you see in a real LLM observability UI
# ───────────────────────────────────────────────────────────
WHAT_YOU_SEE = """\
- Timeline of every call (model, prompt, completion, tokens, cost, latency).
- Filter by user / feature / model / outcome.
- Aggregate: cost over time, P95 latency, cache hit rate.
- Per-trace EVAL scores from a separate grader (file 08).
- Replay any session to a sandbox to debug regressions.
- Tag traces with versions; A/B-test prompts.

Production minimum:
  - Trace EVERY API call (input, output, usage, latency, model, version).
  - Tag with user_id / feature / git_sha for slice-and-dice.
  - REDACT PII before storing free-form text.
  - Link traces to user-visible request IDs in your front-end.
"""
print(WHAT_YOU_SEE)
