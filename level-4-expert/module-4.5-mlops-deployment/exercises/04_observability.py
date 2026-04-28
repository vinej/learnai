"""
Exercise 4 — LLM observability: trace every call, aggregate, alert

Wraps the Anthropic SDK with:
  • A request-scoped trace_id.
  • Per-call structured log to JSONL.
  • An aggregator that reads the log and reports per-user / per-feature stats.
  • A simple "alert" that fires if any user's hourly cost exceeds a budget.

Run: python exercises/04_observability.py    # requires ANTHROPIC_API_KEY

Asserts:
  - At least 3 traces written.
  - Aggregator computes a non-zero cost.
"""

import hashlib
import json
import sys
import time
import uuid
from contextvars import ContextVar
from pathlib import Path

import anthropic


sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent /
                       "module-4.1-llms-prompt-engineering"))
from _common import DEFAULT_MODEL, cost_of_usage, require_api_key


require_api_key()
client = anthropic.Anthropic()

LOG_PATH = Path(__file__).parent / "scratch" / "obs_traces.jsonl"
LOG_PATH.parent.mkdir(exist_ok=True)


# ───────────────────────────────────────────────────────────
# Tracing wrapper
# ───────────────────────────────────────────────────────────
trace_ctx: ContextVar[dict | None] = ContextVar("trace_ctx", default=None)


def hash_text(text: str) -> str:
    return "sha1:" + hashlib.sha1(text.encode()).hexdigest()[:12]


def traced_complete(*, system: str, messages: list[dict],
                    model: str = DEFAULT_MODEL, max_tokens: int = 200) -> str:
    ctx = trace_ctx.get() or {}
    started = time.time()
    response = client.messages.create(
        model=model, max_tokens=max_tokens,
        system=system, messages=messages,
    )
    elapsed = time.time() - started
    cost = cost_of_usage(response.usage, model)

    rec = {
        "trace_id": ctx.get("trace_id"),
        "ts": round(time.time(), 3),
        "user_id": ctx.get("user_id"),
        "feature": ctx.get("feature"),
        "model": model,
        "system_hash": hash_text(system),
        "input_chars": sum(len(m["content"]) for m in messages),
        "stop_reason": response.stop_reason,
        "usage": dict(input=response.usage.input_tokens,
                      output=response.usage.output_tokens),
        "latency_s": round(elapsed, 3),
        "cost_usd": round(cost, 6),
    }
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return "".join(b.text for b in response.content if b.type == "text")


# ───────────────────────────────────────────────────────────
# Two simulated "features"
# ───────────────────────────────────────────────────────────
def feature_summarize(user_id: str, text: str) -> str:
    trace_ctx.set({"trace_id": uuid.uuid4().hex[:12],
                   "user_id":  user_id,
                   "feature":  "summarize"})
    return traced_complete(
        system="Summarize the input in one sentence. No preamble.",
        messages=[{"role": "user", "content": text}],
    )


def feature_translate(user_id: str, text: str, target: str) -> str:
    trace_ctx.set({"trace_id": uuid.uuid4().hex[:12],
                   "user_id":  user_id,
                   "feature":  f"translate_{target}"})
    return traced_complete(
        system=f"Translate the input to {target}. Output only the translation.",
        messages=[{"role": "user", "content": text}],
    )


# ───────────────────────────────────────────────────────────
# Drive a few requests
# ───────────────────────────────────────────────────────────
print("running a mix of feature calls ...")
texts = [
    "Photosynthesis converts sunlight into chemical energy in plants.",
    "The Eiffel Tower was completed in 1889 for the World's Fair in Paris.",
    "Neural networks are trained via gradient descent on a loss function.",
]
for u, t in [("u_alice", texts[0]), ("u_bob", texts[1]), ("u_alice", texts[2])]:
    out = feature_summarize(u, t)
    print(f"  [{u}] summary: {out.strip()}")

print()
fr = feature_translate("u_carol", "Hello, friend.", "French")
print(f"  [u_carol] translation: {fr.strip()}")


# ───────────────────────────────────────────────────────────
# Aggregator
# ───────────────────────────────────────────────────────────
def aggregate():
    rows = [json.loads(line) for line in LOG_PATH.read_text().splitlines() if line.strip()]
    by_user, by_feature = {}, {}
    for r in rows:
        u, f = r["user_id"], r["feature"]
        for d, k in [(by_user, u), (by_feature, f)]:
            d.setdefault(k, {"calls": 0, "input": 0, "output": 0, "cost": 0.0})
            d[k]["calls"]  += 1
            d[k]["input"]  += r["usage"]["input"]
            d[k]["output"] += r["usage"]["output"]
            d[k]["cost"]   += r["cost_usd"]
    return by_user, by_feature, rows


by_user, by_feature, rows = aggregate()

print(f"\n=== aggregates ({len(rows)} traces) ===")
print("by user:")
for u, agg in by_user.items():
    print(f"  {u:<10}  calls={agg['calls']}  tokens={agg['input']+agg['output']}  ${agg['cost']:.5f}")
print("by feature:")
for f, agg in by_feature.items():
    print(f"  {f:<22}  calls={agg['calls']}  tokens={agg['input']+agg['output']}  ${agg['cost']:.5f}")


# ───────────────────────────────────────────────────────────
# Cost alert
# ───────────────────────────────────────────────────────────
HOURLY_USER_BUDGET = 0.001   # $/hour. Demo threshold; bump in prod.

def check_alerts(by_user):
    alerts = []
    for u, agg in by_user.items():
        if agg["cost"] > HOURLY_USER_BUDGET:
            alerts.append((u, agg["cost"]))
    return alerts


alerts = check_alerts(by_user)
if alerts:
    print(f"\nALERTS:")
    for u, c in alerts:
        print(f"  user {u} exceeded budget: ${c:.5f}")
else:
    print(f"\nno users exceeded ${HOURLY_USER_BUDGET}/hour budget")


# ───────────────────────────────────────────────────────────
# Asserts
# ───────────────────────────────────────────────────────────
assert len(rows) >= 3, "expected ≥ 3 traces"
assert sum(r["cost_usd"] for r in rows) > 0, "total cost should be > 0"
print("\nLLM observability harness works ✅")


# ───────────────────────────────────────────────────────────
# In production
# ───────────────────────────────────────────────────────────
PROD_NOTES = """\
- SHIP traces to a real store: Langfuse / Phoenix / a Postgres + dashboards.
- Add a TRACE_ID HTTP header so frontend logs can correlate to backend.
- Sample a small % for QUALITY GRADING (file 4.5.08).
- Set HARD COST CAPS per user / per minute. Reject calls when exceeded.
- Redact PII from any LLM input you store. Hashes are usually enough.
- For batch jobs, log the BATCH ID alongside each call's trace.
"""
print(PROD_NOTES)
