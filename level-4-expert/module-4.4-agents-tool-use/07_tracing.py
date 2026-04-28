"""
07 — Tracing: log every step

Production agents are debugging-hostile by default — they involve LLM
non-determinism, tool side effects, and multi-turn state. The fix:

  Log EVERYTHING for every run.

A useful trace records:
  • timestamp
  • step number
  • model + system prompt + messages (or hashes of long content)
  • tool calls: name, input, output, latency
  • token usage and dollars per step
  • final answer + stop reason
  • errors / retries

This file shows a small home-rolled tracer that writes JSONL. The exercise
(05_trace_viewer.py) builds a CLI viewer for the result.

Run: python 07_tracing.py    # requires ANTHROPIC_API_KEY
"""

import json
import os
import sys
import time
import uuid
from pathlib import Path

import anthropic


sys.path.insert(0, str(Path(__file__).resolve().parent.parent /
                       "module-4.1-llms-prompt-engineering"))
from _common import DEFAULT_MODEL, cost_of_usage, require_api_key


require_api_key()
client = anthropic.Anthropic()


TRACE_DIR = Path(__file__).parent / "traces"
TRACE_DIR.mkdir(exist_ok=True)


# ───────────────────────────────────────────────────────────
# Tracer
# ───────────────────────────────────────────────────────────
class Tracer:
    def __init__(self, run_id: str | None = None):
        self.run_id = run_id or f"run_{uuid.uuid4().hex[:8]}"
        self.path = TRACE_DIR / f"{self.run_id}.jsonl"
        self.f = open(self.path, "w", encoding="utf-8")
        self.start = time.time()

    def log(self, kind: str, **fields):
        rec = {
            "ts": round(time.time() - self.start, 4),
            "kind": kind,
            **fields,
        }
        self.f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


# ───────────────────────────────────────────────────────────
# Tools
# ───────────────────────────────────────────────────────────
def calculator(expression: str) -> str:
    import ast, operator as op
    binops = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
              ast.Div: op.truediv, ast.Pow: op.pow, ast.Mod: op.mod}
    unops  = {ast.USub: op.neg, ast.UAdd: op.pos}
    def _ev(n):
        if isinstance(n, ast.Constant): return n.value
        if isinstance(n, ast.BinOp):    return binops[type(n.op)](_ev(n.left), _ev(n.right))
        if isinstance(n, ast.UnaryOp):  return unops[type(n.op)](_ev(n.operand))
        raise ValueError("unsupported")
    try:
        return json.dumps({"result": _ev(ast.parse(expression, mode="eval").body)})
    except Exception as e:
        return json.dumps({"error": str(e)})


TOOLS = [{
    "name": "calculator",
    "description": "Evaluate a SAFE arithmetic expression.",
    "input_schema": {
        "type": "object",
        "properties": {"expression": {"type": "string"}},
        "required": ["expression"],
    },
}]


# ───────────────────────────────────────────────────────────
# Agent loop with tracing
# ───────────────────────────────────────────────────────────
def run_traced(user_query: str, max_iter: int = 5) -> dict:
    tracer = Tracer()
    tracer.log("run_start", query=user_query, model=DEFAULT_MODEL)
    messages = [{"role": "user", "content": user_query}]

    final_answer = None
    total_cost = 0.0

    for step in range(max_iter):
        t0 = time.time()
        resp = client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=400,
            tools=TOOLS,
            messages=messages,
        )
        latency_ms = (time.time() - t0) * 1000
        cost = cost_of_usage(resp.usage)
        total_cost += cost

        tracer.log(
            "model_call",
            step=step,
            stop_reason=resp.stop_reason,
            latency_ms=round(latency_ms, 1),
            usage=dict(input=resp.usage.input_tokens,
                       output=resp.usage.output_tokens,
                       cache_read=resp.usage.cache_read_input_tokens or 0),
            cost_usd=round(cost, 6),
        )

        if resp.stop_reason == "end_turn":
            final_answer = "".join(b.text for b in resp.content if b.type == "text")
            tracer.log("answer", text=final_answer)
            break

        if resp.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": resp.content})
            results = []
            for b in resp.content:
                if b.type == "tool_use":
                    t1 = time.time()
                    out = calculator(**b.input) if b.name == "calculator" else json.dumps({"error": "unknown tool"})
                    tracer.log(
                        "tool_call",
                        step=step,
                        tool=b.name,
                        input=b.input,
                        output=out,
                        latency_ms=round((time.time() - t1) * 1000, 1),
                    )
                    results.append({"type": "tool_result", "tool_use_id": b.id, "content": out})
            messages.append({"role": "user", "content": results})
            continue

        tracer.log("error", reason=f"unexpected stop: {resp.stop_reason}")
        break

    tracer.log("run_end", total_cost_usd=round(total_cost, 6))
    tracer.close()
    return {"answer": final_answer, "trace_path": str(tracer.path),
            "total_cost": total_cost}


# ───────────────────────────────────────────────────────────
# Try a couple of queries
# ───────────────────────────────────────────────────────────
queries = [
    "What is 53 * 47?",
    "Compute 100 / 3 then add 7. Then compute 12^4. Show all results.",
]

for q in queries:
    print(f"\nQ: {q}")
    out = run_traced(q)
    print(f"  answer: {out['answer'][:120] if out['answer'] else '(no answer)'}")
    print(f"  trace:  {out['trace_path']}")
    print(f"  cost:   ${out['total_cost']:.5f}")


# ───────────────────────────────────────────────────────────
# What to do with traces
# ───────────────────────────────────────────────────────────
WHAT_NEXT = """\
Once you have JSONL traces, you can:

  • Replay a session in a UI to debug failures.
  • Compute aggregate metrics: avg latency / call, cache hit rate, $/query.
  • Detect regressions: if a release changes tool-use distribution, alert.
  • Find pathological queries: longest, most expensive, most tool calls.
  • Export to a tracing service: Langfuse, LangSmith, Phoenix (Arize),
    Helicone — all accept structured traces.
  • Use traces as TEST CASES: replay famous bugs against new model versions.

The exercise (05_trace_viewer.py) builds a small CLI viewer.
"""
print(WHAT_NEXT)
