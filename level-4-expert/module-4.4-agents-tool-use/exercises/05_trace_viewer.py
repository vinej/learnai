"""
Exercise 5 — A CLI viewer for traces produced by 07_tracing.py

Reads a JSONL trace file and prints a readable summary:
  • timeline of model_call / tool_call / answer events
  • per-step latency, tokens, cost
  • totals at the bottom

Then asserts: at least one trace exists, has > 0 steps, and decodes cleanly.

Run: python exercises/05_trace_viewer.py
"""

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
TRACE_DIR = HERE.parent / "traces"


def load_trace(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def render(events: list[dict]) -> None:
    width = 80
    print("=" * width)
    print(f"trace: {len(events)} events")
    print("=" * width)

    total_in, total_out, total_cost, total_tools = 0, 0, 0.0, 0
    for ev in events:
        ts = f"[{ev['ts']:6.2f}s]"
        kind = ev["kind"]
        if kind == "run_start":
            print(f"{ts} START {ev.get('model')}  query={ev.get('query', '')[:60]!r}")
        elif kind == "model_call":
            u = ev.get("usage", {})
            total_in  += u.get("input", 0)
            total_out += u.get("output", 0)
            total_cost += ev.get("cost_usd", 0.0)
            print(f"{ts}   model_call  step={ev['step']:>2}  "
                  f"stop={ev.get('stop_reason'):<10}  "
                  f"in={u.get('input', 0):>4}  out={u.get('output', 0):>3}  "
                  f"latency={ev.get('latency_ms', 0):>5.0f}ms  "
                  f"${ev.get('cost_usd', 0):.5f}")
        elif kind == "tool_call":
            total_tools += 1
            print(f"{ts}   tool_call   step={ev['step']:>2}  "
                  f"{ev['tool']:<14}  in={ev['input']}  "
                  f"({ev.get('latency_ms', 0):.0f}ms)")
            out_summary = ev["output"][:70].replace("\n", " ")
            print(f"{' '*8}     out: {out_summary}{'...' if len(ev['output']) > 70 else ''}")
        elif kind == "answer":
            text = ev.get("text", "")[:width - 16]
            print(f"{ts}   ANSWER:   {text}{'...' if len(ev.get('text', '')) > len(text) else ''}")
        elif kind == "error":
            print(f"{ts}   ERROR     {ev.get('reason')}")
        elif kind == "run_end":
            print(f"{ts} END   total cost ${ev.get('total_cost_usd', 0):.5f}")
        else:
            print(f"{ts}   {kind}     {ev}")

    print("-" * width)
    print(f"totals: tool_calls={total_tools}  input_tokens={total_in}  "
          f"output_tokens={total_out}  cost ${total_cost:.5f}")


def main():
    if not TRACE_DIR.exists() or not list(TRACE_DIR.glob("*.jsonl")):
        print(f"No traces found in {TRACE_DIR}.")
        print("Run `python 07_tracing.py` (in the parent dir) first to generate one.")
        return

    traces = sorted(TRACE_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    print(f"found {len(traces)} trace(s)\n")

    most_recent = traces[0]
    print(f"showing: {most_recent.name}\n")
    events = load_trace(most_recent)
    render(events)

    # ── Assertions ──
    assert len(events) > 0
    assert any(e["kind"] == "model_call" for e in events), \
        "expected at least one model_call event"
    print("\ntrace viewer works ✅")


if __name__ == "__main__":
    main()


# ── Stretch ─────────────────────────────────────────────────────────────────
# • Add `--json-summary` to print a single JSON aggregate (good for CI).
# • Add `--filter step=N` to inspect a single step.
# • Render as HTML — anchor tool_call events to the model_call that triggered them.
# • Stream traces over the network to a real observability stack (Langfuse).
# ─────────────────────────────────────────────────────────────────────────────
