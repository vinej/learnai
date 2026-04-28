"""
05 — Safety and sandboxing

LLM agents take ACTIONS in the world. Without guardrails, they can:

  • run dangerous code (rm -rf, exfil data)
  • call expensive APIs in tight loops
  • get stuck and burn through your budget
  • leak sensitive context to attacker-controlled tools

The defenses come in layers:

  1. SANDBOXING        — physical isolation (Docker, Firecracker, E2B).
                         The model can't escape the box.
  2. PERMISSION GATES  — allow-listed tools / paths / commands.
                         Dangerous ops require human-in-the-loop confirmation.
  3. RESOURCE BUDGETS  — max iterations, max tokens, max wall-clock,
                         max bytes-of-output, max concurrent tools.
  4. AUDIT LOGS         — every action logged with timestamps, args, results.
                         You can replay any session.
  5. INPUT VALIDATION  — Pydantic / JSON schemas catch malformed args.
  6. OUTPUT VALIDATION — sanitize tool outputs before they re-enter the
                         model's context (next file: prompt injection).

This file shows simple instances of (2), (3), (4), and (5).

Run: python 05_safety_sandboxing.py
"""

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ───────────────────────────────────────────────────────────
# A small framework: AgentContext that enforces budgets + logs
# ───────────────────────────────────────────────────────────
@dataclass
class Budget:
    max_iterations:    int   = 10
    max_input_tokens:  int   = 50_000
    max_output_tokens: int   = 5_000
    max_wall_clock_s:  float = 60.0


@dataclass
class AgentContext:
    budget: Budget = field(default_factory=Budget)
    started: float = field(default_factory=time.time)
    iters: int = 0
    in_tokens: int = 0
    out_tokens: int = 0
    log: list = field(default_factory=list)

    def can_continue(self) -> tuple[bool, str]:
        if self.iters >= self.budget.max_iterations:
            return False, "max_iterations"
        if self.in_tokens > self.budget.max_input_tokens:
            return False, "max_input_tokens"
        if self.out_tokens > self.budget.max_output_tokens:
            return False, "max_output_tokens"
        if time.time() - self.started > self.budget.max_wall_clock_s:
            return False, "max_wall_clock"
        return True, ""


@contextmanager
def step(ctx: AgentContext):
    """A logged 'step' in the agent's loop."""
    ctx.iters += 1
    started = time.time()
    record = {"step": ctx.iters, "started_at": started}
    ctx.log.append(record)
    try:
        yield record
    finally:
        record["duration_s"] = time.time() - started


# ───────────────────────────────────────────────────────────
# Permission gating
# ───────────────────────────────────────────────────────────
ALLOWED_PATHS = ["scratch/"]
ALLOWED_COMMANDS = {"calculator", "read_file", "list_dir"}
WRITE_TOOLS = {"write_file", "delete_file", "run_command"}


def is_allowed_tool(tool_name: str, *, require_human: bool = True) -> bool:
    if tool_name in ALLOWED_COMMANDS:
        return True
    if tool_name in WRITE_TOOLS:
        return not require_human or input(f"⚠️  allow {tool_name}? [y/N] ").strip().lower() == "y"
    return False


def safe_path(path: str) -> Path | None:
    p = Path(path).resolve()
    for allowed in ALLOWED_PATHS:
        root = Path(allowed).resolve()
        try:
            p.relative_to(root)
            return p
        except ValueError:
            continue
    return None


# ───────────────────────────────────────────────────────────
# Demo: a tool call that respects budgets, gating, and logging
# ───────────────────────────────────────────────────────────
def execute_tool_safely(ctx: AgentContext, name: str, args: dict[str, Any]) -> str:
    can, reason = ctx.can_continue()
    if not can:
        return json.dumps({"error": "budget_exceeded", "reason": reason})

    if not is_allowed_tool(name, require_human=False):
        return json.dumps({"error": "permission_denied", "tool": name})

    with step(ctx) as rec:
        rec["tool"] = name
        rec["input"] = args
        # Stub: fake the actual call
        result = json.dumps({"ok": True, "echo": args})
        rec["result"] = result[:200]      # truncate for the log
        return result


# ───────────────────────────────────────────────────────────
# Run a few simulated steps
# ───────────────────────────────────────────────────────────
ctx = AgentContext(budget=Budget(max_iterations=3))

print(execute_tool_safely(ctx, "calculator", {"expression": "1+1"}))
print(execute_tool_safely(ctx, "read_file",  {"filename": "scratch/notes.txt"}))
print(execute_tool_safely(ctx, "calculator", {"expression": "2+2"}))
# Next step exceeds the budget:
print(execute_tool_safely(ctx, "calculator", {"expression": "3+3"}))


print(f"\nlog: {len(ctx.log)} entries, ran {ctx.iters} iterations")
for entry in ctx.log:
    print(f"  step {entry['step']}: {entry.get('tool', '?')}({entry.get('input')})")


# ───────────────────────────────────────────────────────────
# Sandboxing for code execution
# ───────────────────────────────────────────────────────────
SANDBOX_NOTES = """\
TOY (in-process) — what we did in file 03's safe_python_math.
  - AST-allowlist for math expressions only.
  - DOES NOT protect against import abuse, stack exhaustion, etc.
  - OK for arithmetic. Not OK for general code.

DOCKER — your default for serious code execution.
  - Run each tool call in a one-shot container with --network=none
    and --read-only --tmpfs /tmp.
  - Capture stdout/stderr, kill on timeout.
  - ~500ms-2s startup overhead per call.

E2B — managed sandbox-as-a-service for LLM agents.
  - Sub-second cold start (snapshot-based).
  - Supports multiple languages and an FS API.
  - The pragmatic choice if you don't want to operate Docker.

FIRECRACKER / gVisor — used by AWS Lambda / Modal under the hood.
  - Stronger isolation than Docker.
  - Requires more setup; usually wrapped by a service like Modal / E2B.

NEVER DO THIS
  - eval(model_output)
  - exec(model_output)
  - subprocess.run(shell=True, model_output, ...)
"""
print(SANDBOX_NOTES)


# ───────────────────────────────────────────────────────────
# Concrete recipe for production agents
# ───────────────────────────────────────────────────────────
RECIPE = """\
1. Allow-list tools by name. NEVER allow arbitrary tool registration at runtime.
2. For each tool, validate inputs with Pydantic + domain rules.
3. Run code in an external sandbox (E2B / Docker). Never in-process.
4. Hard-cap iterations, tokens, and wall-clock. Default to STRICT limits.
5. Require human approval for: writes outside scratch/, network calls to
   any host not on an allow-list, calls to billable APIs.
6. Log every (step, tool, input, output, latency, cost).
7. Periodically replay logs to find regressions.
8. Have a "panic stop" that cancels the agent and tells the user why.
"""
print(RECIPE)
