"""
02 — The agent loop

Every "agent" is some variation of:

  while not done:
      observation = environment_step(...)
      thought, action = model.decide(observation)
      execute(action)

For LLM agents, the loop becomes:

  messages = [user]
  while True:
      response = client.messages.create(model, tools, messages)
      if response.stop_reason == "end_turn":
          return final_text(response)
      messages.append(assistant_with_tool_uses(response))
      messages.append(user_with_tool_results(execute_tools(response)))

This file implements that loop with two tools: a calculator and a
get_current_time. We add iteration limits, a token budget, and basic
logging.

Run: python 02_agent_loop.py    # requires ANTHROPIC_API_KEY
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import anthropic


sys.path.insert(0, str(Path(__file__).resolve().parent.parent /
                       "module-4.1-llms-prompt-engineering"))
from _common import DEFAULT_MODEL, cost_of_usage, require_api_key


require_api_key()
client = anthropic.Anthropic()


# ───────────────────────────────────────────────────────────
# Tools
# ───────────────────────────────────────────────────────────
def calculator(expression: str) -> str:
    """Evaluate a SAFE arithmetic expression. NEVER use Python's eval()."""
    import ast, operator as op
    allowed_binops = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
                      ast.Div: op.truediv, ast.Pow: op.pow, ast.Mod: op.mod}
    allowed_unops = {ast.USub: op.neg, ast.UAdd: op.pos}

    def _eval(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp):
            return allowed_binops[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            return allowed_unops[type(node.op)](_eval(node.operand))
        raise ValueError(f"unsupported: {ast.dump(node)}")

    try:
        return json.dumps({"result": _eval(ast.parse(expression, mode="eval").body)})
    except Exception as e:
        return json.dumps({"error": str(e)})


def get_current_time() -> str:
    return json.dumps({"utc": datetime.utcnow().isoformat() + "Z"})


TOOLS = [
    {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression. Supports + - * / ** %.",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
    {
        "name": "get_current_time",
        "description": "Return the current UTC time as ISO 8601.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
]

LOCAL = {
    "calculator":       lambda **kw: calculator(**kw),
    "get_current_time": lambda **kw: get_current_time(**kw),
}


# ───────────────────────────────────────────────────────────
# The loop, with budgets
# ───────────────────────────────────────────────────────────
def run_agent(
    user_query: str,
    *,
    max_iterations: int = 6,
    max_tokens_per_call: int = 800,
    max_total_input_tokens: int = 20_000,
    timeout_s: float = 60,
):
    messages = [{"role": "user", "content": user_query}]
    started = time.time()
    total_in, total_out = 0, 0
    log = []

    for step in range(max_iterations):
        if time.time() - started > timeout_s:
            return {"status": "timeout", "log": log}
        if total_in > max_total_input_tokens:
            return {"status": "input_token_budget_exceeded", "log": log}

        response = client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=max_tokens_per_call,
            tools=TOOLS,
            messages=messages,
        )
        total_in  += response.usage.input_tokens or 0
        total_out += response.usage.output_tokens or 0
        log.append({"step": step, "stop_reason": response.stop_reason,
                    "usage": dict(input=response.usage.input_tokens,
                                  output=response.usage.output_tokens)})

        if response.stop_reason == "end_turn":
            text = "".join(b.text for b in response.content if b.type == "text")
            return {"status": "ok", "answer": text, "log": log,
                    "total_input": total_in, "total_output": total_out,
                    "cost": cost_of_usage(response.usage) * step}     # rough

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    log.append({"step": step, "tool": block.name, "input": block.input})
                    result = LOCAL[block.name](**block.input)
                    log.append({"step": step, "tool": block.name, "result": result})
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
            messages.append({"role": "user", "content": tool_results})
            continue

        return {"status": f"unexpected_stop:{response.stop_reason}", "log": log}

    return {"status": "max_iterations", "log": log}


# ───────────────────────────────────────────────────────────
# Try it
# ───────────────────────────────────────────────────────────
queries = [
    "What is 17 * 23 + 100?",
    "Use the calculator to evaluate (2 ** 10) - 17, then tell me the current UTC time.",
]

for q in queries:
    print(f"\nQ: {q}")
    out = run_agent(q)
    print(f"  status: {out['status']}")
    if out['status'] == 'ok':
        print(f"  answer: {out['answer'].strip()}")
    print(f"  steps: {len([e for e in out['log'] if 'step' in e and 'tool' not in e])}")


# ───────────────────────────────────────────────────────────
# Patterns built ON TOP of this loop
# ───────────────────────────────────────────────────────────
PATTERNS = """\
ReAct (Reason + Act)
  Inject "Thought: ..." text before each tool call. Encourages the model
  to articulate intent. The default behavior of modern Claude is already
  ReAct-flavored without you adding anything.

Plan-and-Execute
  Step 1: an outer agent writes a numbered plan as text.
  Step 2: a worker agent (or the same agent) executes each step.
  Useful for long-horizon tasks; can rein in wandering.

Reflexion
  After completing a task, ask the model to critique its own work and
  retry. Cheap quality bump on tasks with verifiable outcomes (math, code).

Tree-of-Thought
  Branching exploration: try N candidate continuations at each step,
  evaluate, prune. Expensive; useful for hard reasoning problems.

In production, most useful agents are JUST the basic loop with:
  • A few well-designed tools.
  • A good system prompt.
  • Clear stop conditions.
  • Sane budgets.
That's it. Don't over-engineer.
"""
print(PATTERNS)
