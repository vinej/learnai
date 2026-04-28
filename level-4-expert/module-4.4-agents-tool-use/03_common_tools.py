"""
03 — Common tool patterns

The five most useful tool families:

  WEB SEARCH        — query a search API; return ranked snippets.
  CODE EXECUTION    — run Python in a sandbox; return stdout.
  FILE SYSTEM       — read/write within a permitted directory.
  RAG RETRIEVAL     — search your private docs (Module 4.2).
  HTTP / GraphQL    — call external APIs.

The patterns are similar:
  - Define the schema with TIGHT inputs (enums, paths, max lengths).
  - The local implementation does VALIDATION + SANDBOXING + the actual call.
  - The result is JSON the model can chain on.

This file shows the PATTERN with safe, in-process versions.

Run: python 03_common_tools.py    # requires ANTHROPIC_API_KEY
"""

import json
import re
import sys
from pathlib import Path

import anthropic


sys.path.insert(0, str(Path(__file__).resolve().parent.parent /
                       "module-4.1-llms-prompt-engineering"))
from _common import DEFAULT_MODEL, require_api_key


require_api_key()
client = anthropic.Anthropic()


# ───────────────────────────────────────────────────────────
# 1. Web search — mocked with an in-memory "internet"
# ───────────────────────────────────────────────────────────
WEB = {
    "claude llm":   "Claude is a family of LLMs built by Anthropic, focused on safe and helpful AI assistance.",
    "history paris": "Paris was founded around 3rd century BC by a Celtic tribe called the Parisii.",
    "python language": "Python is a high-level dynamically typed language created by Guido van Rossum in 1991.",
    "transformer attention": "Attention is all you need (2017) introduced the transformer architecture using self-attention.",
}


def web_search(query: str, k: int = 3) -> str:
    """Mocked search. Real impl would call Tavily / Brave / Bing / Google / DuckDuckGo."""
    q = re.findall(r"\w+", query.lower())
    scored = []
    for url, body in WEB.items():
        score = sum(1 for w in q if w in url + " " + body.lower())
        if score:
            scored.append({"score": score, "title": url.title(), "snippet": body})
    scored.sort(key=lambda x: -x["score"])
    return json.dumps({"results": scored[:k]})


# ───────────────────────────────────────────────────────────
# 2. Code execution — sandboxed Python (here: AST-validated math)
# ───────────────────────────────────────────────────────────
# REAL code execution needs Docker / E2B / Firecracker. NEVER use plain
# eval() / exec() in any production agent. The "safe" math evaluator from
# file 02 is the closest to safe you can get without a sandbox.
def safe_python_math(expression: str) -> str:
    import ast, operator as op
    binops = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
              ast.Div: op.truediv, ast.Pow: op.pow, ast.Mod: op.mod}
    unops  = {ast.USub: op.neg, ast.UAdd: op.pos}

    def _ev(node):
        if isinstance(node, ast.Constant): return node.value
        if isinstance(node, ast.BinOp):    return binops[type(node.op)](_ev(node.left), _ev(node.right))
        if isinstance(node, ast.UnaryOp):  return unops[type(node.op)](_ev(node.operand))
        raise ValueError("unsupported expression")

    try:
        return json.dumps({"result": _ev(ast.parse(expression, mode="eval").body)})
    except Exception as e:
        return json.dumps({"error": str(e)})


# ───────────────────────────────────────────────────────────
# 3. File system — restricted to a single directory
# ───────────────────────────────────────────────────────────
SCRATCH = Path(__file__).parent / "scratch"
SCRATCH.mkdir(exist_ok=True)


def safe_read_file(filename: str) -> str:
    target = (SCRATCH / filename).resolve()
    # Prevent path traversal
    if not str(target).startswith(str(SCRATCH.resolve())):
        return json.dumps({"error": "path outside sandbox"})
    if not target.exists():
        return json.dumps({"error": "file not found"})
    if target.stat().st_size > 50_000:
        return json.dumps({"error": "file too large"})
    return json.dumps({"content": target.read_text(encoding="utf-8")[:50_000]})


def safe_write_file(filename: str, content: str) -> str:
    if len(content) > 50_000:
        return json.dumps({"error": "content too long"})
    target = (SCRATCH / filename).resolve()
    if not str(target).startswith(str(SCRATCH.resolve())):
        return json.dumps({"error": "path outside sandbox"})
    target.write_text(content, encoding="utf-8")
    return json.dumps({"ok": True, "wrote": str(target.relative_to(SCRATCH))})


# ───────────────────────────────────────────────────────────
# Tool schemas
# ───────────────────────────────────────────────────────────
TOOLS = [
    {
        "name": "web_search",
        "description": "Search the web for snippets. Returns a JSON list of results.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k":     {"type": "integer", "minimum": 1, "maximum": 5, "default": 3},
            },
            "required": ["query"],
        },
    },
    {
        "name": "calculator",
        "description": "Evaluate a SAFE math expression. Supports + - * / ** %.",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
    {
        "name": "read_file",
        "description": "Read a file from the agent's sandbox directory.",
        "input_schema": {
            "type": "object",
            "properties": {"filename": {"type": "string", "description": "relative path"}},
            "required": ["filename"],
        },
    },
    {
        "name": "write_file",
        "description": "Write a file inside the agent's sandbox directory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {"type": "string"},
                "content":  {"type": "string", "maxLength": 50000},
            },
            "required": ["filename", "content"],
        },
    },
]

LOCAL = {
    "web_search":  lambda **kw: web_search(**kw),
    "calculator":  lambda **kw: safe_python_math(**kw),
    "read_file":   lambda **kw: safe_read_file(**kw),
    "write_file":  lambda **kw: safe_write_file(**kw),
}


# ───────────────────────────────────────────────────────────
# Run a query that uses several tools
# ───────────────────────────────────────────────────────────
def run(user_query: str, max_iter: int = 6) -> str:
    messages = [{"role": "user", "content": user_query}]
    for _ in range(max_iter):
        resp = client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=600,
            tools=TOOLS,
            messages=messages,
        )
        if resp.stop_reason == "end_turn":
            return "".join(b.text for b in resp.content if b.type == "text")
        if resp.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": resp.content})
            results = []
            for b in resp.content:
                if b.type == "tool_use":
                    print(f"  → {b.name}({b.input})")
                    out = LOCAL[b.name](**b.input)
                    print(f"    {out[:120]}")
                    results.append({"type": "tool_result", "tool_use_id": b.id, "content": out})
            messages.append({"role": "user", "content": results})
            continue
        raise RuntimeError(f"unexpected stop: {resp.stop_reason}")
    return "(max iterations)"


print("Q: Search for info about transformer attention, then save a 1-sentence summary to summary.txt.")
ans = run("Search for info about transformer attention, then save a 1-sentence summary to summary.txt.")
print(f"\nfinal: {ans}")
print(f"\nfile written: {(SCRATCH / 'summary.txt').exists()}")


# ───────────────────────────────────────────────────────────
# Real production tools you'll integrate
# ───────────────────────────────────────────────────────────
PROD_TOOLS = """\
WEB SEARCH (paid):
  • Tavily Search API     — purpose-built for LLMs.
  • Brave Search API      — affordable, good coverage.
  • Google Custom Search   — needs allowed domains; hits CSE quotas.
  • SerpAPI                — high-fidelity, expensive.

CODE EXECUTION (sandboxed):
  • E2B (e2b.dev)            — managed Python sandboxes, sub-second startup.
  • Modal / Replit           — full container per call.
  • Docker (DIY)             — strong isolation; you own the container.

KNOWLEDGE / DB:
  • Your RAG pipeline (Module 4.2).
  • Read-only DB views for SQL agents (Module 4.4 ex 2).

API CALLS:
  • Wrap each external API as ONE tool. Don't expose raw HTTP.
  • Always validate args. Always cap response size.
  • For long calls, prefer async + a polling result tool.
"""
print(PROD_TOOLS)
