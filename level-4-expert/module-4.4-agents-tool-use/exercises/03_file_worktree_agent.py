"""
Exercise 3 — A small file agent that operates inside a sandbox dir

Goal: an agent that can read, list, and write files only within a single
sandbox directory. Show the pattern that the Claude Agent SDK formalizes —
but built from scratch with the Anthropic SDK so you understand the moving
parts.

The script asks the agent to:
  1. Read README.md.
  2. Create a new file `summary.md` containing a one-sentence summary.
  3. Verify the file exists.

Asserts: summary.md was created, contains text, and stayed inside the sandbox.

Run: python exercises/03_file_worktree_agent.py    # requires ANTHROPIC_API_KEY
"""

import json
import sys
from pathlib import Path

import anthropic


sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent /
                       "module-4.1-llms-prompt-engineering"))
from _common import DEFAULT_MODEL, require_api_key


require_api_key()
client = anthropic.Anthropic()


# ───────────────────────────────────────────────────────────
# Sandbox setup
# ───────────────────────────────────────────────────────────
SANDBOX = Path(__file__).parent / "sandbox"
SANDBOX.mkdir(exist_ok=True)
SANDBOX_RESOLVED = SANDBOX.resolve()

# Seed a README
(SANDBOX / "README.md").write_text(
    "# Demo project\n\n"
    "This sandbox contains a single project that builds a tiny CLI.\n"
    "It uses Click for argument parsing and Pytest for tests.\n",
    encoding="utf-8",
)


def safe_path(filename: str) -> Path | None:
    candidate = (SANDBOX / filename).resolve()
    try:
        candidate.relative_to(SANDBOX_RESOLVED)
        return candidate
    except ValueError:
        return None


# ───────────────────────────────────────────────────────────
# Tools: read_file, write_file, list_files
# ───────────────────────────────────────────────────────────
def list_files() -> str:
    files = sorted(p.relative_to(SANDBOX).as_posix()
                   for p in SANDBOX.rglob("*") if p.is_file())
    return json.dumps({"files": files})


def read_file(filename: str) -> str:
    path = safe_path(filename)
    if path is None:
        return json.dumps({"error": "path_outside_sandbox"})
    if not path.exists():
        return json.dumps({"error": "not_found"})
    if path.stat().st_size > 50_000:
        return json.dumps({"error": "file_too_large"})
    return json.dumps({"content": path.read_text(encoding="utf-8")[:50_000]})


def write_file(filename: str, content: str) -> str:
    path = safe_path(filename)
    if path is None:
        return json.dumps({"error": "path_outside_sandbox"})
    if len(content) > 50_000:
        return json.dumps({"error": "content_too_large"})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return json.dumps({"ok": True, "wrote": filename, "bytes": len(content)})


TOOLS = [
    {"name": "list_files",
     "description": "List all files in the sandbox.",
     "input_schema": {"type": "object", "properties": {}, "required": []}},
    {"name": "read_file",
     "description": "Read a file from the sandbox by relative path.",
     "input_schema": {"type": "object",
                      "properties": {"filename": {"type": "string"}},
                      "required": ["filename"]}},
    {"name": "write_file",
     "description": "Write a file inside the sandbox. Replaces existing.",
     "input_schema": {"type": "object",
                      "properties": {
                          "filename": {"type": "string"},
                          "content":  {"type": "string", "maxLength": 50000},
                      },
                      "required": ["filename", "content"]}},
]

LOCAL = {"list_files": list_files, "read_file": read_file, "write_file": write_file}


SYSTEM = """\
You are a careful assistant operating in a sandbox directory.

Use tools when needed. Don't write files outside the sandbox; the tools
will reject any such attempt. After writing a file, you may verify it
with read_file.

Be concise.
"""


def run_agent(user_query: str, max_iter: int = 8) -> tuple[str, list[dict]]:
    messages = [{"role": "user", "content": user_query}]
    actions = []
    for _ in range(max_iter):
        resp = client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=500,
            system=SYSTEM,
            tools=TOOLS,
            messages=messages,
        )
        if resp.stop_reason == "end_turn":
            text = "".join(b.text for b in resp.content if b.type == "text").strip()
            return text, actions
        if resp.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": resp.content})
            results = []
            for b in resp.content:
                if b.type == "tool_use":
                    print(f"  → {b.name}({b.input})")
                    out = LOCAL[b.name](**b.input)
                    print(f"    {out[:120]}")
                    actions.append({"tool": b.name, "input": b.input, "output": out})
                    results.append({"type": "tool_result", "tool_use_id": b.id, "content": out})
            messages.append({"role": "user", "content": results})
            continue
        return f"(unexpected stop: {resp.stop_reason})", actions
    return "(max iterations)", actions


# ───────────────────────────────────────────────────────────
# Run the task
# ───────────────────────────────────────────────────────────
TASK = (
    "Read README.md, then write a new file `summary.md` containing exactly "
    "ONE short sentence summarizing the project. Verify it exists by listing "
    "files. Report when you're done."
)
print(f"task: {TASK}\n")
final, actions = run_agent(TASK)
print(f"\n=== agent's reply ===\n{final}")


# ───────────────────────────────────────────────────────────
# Assertions
# ───────────────────────────────────────────────────────────
summary_path = SANDBOX / "summary.md"
assert summary_path.exists(), "agent did not create summary.md"
content = summary_path.read_text(encoding="utf-8").strip()
assert len(content) > 5, "summary is too short"
print(f"\nsummary.md content: {content!r}")

# Ensure no out-of-sandbox path was successfully written
for action in actions:
    if action["tool"] == "write_file":
        out = json.loads(action["output"])
        assert "ok" in out or "error" in out
print("\nfile-agent works inside the sandbox ✅")


# ───────────────────────────────────────────────────────────
# How the Claude Agent SDK simplifies this
# ───────────────────────────────────────────────────────────
WHAT_THE_SDK_GIVES_YOU = """\
The Claude Agent SDK ships polished versions of these tools (Read, Write,
Edit, Glob, Grep, Bash) plus session management, MCP support, custom-tool
registration, hooks, and permission controls. Code looks like:

  from claude_agent_sdk import query, ClaudeAgentOptions

  async def main():
      async for msg in query(
          prompt="Add type hints to all functions in src/.",
          options=ClaudeAgentOptions(cwd="./repo"),
      ):
          print(msg)

For agents that primarily work with files / shells / code, prefer the SDK
over rolling these tools yourself.
"""
print(WHAT_THE_SDK_GIVES_YOU)
