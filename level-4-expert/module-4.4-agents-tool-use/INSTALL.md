# Setup — Module 4.4 (Agents & Tool Use)

This module covers the patterns behind agents that USE TOOLS — calling functions, running code, hitting APIs, walking files. The exercises show what production-grade harnesses do under the hood.

## 1. Python ≥ 3.11

See [../../level-1-beginner/module-1.1-python-essentials/INSTALL.md](../../level-1-beginner/module-1.1-python-essentials/INSTALL.md).

## 2. Create / activate the venv

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1     # Windows PS
# or
source .venv/bin/activate       # macOS/Linux
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

| Package          | Purpose                                          |
|------------------|--------------------------------------------------|
| `anthropic`      | Claude SDK (tool use, streaming)                 |
| `pydantic`       | Tool input validation                            |

Standard library is enough for SQLite, file operations, and the in-memory mock web used in the research agent.

### Optional

```bash
pip install claude-agent-sdk
```

Useful if you want to run the Module 4.4 file-agent exercise on top of Claude Code's harness (recommended for serious agents). Requires Claude Code installed and authenticated.

## 4. ANTHROPIC_API_KEY

Required for every exercise except `05_tracing` (which can run from a logged trace). See [Module 4.1](../module-4.1-llms-prompt-engineering/INSTALL.md) for setup.

## 5. Run the lessons

```bash
python 01_tool_design.py             # API
python 02_agent_loop.py              # API
python 03_common_tools.py            # API
python 04_multi_agent.py             # API
python 05_safety_sandboxing.py       # offline + small API demo
python 06_prompt_injection.py        # API
python 07_tracing.py                 # API
python 08_frameworks.py              # offline
```

## 6. Run the exercises

```bash
python exercises/01_research_agent.py
python exercises/02_sql_agent.py
python exercises/03_file_worktree_agent.py
python exercises/04_prompt_injection_lab.py
python exercises/05_trace_viewer.py
```

## A safety note

The file-touching, code-running, and DB-querying patterns in this module are NOT a recipe for letting a model run free on your machine. Real agents need:

- Sandboxing (Docker, Firecracker, E2B, gVisor).
- Permission gates (allow-listed paths/commands; human-in-the-loop for risky ops).
- Resource budgets (max iterations, max tokens, max wall-clock).
- Audit logs you can inspect after the fact.

The lessons demonstrate the SHAPE of these defenses on small synthetic tasks. Module 4.6 covers production safety in depth.
