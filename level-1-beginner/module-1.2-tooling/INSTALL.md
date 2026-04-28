# Setup — Module 1.2 (Tooling)

This module is about the *tools* you use around Python: editor, debugger, linter, formatter, type checker, version control, and pre-commit hooks.

## 1. Install Python ≥ 3.11

See [../module-1.1-python-essentials/INSTALL.md](../module-1.1-python-essentials/INSTALL.md). On Windows, you can use either `python` or `py`; this doc uses `python`.

## 2. Install Git

- Windows: https://git-scm.com/download/win
- macOS: `brew install git`
- Linux: `sudo apt install git`

Verify:

```bash
git --version
```

Then set your identity once (used in commit metadata):

```bash
git config --global user.name  "Your Name"
git config --global user.email "you@example.com"
```

## 3. Install VS Code + extensions

VS Code: https://code.visualstudio.com/

Install these extensions from the marketplace:

- **Python** (Microsoft)
- **Pylance** (auto-installed with Python)
- **Ruff** (Astral)
- **Jupyter** (Microsoft)
- **GitLens** (optional but excellent)

The included [.vscode/launch.json](.vscode/launch.json) gives you a "Python: Current File" debug configuration — open any `.py` file and press **F5**.

## 4. Create a virtual environment

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Windows cmd:
.venv\Scripts\activate.bat
# macOS / Linux:
source .venv/bin/activate
```

## 5. Install dev tools

```bash
pip install ruff black mypy pre-commit
```

What each does:

| Tool | Role |
|------|------|
| **ruff**   | Fast linter (replaces flake8, isort, pyupgrade...) and a black-compatible formatter |
| **black**  | The original opinionated Python formatter |
| **mypy**   | Static type checker — catches type errors before runtime |
| **pre-commit** | Runs the above automatically on every `git commit` |

## 6. Try them

```bash
# Lint the messy demo file:
ruff check 01_messy_code.py

# Auto-fix what's safe:
ruff check --fix 01_messy_code.py

# Format the file (black-style):
ruff format 01_messy_code.py
# OR with black directly:
black 01_messy_code.py

# Type-check:
mypy 03_type_hints.py
```

## 7. Enable pre-commit hooks

From a folder with a `.git` directory and `.pre-commit-config.yaml`:

```bash
# (initialize git here if not already a repo)
git init

pre-commit install
pre-commit run --all-files
```

Now every `git commit` automatically lints, formats, and checks the files you staged.

## 8. Run the lessons

```bash
python 01_messy_code.py
python 02_breakpoints_and_pdb.py
python 03_type_hints.py
python 04_stack_traces.py
python 05_logging_basics.py
```

## 9. Run the exercises

```bash
python exercises/02_debug_recursion.py
python exercises/03_add_type_hints.py
mypy --strict exercises/03_add_type_hints.py
```

Exercise 1 is in [exercises/01_setup_repo.md](exercises/01_setup_repo.md).
