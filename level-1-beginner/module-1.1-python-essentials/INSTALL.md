# Setup — Module 1.1

Module 1.1 uses **only the Python standard library**. No third-party packages are required.

## 1. Install Python

You need Python **3.11 or newer**.

- Windows: download from [python.org](https://www.python.org/downloads/) (check "Add Python to PATH")
- macOS: `brew install python@3.12`
- Linux: `sudo apt install python3.12 python3.12-venv` (or your distro's equivalent)

Verify:

```bash
python --version
# Windows alternative (the "py launcher", installed with python.org):
py --version
# Some Linux/macOS distros expose it as:
python3 --version
```

> **Windows note:** if `python` isn't on your PATH, use `py` everywhere these
> docs say `python` (e.g. `py 01_variables_and_types.py`).

## 2. Create a virtual environment

A venv keeps each project's packages isolated. Always work inside one.

```bash
# from inside this module folder
python -m venv .venv

# activate it
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (cmd):
.venv\Scripts\activate.bat
# macOS / Linux:
source .venv/bin/activate
```

When activated, your prompt shows `(.venv)`.

## 3. Run the examples

Each numbered file is independent and runnable:

```bash
python 01_variables_and_types.py
python 02_operators.py
python 03_control_flow.py
# ...etc
```

## 4. Run the exercises

```bash
python exercises/01_todo_cli.py add "Buy milk"
python exercises/02_fizzbuzz.py
python exercises/03_word_counter.py exercises/sample.txt
python exercises/04_calculator.py "2 + 3 * 4"
python -m exercises.05_package_example
```

## Optional but recommended dev tools

```bash
pip install ruff black
ruff check .
black .
```

## Deactivating the venv

```bash
deactivate
```
