"""
Exercise 1 — Todo CLI persisted to JSON

Usage:
    python exercises/01_todo_cli.py add "Buy milk"
    python exercises/01_todo_cli.py list
    python exercises/01_todo_cli.py done 1
    python exercises/01_todo_cli.py remove 2
    python exercises/01_todo_cli.py clear

Concepts exercised:
- argparse, file I/O, JSON, pathlib, exceptions, list of dicts
"""

import argparse
import json
import sys
from pathlib import Path

DATA_FILE = Path(__file__).parent / "todos.json"


def load() -> list[dict]:
    if not DATA_FILE.exists():
        return []
    try:
        return json.loads(DATA_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print(f"warning: {DATA_FILE} is corrupt; starting fresh")
        return []


def save(todos: list[dict]) -> None:
    DATA_FILE.write_text(json.dumps(todos, indent=2), encoding="utf-8")


def cmd_add(args):
    todos = load()
    todos.append({"text": args.text, "done": False})
    save(todos)
    print(f"added: {args.text}")


def cmd_list(_args):
    todos = load()
    if not todos:
        print("(no todos)")
        return
    for i, t in enumerate(todos, start=1):
        mark = "x" if t["done"] else " "
        print(f"{i:>2}. [{mark}] {t['text']}")


def cmd_done(args):
    todos = load()
    idx = args.index - 1
    if not 0 <= idx < len(todos):
        sys.exit(f"error: no todo at index {args.index}")
    todos[idx]["done"] = True
    save(todos)
    print(f"marked done: {todos[idx]['text']}")


def cmd_remove(args):
    todos = load()
    idx = args.index - 1
    if not 0 <= idx < len(todos):
        sys.exit(f"error: no todo at index {args.index}")
    removed = todos.pop(idx)
    save(todos)
    print(f"removed: {removed['text']}")


def cmd_clear(_args):
    save([])
    print("cleared all todos")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="A tiny todo CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add", help="add a new todo")
    a.add_argument("text")
    a.set_defaults(func=cmd_add)

    sub.add_parser("list", help="list todos").set_defaults(func=cmd_list)

    d = sub.add_parser("done", help="mark a todo as done")
    d.add_argument("index", type=int)
    d.set_defaults(func=cmd_done)

    r = sub.add_parser("remove", help="remove a todo")
    r.add_argument("index", type=int)
    r.set_defaults(func=cmd_remove)

    sub.add_parser("clear", help="remove all todos").set_defaults(func=cmd_clear)
    return p


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
