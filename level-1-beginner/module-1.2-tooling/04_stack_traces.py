"""
04 — Reading stack traces

When Python raises an unhandled exception, it prints a *traceback*: the chain
of function calls that led to the error.

How to read it:
  1. Read **bottom-up**. The last line is the exception type and message.
  2. Above that, frames are listed *outermost first*. The frame closest to
     the bottom is where the error actually happened.
  3. `^^^^^` markers (3.11+) point at the exact expression that failed.

Run this file and study the output:
    python 04_stack_traces.py
"""

import sys


def load_user(user_id: int) -> dict:
    """Pretend we're loading a user record."""
    users = {1: {"name": "Alice", "age": 30}}
    return users[user_id]               # <-- KeyError if user_id missing


def greet(user_id: int) -> str:
    user = load_user(user_id)
    return f"Hello, {user['nmae']}"     # <-- typo: 'nmae' instead of 'name'


def main() -> None:
    # Pick which demo to run via argv:
    demo = sys.argv[1] if len(sys.argv) > 1 else "1"

    if demo == "1":
        # KeyError from load_user — happens deep in the call chain.
        print(greet(99))

    elif demo == "2":
        # Re-raise with `from e` to chain exceptions cleanly.
        try:
            print(greet(1))   # this hits the typo bug
        except KeyError as e:
            raise RuntimeError("could not greet user") from e

    elif demo == "3":
        # Suppress the original cause with `from None` (rare; usually a bad idea).
        try:
            int("abc")
        except ValueError:
            raise RuntimeError("bad input") from None

    else:
        print("usage: python 04_stack_traces.py [1|2|3]")


if __name__ == "__main__":
    main()


# ── Anatomy of a traceback ───────────────────────────────────────────────────
#
#   Traceback (most recent call last):
#     File "04_stack_traces.py", line 47, in <module>      <- where main was called
#       main()
#     File "04_stack_traces.py", line 33, in main          <- main's call
#       print(greet(99))
#             ^^^^^^^^^                                     <- the failing expression
#     File "04_stack_traces.py", line 23, in greet         <- greet's call to load_user
#       user = load_user(user_id)
#              ^^^^^^^^^^^^^^^^^^
#     File "04_stack_traces.py", line 19, in load_user     <- THE actual error site
#       return users[user_id]
#              ~~~~~^^^^^^^^^
#   KeyError: 99                                            <- the exception itself
#
# Read this BOTTOM-UP:
#   1. KeyError: 99 — that's the problem.
#   2. Site: load_user, line 19, on `users[user_id]`.
#   3. Why was load_user called? See the frame above. Repeat to taste.
# ──────────────────────────────────────────────────────────────────────────────
