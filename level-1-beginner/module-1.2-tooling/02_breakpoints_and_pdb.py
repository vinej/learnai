"""
02 — Debugging with breakpoint() and pdb

Two ways to debug Python code:

1. **VS Code debugger** — open this file, click in the gutter to set a breakpoint,
   press F5. You get variable inspection, watch expressions, a call stack, and
   step controls all in the editor.

2. **pdb** (the command-line debugger) — drop a `breakpoint()` call into your
   code and run it from a terminal. You enter an interactive prompt at that line.

This file demonstrates pdb. Run it:

    python 02_breakpoints_and_pdb.py

When you hit `(Pdb)`, try these commands:

    h or help        — list commands
    l                — list source around current line
    n                — next line (don't step into calls)
    s                — step INTO the next call
    c                — continue until next breakpoint or end
    p <expr>         — print an expression
    pp <expr>        — pretty-print
    w (where)        — show the call stack
    u / d            — move up / down the stack
    b 42             — set a breakpoint at line 42
    !x = 5           — run a Python statement (the `!` lets you assign)
    q                — quit
"""

import math


def quadratic_roots(a: float, b: float, c: float) -> tuple[float, float]:
    """Return the two real roots of ax² + bx + c = 0."""
    discriminant = b * b - 4 * a * c

    # ⬇⬇⬇ Drop into pdb here. Comment this out for normal runs.
    breakpoint()

    sqrt_disc = math.sqrt(discriminant)   # will fail if discriminant < 0
    root1 = (-b + sqrt_disc) / (2 * a)
    root2 = (-b - sqrt_disc) / (2 * a)
    return root1, root2


def main() -> None:
    print(quadratic_roots(1, -3, 2))   # x² - 3x + 2 = 0 → roots 1 and 2

    # Try this one too — it has complex roots and will raise from inside the fn:
    # print(quadratic_roots(1, 2, 5))


if __name__ == "__main__":
    main()


# ── Tips ─────────────────────────────────────────────────────────────────────
# • `breakpoint()` is the modern way (Python 3.7+). It uses pdb by default but
#   respects the PYTHONBREAKPOINT env var, so you can swap to pudb, ipdb, etc.
#
# • Set PYTHONBREAKPOINT=0 to DISABLE all breakpoints in production:
#       Windows:  set PYTHONBREAKPOINT=0
#       bash:     export PYTHONBREAKPOINT=0
#
# • `python -m pdb script.py` enters pdb at the very first line — useful when
#   you can't easily edit the source.
# ──────────────────────────────────────────────────────────────────────────────
