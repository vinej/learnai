"""
Exercise 2 — Debug a buggy recursive function

`sum_digits(n)` should compute the sum of decimal digits of a non-negative
integer:
    sum_digits(123) == 1 + 2 + 3 == 6
    sum_digits(99)  == 9 + 9     == 18
    sum_digits(0)   == 0

Run this file. The assertion fails:

    python exercises/02_debug_recursion.py

Find and fix the bug using one of these debugging approaches.

## Option A — VS Code debugger
1. Open this file in VS Code.
2. Click in the gutter to the left of line numbers inside `sum_digits` to set
   a breakpoint (red dot).
3. Press F5; pick "Debug recursion exercise" or "Python: Current File".
4. Use the Step Over (F10) and Step Into (F11) buttons to advance.
5. Inspect `n` in the Variables panel and the Call Stack panel.

## Option B — pdb from a terminal
1. Uncomment the `breakpoint()` line below.
2. Run: `python exercises/02_debug_recursion.py`
3. At the `(Pdb)` prompt, try:
       p n
       n          (next line)
       s          (step in to recursive call)
       w          (where am I in the stack)
       c          (continue)

When you've found the bug, fix it. The assertion should pass.
"""


def sum_digits(n: int) -> int:
    if n == 0:
        return 0
    # 🐞 BUG: this line returns the wrong thing. What should it be?
    return n + sum_digits(n // 10)


def main() -> None:
    # breakpoint()    # <- uncomment to drop into pdb
    test_cases = [(0, 0), (5, 5), (10, 1), (99, 18), (123, 6), (9999, 36)]
    for n, expected in test_cases:
        got = sum_digits(n)
        assert got == expected, f"sum_digits({n}) -> {got}, expected {expected}"
    print("all tests pass ✅")


if __name__ == "__main__":
    main()
