"""
09 — Exceptions

Python uses exceptions for error signaling. Catch them with try/except.
Don't use exceptions for normal control flow when a simple `if` works.

Run: python 09_exceptions.py
"""

# --- Basic try / except ---
try:
    result = 10 / 0
except ZeroDivisionError as e:
    print("caught:", e)


# --- Multiple exception types ---
def parse_int(text):
    try:
        return int(text)
    except (ValueError, TypeError) as e:
        print(f"can't parse {text!r}: {e}")
        return None

print(parse_int("42"))
print(parse_int("abc"))
print(parse_int(None))


# --- try / except / else / finally ---
# else: runs only if no exception was raised
# finally: ALWAYS runs (cleanup, closing resources)
def read_first_line(path):
    f = None
    try:
        f = open(path, encoding="utf-8")
        line = f.readline()
    except FileNotFoundError:
        print(f"no such file: {path}")
        return None
    else:
        print(f"read {len(line)} chars")
        return line.strip()
    finally:
        if f is not None:
            f.close()
        print("(cleanup done)")

read_first_line(__file__)
read_first_line("does_not_exist.txt")


# --- Raising exceptions ---
def withdraw(balance, amount):
    if amount < 0:
        raise ValueError("amount must be positive")
    if amount > balance:
        raise RuntimeError("insufficient funds")
    return balance - amount

try:
    withdraw(100, -5)
except ValueError as e:
    print("error:", e)


# --- Custom exception classes ---
# Subclass Exception so callers can catch your specific type.
class InsufficientFundsError(Exception):
    """Raised when an account lacks the funds for a withdrawal."""
    def __init__(self, balance, requested):
        super().__init__(f"requested {requested}, balance is {balance}")
        self.balance = balance
        self.requested = requested


def withdraw_v2(balance, amount):
    if amount > balance:
        raise InsufficientFundsError(balance, amount)
    return balance - amount

try:
    withdraw_v2(50, 200)
except InsufficientFundsError as e:
    print(f"need {e.requested - e.balance} more")


# --- Re-raising and exception chaining ---
def load_config(path):
    try:
        with open(path) as f:
            return f.read()
    except FileNotFoundError as e:
        # `from e` preserves the original cause in the traceback
        raise RuntimeError(f"config not loadable: {path}") from e

try:
    load_config("nope.json")
except RuntimeError as e:
    print(type(e).__name__, "->", e)
    print("caused by:", type(e.__cause__).__name__)


# --- Common built-in exceptions to know ---
# ValueError    — right type, wrong value     (int("abc"))
# TypeError     — wrong type                  (1 + "x")
# KeyError      — missing dict key
# IndexError    — list/tuple index out of range
# AttributeError — missing attribute
# FileNotFoundError, PermissionError, IsADirectoryError
# StopIteration  — generator exhausted
# KeyboardInterrupt — user hit Ctrl+C  (don't catch this casually!)


# --- Anti-pattern: bare except ---
# Catches EVERYTHING including KeyboardInterrupt and SystemExit.
# Almost always a bug. Catch specific exceptions, or `except Exception:`.
