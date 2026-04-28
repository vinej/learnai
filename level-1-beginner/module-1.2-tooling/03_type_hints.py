"""
03 — Type hints + mypy

Type hints are optional annotations that tell humans (and tools) what type
each variable / parameter / return value should be. They are NOT enforced at
runtime — mypy checks them ahead of time.

Run:
    mypy 03_type_hints.py
    mypy --strict 03_type_hints.py     # stricter — flags missing hints too

This file is correctly typed. See exercises/03_add_type_hints.py for an
exercise where you add hints yourself.
"""

from collections.abc import Iterable, Sequence
from pathlib import Path


# --- Basic types ---
def add(a: int, b: int) -> int:
    return a + b


# --- Built-in generics (3.9+ — no need for `from typing import List`) ---
def total(numbers: list[float]) -> float:
    return sum(numbers)


def first_key(d: dict[str, int]) -> str:
    return next(iter(d))


# --- Optional / union ---
# `X | None` (3.10+) is the modern way to write Optional[X].
def find_user(user_id: int) -> str | None:
    if user_id == 1:
        return "Alice"
    return None


# --- Union of multiple types ---
def stringify(x: int | float | str) -> str:
    return str(x)


# --- Iterable / Sequence — accept anything iterable, not just lists ---
def average(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("empty sequence")
    return sum(values) / len(values)


def double_all(items: Iterable[int]) -> list[int]:
    return [i * 2 for i in items]


# --- Callables ---
from collections.abc import Callable

def apply(func: Callable[[int], int], value: int) -> int:
    return func(value)


# --- Type aliases (3.12+ syntax shown; older: UserId = int) ---
type UserId = int
type Vector = list[float]

def fetch_name(uid: UserId) -> str:
    return f"user-{uid}"


# --- Literal: allow only specific values ---
from typing import Literal

def set_log_level(level: Literal["DEBUG", "INFO", "WARNING", "ERROR"]) -> None:
    print(f"log level set to {level}")


# --- TypedDict: dict-shaped records ---
from typing import TypedDict

class User(TypedDict):
    name: str
    age: int
    email: str | None

def make_user(name: str, age: int) -> User:
    return {"name": name, "age": age, "email": None}


# --- Generics (3.12+ syntax) ---
def first[T](items: Sequence[T]) -> T:
    return items[0]


# --- Demo ---
if __name__ == "__main__":
    print(add(2, 3))
    print(total([1.0, 2.0, 3.0]))
    print(find_user(1), find_user(99))
    print(average([10, 20, 30]))
    print(double_all(range(5)))
    print(apply(lambda x: x + 1, 10))
    print(make_user("Alice", 30))
    print(first([10, 20, 30]))
    print(first(["a", "b", "c"]))


# ── Common mypy errors and what they mean ────────────────────────────────────
# • "Argument 1 to "f" has incompatible type ..."
#       You passed the wrong type. Fix the call or widen the parameter type.
# • "Incompatible return value type"
#       The function returns a value not matching its declared return type.
# • "Item "None" of "Optional[X]" has no attribute ..."
#       You called a method on something that might be None. Add a None check first.
# • "Need type annotation for ..."
#       In --strict mode, mypy wants explicit hints for variables it can't infer.
# ──────────────────────────────────────────────────────────────────────────────
