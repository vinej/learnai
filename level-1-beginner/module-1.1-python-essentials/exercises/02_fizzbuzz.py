"""
Exercise 2 — FizzBuzz, two ways

Print numbers 1..n, but:
  multiples of 3       -> "Fizz"
  multiples of 5       -> "Buzz"
  multiples of 3 and 5 -> "FizzBuzz"

Concepts exercised: control flow, comprehensions, conditional expressions.

Usage:
    python exercises/02_fizzbuzz.py [n]
"""

import sys


def fizzbuzz_loop(n: int) -> list[str]:
    """Imperative version — easy to read for a beginner."""
    out = []
    for i in range(1, n + 1):
        if i % 15 == 0:
            out.append("FizzBuzz")
        elif i % 3 == 0:
            out.append("Fizz")
        elif i % 5 == 0:
            out.append("Buzz")
        else:
            out.append(str(i))
    return out


def fizzbuzz_comprehension(n: int) -> list[str]:
    """Comprehension version — same logic, one expression."""
    return [
        "FizzBuzz" if i % 15 == 0
        else "Fizz" if i % 3 == 0
        else "Buzz" if i % 5 == 0
        else str(i)
        for i in range(1, n + 1)
    ]


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20

    a = fizzbuzz_loop(n)
    b = fizzbuzz_comprehension(n)
    assert a == b, "the two versions should produce the same output"

    print(", ".join(a))
