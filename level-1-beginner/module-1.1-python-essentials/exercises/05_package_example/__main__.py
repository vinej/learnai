"""
Entry point for `python -m exercises.05_package_example`.
"""

from . import greet, factorial, is_prime


def main():
    print(greet("learner"))
    print("factorial(6) =", factorial(6))
    print("primes up to 20:", [n for n in range(2, 21) if is_prime(n)])


if __name__ == "__main__":
    main()
