"""
Exercise 5 — A minimal package.

Folder layout:
    05_package_example/
      __init__.py        <- this file (marks the folder as a package)
      __main__.py        <- runs when you do `python -m exercises.05_package_example`
      greeter.py
      math_utils.py

Run from the module-1.1 root:
    python -m exercises.05_package_example

Things to notice:
- `__init__.py` runs the FIRST time the package is imported.
- It can re-export names so users can `from pkg import greet` instead of
  `from pkg.greeter import greet`.
"""

from .greeter import greet
from .math_utils import factorial, is_prime

__all__ = ["greet", "factorial", "is_prime"]
__version__ = "0.1.0"
