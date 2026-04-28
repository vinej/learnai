"""
10 — Modules and Packages

A *module* is any .py file. A *package* is a folder with an __init__.py
(or, since 3.3, even without it — namespace packages).

Run: python 10_modules_and_packages.py
"""

# --- Importing ---
import math                          # use as: math.sqrt(2)
import json as j                     # alias
from collections import Counter      # bring a name into local scope
from pathlib import Path, PurePath   # multiple names

print(math.sqrt(2))
print(j.dumps({"a": 1}))
print(Counter("mississippi"))


# --- The __name__ == "__main__" idiom ---
# When a file is RUN directly,    __name__ == "__main__"
# When it is IMPORTED by another, __name__ == "<module name>"
# This lets the same file be both a library and a script.
if __name__ == "__main__":
    print(f"running directly: __name__ = {__name__!r}")


# --- Where does Python look for imports? ---
# `sys.path` is searched in order. It includes:
#  1. the directory of the script being run
#  2. PYTHONPATH (env var)
#  3. installed site-packages
import sys
print("first 3 entries on sys.path:")
for p in sys.path[:3]:
    print(" ", p)


# --- Packages ---
# A package looks like:
#
#   mypkg/
#     __init__.py        <- makes mypkg a package; runs on first import
#     core.py
#     utils/
#       __init__.py
#       text.py
#
# You import like:
#   from mypkg import core
#   from mypkg.utils.text import slugify
#
# See exercises/05_package_example/ for a runnable mini-package.


# --- Useful stdlib modules to know ---
# pathlib       — file paths
# os, sys       — operating system, interpreter
# json, csv     — data formats
# re            — regular expressions
# datetime      — dates and times
# collections   — Counter, defaultdict, deque, namedtuple
# itertools     — iterator combinators
# functools     — reduce, lru_cache, partial
# typing        — type hints
# argparse      — command-line argument parsing
# logging       — structured logging
# unittest, pytest — testing


# --- Don't do this in real code ---
# from somemodule import *   <-- pollutes the namespace, hides origin
