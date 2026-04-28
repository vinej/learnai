"""
01 — Variables and Types

Python is dynamically typed: you don't declare a variable's type up front.
The interpreter figures it out at runtime.

Run: python 01_variables_and_types.py
"""

# --- Basic assignment ---
name = "Alice"        # str
age = 30              # int
height_m = 1.72       # float
is_student = True     # bool
nothing = None        # NoneType (Python's "no value")

print(name, age, height_m, is_student, nothing)


# --- Multiple assignment ---
x, y, z = 1, 2, 3
a = b = c = 0
print(x, y, z, a, b, c)


# --- The built-in types you'll use most ---
samples = [
    "hello",        # str
    42,             # int  (arbitrary precision — try 10**100)
    3.14,           # float
    True,           # bool (subclass of int!)
    None,           # NoneType
    2 + 3j,         # complex
    b"bytes",       # bytes (raw binary data)
]

for value in samples:
    print(f"{value!r:>10} -> {type(value).__name__}")


# --- Type checking ---
print(isinstance(age, int))         # True
print(isinstance(True, int))        # True  — bool IS an int
print(isinstance(age, (int, float)))  # multiple types in a tuple


# --- Casting between types ---
print(int("42"))        # parse a string
print(float("3.14"))
print(str(99))          # "99"
print(bool(0))          # False
print(bool(""))         # False — empty containers are falsy
print(bool("x"))        # True


# --- Type hints (since Python 3.5) ---
# Hints are NOT enforced at runtime. They help tools like mypy and your IDE.
def greet(name: str, age: int = 0) -> str:
    return f"Hello {name}, age {age}"

print(greet("Bob", 25))


# --- None: use `is`, not `==` ---
value = None
if value is None:
    print("value is None")


# --- Mutability matters ---
# Immutable: int, float, str, tuple, frozenset, bool, None
# Mutable:   list, dict, set, most custom objects
nums = [1, 2, 3]
copy_ref = nums           # both names point to the SAME list
copy_ref.append(4)
print(nums)               # [1, 2, 3, 4]  -- surprising if you didn't expect aliasing

real_copy = nums.copy()   # or list(nums), or nums[:]
real_copy.append(99)
print(nums, real_copy)
