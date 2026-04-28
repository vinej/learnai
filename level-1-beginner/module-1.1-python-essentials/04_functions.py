"""
04 — Functions

Run: python 04_functions.py
"""

# --- Basic function ---
def add(a, b):
    return a + b

print(add(2, 3))


# --- Default arguments ---
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

print(greet("Alice"))
print(greet("Bob", "Hi"))
print(greet(name="Carol", greeting="Hey"))   # keyword arguments


# --- *args and **kwargs ---
# *args  — captures extra positional arguments as a tuple
# **kwargs — captures extra keyword arguments as a dict
def report(*args, **kwargs):
    print("positional:", args)
    print("keyword:   ", kwargs)

report(1, 2, 3, name="Alice", age=30)


# --- Unpacking when CALLING a function ---
nums = [1, 2, 3]
config = {"name": "Alice", "greeting": "Hi"}
print(add(*[10, 20]))         # unpack list into positional args
print(greet(**config))        # unpack dict into keyword args


# --- Type hints + docstring ---
def divide(a: float, b: float) -> float:
    """Return a divided by b. Raises ZeroDivisionError if b is zero."""
    return a / b

print(divide.__doc__)


# --- Lambdas (anonymous one-line functions) ---
square = lambda x: x ** 2
print(square(5))

# Lambdas shine as arguments to higher-order functions:
people = [("Alice", 30), ("Bob", 25), ("Carol", 35)]
people.sort(key=lambda person: person[1])
print(people)


# --- Scope: LEGB rule ---
# Local -> Enclosing -> Global -> Built-in
x = "global"

def outer():
    x = "enclosing"
    def inner():
        x = "local"
        print("inner sees:", x)
    inner()
    print("outer sees:", x)

outer()
print("module sees:", x)


# --- nonlocal and global ---
counter = 0
def increment():
    global counter
    counter += 1

increment()
increment()
print("counter:", counter)


# --- A common gotcha: mutable default arguments ---
# Default values are evaluated ONCE, when the function is defined.
def append_bad(item, target=[]):     # don't do this
    target.append(item)
    return target

print(append_bad(1))   # [1]
print(append_bad(2))   # [1, 2]   surprise!

def append_good(item, target=None):
    if target is None:
        target = []
    target.append(item)
    return target

print(append_good(1))  # [1]
print(append_good(2))  # [2]


# --- Functions are first-class objects ---
def apply(func, value):
    return func(value)

print(apply(square, 7))
print(apply(lambda s: s.upper(), "hello"))
