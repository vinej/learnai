"""
03 — Control Flow

Run: python 03_control_flow.py
"""

# --- if / elif / else ---
score = 78
if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
elif score >= 70:
    grade = "C"
else:
    grade = "F"
print(f"score={score} -> grade={grade}")


# --- Ternary expression ---
status = "adult" if score >= 18 else "minor"
print(status)


# --- for loops ---
for i in range(5):              # 0..4
    print(i, end=" ")
print()

for i in range(2, 10, 2):       # start, stop, step
    print(i, end=" ")
print()

# Iterate over a sequence
fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
    print(fruit)

# enumerate gives you (index, value)
for i, fruit in enumerate(fruits, start=1):
    print(f"{i}. {fruit}")

# zip pairs sequences together
prices = [1.0, 0.5, 2.5]
for fruit, price in zip(fruits, prices):
    print(f"{fruit}: ${price}")


# --- while loops ---
n = 5
while n > 0:
    print(n, end=" ")
    n -= 1
print()


# --- break and continue ---
for i in range(10):
    if i == 3:
        continue   # skip 3
    if i == 7:
        break      # stop entirely
    print(i, end=" ")
print()


# --- Loop with else ---
# The `else` block runs if the loop completes WITHOUT hitting `break`.
# Useful for "search" patterns.
target = 4
for x in [1, 2, 3, 4, 5]:
    if x == target:
        print("found!")
        break
else:
    print("not found")


# --- match / case (Python 3.10+) ---
# Structural pattern matching — far more powerful than a switch statement.
def describe(thing):
    match thing:
        case 0:
            return "zero"
        case int() if thing < 0:
            return "negative int"
        case int():
            return "positive int"
        case [a, b]:
            return f"pair: {a}, {b}"
        case {"name": name}:
            return f"named: {name}"
        case _:
            return "something else"

print(describe(0))
print(describe(-5))
print(describe([1, 2]))
print(describe({"name": "Alice", "age": 30}))
print(describe("hello"))
