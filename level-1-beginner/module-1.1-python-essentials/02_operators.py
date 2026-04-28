"""
02 — Operators

Run: python 02_operators.py
"""

# --- Arithmetic ---
print(7 + 3)      # 10
print(7 - 3)      # 4
print(7 * 3)      # 21
print(7 / 3)      # 2.333... (true division, always float)
print(7 // 3)     # 2        (floor division)
print(7 % 3)      # 1        (modulo / remainder)
print(7 ** 3)     # 343      (power)
print(-7 // 3)    # -3       (floors toward negative infinity!)


# --- Comparison ---
print(2 == 2.0)   # True   (value equality)
print(2 != 3)     # True
print(1 < 2 < 3)  # True   — chained comparisons are allowed and idiomatic


# --- Logical: and / or / not ---
# `and` and `or` short-circuit AND return the operand, not just True/False.
print(True and "hello")    # "hello"
print(0 or "fallback")     # "fallback"  — handy default-value pattern
print(not 0)               # True


# --- Identity vs equality ---
# `==` checks values; `is` checks the same object in memory.
a = [1, 2, 3]
b = [1, 2, 3]
print(a == b)   # True  (equal values)
print(a is b)   # False (different objects)
print(a is a)   # True

# Use `is` for None, True, False — never `== None`.
x = None
print(x is None)


# --- Membership ---
print("a" in "alphabet")       # True
print(3 in [1, 2, 3])          # True
print("key" in {"key": 1})     # True (checks keys)


# --- Bitwise (less common in app code, common in low-level / ML) ---
print(0b1100 & 0b1010)   # 8   (AND)
print(0b1100 | 0b1010)   # 14  (OR)
print(0b1100 ^ 0b1010)   # 6   (XOR)
print(~0b1100)           # -13 (NOT)
print(0b0001 << 3)       # 8   (left shift)


# --- Walrus operator := (since 3.8) ---
# Assign and use in the same expression. Useful in `while` and comprehensions.
data = "hello world"
if (n := len(data)) > 5:
    print(f"length is {n}, exceeds 5")


# --- Augmented assignment ---
total = 0
total += 5      # total = total + 5
total *= 2
print(total)    # 10
