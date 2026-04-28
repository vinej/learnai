"""
05 — Comprehensions

Comprehensions are concise expressions that build lists, dicts, sets,
or generators in a single readable line.

Run: python 05_comprehensions.py
"""

# --- List comprehension ---
squares = [x ** 2 for x in range(10)]
print(squares)

# With a filter
even_squares = [x ** 2 for x in range(10) if x % 2 == 0]
print(even_squares)

# Equivalent imperative version (more verbose):
even_squares_loop = []
for x in range(10):
    if x % 2 == 0:
        even_squares_loop.append(x ** 2)
print(even_squares_loop)


# --- Dict comprehension ---
word = "hello"
char_counts = {ch: word.count(ch) for ch in set(word)}
print(char_counts)

# Inverting a dict
original = {"a": 1, "b": 2, "c": 3}
inverted = {v: k for k, v in original.items()}
print(inverted)


# --- Set comprehension ---
unique_lengths = {len(w) for w in ["hi", "hello", "hey", "hola"]}
print(unique_lengths)


# --- Generator expression (lazy — saves memory) ---
# Looks like a list comprehension but with parentheses.
gen = (x ** 2 for x in range(1_000_000))
# Nothing computed yet. Values are produced one at a time as you iterate.
print(sum(gen))   # consumes the generator

# Common use: feed straight into a function that takes an iterable.
print(sum(x ** 2 for x in range(10)))


# --- Conditional expression inside the body ---
labels = ["even" if x % 2 == 0 else "odd" for x in range(5)]
print(labels)


# --- Nested comprehensions ---
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

# Flatten
flat = [n for row in matrix for n in row]
print(flat)

# Transpose
transposed = [[row[i] for row in matrix] for i in range(3)]
print(transposed)


# --- When NOT to use a comprehension ---
# If your comprehension needs more than one filter + a tricky transform,
# a regular `for` loop is usually clearer to read. Optimize for the
# next person who will read your code.
