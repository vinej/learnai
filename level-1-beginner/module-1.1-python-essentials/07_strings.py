"""
07 — Strings

Strings are immutable sequences of Unicode code points.

Run: python 07_strings.py
"""

# --- Literals ---
single = 'single quotes'
double = "double quotes"
triple = """multi-line
string"""
print(single, double)
print(triple)

# Raw string — backslashes are literal (handy for regex and Windows paths)
path = r"C:\Users\Alice\Documents"
print(path)

# Bytes literal — sequence of bytes, NOT text
data = b"hello"
print(data, type(data))


# --- f-strings (formatted string literals, 3.6+) ---
name = "Alice"
age = 30
print(f"{name} is {age} years old")

# Expressions inside braces
print(f"{name.upper()} - next year: {age + 1}")

# Format specifiers
pi = 3.14159265
print(f"{pi:.2f}")        # 3.14
print(f"{pi:10.3f}")      # right-aligned in width 10
print(f"{pi:<10.3f}")     # left-aligned
print(f"{1234567:,}")     # 1,234,567
print(f"{0.85:.1%}")      # 85.0%
print(f"{255:#x}")        # 0xff (hex)

# Self-documenting (3.8+) — useful while debugging
value = 42
print(f"{value=}")        # value=42


# --- Common string methods ---
s = "  Hello, World!  "
print(s.strip())              # remove leading/trailing whitespace
print(s.lower())
print(s.upper())
print(s.replace("World", "Python"))
print(s.startswith("  Hello"))
print("Hello".find("l"))      # 2 (first index)
print("Hello".count("l"))     # 2

# split / join — inverse operations
csv = "alice,bob,carol"
parts = csv.split(",")
print(parts)
print(" | ".join(parts))


# --- Indexing & slicing (strings are sequences too) ---
text = "Python"
print(text[0])     # 'P'
print(text[-1])    # 'n'
print(text[1:4])   # 'yth'
print(text[::-1])  # reversed: 'nohtyP'

# Strings are immutable: `text[0] = "Q"` would raise TypeError.


# --- Useful checks ---
print("123".isdigit())
print("abc".isalpha())
print("abc123".isalnum())
print("   ".isspace())


# --- Concatenation performance ---
# DON'T build big strings with += in a loop — quadratic time.
# DO use ''.join(...).
parts = [str(i) for i in range(100)]
result = "".join(parts)
print(len(result))


# --- Encoding / decoding ---
# str <-> bytes via an explicit encoding (almost always utf-8)
encoded = "café".encode("utf-8")
print(encoded)
decoded = encoded.decode("utf-8")
print(decoded)
