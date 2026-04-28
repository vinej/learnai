"""
06 — Built-in Data Structures

list   — ordered, mutable, allows duplicates
tuple  — ordered, IMMUTABLE, allows duplicates
dict   — key-value, ordered (insertion order, since 3.7), mutable
set    — unordered, unique elements, mutable
frozenset — immutable set (hashable, can be a dict key)

Run: python 06_data_structures.py
"""

# ============== LIST ==============
fruits = ["apple", "banana", "cherry"]
fruits.append("date")           # add to end
fruits.insert(0, "apricot")     # insert at index
fruits.remove("banana")         # remove first occurrence
last = fruits.pop()             # remove and return last
print(fruits, "popped:", last)

# Slicing
nums = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print(nums[2:5])      # [2, 3, 4]
print(nums[:3])       # [0, 1, 2]
print(nums[-3:])      # [7, 8, 9]
print(nums[::2])      # every 2nd: [0, 2, 4, 6, 8]
print(nums[::-1])     # reversed

# Sorting
words = ["pear", "apple", "banana"]
print(sorted(words))                 # returns NEW list
words.sort(reverse=True)             # sorts IN PLACE
print(words)


# ============== TUPLE ==============
# Immutable. Use for fixed collections (coords, db rows, return values).
point = (3, 4)
x, y = point         # tuple unpacking
print(x, y)

# Single-element tuple needs the trailing comma:
not_a_tuple = (5)        # this is just int 5
yes_a_tuple = (5,)
print(type(not_a_tuple), type(yes_a_tuple))

# Named tuples — readable, lightweight records
from collections import namedtuple
Person = namedtuple("Person", ["name", "age"])
alice = Person("Alice", 30)
print(alice.name, alice.age)


# ============== DICT ==============
user = {"name": "Alice", "age": 30, "email": "a@example.com"}

# Access
print(user["name"])
print(user.get("phone"))                  # None — no KeyError
print(user.get("phone", "not provided"))  # default value

# Mutate
user["age"] = 31
user["city"] = "Paris"
del user["email"]

# Iterate
for key in user:               # iterates over keys
    print(key, "->", user[key])

for key, value in user.items():
    print(f"{key}={value}")

# Useful patterns
counts = {}
for word in ["a", "b", "a", "c", "b", "a"]:
    counts[word] = counts.get(word, 0) + 1
print(counts)

# Or use defaultdict / Counter (in stdlib)
from collections import Counter, defaultdict
print(Counter(["a", "b", "a", "c", "b", "a"]))

groups = defaultdict(list)
for n in range(10):
    groups["even" if n % 2 == 0 else "odd"].append(n)
print(dict(groups))

# Merge dicts (3.9+)
defaults = {"theme": "light", "lang": "en"}
overrides = {"lang": "fr"}
print(defaults | overrides)


# ============== SET ==============
a = {1, 2, 3, 4}
b = {3, 4, 5, 6}

print(a | b)   # union:        {1, 2, 3, 4, 5, 6}
print(a & b)   # intersection: {3, 4}
print(a - b)   # difference:   {1, 2}
print(a ^ b)   # symmetric:    {1, 2, 5, 6}

# Quick deduplication
items = [1, 2, 2, 3, 3, 3]
print(list(set(items)))

# Membership in a set is O(1) — much faster than `in list` for large data.


# ============== FROZENSET ==============
# Immutable set. Hashable, so usable as a dict key or set element.
key = frozenset({"red", "blue"})
table = {key: "purple-ish"}
print(table)
