"""
08 — File I/O

Modern Python: prefer `pathlib.Path` over raw string paths.
Always use the `with` statement so files are closed automatically.

Run: python 08_file_io.py
"""

import json
from pathlib import Path

# Work in a temporary folder next to this script
HERE = Path(__file__).parent
SCRATCH = HERE / "scratch"
SCRATCH.mkdir(exist_ok=True)


# --- Writing text ---
note = SCRATCH / "note.txt"
note.write_text("Hello, file!\nSecond line.\n", encoding="utf-8")

# Or with the lower-level open() context manager:
with open(note, "a", encoding="utf-8") as f:        # 'a' = append
    f.write("Third line, appended.\n")


# --- Reading text ---
content = note.read_text(encoding="utf-8")
print("--- full content ---")
print(content)

# Line by line (memory-friendly for big files)
print("--- line by line ---")
with open(note, encoding="utf-8") as f:
    for i, line in enumerate(f, start=1):
        print(f"{i}: {line.rstrip()}")


# --- JSON ---
config_path = SCRATCH / "config.json"
config = {
    "name": "demo",
    "features": ["a", "b", "c"],
    "settings": {"verbose": True, "retries": 3},
}

# Write JSON
with open(config_path, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)

# Read JSON
with open(config_path, encoding="utf-8") as f:
    loaded = json.load(f)
print("loaded:", loaded)


# --- pathlib useful operations ---
print("exists:", note.exists())
print("size:", note.stat().st_size, "bytes")
print("name:", note.name, "stem:", note.stem, "suffix:", note.suffix)
print("parent:", note.parent)

# List files (non-recursive)
for p in SCRATCH.iterdir():
    print(" ", p.name)

# Glob (recursive with **)
for p in SCRATCH.glob("*.txt"):
    print("found txt:", p)


# --- Binary I/O ---
binary_path = SCRATCH / "bytes.bin"
binary_path.write_bytes(b"\x00\x01\x02\x03")
print("binary contents:", binary_path.read_bytes())


# --- Cleanup (optional) ---
# Uncomment to delete the scratch folder when done:
# import shutil; shutil.rmtree(SCRATCH)
