"""
Exercise 3 — Word frequency counter

Counts words in a text file, ignoring case and punctuation.
Prints the top N most frequent words.

Concepts exercised: file I/O, strings, dict / Counter, sorting, argparse.

Usage:
    python exercises/03_word_counter.py exercises/sample.txt
    python exercises/03_word_counter.py exercises/sample.txt --top 5
"""

import argparse
import re
import sys
from collections import Counter
from pathlib import Path


WORD_RE = re.compile(r"[a-z']+")


def count_words(text: str) -> Counter:
    words = WORD_RE.findall(text.lower())
    return Counter(words)


def main():
    parser = argparse.ArgumentParser(description="Count words in a text file")
    parser.add_argument("path", type=Path)
    parser.add_argument("--top", type=int, default=10, help="how many to show")
    args = parser.parse_args()

    if not args.path.exists():
        sys.exit(f"error: file not found: {args.path}")

    text = args.path.read_text(encoding="utf-8")
    counts = count_words(text)

    print(f"{len(counts)} unique words, {sum(counts.values())} total")
    print(f"top {args.top}:")
    for word, n in counts.most_common(args.top):
        print(f"  {n:>5}  {word}")


if __name__ == "__main__":
    main()
