"""
Exercise 3 — Add type hints and pass `mypy --strict`

This module has no type hints. Your job:

1. Add hints to every function parameter and return value.
2. Run:
       mypy --strict exercises/03_add_type_hints.py
3. Fix every error mypy reports — without changing the runtime behavior.

Hints (no peeking until you've tried):
  • parse_score takes a string, returns an int.
  • average takes a sequence of ints (try `from collections.abc import Sequence`)
    and returns a float.
  • grade returns one of "A", "B", "C", "D", "F". You can use Literal for that.
  • grade_all takes a sequence and returns a list of grades.
  • Watch out for `int / int` which returns a float — make sure your return
    type matches what's actually returned.

When you're done, also run the script directly to confirm it still works:

    python exercises/03_add_type_hints.py
"""


def parse_score(text):
    return int(text.strip())


def average(scores):
    if not scores:
        raise ValueError("no scores to average")
    return sum(scores) / len(scores)


def grade(score):
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"


def grade_all(scores):
    return [grade(s) for s in scores]


def best_student(name_to_score):
    return max(name_to_score, key=lambda name: name_to_score[name])


if __name__ == "__main__":
    raw = ["88 ", " 92", "75", "63", "55"]
    parsed = [parse_score(s) for s in raw]

    print("parsed:  ", parsed)
    print("average: ", average(parsed))
    print("grades:  ", grade_all(parsed))

    students = {"Alice": 88, "Bob": 92, "Carol": 75}
    print("best:    ", best_student(students))
