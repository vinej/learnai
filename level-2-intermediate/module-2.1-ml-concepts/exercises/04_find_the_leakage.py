"""
Exercise 4 — Find the data leakage

The function `train_buggy()` below trains a classifier and reports test
accuracy. The number it prints will look great. There are TWO leakage bugs.

  1. Run the script — observe how good the metrics look.
  2. Find both bugs.
  3. Implement `train_correct()` that fixes them. The script asserts that
     the corrected accuracy is LOWER (because the leaks were inflating it).

Hints:
  • Where is the scaler fit?
  • What does `helpful_feature` actually contain?

Run: python exercises/04_find_the_leakage.py
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


rng = np.random.default_rng(0)


def make_dataset():
    X, y = make_classification(
        n_samples=2000, n_features=10, n_informative=5,
        random_state=0,
    )
    return X, y


# ═══════════════════════════════════════════════════════════
# THE BUGGY VERSION — read carefully
# ═══════════════════════════════════════════════════════════
def train_buggy() -> float:
    X, y = make_dataset()

    # 🐞 BUG #1
    helpful_feature = y + rng.normal(0, 0.05, y.shape)
    X = np.column_stack([X, helpful_feature])

    # 🐞 BUG #2
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=0, stratify=y,
    )
    model = LogisticRegression(max_iter=2000).fit(X_train, y_train)
    return accuracy_score(y_test, model.predict(X_test))


# ═══════════════════════════════════════════════════════════
# YOUR CORRECTED VERSION
# ═══════════════════════════════════════════════════════════
def train_correct() -> float:
    X, y = make_dataset()
    # ▼▼▼ Fix the bugs here ▼▼▼

    # Fix 1: do NOT add a feature derived from y. Drop it entirely.
    # (`helpful_feature` is removed.)

    # Fix 2: scale INSIDE a Pipeline, AFTER the split, on training data only.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y,
    )
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    model.fit(X_train, y_train)
    return accuracy_score(y_test, model.predict(X_test))


def main() -> None:
    buggy_acc   = train_buggy()
    correct_acc = train_correct()

    print(f"buggy   accuracy: {buggy_acc:.3f}")
    print(f"correct accuracy: {correct_acc:.3f}")

    assert correct_acc < buggy_acc, (
        "the buggy version should look better — that's the trap. "
        "If your 'correct' version still inflates, you haven't fixed both bugs."
    )
    print("leakage successfully removed ✅")


if __name__ == "__main__":
    main()


# ── Lessons ──────────────────────────────────────────────────────────────────
# • Always ask: "could this feature have been computed at PREDICTION TIME, in
#   production, without knowing the answer?" If no, it's leakage.
# • Always ask: "did any preprocessor see data outside its training fold?"
# • Pipelines aren't just convenience — they are LEAKAGE INSURANCE.
# ─────────────────────────────────────────────────────────────────────────────
