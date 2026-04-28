"""
Exercise 3 — Gradient descent on f(x) = x² + 3x + 2

Find the minimum of  f(x) = x² + 3x + 2  using gradient descent.

  • f'(x) = 2x + 3
  • f'(x) = 0  →  x = -1.5     (the analytic minimum)

The script:
  1. Starts at some x0.
  2. Repeatedly steps:  x ← x - learning_rate * f'(x).
  3. Plots f(x) and overlays the path of x values.

Run: python exercises/03_gradient_descent.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def f(x):
    return x ** 2 + 3 * x + 2


def f_prime(x):
    return 2 * x + 3


def gradient_descent(start, lr=0.1, steps=30):
    """Returns the list of x values visited (including the start)."""
    history = [start]
    x = start
    for _ in range(steps):
        x = x - lr * f_prime(x)
        history.append(x)
    return history


if __name__ == "__main__":
    history = gradient_descent(start=4.0, lr=0.1, steps=30)
    print(f"final x: {history[-1]:.6f}    (analytic min: -1.5)")
    print(f"final f(x): {f(history[-1]):.6f}    (analytic min value: -0.25)")

    # ── plot ──
    xs = np.linspace(-5, 5, 200)
    ys = f(xs)

    history = np.array(history)
    history_y = f(history)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, ys, label="f(x) = x² + 3x + 2")
    ax.plot(history, history_y, "ro-", markersize=5, label="GD path")

    # annotate first and last
    ax.annotate("start", (history[0], history_y[0]),
                xytext=(10, 10), textcoords="offset points")
    ax.annotate("end", (history[-1], history_y[-1]),
                xytext=(10, -15), textcoords="offset points")

    ax.axvline(-1.5, color="green", linestyle="--", alpha=0.5,
               label="true minimum x = -1.5")
    ax.set_title("Gradient descent on a parabola")
    ax.legend()
    ax.grid(True, alpha=0.3)

    out = Path(__file__).parent.parent / "figures"
    out.mkdir(exist_ok=True)
    path = out / "exercise_03_gradient_descent.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"figure saved to: {path}")


# ── Bonus questions ──────────────────────────────────────────────────────────
# 1. Try lr = 0.01, lr = 0.5, lr = 1.1 — what happens with each?
#    (Tiny lr → very slow; large lr → can overshoot or even diverge.)
#
# 2. Modify gradient_descent to STOP early when |f'(x)| is below a tolerance.
#
# 3. Generalize to a function of two variables, e.g. f(x, y) = x² + y².
#    You'll need a 2-D contour plot and a vector-valued gradient.
# ─────────────────────────────────────────────────────────────────────────────
