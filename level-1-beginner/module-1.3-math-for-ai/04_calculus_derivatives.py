"""
04 — Derivatives (numerical)

The derivative f'(x) is the slope of f at x. It tells you how fast f changes.

Analytically:
    if f(x) = x²,  then f'(x) = 2x

Numerically (handy when you don't have an analytic form):
    f'(x) ≈ (f(x + h) - f(x - h)) / (2h)        # central difference

This script plots f(x) = x² and overlays its analytic and numerical
derivatives, then saves a PNG.

Run: python 04_calculus_derivatives.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def f(x):
    return x ** 2


def f_prime_analytic(x):
    return 2 * x


def numerical_derivative(func, x, h=1e-5):
    """Central-difference derivative — accurate to second order in h."""
    return (func(x + h) - func(x - h)) / (2 * h)


x = np.linspace(-3, 3, 200)

y = f(x)
dy_analytic = f_prime_analytic(x)
dy_numeric = numerical_derivative(f, x)

# How close is the numerical derivative to the truth?
max_err = np.max(np.abs(dy_analytic - dy_numeric))
print(f"max error between analytic and numerical: {max_err:.2e}")


# --- Plot ---
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, y,           label="f(x) = x²")
ax.plot(x, dy_analytic, label="f'(x) = 2x  (analytic)")
ax.plot(x, dy_numeric,  label="f'(x) (numerical)", linestyle="--")
ax.axhline(0, color="black", linewidth=0.5)
ax.axvline(0, color="black", linewidth=0.5)
ax.grid(True, alpha=0.3)
ax.set_title("A function and its derivative")
ax.legend()

out = Path(__file__).parent / "figures"
out.mkdir(exist_ok=True)
path = out / "04_derivatives.png"
fig.savefig(path, dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"figure saved to: {path}")


# --- Why care about derivatives in ML? ---
# Training a model = MINIMIZING a loss function L(θ) over the parameters θ.
# To do that we need the gradient ∂L/∂θ — which tells us which way to move
# the parameters to reduce L. That's gradient descent. See file 05.
