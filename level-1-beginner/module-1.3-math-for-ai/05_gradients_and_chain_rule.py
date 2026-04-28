"""
05 — Gradients and the chain rule

A *gradient* is the multivariable generalization of a derivative.
For f(x, y), the gradient is the vector of partial derivatives:

    ∇f = ( ∂f/∂x , ∂f/∂y )

It points in the direction of *steepest increase* of f.
Gradient DESCENT walks in the OPPOSITE direction to minimize f.

The *chain rule* lets you differentiate a composition of functions:

    if z = g(y)  and  y = h(x),  then  dz/dx = (dz/dy) * (dy/dx)

Backpropagation is the chain rule applied automatically through every
layer of a neural network.

Run: python 05_gradients_and_chain_rule.py
"""

import numpy as np


# ───────────────────────────────────────────────────────────
# 1. Partial derivatives
# ───────────────────────────────────────────────────────────
# f(x, y) = x² + 3xy + y²
# ∂f/∂x = 2x + 3y
# ∂f/∂y = 3x + 2y
def f(x, y):
    return x ** 2 + 3 * x * y + y ** 2

def grad_f(x, y):
    return np.array([2 * x + 3 * y, 3 * x + 2 * y])

print("f(1, 2) =", f(1, 2))
print("∇f(1, 2) =", grad_f(1, 2))


# ───────────────────────────────────────────────────────────
# 2. Numerical gradient (a sanity check)
# ───────────────────────────────────────────────────────────
def numerical_gradient(func, point, h=1e-5):
    point = np.asarray(point, dtype=float)
    grad = np.zeros_like(point)
    for i in range(len(point)):
        plus, minus = point.copy(), point.copy()
        plus[i]  += h
        minus[i] -= h
        grad[i] = (func(*plus) - func(*minus)) / (2 * h)
    return grad

print("numerical ∇f(1, 2):", numerical_gradient(f, [1, 2]))
print("analytic  ∇f(1, 2):", grad_f(1, 2))


# ───────────────────────────────────────────────────────────
# 3. The chain rule, by hand
# ───────────────────────────────────────────────────────────
# Let y = 3x + 1   and   z = y²
# Then z = (3x + 1)²
#
# Direct derivative:    dz/dx = 2(3x + 1) * 3 = 6(3x + 1)
# By the chain rule:    dz/dx = (dz/dy)(dy/dx) = 2y * 3 = 6y = 6(3x + 1)  ✓
def y_of(x): return 3 * x + 1
def z_of(y): return y ** 2

x = 4.0
y = y_of(x)
z = z_of(y)

dz_dy = 2 * y           # local derivative
dy_dx = 3               # local derivative
dz_dx_chain = dz_dy * dy_dx
dz_dx_direct = 6 * (3 * x + 1)

print(f"\nat x={x}:  dz/dx (chain) = {dz_dx_chain}, "
      f"dz/dx (direct) = {dz_dx_direct}")


# ───────────────────────────────────────────────────────────
# 4. Why this matters for neural networks
# ───────────────────────────────────────────────────────────
# A neural net is just a deep composition of differentiable functions.
# Forward pass:    x → layer1 → layer2 → ... → layerN → loss
# Backward pass:   apply the chain rule from loss back to each parameter.
#                  That gives ∂loss/∂param — the signal that updates weights.
# Frameworks like PyTorch do this automatically (autograd). Here you've
# done a one-step version by hand.
