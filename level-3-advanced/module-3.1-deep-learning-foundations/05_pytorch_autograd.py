"""
05 — PyTorch autograd

`autograd` records every operation on a tensor with `requires_grad=True`,
building a computation graph. Calling `.backward()` walks the graph and
fills each parameter's `.grad` attribute via the chain rule.

This is the magic that makes `loss.backward()` work — file 03 did it by hand.

Run: python 05_pytorch_autograd.py
"""

import torch


# ───────────────────────────────────────────────────────────
# A toy: minimize f(x) = (x - 3)² with autograd-driven gradient descent
# ───────────────────────────────────────────────────────────
x = torch.tensor(10.0, requires_grad=True)
lr = 0.1

for step in range(20):
    loss = (x - 3) ** 2
    loss.backward()                 # populates x.grad
    with torch.no_grad():
        x -= lr * x.grad            # gradient step
    x.grad.zero_()                  # PyTorch ACCUMULATES grads — clear after each step

    if step % 5 == 0 or step == 19:
        print(f"step {step:>2}: x={x.item():.5f}, loss={loss.item():.5f}")

print(f"\nfinal x: {x.item():.4f} (true minimum is 3.0)")


# ───────────────────────────────────────────────────────────
# Autograd in 2-D: minimize f(x, y) = x² + (y - 5)²
# ───────────────────────────────────────────────────────────
xy = torch.tensor([4.0, 0.0], requires_grad=True)
optimizer = torch.optim.SGD([xy], lr=0.1)

for _ in range(30):
    optimizer.zero_grad()
    loss = (xy[0] ** 2) + (xy[1] - 5) ** 2
    loss.backward()
    optimizer.step()

print(f"\noptimized xy: {xy.detach().tolist()}  (target [0, 5])")


# ───────────────────────────────────────────────────────────
# requires_grad — tracking vs. not
# ───────────────────────────────────────────────────────────
a = torch.tensor([1.0, 2.0])               # not tracked by default
b = torch.tensor([3.0, 4.0], requires_grad=True)
c = a + b
print(f"\nc.requires_grad = {c.requires_grad} (any input had grad → output has grad)")


# ───────────────────────────────────────────────────────────
# torch.no_grad() — disable tracking inside a block
# ───────────────────────────────────────────────────────────
# Use this for INFERENCE: it's faster and uses less memory because nothing
# is recorded for backprop.
model = torch.nn.Linear(4, 2)
x = torch.randn(8, 4)
with torch.no_grad():
    y = model(x)
print(f"\ninference output requires_grad: {y.requires_grad}")


# ───────────────────────────────────────────────────────────
# .detach() — get a copy without grad tracking
# ───────────────────────────────────────────────────────────
# Useful when you want to use a tensor's value but break the gradient flow
# (e.g., target networks in RL, "detaching" a hidden state in RNNs).
z = torch.tensor([1.0, 2.0], requires_grad=True)
w = z * 2
detached = w.detach()
print(f"\nw.requires_grad = {w.requires_grad}, detached = {detached.requires_grad}")


# ───────────────────────────────────────────────────────────
# Gradient accumulation — why we call zero_grad()
# ───────────────────────────────────────────────────────────
# .backward() ADDS to .grad. If you don't zero, gradients from earlier
# minibatches keep adding up.
p = torch.tensor(1.0, requires_grad=True)
(p * 2).backward()
print(f"\nafter 1st backward: p.grad = {p.grad.item()}")
(p * 3).backward()
print(f"after 2nd (no zero): p.grad = {p.grad.item()}  (accumulated!)")
p.grad.zero_()
(p * 3).backward()
print(f"after zero + backward:  p.grad = {p.grad.item()}")


# ───────────────────────────────────────────────────────────
# Common errors and fixes
# ───────────────────────────────────────────────────────────
# • "RuntimeError: ... requires_grad" — you tried to modify a leaf tensor in-place
#   without no_grad. Wrap the update in `with torch.no_grad():` or use an optimizer.
#
# • "Tried to call backward through the graph a second time" —
#   each backward consumes the graph. Call backward(retain_graph=True) if you
#   really need to, or recompute the forward.
#
# • "element 0 of tensors does not require grad" —
#   you're calling .backward() on a tensor that wasn't computed from a leaf
#   with requires_grad=True. Check inputs.
#
# • Gradients are zero / NaN — check for ReLU dead units, exploding logits,
#   bad init, or wrong loss/output pairing (see file 02).
