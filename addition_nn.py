"""
Teaching a neural network to add two numbers from 1 to 100.
Pure NumPy implementation -- every matmul and gradient is visible.

The network never sees the rule "a + b = c". It only sees thousands of
examples and has to figure out the pattern using dot products and
gradient descent.
"""

import numpy as np

rng = np.random.default_rng(42)


# -----------------------------------------------------------------------------
# 1. Generate training data
# -----------------------------------------------------------------------------
def make_dataset(n_samples, low=1, high=100, seed=0):
    r = np.random.default_rng(seed)
    a = r.integers(low, high + 1, size=n_samples)
    b = r.integers(low, high + 1, size=n_samples)
    X = np.stack([a, b], axis=1).astype(np.float32)
    y = (a + b).astype(np.float32).reshape(-1, 1)
    return X, y


# Networks train MUCH better when inputs are roughly in [0, 1].
# Without scaling, gradients explode and training fails.
SCALE = 200.0  # max possible sum is 100 + 100 = 200

X_train, y_train = make_dataset(10_000, seed=42)
X_val,   y_val   = make_dataset(1_000,  seed=7)

X_train_s = X_train / SCALE
y_train_s = y_train / SCALE
X_val_s   = X_val   / SCALE
y_val_s   = y_val   / SCALE


# -----------------------------------------------------------------------------
# 2. Define the model -- explicit matmuls
# -----------------------------------------------------------------------------
# Architecture: 2 -> 16 -> 16 -> 1, with ReLU between hidden layers.
# Each layer is exactly: H = ReLU(X @ W + b)  -- the matmul we discussed.

HIDDEN = 16

def init_layer(in_dim, out_dim):
    # He initialization (good for ReLU): scale by sqrt(2/in_dim)
    W = rng.standard_normal((in_dim, out_dim)).astype(np.float32) * np.sqrt(2.0 / in_dim)
    b = np.zeros((out_dim,), dtype=np.float32)
    return W, b

W1, b1 = init_layer(2, HIDDEN)
W2, b2 = init_layer(HIDDEN, HIDDEN)
W3, b3 = init_layer(HIDDEN, 1)

n_params = W1.size + b1.size + W2.size + b2.size + W3.size + b3.size
print(f"Model has {n_params} parameters")


def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(np.float32)


def forward(X):
    """Returns the prediction AND the intermediate values (needed for backprop)."""
    Z1 = X @ W1 + b1          # matmul #1: (N, 2) @ (2, 16) -> (N, 16)
    H1 = relu(Z1)
    Z2 = H1 @ W2 + b2         # matmul #2: (N, 16) @ (16, 16) -> (N, 16)
    H2 = relu(Z2)
    Z3 = H2 @ W3 + b3         # matmul #3: (N, 16) @ (16, 1) -> (N, 1)
    return Z3, (X, Z1, H1, Z2, H2)


# -----------------------------------------------------------------------------
# 3. Train with manual gradient descent
# -----------------------------------------------------------------------------
# Loss = MSE = (1/N) * ||y_pred - y_true||_2^2  -- literally an L2 norm.
# We compute gradients via the chain rule and step against them.

LR = 0.01
N_EPOCHS = 3000

for epoch in range(N_EPOCHS):
    # Forward pass
    y_pred, cache = forward(X_train_s)
    X, Z1, H1, Z2, H2 = cache
    N = X.shape[0]

    # Loss
    error = y_pred - y_train_s             # shape (N, 1)
    loss = np.mean(error ** 2)             # MSE = squared L2 norm / N

    # Backward pass (chain rule, all matmuls)
    dZ3 = (2.0 / N) * error                # gradient of MSE wrt Z3
    dW3 = H2.T @ dZ3                       # matmul: how W3 should change
    db3 = dZ3.sum(axis=0)

    dH2 = dZ3 @ W3.T                       # matmul: gradient flowing back
    dZ2 = dH2 * relu_grad(Z2)
    dW2 = H1.T @ dZ2
    db2 = dZ2.sum(axis=0)

    dH1 = dZ2 @ W2.T
    dZ1 = dH1 * relu_grad(Z1)
    dW1 = X.T @ dZ1
    db1 = dZ1.sum(axis=0)

    # Gradient descent step
    W1 -= LR * dW1; b1 -= LR * db1
    W2 -= LR * dW2; b2 -= LR * db2
    W3 -= LR * dW3; b3 -= LR * db3

    if (epoch + 1) % 300 == 0:
        val_pred, _ = forward(X_val_s)
        val_loss = np.mean((val_pred - y_val_s) ** 2)
        mae = np.mean(np.abs(val_pred * SCALE - y_val_s * SCALE))
        print(f"Epoch {epoch+1:4d} | train_loss={loss:.6f} "
              f"| val_loss={val_loss:.6f} | val MAE={mae:.3f}")


# -----------------------------------------------------------------------------
# 4. Try it out
# -----------------------------------------------------------------------------
def predict(a, b):
    x = np.array([[a, b]], dtype=np.float32) / SCALE
    out, _ = forward(x)
    return out.item() * SCALE

print("\n--- Test cases (in-distribution: both inputs in [1, 100]) ---")
for a, b in [(2, 3), (17, 25), (50, 50), (99, 100), (1, 1), (73, 88)]:
    pred = predict(a, b)
    true = a + b
    print(f"  {a:3d} + {b:3d} = {true:3d}  |  predicted {pred:7.3f}  "
          f"(error {pred - true:+.3f})")


# -----------------------------------------------------------------------------
# 5. Out-of-distribution: numbers the network has never seen
# -----------------------------------------------------------------------------
print("\n--- Out-of-distribution (numbers > 100, never seen in training) ---")
for a, b in [(150, 200), (500, 500), (1000, 1)]:
    pred = predict(a, b)
    true = a + b
    print(f"  {a:4d} + {b:4d} = {true:5d}  |  predicted {pred:9.3f}  "
          f"(error {pred - true:+.2f})")


# -----------------------------------------------------------------------------
# 6. Bonus: a single linear layer is the OPTIMAL solution
# -----------------------------------------------------------------------------
# For pure addition, y = 1*a + 1*b + 0. A single Linear(2, 1) layer with
# no nonlinearity should find exactly that.
print("\n--- A linear model finds the exact rule ---")
W = (rng.standard_normal((2, 1)).astype(np.float32)) * 0.1
b = np.zeros((1,), dtype=np.float32)
for _ in range(3000):
    pred = X_train_s @ W + b
    err = pred - y_train_s
    dW = X_train_s.T @ err * (2.0 / X_train_s.shape[0])
    db = err.mean(axis=0) * 2.0
    W -= 0.05 * dW
    b -= 0.05 * db

# Because we scaled inputs and outputs by the same factor, the learned
# weights should converge to ~[1, 1] and bias to ~0.
print(f"  Learned weights: [{W[0,0]:.4f}, {W[1,0]:.4f}]   (true: [1.0, 1.0])")
print(f"  Learned bias:    {b[0]:.4f}                  (true: 0.0)")

# And test it on out-of-distribution -- it generalizes perfectly!
print("\n  Linear model on OOD inputs (it generalizes because it learned the RULE):")
for a, bb in [(150, 200), (500, 500), (1000, 1)]:
    x = np.array([[a, bb]], dtype=np.float32) / SCALE
    p = (x @ W + b).item() * SCALE
    print(f"    {a:4d} + {bb:4d} = {a+bb:5d}  |  predicted {p:9.3f}")
