# Module 3.1 — Deep Learning Foundations

**Level:** 3 — Advanced
**Estimated time:** 3 weeks

## Goal
Understand how neural networks learn, and be able to write a training loop in PyTorch from scratch.

## Topics
### Theory
- The perceptron and multi-layer perceptron (MLP)
- Activation functions: ReLU, GELU, sigmoid, tanh, softmax
- Forward pass, loss functions (cross-entropy, MSE, BCE)
- Backpropagation & the chain rule (revisit)
- Optimizers: SGD, Momentum, RMSProp, Adam, AdamW
- Learning rate, schedulers, warmup
- Regularization: dropout, weight decay, batch norm, layer norm, early stopping
- Initialization: Xavier, Kaiming

### PyTorch
- Tensors, devices (CPU/GPU/MPS), `dtype`
- Autograd: `requires_grad`, `backward`, `grad`
- `nn.Module`, `nn.Linear`, `nn.Sequential`
- `Dataset`, `DataLoader`, `collate_fn`
- The training loop: forward → loss → backward → step → zero_grad
- Saving / loading checkpoints
- TensorBoard integration

## Exercises
1. Implement a 2-layer MLP from scratch (NumPy only) — including backprop.
2. Reimplement it in PyTorch using `nn.Module`.
3. Train it on MNIST; reach >97% test accuracy.
4. Compare optimizers (SGD vs Adam) and learning rates on the same problem.
5. Add dropout and batch norm; observe effect on overfitting.

## Resources
- Book: *Deep Learning* — Goodfellow, Bengio, Courville (Ch. 6-8)
- PyTorch tutorials: https://pytorch.org/tutorials/
- fast.ai Practical Deep Learning course
- Andrej Karpathy's "Neural Networks: Zero to Hero" (YouTube)

## Checkpoint
You can write a PyTorch training loop from a blank file, train a model on a non-trivial dataset, and explain what happens in each line of `loss.backward()` → `optimizer.step()`.
