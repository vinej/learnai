# Module 2.1 — ML Concepts

**Level:** 2 — Intermediate
**Estimated time:** 2 weeks

## Goal
Develop the mental model of machine learning: what a model does, how to evaluate it, and how to avoid fooling yourself.

## Topics
- What "learning" means: parameters, loss, optimization
- Categories: supervised, unsupervised, semi-supervised, reinforcement
- Splits: train/validation/test, k-fold cross-validation, stratification
- Bias-variance tradeoff
- Overfitting & underfitting — recognizing both
- Regularization (intuition: L1, L2, early stopping)
- Metrics:
  - **Classification:** accuracy, precision, recall, F1, ROC-AUC, PR-AUC, log-loss, confusion matrix
  - **Regression:** MAE, MSE, RMSE, R²
- Baselines (always start with one!)
- Data leakage — what it is and how it sneaks in

## Exercises
1. For a classification dataset, plot the confusion matrix and compute precision/recall/F1 by class.
2. Train a model with and without regularization — plot training vs validation loss to see overfitting.
3. Build a "dummy" baseline (most-frequent / mean predictor) and beat it.
4. Find the data leakage: given a notebook with leakage planted, identify and fix it.

## Resources
- Book: *Hands-On Machine Learning* — Aurélien Géron (Ch. 1-4)
- Andrew Ng's *Machine Learning Yearning* (free)
- Google ML Crash Course

## Checkpoint
You can explain — without notes — when to use precision vs recall, what overfitting looks like on a learning curve, and why a held-out test set must remain untouched until the very end.
