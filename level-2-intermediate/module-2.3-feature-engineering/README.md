# Module 2.3 — Feature Engineering

**Level:** 2 — Intermediate
**Estimated time:** 2 weeks

## Goal
Turn raw data into signals models can learn from. In classical ML, feature engineering is often *more* important than the model.

## Topics
### Numerical
- Scaling: standardization, min-max, robust scaling
- Log / Box-Cox / Yeo-Johnson transforms for skewed data
- Binning / discretization
- Polynomial & interaction features

### Categorical
- One-hot, ordinal, label encoding
- Target / mean encoding (and how to avoid leakage)
- Hashing trick for high-cardinality features
- Embeddings (preview of deep learning approaches)

### Missing data
- Mean/median/mode imputation
- KNN imputation, iterative imputer
- Missingness as a signal

### Imbalanced classes
- Class weights
- Resampling: SMOTE, ADASYN, random under/over-sampling
- Threshold tuning

### Time series
- Lag features, rolling windows, expanding windows
- Date/time features (hour, weekday, holiday)
- Differencing, seasonality decomposition

### Selection
- Filter methods (variance, correlation, mutual info)
- Wrapper (RFE)
- Embedded (Lasso, tree feature importance, SHAP)

## Exercises
1. Take a real-world tabular dataset and engineer 10+ new features; measure model lift.
2. Compare one-hot vs target encoding on a high-cardinality column.
3. Handle a class imbalance (e.g., fraud detection) with three different strategies; compare PR-AUC.
4. Engineer time-series features for a forecasting task.

## Resources
- Book: *Feature Engineering for Machine Learning* — Zheng & Casari
- Kaggle "Feature Engineering" course

## Checkpoint
Given a raw tabular dataset, you can produce a feature matrix that materially improves model performance over a naive baseline.
