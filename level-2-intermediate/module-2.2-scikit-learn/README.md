# Module 2.2 — Scikit-learn

**Level:** 2 — Intermediate
**Estimated time:** 3 weeks

## Goal
Master the workhorse library of classical ML and be able to ship a trained model end-to-end.

## Topics
### Algorithms
- Linear regression, ridge, lasso, elastic net
- Logistic regression
- Decision trees, random forests
- Gradient boosting: `sklearn.ensemble`, **XGBoost**, **LightGBM**, **CatBoost**
- k-Nearest Neighbors
- Support Vector Machines (intuition + practical use)
- k-Means, DBSCAN, hierarchical clustering
- PCA, t-SNE, UMAP for dimensionality reduction

### Workflow
- The estimator API: `fit`, `predict`, `transform`, `score`
- `Pipeline` and `ColumnTransformer`
- Cross-validation: `cross_val_score`, `KFold`, `StratifiedKFold`, `TimeSeriesSplit`
- Hyperparameter search: `GridSearchCV`, `RandomizedSearchCV`, **Optuna**
- Saving models: `joblib`, ONNX (intro)

## Exercises
1. Predict house prices on the California Housing dataset — try linear, RF, and XGBoost; compare RMSE.
2. Classify the Wine dataset with a full `Pipeline` (impute → scale → encode → model).
3. Tune XGBoost with Optuna and beat your baseline.
4. Cluster customers with k-Means; visualize with PCA.

## Resources
- scikit-learn user guide: https://scikit-learn.org/stable/user_guide.html
- XGBoost docs: https://xgboost.readthedocs.io/
- Book: *Hands-On Machine Learning* — Aurélien Géron (Ch. 5-9)

## Checkpoint
You can: build a `Pipeline` from raw data → predictions, run cross-validated hyperparameter search, and persist the trained model for reuse.
