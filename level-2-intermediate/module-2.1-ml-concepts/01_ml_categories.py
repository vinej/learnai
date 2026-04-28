"""
01 — Categories of machine learning

Three broad paradigms:

  SUPERVISED       — labeled data:  X → y.   Most of what you'll do.
                     Subtypes: classification (discrete y), regression (continuous y).
  UNSUPERVISED     — only X, no labels. Find structure: clustering, PCA, topic models.
  REINFORCEMENT    — an agent acting in an environment, learning from rewards.
                     (We won't implement RL here — it deserves its own course.)

Run: python 01_ml_categories.py
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris, make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


rng = np.random.default_rng(0)


# ───────────────────────────────────────────────────────────
# 1. Supervised — Classification (discrete labels)
# ───────────────────────────────────────────────────────────
iris = load_iris()
X, y = iris.data, iris.target          # X: 150×4 features, y: 150 class labels (0/1/2)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y,
)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("=== Supervised classification (Iris) ===")
print(f"  test accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"  classes: {iris.target_names.tolist()}")


# ───────────────────────────────────────────────────────────
# 2. Supervised — Regression (continuous target)
# ───────────────────────────────────────────────────────────
X, y = make_regression(n_samples=200, n_features=3, noise=10.0, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

reg = LinearRegression().fit(X_train, y_train)
y_pred = reg.predict(X_test)

print("\n=== Supervised regression (synthetic) ===")
print(f"  RMSE on test:   {mean_squared_error(y_test, y_pred) ** 0.5:.3f}")
print(f"  learned coefs:  {reg.coef_.round(2)}")


# ───────────────────────────────────────────────────────────
# 3. Unsupervised — Clustering
# ───────────────────────────────────────────────────────────
# Same Iris features, but pretend we've LOST the labels.
# k-means tries to find 3 groups using only X.
X = iris.data
clusters = KMeans(n_clusters=3, n_init="auto", random_state=0).fit_predict(X)

print("\n=== Unsupervised clustering (Iris, no labels) ===")
print(f"  cluster sizes: {np.bincount(clusters)}")
# Note: cluster IDs don't necessarily match the true labels (0,1,2). We'd need
# to align them to compute "accuracy". The point is: structure was discovered
# without ANY labels.


# ───────────────────────────────────────────────────────────
# 4. Reinforcement — concept only
# ───────────────────────────────────────────────────────────
# An RL agent observes a STATE, picks an ACTION, gets a REWARD, and updates
# its policy to maximize cumulative reward. Examples:
#   • AlphaGo / chess engines     • Robotics / control
#   • RLHF for aligning LLMs       • Recommender bandits
# Libraries to know: stable-baselines3, gymnasium, RLlib.
print("\n=== Reinforcement learning ===")
print("  conceptual only — see Module 4.3 (RLHF) and external RL courses.")


# ───────────────────────────────────────────────────────────
# Other useful flavors
# ───────────────────────────────────────────────────────────
# • Semi-supervised: small labeled set + large unlabeled set.
# • Self-supervised: labels derived from the data itself (masked-LM, contrastive).
#   This is how foundation models (BERT, GPT, CLIP) are pre-trained.
# • Online learning: model updates one sample at a time as data streams in.
