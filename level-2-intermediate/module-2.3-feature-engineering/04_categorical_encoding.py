"""
04 — Encoding categorical features

ML models eat numbers. Categories must be turned into numeric form somehow.

  ONE-HOT ENCODING        — one binary column per level. Most universal default.
                            Explodes for high cardinality.
  ORDINAL ENCODING        — map levels to integers. Implies an ORDER, so use only
                            when the categories really are ordered (low/med/high).
  LABEL ENCODING          — same as ordinal but for the TARGET (in classification).
  HASHING TRICK           — hash levels into N buckets. Fast, fixed dimension,
                            collisions OK in practice.
  TARGET / MEAN ENCODING  — replace category with its mean target value.
                            Powerful but leakage-prone — covered in file 05.

Run: python 04_categorical_encoding.py
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder


# Toy dataset
df = pd.DataFrame({
    "city":    ["Paris", "London", "Paris", "Berlin", "Madrid", "Berlin"],
    "size":    ["small", "large",  "medium","large",  "small",  "large"],
    "country": ["FR",    "UK",     "FR",    "DE",     "ES",     "DE"],
})
print("raw:")
print(df)


# ───────────────────────────────────────────────────────────
# 1. One-hot encoding
# ───────────────────────────────────────────────────────────
# Always pass `handle_unknown='ignore'` so unseen test-time categories don't crash.
ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
X_ohe = ohe.fit_transform(df[["city"]])
print(f"\none-hot 'city':   shape {X_ohe.shape}")
print(pd.DataFrame(X_ohe, columns=ohe.get_feature_names_out(["city"])))


# ───────────────────────────────────────────────────────────
# 2. Ordinal encoding (when the order is meaningful)
# ───────────────────────────────────────────────────────────
size_order = [["small", "medium", "large"]]
oe = OrdinalEncoder(categories=size_order)
X_size = oe.fit_transform(df[["size"]])
print(f"\nordinal 'size' (small=0, medium=1, large=2):")
print(np.column_stack([df["size"].values, X_size.ravel()]))


# ───────────────────────────────────────────────────────────
# 3. Label encoding (target only!)
# ───────────────────────────────────────────────────────────
# LabelEncoder is for the y vector, not features. It's a 1-D ordinal encoder.
y = ["spam", "ham", "spam", "ham", "spam"]
le = LabelEncoder().fit(y)
print(f"\nlabel encoded y: {le.transform(y).tolist()}")
print(f"classes:         {le.classes_.tolist()}")


# ───────────────────────────────────────────────────────────
# 4. Hashing trick — for high-cardinality categoricals
# ───────────────────────────────────────────────────────────
# Imagine 100,000+ unique zip codes. One-hot is impractical. Hashing maps
# each value into a fixed number of buckets.
# Tradeoff: collisions (different categories hash to the same bucket).
# Usually OK because the model can still learn from the partial signal.
hasher = FeatureHasher(n_features=8, input_type="string")
zips = ["94110", "94107", "10001", "94110", "94110", "10002"]
hashed = hasher.transform([[z] for z in zips]).toarray()
print(f"\nhashed zips into 8 buckets:\n{hashed.astype(int)}")


# ───────────────────────────────────────────────────────────
# 5. Frequency / count encoding (DIY, often surprisingly useful)
# ───────────────────────────────────────────────────────────
# Replace each category with how often it appears.
freq = df["city"].value_counts(normalize=True)
df_with_freq = df.assign(city_freq=df["city"].map(freq))
print("\nfrequency-encoded city:")
print(df_with_freq[["city", "city_freq"]])


# ───────────────────────────────────────────────────────────
# Picking an encoder
# ───────────────────────────────────────────────────────────
# • Cardinality < ~50 levels                 → OneHotEncoder.
# • Truly ordered categories                  → OrdinalEncoder (with `categories`).
# • High cardinality + linear model           → Hashing or Target encoding.
# • High cardinality + tree model             → Often OrdinalEncoder works fine
#   (trees can split on integer codes).
# • Very high cardinality + neural net        → Embedding layer (Module 3.3).


# ───────────────────────────────────────────────────────────
# Pitfalls
# ───────────────────────────────────────────────────────────
# • DON'T use OrdinalEncoder for unordered categories (city, color) before a
#   LINEAR model — you imply false ordering. Trees can still cope.
# • DON'T fit OneHotEncoder on the full dataset before splitting — that's
#   leakage if the test set has categories the train set didn't. Use a Pipeline.
# • Always set `handle_unknown='ignore'` (or `infrequent_if_exist` in newer
#   sklearn) so production data with new levels doesn't crash predict().
