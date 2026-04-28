"""
09 — Saving and loading models

After training, you need to persist the trained model so you can:
  • predict on new data without retraining
  • deploy to a server
  • compare models over time

Tools:
  joblib    — sklearn's recommended way; handles big NumPy arrays efficiently.
  pickle    — generic Python; works but joblib is faster for ML objects.
  ONNX      — open, framework-agnostic format. Great for deployment to
              C++, JavaScript, edge devices. (Brief mention here.)
  cloudpickle — like pickle but handles more weird closures; AI frameworks
                often rely on it under the hood.

Run: python 09_model_persistence.py
"""

from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ───────────────────────────────────────────────────────────
# Train a model
# ───────────────────────────────────────────────────────────
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

pipe = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=200, random_state=0))
pipe.fit(X_train, y_train)
print(f"trained pipeline accuracy: {pipe.score(X_test, y_test):.3f}")


# ───────────────────────────────────────────────────────────
# Save & load with joblib
# ───────────────────────────────────────────────────────────
out = Path(__file__).parent / "models"
out.mkdir(exist_ok=True)
model_path = out / "iris_pipeline.joblib"

joblib.dump(pipe, model_path)
print(f"saved → {model_path}  ({model_path.stat().st_size / 1024:.1f} KB)")

# In another process / day / server:
reloaded = joblib.load(model_path)
print(f"reloaded accuracy:        {reloaded.score(X_test, y_test):.3f}")


# ───────────────────────────────────────────────────────────
# Compression
# ───────────────────────────────────────────────────────────
# Useful when models are large.
joblib.dump(pipe, out / "iris_compressed.joblib", compress=3)
print(f"compressed: {(out / 'iris_compressed.joblib').stat().st_size / 1024:.1f} KB")


# ───────────────────────────────────────────────────────────
# Always save metadata alongside the model
# ───────────────────────────────────────────────────────────
# A bare .joblib has no record of:
#   • which library versions trained it
#   • which features it expects
#   • what the target classes mean
# Save a sidecar JSON with this info. Future you (or the on-call engineer)
# will be very grateful.
import json
import sklearn

metadata = {
    "sklearn_version":  sklearn.__version__,
    "feature_names":    load_iris().feature_names,
    "target_classes":   load_iris().target_names.tolist(),
    "training_score":   float(pipe.score(X_test, y_test)),
    "trained_at":       "2026-01-01T00:00:00Z",
}
(out / "iris_pipeline.metadata.json").write_text(json.dumps(metadata, indent=2))
print("metadata saved.")


# ───────────────────────────────────────────────────────────
# Cross-version risks
# ───────────────────────────────────────────────────────────
# • joblib/pickle files saved with sklearn X.Y are NOT guaranteed to load
#   in sklearn X.Z. Pin your library versions in production.
# • Untrusted pickle/joblib files can execute arbitrary code on load.
#   Never load model files from a source you don't fully trust.


# ───────────────────────────────────────────────────────────
# ONNX — for cross-language / cross-framework deployment
# ───────────────────────────────────────────────────────────
# `pip install skl2onnx onnxruntime`, then:
#
#   from skl2onnx import to_onnx
#   onnx_model = to_onnx(pipe, X_train[:1].astype(np.float32))
#   Path("model.onnx").write_bytes(onnx_model.SerializeToString())
#
# Now JavaScript/C++/Rust runtimes can load model.onnx and predict.


# ───────────────────────────────────────────────────────────
# A minimum production checklist
# ───────────────────────────────────────────────────────────
# 1. Save the FULL pipeline, not just the final model.
# 2. Save versions (sklearn, numpy, pandas) and a sample input.
# 3. Have an automated test that loads the model and asserts a known prediction.
# 4. Track models with a registry (MLflow, see module 2.4).
