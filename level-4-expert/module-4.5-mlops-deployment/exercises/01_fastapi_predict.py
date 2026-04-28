"""
Exercise 1 — Train + serve a model with FastAPI

A complete end-to-end CLI:

  python 01_fastapi_predict.py train     # train model.joblib
  python 01_fastapi_predict.py serve     # start the API on :8000
  python 01_fastapi_predict.py demo      # send a few requests, assert OK

Asserts (in `demo`):
  - /health returns ok
  - /predict on a valid input returns a price
  - /predict on a bad input returns 422
"""

import sys
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).parent
MODEL_PATH = HERE / "scratch" / "model.joblib"
MODEL_PATH.parent.mkdir(exist_ok=True)


# ───────────────────────────────────────────────────────────
# Train
# ───────────────────────────────────────────────────────────
def train():
    rng = np.random.default_rng(0)
    n = 5000
    X = rng.uniform(0, 1, (n, 4))   # features in [0, 1]
    bedrooms = (X[:, 0] * 5 + 1).round()
    bathrooms = (X[:, 1] * 4 + 1)
    sqft = (X[:, 2] * 4000 + 500).round()
    lot = (X[:, 3] * 50_000 + 5_000).round()
    feats = np.stack([bedrooms, bathrooms, sqft, lot], axis=1)
    price = (
        bedrooms * 30_000
        + bathrooms * 20_000
        + sqft * 200
        + lot * 1.5
        + rng.normal(0, 20_000, n)
    )

    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("model", GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=0)),
    ])
    pipe.fit(feats, price)
    joblib.dump(pipe, MODEL_PATH)
    print(f"trained → {MODEL_PATH}  R²={pipe.score(feats, price):.3f}")


# ───────────────────────────────────────────────────────────
# Serve
# ───────────────────────────────────────────────────────────
class HouseFeatures(BaseModel):
    bedrooms:  int   = Field(ge=0, le=20)
    bathrooms: float = Field(ge=0, le=20)
    sqft:      int   = Field(gt=0, le=20_000)
    lot_size:  int   = Field(ge=0, le=1_000_000)


class Prediction(BaseModel):
    predicted_price: float
    model_version: str = "0.1.0"


def make_app() -> FastAPI:
    if not MODEL_PATH.exists():
        raise RuntimeError(f"missing {MODEL_PATH}; run `... train` first.")

    app = FastAPI(title="House price predictor", version="0.1.0")
    app.state.model = joblib.load(MODEL_PATH)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/predict", response_model=Prediction)
    def predict(features: HouseFeatures) -> Prediction:
        try:
            x = [[features.bedrooms, features.bathrooms,
                  features.sqft, features.lot_size]]
            y = float(app.state.model.predict(x)[0])
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        return Prediction(predicted_price=round(y, 2))

    return app


def serve():
    import uvicorn
    app = make_app()
    print("serving on http://127.0.0.1:8000  (docs at /docs)")
    uvicorn.run(app, host="127.0.0.1", port=8000)


# ───────────────────────────────────────────────────────────
# Demo client (assumes serve is running in another terminal)
# ───────────────────────────────────────────────────────────
def demo():
    import httpx

    base = "http://127.0.0.1:8000"
    h = httpx.get(f"{base}/health", timeout=5)
    assert h.status_code == 200 and h.json() == {"status": "ok"}
    print(f"  /health → {h.json()}")

    good = httpx.post(f"{base}/predict",
                      json={"bedrooms": 3, "bathrooms": 2, "sqft": 1800, "lot_size": 7000},
                      timeout=5)
    assert good.status_code == 200
    print(f"  /predict (valid) → {good.json()}")

    bad = httpx.post(f"{base}/predict",
                     json={"bedrooms": -1, "bathrooms": 2, "sqft": 0, "lot_size": -5},
                     timeout=5)
    assert bad.status_code == 422
    print(f"  /predict (invalid) → 422 (expected)")
    print("\nFastAPI service works end-to-end ✅")


# ───────────────────────────────────────────────────────────
# Dispatch
# ───────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    action = sys.argv[1]
    if action == "train":
        train()
    elif action == "serve":
        serve()
    elif action == "demo":
        demo()
    else:
        print(f"unknown action: {action}")
        sys.exit(1)


if __name__ == "__main__":
    main()


# ── Stretch ─────────────────────────────────────────────────────────────────
# • Containerize: write a Dockerfile (file 03) and `docker run` it.
# • Add API key auth via FastAPI dependency.
# • Add /metrics with prometheus-client (exercise 2).
# ─────────────────────────────────────────────────────────────────────────────
