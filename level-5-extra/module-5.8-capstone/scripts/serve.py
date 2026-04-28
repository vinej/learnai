"""Convenience launcher for the FastAPI service."""
from __future__ import annotations

import uvicorn

if __name__ == "__main__":
    uvicorn.run("forecaster.serving.api:app", host="0.0.0.0", port=8000, reload=True)
