"""CQR / split-conformal calibration on top of the ensemble's point + raw PI."""
from __future__ import annotations

import numpy as np


def split_conformal_pad(residuals: np.ndarray, alpha: float = 0.1) -> float:
    n = len(residuals)
    q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(np.abs(residuals), q_level))


def cqr_pad(low_cal: np.ndarray, high_cal: np.ndarray, y_cal: np.ndarray,
             alpha: float = 0.1) -> float:
    """Conformal pad to add to (lo, hi) for guaranteed coverage."""
    s = np.maximum(low_cal - y_cal, y_cal - high_cal)
    n = len(s)
    q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(s, q_level))
