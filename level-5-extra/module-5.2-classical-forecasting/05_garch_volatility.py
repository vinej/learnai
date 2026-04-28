"""
05 — GARCH for volatility forecasting

Returns are nearly unforecastable. Volatility is much more so. The
classic model:

    r_t = mu + epsilon_t,    epsilon_t = sigma_t * z_t,  z_t ~ iid
    sigma_t^2 = omega + alpha * epsilon_{t-1}^2 + beta * sigma_{t-1}^2

We fit GARCH(1,1) on SPY and BTC log returns, evaluate on rolling
1-day-ahead variance vs realized squared returns. Use the `arch` package.

Run: python 05_garch_volatility.py
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from _common import fetch_market

warnings.filterwarnings("ignore")


def garch_forecast(returns: pd.Series, n_test: int = 250) -> pd.DataFrame:
    """Walk-forward 1-day-ahead variance forecasts via refit-each-step."""
    from arch import arch_model

    r = (returns.dropna() * 100)  # arch likes returns in percent for stability
    out = []
    for i in range(len(r) - n_test, len(r)):
        train = r.iloc[:i]
        am = arch_model(train, mean="Constant", vol="GARCH", p=1, q=1, dist="t")
        res = am.fit(disp="off")
        f = res.forecast(horizon=1, reindex=False)
        sigma2_hat = float(f.variance.values[-1, 0]) / 1e4   # back to decimal^2
        out.append((r.index[i], sigma2_hat, (r.iloc[i] / 100) ** 2))
    return pd.DataFrame(out, columns=["date", "sigma2_hat", "realized_r2"]).set_index("date")


if __name__ == "__main__":
    try:
        import arch  # noqa: F401
    except ImportError:
        print("Install: pip install arch")
        raise SystemExit(0)

    for ticker in ("SPY", "BTC-USD"):
        px = fetch_market(ticker)["Close"]
        r = np.log(px / px.shift(1)).dropna()
        df = garch_forecast(r, n_test=200)
        # MSE between predicted variance and realized squared return
        mse = ((df["sigma2_hat"] - df["realized_r2"]) ** 2).mean()
        # vs naive: rolling sample variance of last 30 days
        rolling_var = (r ** 2).rolling(30).mean().shift(1)
        rolling_var = rolling_var.reindex(df.index)
        mse_naive = ((rolling_var - df["realized_r2"]) ** 2).mean()
        print(f"{ticker:<8}  GARCH MSE: {mse:.2e}    rolling-30 MSE: {mse_naive:.2e}")

    # GARCH usually beats rolling variance on equities and crypto for short
    # horizons. For h>5 days the advantage shrinks fast.
