"""
Exercise 6 — Bayesian analysis of a stock signal

The hypothesis: "after a big down week, the stock is more likely to bounce."
Sounds plausible. Is it true? Bayes' rule, applied to actual data, gives
us an honest answer.

Pipeline:
  1. Get weekly returns (synthetic random walk by default; yfinance if you
     want real data — see TICKER below).
  2. Define a SIGNAL:    S = "this week's return < -3%"
  3. Define an OUTCOME:  U = "next week's return > 0"
  4. Measure empirically:
        prior        P(U)
        likelihoods  P(S | U), P(S | not U)
  5. Apply Bayes to get  P(U | S) — the posterior we care about.

What you'll usually see:
  • Prior  P(U) ≈ 0.52   (slight upward drift, almost a coin flip)
  • Posterior P(U | S) ≈ 0.50–0.55   on most tickers
  • i.e. the signal moves the needle by ~1–3 percentage points
  • on synthetic data (where there IS no signal by construction) the
    posterior is statistically indistinguishable from the prior

That's the lesson: Bayes' rule is honest. If the world doesn't contain a
signal, the math won't invent one. The framework is great for forecasting;
the inputs are the hard part.

Run: python exercises/06_bayesian_stock_signal.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------
# Configuration — edit these and rerun
# ---------------------------------------------------------------
TICKER          = 'AAPL'          # e.g. "SPY", "GIB", "AAPL" — needs `pip install yfinance`
YEARS           = 15            # how much history to use
DOWN_THRESHOLD  = -0.03         # signal fires when weekly return < this
SEED            = 7             # for reproducible synthetic data


def synthetic_weekly_returns(n_weeks: int, seed: int) -> np.ndarray:
    """A random walk in log-returns. By construction: no signal."""
    rng = np.random.default_rng(seed)
    # mu and sigma roughly match a broad equity index in weekly log-returns
    return rng.normal(loc=0.001, scale=0.025, size=n_weeks)


def fetch_real_returns(ticker: str, years: int) -> np.ndarray | None:
    """Return weekly log-returns for `ticker`, or None if yfinance unavailable."""
    try:
        import yfinance as yf
    except ImportError:
        print(f"(yfinance not installed — falling back to synthetic data)")
        return None

    df = yf.download(ticker, period=f"{years}y", interval="1wk",
                     progress=False, auto_adjust=True)
    if df is None or df.empty:
        print(f"(no data returned for {ticker} — falling back to synthetic)")
        return None

    close = df["Close"].to_numpy().squeeze()
    return np.diff(np.log(close))


def bayes_update(prior: float, p_signal_given_up: float,
                 p_signal_given_down: float) -> float:
    """P(up | signal) by Bayes' rule."""
    p_signal = p_signal_given_up * prior + p_signal_given_down * (1 - prior)
    return p_signal_given_up * prior / p_signal


def main() -> None:
    n_weeks = YEARS * 52

    # 1. Get returns
    returns = None
    if TICKER is not None:
        returns = fetch_real_returns(TICKER, YEARS)
    if returns is None:
        returns = synthetic_weekly_returns(n_weeks, SEED)
        source = "synthetic random walk"
    else:
        source = f"{TICKER} (yfinance)"

    # Pair each week with the NEXT week's outcome (drop last row, no future).
    this_week = returns[:-1]
    next_week = returns[1:]

    # 2. Signal and outcome
    signal  = this_week < DOWN_THRESHOLD     # "big down week"
    up_next = next_week > 0                  # "next week closes higher"

    n = len(this_week)

    # 3. Empirical probabilities
    prior = up_next.mean()                                            # P(U)
    p_signal_given_up   = signal[up_next].mean()                      # P(S | U)
    p_signal_given_down = signal[~up_next].mean()                     # P(S | ¬U)

    # 4. Apply Bayes
    posterior = bayes_update(prior, p_signal_given_up, p_signal_given_down)

    # 5. Sanity-check directly: P(U | S) just by counting
    direct = up_next[signal].mean() if signal.any() else float("nan")

    n_signal = int(signal.sum())

    print(f"\nsource: {source}   ({n} weekly transitions)")
    print(f"signal: this week's return < {DOWN_THRESHOLD:+.0%}   "
          f"(fired on {n_signal} weeks, {n_signal/n:.1%})")
    print()
    print(f"  prior  P(up next week)            = {prior:.4f}")
    print(f"  P(signal | up next)               = {p_signal_given_up:.4f}")
    print(f"  P(signal | down next)             = {p_signal_given_down:.4f}")
    print(f"  posterior P(up next | signal)     = {posterior:.4f}   <- Bayes")
    print(f"  direct count P(up next | signal)  = {direct:.4f}   <- sanity check")
    print()

    delta_pp = (posterior - prior) * 100
    if abs(delta_pp) < 1.0:
        verdict = "essentially zero -- the signal carries no information"
    elif abs(delta_pp) < 3.0:
        verdict = "marginal -- within typical sampling noise"
    else:
        verdict = "noticeable -- but check sample size before betting on it"
    print(f"  signal moves the prior by {delta_pp:+.1f} percentage points -> {verdict}")

    # ── Final verdict: NEGATIVE / NEUTRAL / POSITIVE ─────────────────────────
    # Use as ONE confidence input alongside other techniques (technicals,
    # fundamentals, macro). Never a standalone trade decision.
    if abs(delta_pp) < 1.0:
        label = "NEUTRAL"
    elif delta_pp > 0:
        label = "POSITIVE"
    else:
        label = "NEGATIVE"
    caveat = "" if n_signal >= 30 else "  (low sample size -- treat with skepticism)"
    print(f"\n  >> SIGNAL: {label}{caveat}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].hist(this_week, bins=50, color="steelblue", alpha=0.8, edgecolor="white")
    axes[0].axvline(DOWN_THRESHOLD, color="red", linestyle="--",
                    label=f"signal threshold = {DOWN_THRESHOLD:+.0%}")
    axes[0].set_xlabel("weekly log-return")
    axes[0].set_ylabel("count")
    axes[0].set_title(f"Distribution of weekly returns ({source})")
    axes[0].legend()

    bars = axes[1].bar(["prior\nP(up)", "posterior\nP(up | signal)"],
                       [prior, posterior],
                       color=["gray", "C0"], edgecolor="black")
    axes[1].axhline(0.5, color="black", linestyle=":", alpha=0.4, label="coin flip")
    for bar, val in zip(bars, [prior, posterior]):
        axes[1].text(bar.get_x() + bar.get_width()/2, val + 0.01,
                     f"{val:.3f}", ha="center", fontsize=11)
    axes[1].set_ylim(0, max(prior, posterior) * 1.25)
    axes[1].set_ylabel("probability")
    axes[1].set_title("Does the signal change our belief?")
    axes[1].legend()

    fig.tight_layout()
    out = Path(__file__).parent.parent / "figures"
    out.mkdir(exist_ok=True)
    path = out / "exercise_06_bayes_stock.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"figure saved to: {path}")


if __name__ == "__main__":
    main()


# ── Lessons ──────────────────────────────────────────────────────────────────
# 1. Bayes' rule needs three numbers: a prior and two likelihoods. For
#    stocks, the prior is roughly 0.52 (markets drift up). The likelihoods
#    have to be MEASURED — guessing them is just opinion-laundering.
#
# 2. On synthetic random-walk data the posterior should ≈ prior, because
#    by construction there is no signal. If you see a big shift on
#    synthetic data, you've found a bug, not alpha.
#
# 3. On real liquid tickers, simple signals (down-week, RSI<30, MA cross,
#    etc.) typically move the prior by < 3 percentage points. After
#    transaction costs and the variance of the estimate itself, that's
#    not enough to trade on.
#
# 4. This is why "ask an LLM for P(GIB up next week)" is a fool's errand
#    UNLESS you give it real, measured conditional probabilities — at
#    which point you've already done the hard work yourself.
#
# 5. The honest use of Bayesian thinking in markets: keep priors that
#    reflect base rates (most strategies have negative edge after costs),
#    require strong evidence to update, and shrink your posteriors
#    toward the prior when sample sizes are small. That's what
#    quants call "Bayesian shrinkage" and it's the difference between
#    overfitting noise and finding real (small) edges.
# ─────────────────────────────────────────────────────────────────────────────
