"""
08 — Monitoring quality (drift, hallucinations, regressions)

System metrics tell you the SERVICE is up. They don't tell you the MODEL
is correct. ML / LLM monitoring needs additional axes:

  DATA DRIFT
    The distribution of inputs has changed since training.
    Detect: PSI (Population Stability Index), KS-test on continuous,
    chi-square on categorical. Alert if PSI > 0.2 sustained.

  PREDICTION DRIFT
    The distribution of outputs has changed.
    A spike in "I don't know" responses, or a class imbalance shift.

  CONCEPT DRIFT
    The ground-truth relationship has shifted (e.g., post-COVID housing
    prices). Catches you off-guard if you only track input drift.

  QUALITY (LLM)
    A separate LLM grader scores a SAMPLE of responses for faithfulness,
    relevance, format. Track the grader score over time.

  COST / LATENCY
    Already in file 05. Treat them as quality signals — sudden token
    inflation often means the prompt has drifted or the model is wandering.

This file shows PSI computation and a sketch of online quality monitoring.

Run: python 08_monitoring_quality.py
"""

import numpy as np


# ───────────────────────────────────────────────────────────
# PSI — Population Stability Index
# ───────────────────────────────────────────────────────────
# PSI compares the distribution of a feature in two populations
# (training vs production). Bin both, compute Σ (p_p - p_t) * log(p_p / p_t).
def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Return PSI between reference (training) and current (live) data."""
    edges = np.linspace(reference.min(), reference.max(), bins + 1)
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    ref_hist, _ = np.histogram(reference, bins=edges)
    cur_hist, _ = np.histogram(current,   bins=edges)

    # Avoid divide-by-zero
    p_ref = (ref_hist + 1) / (ref_hist.sum() + bins)
    p_cur = (cur_hist + 1) / (cur_hist.sum() + bins)

    return float(np.sum((p_cur - p_ref) * np.log(p_cur / p_ref)))


rng = np.random.default_rng(0)
reference = rng.normal(loc=0, scale=1, size=10_000)

print("=== PSI examples ===")
print(f"  same distribution:    {psi(reference, rng.normal(0, 1, 10_000)):.3f}  (~0.00)")
print(f"  small shift:           {psi(reference, rng.normal(0.2, 1, 10_000)):.3f}  (~0.04)")
print(f"  big shift:             {psi(reference, rng.normal(1.0, 1, 10_000)):.3f}  (>0.25)")
print(f"  scale change:          {psi(reference, rng.normal(0, 2, 10_000)):.3f}")


THRESHOLDS = """\
PSI rules of thumb (per feature):
  PSI < 0.10        — no significant change.
  0.10-0.25         — small change. Watch.
  > 0.25            — major change. Investigate.

Compute PSI nightly (or hourly) for every important feature. Alert when
sustained over the threshold for several consecutive periods (don't page on
a single spike).
"""
print(THRESHOLDS)


# ───────────────────────────────────────────────────────────
# Online quality monitoring — LLM
# ───────────────────────────────────────────────────────────
SAMPLED_GRADING = """\
Run this on a SAMPLE (1-5%) of production traffic:

  1. Sample request + response from the trace store.
  2. Send (question, answer, retrieved docs) to a STRONG judge model
     (Sonnet / Opus / GPT-4o).
  3. Get scores for: faithful (bool), relevant (bool), format_ok (bool),
     unsafe (bool).
  4. Store in a metrics database.
  5. Graph it. Alert if faithful% drops > 5pp WoW or unsafe% > 0.5%.

Cost: ~$1-5 / 1000 sampled responses with Haiku-as-judge. Worth every cent.
"""
print(SAMPLED_GRADING)


# ───────────────────────────────────────────────────────────
# Detecting regressions across releases
# ───────────────────────────────────────────────────────────
REGRESSIONS = """\
WHEN YOU SHIP A NEW PROMPT / MODEL / RAG INDEX:

  - Run the GOLDEN EVAL SET (Module 4.2.07.exercise_04) before deploy.
  - Compare aggregate scores vs the production baseline. Block deploy
    if regression > Δ.
  - After deploy, compare LIVE quality scores week-over-week for ~14 days.
  - Tag each trace with `prompt_version` / `model_version` / `index_version`
    so you can A/B compare easily.

ALERT WHEN:
  - Eval set faithful% drops by ≥ 5pp.
  - Live quality grader score drops by ≥ 5pp.
  - "I don't know" rate spikes (RAG retrieval got worse?).
  - Refusal rate spikes (safety filter mis-triggered?).
  - P95 latency drifts up (model swap? caching broken?).
  - Token usage drifts up (prompt got bloated? tools loop?).
"""
print(REGRESSIONS)


# ───────────────────────────────────────────────────────────
# What "responsible deployment" actually looks like
# ───────────────────────────────────────────────────────────
RESPONSIBLE = """\
DEPLOYMENT
  - Canary or shadow.
  - Prepare a rollback plan BEFORE you deploy. Test it.
  - Owner-on-call for 24h after every deploy.

OBSERVATION
  - System metrics: latency, errors, saturation (file 05).
  - LLM metrics: tokens, cost, cache (file 06).
  - Quality metrics: judged sample, drift (this file).
  - Logs / traces: every request reproducible (file 06).

GOVERNANCE
  - Audit log: who changed what, when.
  - Approval gates for production model swaps.
  - Documented model cards and dataset cards.
  - Privacy review (Module 4.6).

Module 4.6 closes the loop on the safety / compliance side.
"""
print(RESPONSIBLE)
