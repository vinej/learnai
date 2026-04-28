"""
Exercise 5 — Bayes' rule on a medical test

A disease has prevalence `prior` in the population. A test has:
  • sensitivity = P(positive | sick)
  • specificity = P(negative | healthy)

Compute  P(sick | positive)  using Bayes' rule, and verify with a simulation.

Try editing the numbers below — what happens when:
  • the disease becomes 10× rarer?
  • the test becomes 99% specific instead of 90%?

Run: python exercises/05_medical_test_bayes.py
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class TestSetup:
    prior: float          # P(disease) in the population
    sensitivity: float    # P(test+ | disease)
    specificity: float    # P(test- | healthy)

    @property
    def false_positive_rate(self) -> float:
        return 1 - self.specificity

    def posterior_given_positive(self) -> float:
        """P(D | T+) by Bayes' rule."""
        p_pos_given_d = self.sensitivity
        p_pos_given_nd = self.false_positive_rate
        p_pos = p_pos_given_d * self.prior + p_pos_given_nd * (1 - self.prior)
        return p_pos_given_d * self.prior / p_pos

    def posterior_given_negative(self) -> float:
        """P(D | T-) — the probability you're STILL sick after a negative test."""
        p_neg_given_d = 1 - self.sensitivity
        p_neg_given_nd = self.specificity
        p_neg = p_neg_given_d * self.prior + p_neg_given_nd * (1 - self.prior)
        return p_neg_given_d * self.prior / p_neg


def simulate(setup: TestSetup, n: int = 1_000_000, seed: int = 0) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    sick = rng.random(n) < setup.prior
    test_positive = np.where(
        sick,
        rng.random(n) < setup.sensitivity,
        rng.random(n) < setup.false_positive_rate,
    )
    pos_count = int(test_positive.sum())
    neg_count = n - pos_count
    return {
        "P(D | T+)": (sick & test_positive).sum() / max(pos_count, 1),
        "P(D | T-)": (sick & ~test_positive).sum() / max(neg_count, 1),
    }


def report(label: str, setup: TestSetup):
    print(f"\n=== {label} ===")
    print(f"  prior {setup.prior:.4f}, "
          f"sensitivity {setup.sensitivity}, "
          f"specificity {setup.specificity}")

    analytic = {
        "P(D | T+)": setup.posterior_given_positive(),
        "P(D | T-)": setup.posterior_given_negative(),
    }
    empirical = simulate(setup)

    for key in analytic:
        print(f"  {key}: analytic = {analytic[key]:.4f}, "
              f"simulated ≈ {empirical[key]:.4f}")


if __name__ == "__main__":
    base = TestSetup(prior=0.01, sensitivity=0.95, specificity=0.90)
    report("Rare disease, decent test", base)

    rarer = TestSetup(prior=0.001, sensitivity=0.95, specificity=0.90)
    report("10× rarer", rarer)

    very_specific = TestSetup(prior=0.01, sensitivity=0.95, specificity=0.99)
    report("Same prior, much more specific test", very_specific)


# ── Lessons ──────────────────────────────────────────────────────────────────
# 1. With a low prior, the posterior P(D|T+) is dominated by FALSE positives.
# 2. Increasing SPECIFICITY (= reducing FPR) improves P(D|T+) more than
#    increasing sensitivity, when the disease is rare.
# 3. This is why screening tests are often followed by a more specific
#    confirmatory test — that's iterating Bayes' rule.
# ─────────────────────────────────────────────────────────────────────────────
