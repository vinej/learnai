"""
08 — Conditional probability and Bayes' rule

CONDITIONAL probability:
    P(A | B) = "probability of A given that B happened"
             = P(A and B) / P(B)

BAYES' RULE — flip a conditional:
    P(A | B) = P(B | A) * P(A) / P(B)

This is the foundation of probabilistic reasoning, naive Bayes classifiers,
Kalman filters, MCMC, and modern generative models.

We'll use the classic medical-test example, then verify it with a simulation.

Run: python 08_bayes_rule.py
"""

import numpy as np


# ---------------------------------------------------------------
# The setup (these numbers are made up but realistic)
# ---------------------------------------------------------------
# A disease affects 1% of the population.        P(D) = 0.01
# A test is 95% sensitive (TPR):                 P(T+ | D)  = 0.95
# A test is 90% specific (1 - FPR):              P(T+ | ¬D) = 0.10
#
# Q: someone tested POSITIVE. What's the probability they're sick?
#    i.e.   P(D | T+) = ?
# ---------------------------------------------------------------

P_D    = 0.01           # prior: disease prevalence
P_TpD  = 0.95           # sensitivity:    P(T+ given disease)
P_TpND = 0.10           # false-positive: P(T+ given no disease)

# By total probability:
# P(T+) = P(T+|D)P(D) + P(T+|¬D)P(¬D)
P_Tp = P_TpD * P_D + P_TpND * (1 - P_D)
print(f"P(T+) = {P_Tp:.4f}")

# Bayes' rule:
P_D_given_Tp = P_TpD * P_D / P_Tp
print(f"P(D | T+) = {P_D_given_Tp:.4f}")

print()
print(f"Even with a 95%-accurate test, a positive result means only "
      f"{P_D_given_Tp:.1%} chance of actually being sick.")
print("Why? Because the disease is rare → most positives are FALSE positives.")


# ---------------------------------------------------------------
# Sanity-check with a simulation
# ---------------------------------------------------------------
rng = np.random.default_rng(0)
N = 1_000_000

# 1. Generate sick / healthy people
sick = rng.random(N) < P_D                # boolean mask

# 2. Test them. Sensitivity for the sick, false-positive rate for the healthy.
test_positive = np.where(
    sick,
    rng.random(N) < P_TpD,
    rng.random(N) < P_TpND,
)

# Empirical  P(D | T+)  =  count(sick AND positive) / count(positive)
positive_count = test_positive.sum()
sick_and_positive = (sick & test_positive).sum()
emp = sick_and_positive / positive_count

print(f"\nsimulated   P(D | T+) ≈ {emp:.4f}   (analytic: {P_D_given_Tp:.4f})")


# ---------------------------------------------------------------
# Why this matters for ML
# ---------------------------------------------------------------
# • A "naive Bayes" text classifier multiplies likelihoods and applies
#   Bayes' rule to pick the most probable class.
# • In RAG, you're often computing P(document | query) and ranking — same
#   reasoning at heart.
# • Whenever you see "prior", "likelihood", "posterior" — that's Bayes.
