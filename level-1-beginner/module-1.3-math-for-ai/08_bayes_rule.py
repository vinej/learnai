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

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
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
# Exercise 1 — Sweep the prior
# ---------------------------------------------------------------
# How much does the answer depend on the prior P(D)? A LOT.
# Same test (95% sensitive, 90% specific) — vary disease prevalence
# from 1-in-10,000 to 1-in-2 and watch what happens to P(D | T+).
#
# This is the "base-rate fallacy" made visible.
# ---------------------------------------------------------------

priors = np.logspace(-4, np.log10(0.5), 200)        # 0.0001 → 0.5
posteriors = (P_TpD * priors) / (P_TpD * priors + P_TpND * (1 - priors))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(priors, posteriors, linewidth=2)
ax.axvline(0.01, color="red", linestyle="--", alpha=0.6, label="our example: P(D)=0.01")
ax.axhline(P_D_given_Tp, color="red", linestyle=":", alpha=0.6)
ax.set_xscale("log")
ax.set_xlabel("prior  P(D)   (disease prevalence, log scale)")
ax.set_ylabel("posterior  P(D | T+)")
ax.set_title("Same test, different prior → very different posterior")
ax.grid(True, alpha=0.3)
ax.legend()

out = Path(__file__).parent / "figures"
out.mkdir(exist_ok=True)
path = out / "08_bayes_rule.png"
fig.savefig(path, dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"\nfigure saved to: {path}")


# ---------------------------------------------------------------
# Exercise 2 — Sequential updating (the second test)
# ---------------------------------------------------------------
# Bayes' rule is composable: today's posterior is tomorrow's prior.
# After one positive, our belief jumped from 1% → 8.8%.
# What if we run a SECOND independent test and it's also positive?
# ---------------------------------------------------------------

# Posterior from test 1 becomes the prior for test 2.
prior_2     = P_D_given_Tp
P_Tp_2      = P_TpD * prior_2 + P_TpND * (1 - prior_2)
posterior_2 = P_TpD * prior_2 / P_Tp_2

print(f"\nafter 1 positive:  P(D) = {P_D_given_Tp:.4f}")
print(f"after 2 positives: P(D) = {posterior_2:.4f}")

# What about a NEGATIVE second test? Sensitivity flips to (1 - P_TpD),
# specificity flips to (1 - P_TpND).
P_Tn_2      = (1 - P_TpD) * prior_2 + (1 - P_TpND) * (1 - prior_2)
posterior_neg = (1 - P_TpD) * prior_2 / P_Tn_2
print(f"after 1 positive then 1 negative: P(D) = {posterior_neg:.4f}")
print("→ a single negative crushes the belief built up by a positive.")


# ---------------------------------------------------------------
# Exercise 3 — Tiny naive Bayes text classifier
# ---------------------------------------------------------------
# "Naive" because we assume words are conditionally independent given
# the class. That's clearly wrong (words come in phrases) — but the
# math becomes a simple product, and it works surprisingly well.
#
#   P(class | words) ∝ P(class) * Π P(word | class)
#
# We use ADD-ONE (Laplace) smoothing so unseen words don't zero out
# the whole product, and we work in LOG space so tiny probabilities
# don't underflow.
# ---------------------------------------------------------------

train = [
    ("pos", "i love this movie it was great"),
    ("pos", "fantastic acting and a beautiful story"),
    ("pos", "loved every minute really enjoyed it"),
    ("pos", "great direction and brilliant performance"),
    ("pos", "wonderful film highly recommend"),
    ("neg", "terrible movie i hated every minute"),
    ("neg", "boring story and awful acting"),
    ("neg", "what a waste of time really bad"),
    ("neg", "i disliked this it was awful"),
    ("neg", "poor direction and a bad script"),
]

# Count word occurrences per class and build the vocabulary.
counts = {"pos": Counter(), "neg": Counter()}
for label, text in train:
    counts[label].update(text.split())

vocab = set(counts["pos"]) | set(counts["neg"])
V = len(vocab)

# Class priors:  P(class) = (#docs of class) / (total #docs)
n_pos = sum(1 for label, _ in train if label == "pos")
n_neg = len(train) - n_pos
log_prior = {
    "pos": np.log(n_pos / len(train)),
    "neg": np.log(n_neg / len(train)),
}

# Likelihoods with add-1 smoothing:
#   P(word | class) = (count(word in class) + 1) / (total words in class + V)
totals = {c: sum(counts[c].values()) for c in counts}

def log_likelihood(word, cls):
    return np.log((counts[cls][word] + 1) / (totals[cls] + V))

def classify(sentence):
    scores = {}
    for cls in ("pos", "neg"):
        scores[cls] = log_prior[cls] + sum(
            log_likelihood(w, cls) for w in sentence.split()
        )
    return max(scores, key=scores.get), scores

print("\n--- naive Bayes predictions ---")
for sentence in [
    "i really loved this great movie",
    "what an awful boring waste",
    "the acting was brilliant",
    "a bad and boring story",
]:
    pred, scores = classify(sentence)
    print(f"[{pred}]  {sentence!r:45s}  "
          f"(log P pos={scores['pos']:.2f}, neg={scores['neg']:.2f})")


# ---------------------------------------------------------------
# Why this matters for ML
# ---------------------------------------------------------------
# • A "naive Bayes" text classifier multiplies likelihoods and applies
#   Bayes' rule to pick the most probable class.
# • In RAG, you're often computing P(document | query) and ranking — same
#   reasoning at heart.
# • Sequential updating (Exercise 2) is the entire idea behind Kalman
#   filters, online learning, and any system that updates as data arrives.
# • Whenever you see "prior", "likelihood", "posterior" — that's Bayes.
