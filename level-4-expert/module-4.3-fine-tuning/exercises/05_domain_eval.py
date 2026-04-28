"""
Exercise 5 — Domain-specific eval: pre-fine-tune vs post-fine-tune

Build a 15-question custom eval set covering YOUR domain. Compare two
"versions" of a model on it, scored 1-5 by Claude as judge.

Here the two "versions" are HARDCODED ANSWERS to simulate before/after a
fine-tune (so the exercise runs in seconds and demonstrates the eval loop
without needing 30 minutes of training). The pattern is the same when
comparing real checkpoints.

The script asserts the "fine-tuned" version's average score > base.

Run: python exercises/05_domain_eval.py    # requires ANTHROPIC_API_KEY
"""

import json
import sys
from pathlib import Path

import anthropic


sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent /
                       "module-4.1-llms-prompt-engineering"))
from _common import DEFAULT_MODEL, require_api_key


require_api_key()
client = anthropic.Anthropic()


# ───────────────────────────────────────────────────────────
# 15-question domain eval — about a fictional product
# ───────────────────────────────────────────────────────────
EVAL = [
    {
        "q": "What's the refund policy?",
        "rubric": "States: refunds within 30 days, pro-rata; no refunds after 30 days.",
        "base":   "Refunds happen sometimes.",
        "tuned":  "Refunds are pro-rata within 30 days. No refunds after 30 days.",
    },
    {
        "q": "How long is the free trial?",
        "rubric": "Says 14 days, no credit card required, full Team-plan features.",
        "base":   "There's a free trial period.",
        "tuned":  "The free trial is 14 days, full Team-plan features, no credit card required.",
    },
    {
        "q": "Which plans include SSO?",
        "rubric": "Names Team and Enterprise plans only.",
        "base":   "All plans support SSO.",
        "tuned":  "SAML SSO is on Team and Enterprise plans only.",
    },
    {
        "q": "What rate limit applies to Starter?",
        "rubric": "60 requests per minute per IP.",
        "base":   "There's a rate limit.",
        "tuned":  "Starter is limited to 60 requests per minute per IP.",
    },
    {
        "q": "Are backups available on Starter?",
        "rubric": "Backups run nightly with 30-day retention. PITR is Enterprise-only.",
        "base":   "Backups happen sometimes.",
        "tuned":  "Backups run nightly with 30-day retention. Point-in-time recovery is Enterprise only.",
    },
    {
        "q": "Do you support Okta SSO?",
        "rubric": "Yes (Team/Enterprise). Mentions Okta as a supported provider.",
        "base":   "Yes we support some SSO providers.",
        "tuned":  "Yes — Okta is supported on Team and Enterprise plans.",
    },
    {
        "q": "Is HIPAA-eligible?",
        "rubric": "HIPAA-eligible on Enterprise; not on lower plans.",
        "base":   "We are very secure.",
        "tuned":  "HIPAA eligibility is available on the Enterprise plan only.",
    },
    {
        "q": "What auth providers do you support?",
        "rubric": "Lists Okta, Azure AD, Google Workspace, generic SAML 2.0.",
        "base":   "Several providers.",
        "tuned":  "Okta, Azure AD, Google Workspace, and generic SAML 2.0.",
    },
    {
        "q": "What support hours are there?",
        "rubric": "24/7 chat for all; phone Mon-Fri 9-6 PT, Enterprise only.",
        "base":   "We have support.",
        "tuned":  "24/7 chat for everyone. Phone support, Mon-Fri 9-6 PT, on Enterprise only.",
    },
    {
        "q": "Can I deploy in EU?",
        "rubric": "Yes — eu-west region available.",
        "base":   "Probably.",
        "tuned":  "Yes, eu-west is one of the four available regions.",
    },
    {
        "q": "How long are backups retained?",
        "rubric": "30 days.",
        "base":   "A while.",
        "tuned":  "Backups are retained for 30 days.",
    },
    {
        "q": "Is 2FA free?",
        "rubric": "Yes, on all plans.",
        "base":   "Some accounts have it.",
        "tuned":  "Yes, two-factor authentication is free on all Acme plans.",
    },
    {
        "q": "Which 2FA methods?",
        "rubric": "TOTP and WebAuthn.",
        "base":   "Several methods.",
        "tuned":  "TOTP and WebAuthn are both supported.",
    },
    {
        "q": "Are you SOC 2?",
        "rubric": "SOC 2 Type II certified.",
        "base":   "We have certifications.",
        "tuned":  "Yes, Acme is SOC 2 Type II certified.",
    },
    {
        "q": "Is GDPR compliant?",
        "rubric": "Yes, GDPR-ready.",
        "base":   "We follow regulations.",
        "tuned":  "Yes, Acme is GDPR-compliant.",
    },
]


# ───────────────────────────────────────────────────────────
# Judge — score 1-5 against the rubric
# ───────────────────────────────────────────────────────────
JUDGE = """\
Score this answer 1-5 against the RUBRIC. Output ONLY JSON:
  {"score": 1|2|3|4|5}

5 = nails every required detail.
3 = correct but missing key details.
1 = wrong, vague, or off-topic.
"""


def score(question: str, ans: str, rubric: str) -> int:
    resp = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=40,
        temperature=0,
        system=JUDGE,
        messages=[{"role": "user", "content":
                   f"QUESTION: {question}\nRUBRIC: {rubric}\nANSWER: {ans}"}],
    )
    raw = resp.content[0].text.strip().replace("```json", "").replace("```", "").strip()
    return int(json.loads(raw)["score"])


# ───────────────────────────────────────────────────────────
# Run
# ───────────────────────────────────────────────────────────
print(f"running {len(EVAL)}-question domain eval, judged by {DEFAULT_MODEL} ...\n")
print(f"{'#':>2}  {'base':>5}  {'tuned':>5}  Q")
base_scores, tuned_scores = [], []
for i, case in enumerate(EVAL, start=1):
    b = score(case["q"], case["base"],  case["rubric"])
    t = score(case["q"], case["tuned"], case["rubric"])
    base_scores.append(b)
    tuned_scores.append(t)
    print(f"{i:>2}  {b:>5}  {t:>5}  {case['q']}")

base_avg  = sum(base_scores)  / len(base_scores)
tuned_avg = sum(tuned_scores) / len(tuned_scores)
print(f"\nbase avg:  {base_avg:.2f}/5")
print(f"tuned avg: {tuned_avg:.2f}/5")
print(f"lift:      {tuned_avg - base_avg:+.2f}")

assert tuned_avg > base_avg + 1.0, \
    f"expected fine-tune to lift average score by ≥ 1.0; got {tuned_avg - base_avg:.2f}"
print("\nfine-tuned model wins on the domain eval ✅")


# ── How to use this in real life ────────────────────────────────────────────
# 1. Build the EVAL list ONCE, BEFORE fine-tuning. ~50-200 questions ideally.
# 2. Compute baseline scores with the base model.
# 3. After EACH training run / hyperparam change, re-score and compare.
# 4. Plot the score over training runs — that's your "is this getting better?".
# 5. Track per-question regressions: which questions did the new model GET WORSE on?
#    Those reveal training data gaps.
# ─────────────────────────────────────────────────────────────────────────────
