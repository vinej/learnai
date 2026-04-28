"""
Exercise 3 — Build a PII redaction pipeline

A two-stage redactor:

  STAGE 1: Regex pass for unambiguous patterns (email, phone, SSN, credit
           card, IP, URL).
  STAGE 2: LLM verifier — review the redacted output and catch missed PII
           (names, addresses, account numbers, free-form identifiers).

The script asserts the final output contains NO original PII strings from
the seed test cases.

Run: python exercises/03_pii_redaction.py    # requires ANTHROPIC_API_KEY
"""

import re
import sys
from pathlib import Path

import anthropic


sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent /
                       "module-4.1-llms-prompt-engineering"))
from _common import DEFAULT_MODEL, require_api_key


require_api_key()
client = anthropic.Anthropic()


# ───────────────────────────────────────────────────────────
# Stage 1: regex
# ───────────────────────────────────────────────────────────
PATTERNS = {
    "EMAIL":      re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    "PHONE":      re.compile(r"\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b"),
    "SSN":        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    "IPV4":       re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "URL":        re.compile(r"\bhttps?://[^\s)]+"),
}


def regex_redact(text: str) -> str:
    out = text
    for label, pattern in PATTERNS.items():
        out = pattern.sub(f"[{label}]", out)
    return out


# ───────────────────────────────────────────────────────────
# Stage 2: LLM verifier
# ───────────────────────────────────────────────────────────
LLM_PROMPT = """\
You receive a text that has already had OBVIOUS PII patterns redacted
(emails, phones, SSNs, etc.). Your job: find any REMAINING personal
information — names, postal addresses, account numbers, IDs, dates of
birth — and replace them with bracketed labels like [NAME], [ADDRESS],
[ACCOUNT], [DOB], [ID].

Rules:
- Output the rewritten text only, no preamble.
- DO NOT add new content or commentary.
- Preserve all non-PII words EXACTLY as written.
- If a name is already redacted (e.g. [NAME]), leave it.
"""


def llm_redact(text: str) -> str:
    resp = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=600,
        temperature=0,
        system=LLM_PROMPT,
        messages=[{"role": "user", "content": text}],
    )
    return resp.content[0].text.strip()


def two_stage_redact(text: str) -> str:
    return llm_redact(regex_redact(text))


# ───────────────────────────────────────────────────────────
# Test cases — each tagged with the leakable substrings
# ───────────────────────────────────────────────────────────
CASES = [
    {
        "text": (
            "Hi, please contact Alice Smith at alice.smith@example.com or "
            "+1 (415) 555-7890. Her SSN is 123-45-6789, account 4567890012."
        ),
        "must_not_appear": [
            "alice.smith@example.com", "415", "123-45-6789",
            "Alice Smith", "4567890012",
        ],
    },
    {
        "text": (
            "The patient John Doe (DOB 1972-03-14) lives at 42 Maple Ave, "
            "Springfield. His insurance ID is INS-9988-XX. Email "
            "john.doe@example.org."
        ),
        "must_not_appear": [
            "John Doe", "1972-03-14", "42 Maple Ave",
            "INS-9988-XX", "john.doe@example.org",
        ],
    },
    {
        "text": (
            "We received a payment from card 4111-1111-1111-1111 issued to "
            "Bob Marley at 123 Jericho Rd, Negril. Card was last used at "
            "https://example.com/receipt/4982."
        ),
        "must_not_appear": [
            "4111-1111-1111-1111", "Bob Marley", "123 Jericho Rd",
            "https://example.com/receipt/4982",
        ],
    },
]


# ───────────────────────────────────────────────────────────
# Run
# ───────────────────────────────────────────────────────────
print("running two-stage redaction on test cases ...\n")
all_clean = True

for i, case in enumerate(CASES, start=1):
    redacted = two_stage_redact(case["text"])
    print(f"=== case {i} ===")
    print("input:    ", case["text"])
    print("redacted: ", redacted)

    leaked = [s for s in case["must_not_appear"] if s in redacted]
    if leaked:
        print(f"  ⚠  LEAKS: {leaked}")
        all_clean = False
    else:
        print(f"  ✓ no PII leaks detected")
    print()


assert all_clean, "PII leaked through in at least one case"
print("two-stage redaction works ✅")


# ───────────────────────────────────────────────────────────
# Production hardening
# ───────────────────────────────────────────────────────────
HARDENING = """\
1. STAGE 1 must be FAST — every input passes through it. Keep it regex-only.
2. STAGE 2 is the LLM verifier. If you can't afford it on every request,
   sample (e.g. 10%) and use that to find systematic regex gaps.
3. NEVER trust an LLM to be 100% complete. Use defense-in-depth:
   regex + Presidio + LLM verifier + output filters before display.
4. LOG WHAT WAS REDACTED (categorically), not the content. "Redacted 3
   emails, 1 SSN" is fine; the strings themselves are PII.
5. For HEALTHCARE: use a HIPAA-eligible LLM endpoint, signed BAA.
6. For AUDIT: store the original text encrypted in a separate vault with
   strict access. Most teams need to be able to investigate later.
"""
print(HARDENING)
