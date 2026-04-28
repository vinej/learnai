"""
03 — Privacy: PII detection and redaction

If your AI system handles user-generated text, you WILL receive emails,
phone numbers, SSNs, addresses, and unfiltered free-form data. Sending it
unredacted to third-party APIs can violate GDPR, HIPAA, CCPA, and your
own DPAs.

Strategies:
  REGEX-BASED        — fast, deterministic, low-recall on edge cases
                        (foreign formats, typos, contextual PII).
  NER-BASED          — Microsoft Presidio, spaCy + rule sets. Higher recall.
  LLM-BASED          — ask Claude to find and redact. Most flexible, slowest.
  HYBRID              — combine regex + NER + (optional) LLM verifier.

This file shows a regex-based redactor and a sketch of LLM-based redaction.

Run: python 03_privacy_pii.py
"""

import re


# ───────────────────────────────────────────────────────────
# Regex patterns — fast first pass
# ───────────────────────────────────────────────────────────
PATTERNS = {
    "EMAIL":      re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    "PHONE_US":   re.compile(r"\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b"),
    "SSN":        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    "IPV4":       re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "URL":        re.compile(r"\bhttps?://[^\s)]+"),
}


def redact_pii(text: str) -> tuple[str, list[dict]]:
    """Return (redacted_text, list_of_findings)."""
    findings = []
    out = text
    for label, pattern in PATTERNS.items():
        for m in pattern.finditer(text):
            findings.append({"label": label, "value": m.group(), "span": m.span()})
        out = pattern.sub(f"[{label}]", out)
    return out, findings


# ───────────────────────────────────────────────────────────
# Demo
# ───────────────────────────────────────────────────────────
INPUT = """\
Hi, please contact Alice at alice.smith@example.com or call +1 (415) 555-7890.
Her SSN is 123-45-6789 and the corporate IP is 10.0.0.1.
Her card was 4111 1111 1111 1111. The release notes are at https://example.com/rn.
"""

clean, findings = redact_pii(INPUT)

print("=== input ===")
print(INPUT)
print("=== redacted ===")
print(clean)
print("=== findings ===")
for f in findings:
    print(f"  {f['label']:<12}  {f['value']!r}")


# ───────────────────────────────────────────────────────────
# Production-grade redaction: Presidio
# ───────────────────────────────────────────────────────────
PRESIDIO_SNIPPET = """\
# pip install presidio-analyzer presidio-anonymizer
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

results = analyzer.analyze(text=INPUT, language="en")
clean = anonymizer.anonymize(
    text=INPUT,
    analyzer_results=results,
    operators={"DEFAULT": OperatorConfig("replace", {"new_value": "[REDACTED]"})}
).text
"""
print("\n=== Presidio (production option) ===")
print(PRESIDIO_SNIPPET)


# ───────────────────────────────────────────────────────────
# LLM-based redaction (sketch)
# ───────────────────────────────────────────────────────────
LLM_SNIPPET = """\
# Use only when regex/NER fails. Slower, costs money, can hallucinate.
# Best as a VERIFIER on top of pattern-based redaction, not a primary tool.

from anthropic import Anthropic
client = Anthropic()

REDACT_PROMPT = '''
You replace any personal information in the text with [REDACTED:<TYPE>]
markers. Preserve all other words. Output only the redacted text.

Types: NAME, EMAIL, PHONE, SSN, ADDRESS, CARD, IP, URL.
'''

resp = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=1000,
    temperature=0,
    system=REDACT_PROMPT,
    messages=[{"role": "user", "content": original_text}],
)
"""
print("\n=== LLM-based redaction (sketch) ===")
print(LLM_SNIPPET)


# ───────────────────────────────────────────────────────────
# Privacy patterns beyond redaction
# ───────────────────────────────────────────────────────────
PATTERNS_NOTES = """\
DIFFERENTIAL PRIVACY
  Add CALIBRATED NOISE to outputs / gradients so individual records can't be
  inferred. Mathematically rigorous; bites into model accuracy.
  Tools: PyTorch Opacus, Google's TensorFlow Privacy, OpenDP.

FEDERATED LEARNING
  Model trains on devices; only aggregate updates leave the client. No raw
  data centralized. Use case: phone keyboards, hospital data.
  Tools: TensorFlow Federated, Flower, PySyft.

DATA MINIMIZATION
  Don't store what you don't need. Hash IDs that don't need to be reversed.
  Set retention timers on logs.

ENCRYPTION AT REST AND IN TRANSIT
  TLS for every network hop. Disk encryption for stores. Key management
  via Vault / AWS KMS / GCP KMS.

ACCESS CONTROL
  Least privilege. Audit who reads what. Assume insider risk.

AGREEMENTS
  Have a Data Processing Agreement (DPA) with every API provider you send
  user data to. Anthropic, OpenAI, etc. all offer enterprise DPAs.
"""
print(PATTERNS_NOTES)
