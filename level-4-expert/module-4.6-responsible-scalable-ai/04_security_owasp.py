"""
04 — LLM security: OWASP Top 10

The OWASP Foundation maintains the Top 10 risks for LLM applications.
This file walks them with concrete mitigations and what's already covered
elsewhere in this curriculum.

  https://owasp.org/www-project-top-10-for-large-language-model-applications/

Run: python 04_security_owasp.py
"""

OWASP_TOP_10 = [
    {
        "id": "LLM01",
        "name": "Prompt Injection",
        "what":
            "Crafted inputs cause the LLM to ignore its instructions or follow "
            "an attacker's. Direct (in user input) or indirect (via tool output, "
            "documents, web pages).",
        "covered_in": "Module 4.4.06 + 4.4.04 lab",
        "mitigations": [
            "System prompts that explicitly say tool output is untrusted data.",
            "Structured tool calls (typed) — model can't 'argue' into actions.",
            "Privilege separation: sensitive tools live in separate agents.",
            "Output validation: filter the model's reply before returning to the user.",
            "Don't put secrets in the system prompt.",
        ],
    },
    {
        "id": "LLM02",
        "name": "Insecure Output Handling",
        "what":
            "The application TRUSTS the LLM's output and does dangerous things "
            "with it (executes code, builds SQL, renders HTML).",
        "mitigations": [
            "Treat LLM output as USER INPUT — sanitize before downstream use.",
            "Never `eval(model_output)` or execute commands directly.",
            "If output goes into HTML/SQL/shell, escape it the same way you would user input.",
            "Validate against a schema (Pydantic, JSON Schema).",
        ],
    },
    {
        "id": "LLM03",
        "name": "Training Data Poisoning",
        "what":
            "An attacker contaminates fine-tuning or pre-training data so the "
            "model misbehaves on specific triggers.",
        "mitigations": [
            "Vet data sources; prefer curated corpora.",
            "Inspect a random sample BEFORE training.",
            "Test for trigger backdoors (give known-malicious inputs at eval time).",
            "Pin models to a specific verified checkpoint.",
        ],
    },
    {
        "id": "LLM04",
        "name": "Model Denial of Service",
        "what":
            "Resource-exhausting inputs (huge contexts, deeply recursive tool "
            "calls) burn through your compute or quota.",
        "covered_in": "Module 4.4.05 + 4.5.07",
        "mitigations": [
            "Hard caps: max_tokens, max_iterations, max_wall_clock, max input length.",
            "Rate limiting per user / API key.",
            "Cost budgets per user / per minute / per day.",
            "Reject unreasonable inputs at the boundary (length, complexity).",
        ],
    },
    {
        "id": "LLM05",
        "name": "Supply Chain Vulnerabilities",
        "what":
            "Compromised model weights, poisoned datasets, malicious Python "
            "packages, leaky base images.",
        "mitigations": [
            "Pin EVERY dependency to a SHA. Use lockfiles.",
            "Scan images (trivy, Snyk, Docker Scout).",
            "Download model weights from trusted sources; verify hashes.",
            "Run dependency auditing in CI (pip-audit, OSV-Scanner).",
        ],
    },
    {
        "id": "LLM06",
        "name": "Sensitive Information Disclosure",
        "what":
            "The model leaks PII, secrets, or proprietary data — either from "
            "training data or from contexts you fed it.",
        "covered_in": "Module 4.6.03",
        "mitigations": [
            "Redact PII before it reaches the model.",
            "Don't put secrets in prompts.",
            "Fine-tune with PII-free data; ask vendors about training-data isolation.",
            "Output filter on the way back to the user.",
        ],
    },
    {
        "id": "LLM07",
        "name": "Insecure Plugin Design",
        "what":
            "Tools given to the model are too permissive (can read any file, "
            "make arbitrary HTTP calls, run shell commands).",
        "covered_in": "Module 4.4.01 + 4.4.05",
        "mitigations": [
            "Tightly typed inputs (enums, max lengths, path checks).",
            "Allow-listed actions only.",
            "Sandboxed code execution (E2B, Docker).",
            "Permission gates for write/destructive ops.",
        ],
    },
    {
        "id": "LLM08",
        "name": "Excessive Agency",
        "what":
            "Agent has too much authority; can take dangerous actions without "
            "human review (transfer money, delete files, send emails).",
        "mitigations": [
            "Principle of least privilege per tool.",
            "Human-in-the-loop checkpoints for irreversible / costly actions.",
            "Explicit confirmation tools (`request_user_approval`).",
            "Per-agent / per-feature rate caps.",
        ],
    },
    {
        "id": "LLM09",
        "name": "Overreliance",
        "what":
            "Users (or downstream systems) trust LLM output as authoritative "
            "without verification, leading to hallucination-driven errors.",
        "mitigations": [
            "Cite sources where possible (RAG with [doc-id] markers).",
            "Show CONFIDENCE / UNCERTAINTY signals to users.",
            "Pair LLM output with deterministic checks for critical use.",
            "Human review for high-impact decisions.",
        ],
    },
    {
        "id": "LLM10",
        "name": "Model Theft",
        "what":
            "Adversary extracts your fine-tuned weights via aggressive querying "
            "(model distillation attack) or stolen credentials.",
        "mitigations": [
            "Rate limit API access. Detect anomalous query patterns.",
            "Add output noise (where acceptable) to limit extraction fidelity.",
            "Strong API auth (rotating keys, OAuth, mTLS).",
            "Watermark outputs where feasible.",
        ],
    },
]


for risk in OWASP_TOP_10:
    print(f"\n=== {risk['id']}: {risk['name']} ===")
    print(f"  WHAT: {risk['what']}")
    if "covered_in" in risk:
        print(f"  COVERED IN: {risk['covered_in']}")
    print(f"  MITIGATIONS:")
    for m in risk["mitigations"]:
        print(f"    - {m}")


# ───────────────────────────────────────────────────────────
# A practical security checklist for shipping an LLM app
# ───────────────────────────────────────────────────────────
CHECKLIST = """\
PRE-PROD CHECKLIST

[ ] System prompt NEVER contains secrets, keys, or SSO tokens.
[ ] All tools have tight schemas (enums, lengths, paths).
[ ] All write tools are gated behind explicit per-call permission or
    human confirmation.
[ ] All code execution runs in a sandboxed container with --network=none
    by default.
[ ] PII redaction at the BOUNDARY before the model sees user input.
[ ] PII redaction on the way OUT before showing model output to others.
[ ] Hard budgets: max iterations, max tokens, max wall clock, max $.
[ ] Per-user / per-key rate limiting.
[ ] Logs capture every tool call, redacted.
[ ] Anomaly detection: spike in tool calls, retries, unusual patterns.
[ ] Red-team test corpus is run on every release (Module 4.4.04).
[ ] Pen test before every major version bump.
[ ] Incident response runbook documented and tested.
"""
print(CHECKLIST)
