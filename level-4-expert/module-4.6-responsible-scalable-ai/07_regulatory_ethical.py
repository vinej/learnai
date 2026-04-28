"""
07 — Regulatory and ethical

This isn't legal advice. The goal: know which laws and norms exist, what
they ask for, and how engineers operationalize them.

Run: python 07_regulatory_ethical.py
"""

REGULATIONS = """\
=== EU AI ACT (2024, phased rollout 2025-2027) ===

Risk-tiered for AI systems sold or used in the EU:

  PROHIBITED              social scoring, real-time public biometric ID
                          (with narrow exceptions), exploitative AI.
  HIGH RISK                AI in employment, education, credit, critical
                          infrastructure. Heavy obligations: risk
                          management, data governance, transparency,
                          human oversight, robustness, registration.
  LIMITED RISK             chatbots / deepfakes — must DISCLOSE to users.
  MINIMAL RISK             most consumer AI. Voluntary code of conduct.

GENERAL-PURPOSE AI (GPAI / foundation models): separate obligations
including documentation of training data, copyright compliance,
incident reporting; tighter rules for "systemic risk" models.

For builders:
  - Identify which tier your application falls into BEFORE you build.
  - Document the dataset sources and processing.
  - Keep a human-in-the-loop for high-risk decisions.
  - Log outputs for audit replay.


=== GDPR (EU) ===

For any system processing EU residents' personal data:
  • LAWFUL BASIS for processing.
  • DATA SUBJECT RIGHTS: access, rectification, deletion, portability.
  • DPO (Data Protection Officer) if you process at scale.
  • DATA BREACHES reported within 72 hours.
  • DATA PROCESSING AGREEMENT with each provider you send data to.
  • DATA RESIDENCY when contractually required.

For LLM apps specifically:
  - Right to erasure is HARD when a model has trained on your data. Most
    teams MITIGATE via per-user retrieval (RAG instead of fine-tuning),
    so deletion = remove the user's docs.
  - Use vendors with regional endpoints (Anthropic EU, AWS Bedrock Frankfurt).


=== HIPAA (US healthcare) ===

If you handle Protected Health Information (PHI):
  - BAA (Business Associate Agreement) with every vendor.
  - Audit logs of who accessed what PHI when.
  - Encryption at rest and in transit.
  - Limit access on a least-privilege basis.

Anthropic, OpenAI, and major clouds offer HIPAA-eligible deployments
under signed BAAs. Don't send PHI without one.


=== CCPA / CPRA (California) ===

Similar to GDPR for California residents:
  - Right to know what data is collected.
  - Right to delete.
  - Right to opt out of "selling" or "sharing" personal information.


=== INDUSTRY-SPECIFIC ===

  FINRA / SEC      financial advice: keep records, disclose AI use.
  FAR / DFARS       government contracts: data must stay in-country, etc.
  COPPA              children under 13 — strict consent rules.
  SOX                public companies: financial controls integrity.

A useful mental check: is this AI in a place where humans CARE if it's
wrong? If yes, regulation probably touches it.
"""

print(REGULATIONS)


MODEL_CARDS = """\
=== MODEL CARDS AND DATASHEETS ===

A MODEL CARD is a one-pager that documents:
  • What the model does (intended use).
  • What it does NOT do (out-of-scope use).
  • Training data: source, size, sampling biases.
  • Performance metrics on representative subgroups.
  • Known limitations and failure modes.
  • Licensing and contact for issues.

A DATASHEET FOR DATASETS does the same for training data:
  • Who collected it, why, when.
  • Composition (records, format, size).
  • Sampling / labeling process.
  • Privacy treatment of subjects.
  • Maintenance (will it be updated? how?).

These are NOT legal documents but they are the foundation regulators ask
for. Templates:
  - HuggingFace Model Card template.
  - Google Model Card Toolkit.
  - Datasheets for Datasets (Gebru et al. 2018).

Real-world examples to study:
  - Llama 3 model card.
  - Anthropic Claude Constitutional AI paper / system card.
  - OpenAI GPT-4 system card.
"""
print(MODEL_CARDS)


RESPONSIBLE_RELEASE = """\
=== RESPONSIBLE RELEASE PROCESS ===

Stages an LLM-app team should run for a major release:

  1. THREAT MODELING
     What can go wrong? Who could be harmed? Sensitive populations?
     Output: a list of risks ranked by severity × likelihood.

  2. INTERNAL RED-TEAM
     A separate team tries to break it: prompt injection, jailbreaks,
     data extraction, harmful content. (Module 4.4.06 is the technical
     starting point.)

  3. EVAL SUITE
     Domain golden set + safety golden set + fairness slices.
     Refuse to ship if regression > Δ on any subgroup.

  4. STAGED ROLLOUT
     Start with internal users → 1% of prod → 10% → 50% → 100%.
     Watch the dashboards at every stage.

  5. INCIDENT RESPONSE PLAN
     Pre-written runbook for: PII leak, harmful content, accuracy regression,
     downtime. Who's on call. How to roll back. How to communicate.

  6. POST-LAUNCH MONITORING
     Live dashboards for quality (file 4.5.08). Sampled human review.
     User feedback collection / triage.

Don't skip stages because the model 'seems' fine. Most production
LLM incidents come from edge cases that look benign in dev.
"""
print(RESPONSIBLE_RELEASE)


ETHICS_PRINCIPLES = """\
=== A SHORT LIST OF QUESTIONS TO ASK BEFORE BUILDING ===

  1. WHO BENEFITS? Who bears the risk?
  2. WHAT'S THE FAILURE MODE? What happens if the model is wrong?
     What if it's wrong systematically against a group?
  3. WHO'S ACCOUNTABLE? When something goes wrong, who fixes it?
  4. CAN A USER OPT OUT? Can they get a HUMAN instead?
  5. CAN YOU EXPLAIN OUTCOMES? Even at a high level?
  6. WHAT'S THE FEEDBACK LOOP? How do users tell you it's wrong?
  7. WHAT'S THE SUNSET PLAN? When you turn it off, what happens to
     people who depend on it?

If any of these has an unsatisfying answer — that's not a tooling problem.
Slow down. The right ETHICAL FAILURE MODE is "we did less than we could."
"""
print(ETHICS_PRINCIPLES)
