"""
08 — Capstone: ship a production-grade LLM application

You've finished every Level 4 module. The capstone is a SYNTHESIS PROJECT:
you build, deploy, monitor, and evaluate an LLM application that uses
techniques from across the curriculum.

This file is a brief / rubric. Treat it as the spec for your own project.

Run: python 08_capstone_overview.py
"""


SPEC = """\
================================================================================
LEVEL 4 CAPSTONE PROJECT
================================================================================

GOAL
  Ship a real, useful LLM application your friends or colleagues can
  actually use. Combine RAG + agents + careful prompting + MLOps + safety.

IDEAS (pick ONE; or invent your own)
  1. "Chat with my docs" — RAG over a personal corpus (notes, papers, code).
  2. SQL agent that answers questions about a real dataset (Kaggle, your DB).
  3. Code-review assistant for PRs in a specific language / style.
  4. Customer-support bot for a fictional product, with a small KB.
  5. Research assistant that web-searches + synthesizes with citations.
  6. Task planner that breaks user goals into steps + tracks them in a DB.

================================================================================
HARD REQUIREMENTS
================================================================================

ARCHITECTURE
  [ ] An /api endpoint (FastAPI) — Module 4.5.01.
  [ ] At least one RAG component (FAISS / Chroma / pgvector) — Module 4.2.
  [ ] At least one TOOL the model can call — Module 4.4.
  [ ] Streaming responses for any user-facing text — Module 4.5.02.

PROMPTING
  [ ] System prompt that explicitly handles "I don't know" — Module 4.1.
  [ ] Prompt caching enabled for the system prompt — Module 4.1.06.
  [ ] Citations rendered in the UI / response.

EVALUATION
  [ ] Golden eval set (≥ 30 questions) checked into the repo — Module 4.2.07.
  [ ] CI runs the eval on every PR; blocks merge below quality threshold —
      Module 4.5.04.
  [ ] LLM-as-judge for at least one quality axis (faithful / relevant).

OBSERVABILITY
  [ ] Trace EVERY model call (input hash, tokens, latency, cost) — Module 4.5.06.
  [ ] /metrics endpoint with Prometheus counters & latency histograms.
  [ ] Per-user / per-feature cost dashboard (or a daily report).

SAFETY
  [ ] Untrusted-input defense: tool output / retrieved docs labeled as such —
      Module 4.4.06.
  [ ] PII redaction at the input boundary — Module 4.6.03.
  [ ] Hard budgets: max_iterations, max_tokens, request timeout — Module 4.4.05.
  [ ] No secrets in the system prompt.

DEPLOYMENT
  [ ] Dockerfile + docker-compose.yml for local dev — Module 4.5.03.
  [ ] CI/CD workflow (GitHub Actions) — Module 4.5.04.
  [ ] Deployed to a real URL (Fly.io / Cloud Run / Modal / Railway / your VPS).

================================================================================
STRETCH REQUIREMENTS
================================================================================

  [ ] Model cascade: Haiku-first, Sonnet on escalate — Module 4.6.05.
  [ ] Reranker on top of dense retrieval — Module 4.2.05.
  [ ] Fine-tune a small adapter for one specific behavior — Module 4.3.
  [ ] Fairness check on retrieval (equal recall across topic clusters).
  [ ] Red-team test corpus + automated injection tests on every release.
  [ ] Multi-region read replicas / CDN for static assets.

================================================================================
DELIVERABLES
================================================================================

  1. PUBLIC GIT REPO with everything: code, eval set, README.
  2. README that includes:
     - the problem statement
     - architecture diagram
     - setup instructions
     - eval scores (table over time)
     - cost-per-query estimate
     - known limitations
  3. 60-second demo video (Loom / screencap).
  4. A LIVE URL anyone can try.
  5. WRITEUP (≤ 1500 words) covering:
     - what worked, what didn't
     - one thing you'd do differently
     - what the next iteration would add

================================================================================
RUBRIC (for self-grading or peer review)
================================================================================

CORRECTNESS (40)
  - Answers questions accurately on the eval set.
  - Handles "out of scope" gracefully (no hallucination).
  - Citations point to the right sources.

ENGINEERING QUALITY (25)
  - Clean code, type hints, tests.
  - Readable architecture.
  - CI green; eval green.

OBSERVABILITY (15)
  - You can answer "how many requests yesterday?" "which were slow?"
    "which were the most expensive?" from your dashboards.

SAFETY (10)
  - Red-team prompts don't leak system prompt or call destructive tools.
  - PII redaction works on a sample.

PRODUCT (10)
  - It's PLEASANT to use. The streaming feels fast. Citations are useful.
  - Errors are friendly, not stack-traced.

================================================================================
TIMELINE (suggested)
================================================================================

Week 1
  - Pick an idea. Collect a small corpus.
  - Build the simplest RAG end-to-end (file 4.2.04).
  - Write 30 eval questions WITH expected answers.

Week 2
  - Add tools as needed (tool design from 4.4.01).
  - Wire CI/CD + Docker (Module 4.5).
  - Deploy v0.1 somewhere public.

Week 3
  - Add caching, observability, fallbacks.
  - Run the red-team / safety harness.
  - Iterate on prompts based on eval failures.

Week 4
  - Polish: streaming UI, citations rendering, friendly errors.
  - Write the README + writeup.
  - Record the demo video.

================================================================================
WHAT THIS LEAVES YOU WITH
================================================================================

A portfolio piece you can show to anyone — recruiters, collaborators,
clients. More importantly, you've built every part of the modern AI stack
once with your own hands. The next problem you face will feel familiar.

Good luck — and ship it.
"""

print(SPEC)
