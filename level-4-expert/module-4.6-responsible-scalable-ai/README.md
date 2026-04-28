# Module 4.6 — Responsible & Scalable AI

**Level:** 4 — Expert
**Estimated time:** 2 weeks

## Goal
Build AI systems that are safe, fair, secure, and economically viable at scale.

## Topics
### Bias & fairness
- Sources of bias: data, labeling, deployment
- Fairness metrics: demographic parity, equal opportunity, equalized odds
- Mitigation: reweighting, adversarial debiasing, post-processing
- Tools: **Fairlearn**, **AIF360**

### Explainability
- Global vs local explanations
- **SHAP**, **LIME**, integrated gradients
- For LLMs: chain-of-thought, attention rollout (and their limits)
- Communicating uncertainty to users

### Privacy
- PII detection & redaction (Presidio, regex, NER models)
- Differential privacy (intuition + Opacus for PyTorch)
- Federated learning (when it makes sense)
- Secrets management for keys & data

### Security (LLM-specific)
- **OWASP Top 10 for LLM Applications**
- Prompt injection (direct, indirect via retrieved content)
- Data exfiltration, jailbreaks
- Supply-chain risks (compromised models, datasets)
- Output handling: never `eval` model output; sandbox tool calls
- Rate limiting, abuse detection

### Cost & latency optimization
- Model cascades / routing (small model first, big on fallback)
- Caching: prompt cache, semantic cache, full response cache
- Batching strategies
- Distillation: train a small model on big-model outputs
- Right-sizing: smallest model that hits quality bar

### Scaling patterns
- Horizontal scaling, autoscaling
- Multi-region serving
- GPU sharing, fractional GPUs
- Spot vs on-demand strategy

### Regulatory & ethical
- EU AI Act, GDPR, HIPAA (overview, not legal advice)
- Model cards & datasheets for datasets
- Responsible release: red-teaming, staged rollouts

## Exercises
1. Audit a classifier from Module 2.x for fairness across a sensitive attribute; mitigate with Fairlearn.
2. Run SHAP on a tabular model and explain the top-3 drivers of one prediction.
3. Build a PII redaction pipeline before sending text to an LLM.
4. Red-team a previous agent: try 10 prompt-injection attacks and patch the holes.
5. Add a model cascade: route easy queries to a 4o-mini-class model, hard ones to Claude Opus; measure cost savings.

## Capstone (Level 4 — Final)
Ship a production-grade LLM application combining everything from Level 4:
- RAG over a real corpus (Module 4.2)
- Tool-using agent (Module 4.4)
- A fine-tuned or carefully prompted model (Modules 4.1 / 4.3)
- FastAPI deployment with Docker + CI/CD (Module 4.5)
- Tracing, eval suite in CI, cost dashboard (Modules 4.5 / this one)
- Prompt-injection defenses and PII redaction (this module)

Deliverable: a public repo, a deployed demo URL, and a write-up covering architecture, evals, costs, and known limitations.

## Resources
- OWASP Top 10 for LLM Apps: https://owasp.org/www-project-top-10-for-large-language-model-applications/
- Anthropic responsible scaling policy & safety research
- Book: *Trustworthy Machine Learning* — Kush Varshney (free)

## Checkpoint
You can take ownership of an AI system end-to-end: build it, deploy it, monitor it, defend it, and explain its behavior and limitations to non-engineers.
