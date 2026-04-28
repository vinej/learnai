# Setup — Module 4.6 (Responsible & Scalable AI)

The final module. Covers fairness, explainability, privacy, LLM security, cost/latency optimization, scaling patterns, and a Level 4 capstone.

## 1. Python ≥ 3.11

See [../../level-1-beginner/module-1.1-python-essentials/INSTALL.md](../../level-1-beginner/module-1.1-python-essentials/INSTALL.md).

## 2. Create / activate the venv

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1     # Windows PS
# or
source .venv/bin/activate       # macOS/Linux
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

| Package         | Purpose                                          |
|-----------------|--------------------------------------------------|
| `scikit-learn`  | Models for fairness / explainability demos       |
| `fairlearn`     | Fairness metrics and mitigations                 |
| `numpy`, `pandas`, `matplotlib` | Data + plots                       |
| `anthropic`     | LLM-side exercises (PII redaction, model cascade) |

### Optional

```bash
pip install shap                    # explainability concept file 02 stretch
pip install presidio-analyzer presidio-anonymizer    # production PII (Module 4.6.03)
```

## 4. Run the lessons

```bash
python 01_bias_fairness.py
python 02_explainability.py
python 03_privacy_pii.py
python 04_security_owasp.py
python 05_cost_optimization.py
python 06_scaling_patterns.py
python 07_regulatory_ethical.py
python 08_capstone_overview.py
```

## 5. Run the exercises

```bash
python exercises/01_fairness_audit.py
python exercises/02_explanations.py
python exercises/03_pii_redaction.py
python exercises/04_red_team_agent.py
python exercises/05_model_cascade.py
```

## ANTHROPIC_API_KEY

Required for: `03_privacy_pii.py`, `04_red_team_agent.py`, `05_cost_optimization.py`, `05_model_cascade.py`. The rest run offline.

## After this module

You've reached the end of the curriculum. The capstone in [08_capstone_overview.py](08_capstone_overview.py) is the synthesis project — building a production LLM app that combines RAG, agents, fine-tuning (or careful prompting), MLOps, and the safety practices from this module. It's deliberately open-ended.
