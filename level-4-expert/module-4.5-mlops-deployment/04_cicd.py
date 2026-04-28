"""
04 — CI/CD for ML

A good ML CI pipeline runs on EVERY PR:

  1. LINT — ruff, black, mypy (Module 1.2).
  2. UNIT TESTS — pytest. Cover preprocessing, postprocessing, business
                  logic. Mock the model itself.
  3. EVAL — run a small held-out eval set against the new model artifact
            (or the new prompts). Fail the build if metrics regress.
  4. BUILD — Docker image; push to a registry on the main branch.
  5. DEPLOY — promote to staging automatically; prod requires a human
              approval (or a passing canary).

This file shows the canonical GitHub Actions workflow + the patterns it
encodes.

Run: python 04_cicd.py
"""

# ───────────────────────────────────────────────────────────
# Canonical GitHub Actions workflow
# ───────────────────────────────────────────────────────────
WORKFLOW = """\
# .github/workflows/ci.yml
name: CI

on:
  pull_request:
  push:
    branches: [main]

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11", cache: "pip" }
      - run: pip install -r requirements.txt -r requirements-dev.txt
      - run: ruff check .
      - run: ruff format --check .
      - run: mypy .
      - run: pytest -q --cov=src --cov-fail-under=80

  ml-eval:
    needs: lint-and-test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11", cache: "pip" }
      - run: pip install -r requirements.txt
      - name: Run held-out eval
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: python scripts/run_eval.py --threshold 0.85
      - uses: actions/upload-artifact@v4
        with:
          name: eval-report
          path: eval-report.json

  build-and-push:
    needs: ml-eval
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    steps:
      - uses: actions/checkout@v4
      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: |
            ghcr.io/${{ github.repository }}:latest
            ghcr.io/${{ github.repository }}:${{ github.sha }}

  deploy-staging:
    needs: build-and-push
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    environment: staging
    steps:
      - run: |
          # call your deployer — kubectl apply, fly deploy, etc.
          echo "Deploying ${{ github.sha }} to staging"
"""
print("=== .github/workflows/ci.yml ===")
print(WORKFLOW)


# ───────────────────────────────────────────────────────────
# What each job actually checks
# ───────────────────────────────────────────────────────────
JOB_NOTES = """\
LINT-AND-TEST
  - Should run in <2 min. Fail fast for bad code.
  - cov-fail-under=80 enforces test coverage.
  - Run on EVERY commit, not just main.

ML-EVAL
  - Use a SMALL but representative eval set (50-200 examples).
  - Compare against a baseline; fail if accuracy / faithful% drops by Δ.
  - Upload the report so reviewers can inspect failures.
  - Keep API costs manageable (Haiku, small subset).

BUILD-AND-PUSH (main only)
  - Use docker/build-push-action with cache-from for faster builds.
  - Tag :latest AND :<sha> — :sha is what staging deploys.

DEPLOY-STAGING
  - Automatic on green main. Always push to staging first.
  - Smoke-test the new build (curl /health, run a sample request).

DEPLOY-PROD
  - Manual approval (`environment: production` + required reviewers).
  - Or fully automatic AFTER N hours of green staging metrics.
  - Use a canary or shadow deploy; promote on success.
"""
print(JOB_NOTES)


# ───────────────────────────────────────────────────────────
# Promotion strategies
# ───────────────────────────────────────────────────────────
PROMOTION = """\
PROMOTION STRATEGIES (production rollouts)

CANARY
  Send a small % (1-10%) of traffic to the new version. Compare error and
  latency rates. Promote to 100% if green; rollback if not.
  Tools: Argo Rollouts, Flagger, AWS CodeDeploy, your own LB rules.

SHADOW
  New version receives a COPY of every request. Predictions are logged but
  not returned to the user. Use to validate quality / latency safely.
  Common in ML, where "wrong" predictions don't visibly break the UI.

BLUE-GREEN
  Two identical environments (blue + green). Route all traffic to one;
  test the other with prod data; flip the load balancer. Easy rollback.

ROLLING
  Replace pods one at a time. Default for k8s deployments.

CHAMPION-CHALLENGER
  Two model versions answer in parallel; an internal scorer picks one
  per request OR you A/B test based on user.
  Best when "quality" is hard to measure offline.
"""
print(PROMOTION)


# ───────────────────────────────────────────────────────────
# Model registry vs git
# ───────────────────────────────────────────────────────────
REGISTRY = """\
MODEL ARTIFACT WORKFLOW

GIT  — your CODE: training scripts, eval harness, prompts.
       NEVER commit big binaries here.

MODEL REGISTRY — the trained MODEL, with metadata:
       - MLflow Registry (Module 2.4)
       - Hugging Face Hub
       - SageMaker Model Registry, Vertex AI Model Registry
       - Weights & Biases Artifacts

CI flow:
  1. Code commit on main → CI runs full eval.
  2. If eval passes a quality bar, REGISTER the model with new version.
  3. Tag it with the git SHA + dataset hash + metrics.
  4. Deploy job pulls FROM the registry (not from CI build artifacts).
  5. Production loads `models:/my-model@champion` (Module 2.4).
"""
print(REGISTRY)


# ───────────────────────────────────────────────────────────
# Things that surprise people the first time
# ───────────────────────────────────────────────────────────
GOTCHAS = """\
- API rate limits in CI: provider quotas hit hard if you eval too often.
  Cache evaluator outputs by content-hash of the model + question.

- LLM nondeterminism: the same prompt with T>0 gives different outputs each
  run. Set T=0 in eval, or use a tolerance threshold.

- Floating-point drift: training in CI on different hardware can produce
  slightly different weights. Use seeds + accept small diffs.

- Secrets: NEVER print env vars. NEVER commit .env. Rotate any key you
  see in a public log.

- Build time: a 10-minute CI run kills iteration speed. Cache pip, cache
  Docker layers, parallelize independent jobs.

- Eval flake: an LLM-as-judge can rate the same answer differently across
  runs. Use majority-of-3 or a deterministic grader where possible.
"""
print(GOTCHAS)
