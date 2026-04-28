"""
Exercise 3 — Inspect a real CI/CD workflow

Open `03_ci_workflow.yml` next to this file. It's the GitHub Actions
workflow you'd commit to `.github/workflows/ci.yml` in your repo.

This script just prints / explains its sections so you can read along.

Asserts:
  - The workflow has the expected jobs (lint, test, ml-eval, build-image,
    deploy-staging, deploy-prod).
  - Production deploy is gated behind an environment.

Run: python exercises/03_ci_workflow.py
"""

from pathlib import Path
import re


WORKFLOW_PATH = Path(__file__).parent / "03_ci_workflow.yml"


def main():
    wf = WORKFLOW_PATH.read_text(encoding="utf-8")

    # Find job names — lines matching r"^  (\w[\w-]+):" under the `jobs:` block.
    jobs = re.findall(r"^  (\w[\w-]+):", wf, flags=re.MULTILINE)
    print(f"workflow file: {WORKFLOW_PATH.name}\n")
    print("jobs found:")
    for j in jobs:
        if j in {"on", "name", "permissions"}:
            continue
        print(f"  • {j}")

    expected = {"lint", "test", "ml-eval", "build-image", "deploy-staging", "deploy-prod"}
    missing = expected - set(jobs)
    assert not missing, f"missing jobs: {missing}"

    # Production deploy must be in an `environment:` block (gives manual approval).
    prod_block = re.search(r"deploy-prod:.*?(?=\njobs:|\Z)", wf, re.DOTALL)
    assert prod_block, "deploy-prod not found"
    assert "environment:" in prod_block.group() and "production" in prod_block.group(), (
        "deploy-prod should reference an `environment: production` for approval gating"
    )

    print("\nworkflow has the expected jobs and prod-approval gate ✅")

    print("\n=== quick guide ===")
    print(GUIDE)


GUIDE = """\
HOW THIS WORKFLOW WORKS

1. Trigger: every PR + every push to main.
2. lint / test / ml-eval run in PARALLEL paths but block each other via `needs`.
3. Eval uploads `eval-report.json` as a GitHub artifact for reviewers.
4. On merge to main:
     a. Build a Docker image and push to ghcr.io with two tags
        (:latest and :<sha>).
     b. Auto-deploy to STAGING.
     c. Wait for human approval (configured under
        Settings → Environments → production).
     d. Deploy to PROD.

CONFIGURE BEFORE USE
  - Add ANTHROPIC_API_KEY and STAGING_TOKEN as repo secrets.
  - Create environments `staging` and `production` under repo Settings.
  - Add required reviewers to `production` so prod deploys require approval.
  - Replace the placeholder deploy commands with your real deployer.

WHAT TO ADD NEXT
  - Vulnerability scan (docker scout / trivy) on the built image.
  - Schema migration job before deploy-staging.
  - Smoke test against staging before deploy-prod.
  - Slack / Discord notification on deploy.
"""


if __name__ == "__main__":
    main()
