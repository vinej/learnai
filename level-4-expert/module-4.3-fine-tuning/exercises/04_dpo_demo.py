"""
Exercise 4 — DPO toward concise answers

Take a base distilgpt2 and DPO-train it on a tiny preference dataset that
prefers concise answers over verbose ones. Then verify the policy's
log-probability for "chosen" responses goes UP relative to "rejected".

The script asserts that, on the training set, the chosen-vs-rejected log
probability ratio improves after training.

Run: python exercises/04_dpo_demo.py
"""

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


torch.manual_seed(0)
MODEL_NAME = "distilgpt2"


# ───────────────────────────────────────────────────────────
# Preference dataset: prefer concise over verbose
# ───────────────────────────────────────────────────────────
PAIRS = [
    {
        "prompt":   "Question: What is two plus two?\nAnswer:",
        "chosen":   " Four.",
        "rejected": " The sum of two and two, when added together using standard arithmetic operations and considering them as positive integers, equals exactly four.",
    },
    {
        "prompt":   "Question: What color is the sky?\nAnswer:",
        "chosen":   " Blue.",
        "rejected": " On a clear, cloudless day during the daytime hours, the sky generally appears to be a shade of blue, though it can vary based on time of day, weather conditions, and pollution levels.",
    },
    {
        "prompt":   "Question: How many days are in a week?\nAnswer:",
        "chosen":   " Seven.",
        "rejected": " A typical week, as defined by most cultures and calendars throughout the world, consists of seven days, beginning either on Sunday or Monday depending on tradition.",
    },
    {
        "prompt":   "Question: What is the chemical symbol for water?\nAnswer:",
        "chosen":   " H2O.",
        "rejected": " The chemical symbol for water, which is composed of two hydrogen atoms covalently bonded to a single oxygen atom, is conventionally written as H2O in scientific notation.",
    },
] * 6


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

policy = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
ref    = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
for p in ref.parameters():
    p.requires_grad = False


# ───────────────────────────────────────────────────────────
# Helper: log-prob ratio chosen/rejected
# ───────────────────────────────────────────────────────────
def logprob_of_completion(model, prompt: str, completion: str) -> float:
    """Sum of log-probs of completion tokens given prompt."""
    full = prompt + completion
    full_ids = tokenizer(full, return_tensors="pt").input_ids
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
    n_completion = full_ids.shape[1] - prompt_ids.shape[1]

    with torch.no_grad():
        logits = model(full_ids).logits         # (1, T, V)

    logp = 0.0
    for i in range(n_completion):
        pos = prompt_ids.shape[1] + i - 1
        token = full_ids[0, pos + 1].item()
        logp += torch.log_softmax(logits[0, pos], dim=-1)[token].item()
    return logp


def chosen_minus_rejected(model) -> float:
    """Average log-prob preference for chosen over rejected, across data."""
    diffs = []
    for p in PAIRS[:8]:                          # sample first 8 unique
        c = logprob_of_completion(model, p["prompt"], p["chosen"])
        r = logprob_of_completion(model, p["prompt"], p["rejected"])
        diffs.append(c - r)
    return sum(diffs) / len(diffs)


# ───────────────────────────────────────────────────────────
# Measure BEFORE
# ───────────────────────────────────────────────────────────
before = chosen_minus_rejected(policy)
print(f"before DPO: avg log P(chosen) - log P(rejected) = {before:+.3f}")


# ───────────────────────────────────────────────────────────
# Train DPO (very short)
# ───────────────────────────────────────────────────────────
config = DPOConfig(
    output_dir="scratch/dpo-concise",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    learning_rate=5e-5,
    beta=0.1,
    max_length=256,
    max_prompt_length=64,
    save_strategy="no",
    logging_steps=4,
    report_to="none",
)
trainer = DPOTrainer(
    model=policy,
    ref_model=ref,
    args=config,
    train_dataset=Dataset.from_list(PAIRS),
    tokenizer=tokenizer,
)
print("\nrunning DPO ...")
trainer.train()
print("done")


# ───────────────────────────────────────────────────────────
# Measure AFTER
# ───────────────────────────────────────────────────────────
after = chosen_minus_rejected(policy)
print(f"\nafter DPO:  avg log P(chosen) - log P(rejected) = {after:+.3f}")
print(f"improvement: {after - before:+.3f}  (positive = DPO worked)")

assert after > before, (
    "DPO should INCREASE the log-prob ratio of chosen vs rejected. "
    "If it didn't, training didn't run long enough or beta is too low."
)
print("DPO toward concise answers works ✅")


# ── Stretch ─────────────────────────────────────────────────────────────────
# • Sample completions from `policy` BEFORE and AFTER. They should get shorter.
# • Try beta = 0.01 vs 0.5. Higher beta → stronger preference signal but
#   more risk of forgetting base capabilities (run a sanity check on
#   unrelated prompts).
# • Replace DPO with KTO: only requires a binary "chosen" or "rejected"
#   label per response. Cheaper to label.
# ─────────────────────────────────────────────────────────────────────────────
