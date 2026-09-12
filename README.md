# Hiver Support Agent (Take-Home Assignment)

An AI customer-support agent for **DropboxSupport**, built from the
[Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
Kaggle dataset. The agent: (1) classifies an incoming customer message
into an intent, (2) drafts a reply grounded in how this brand has
historically resolved similar issues, and (3) decides whether to
auto-handle or escalate to a human, with a stated reason.

## Status: end-to-end pipeline built and evaluated against a 189-example hand-labeled golden set

This README is written *as we go*, not after the fact -- it reflects
what's built right now, including the things that didn't work.

## Headline results

Measured on 189 hand-labeled golden examples (`data/golden_labels.csv`).
Reproduce with `scripts/12_evaluate.py` -- no API calls, everything is
computed from stored predictions.

**Intent classification**

| approach | accuracy |
|---|---|
| Trivial baseline (majority class) | 15.9% |
| Simple baseline (TF-IDF + LogisticRegression, 5-fold CV) | 40.7% |
| **LLM (Groq `openai/gpt-oss-20b`, zero-shot prompted)** | **62.4%** |

**Escalation decision** (model intent -> policy table -> decision):
78.3% end-to-end. Split by cost, because the two errors aren't
symmetric: 15 missed escalations (21% of what a human would escalate --
the expensive failure) vs. 26 over-escalations (cheap).

**Reply quality** (LLM-as-judge, `qwen/qwen3.8-27b` -- deliberately a
different model family from the drafter):

| system | overall (1-5) | deflection rate |
|---|---|---|
| **Grounded LLM agent** | **4.92** | **0%** |
| Simple baseline (1-NN retrieval) | 1.38 | 46% |
| Trivial baseline (canned reply) | 3.54 | 38% |

The deflection column is the one that matters -- see the policy section
below.

**What's misleading about these numbers** is documented in its own
section near the end, and printed by `12_evaluate.py` itself. Read it
before quoting any of the above.

## Approach

- **No labeled training data exists up front.** The dataset has real
  customer<->brand conversations but no intent labels, no quality
  labels, nothing. So: use a prompted (not fine-tuned) LLM for
  classification and drafting, and hand-build a small golden set to
  *evaluate* it.
- **Intent classification**: prompted LLM call against the fixed
  category list in `intents.json`, with structured JSON output.
- **Grounded reply drafting**: TF-IDF retrieval over past
  DropboxSupport threads, then prompt the LLM to draft a reply
  consistent with how the brand actually responded -- not the model's
  generic instincts.
- **Auto-handle vs. escalate**: intent-based default rules, not
  open-ended LLM discretion, with a stated reason on every decision.
  The table was then *calibrated against the golden set* (see below).
- **Baselines**: defined before the LLM agent was built, so the results
  table isn't retrofitted.
- **Evaluation**: hand-labeled golden set, automated metrics, plus an
  LLM-as-judge for reply quality with judge-vs-judge and (pending)
  judge-vs-human agreement evidence.

LLM provider: **Groq** (`openai/gpt-oss-20b`), key in `.env`
(gitignored). See "Why Groq instead of Gemini" in `PROGRESS.md`.

### Policy: what "good" means here (the DM-deflection problem)

Read-through showed many DropboxSupport replies are just "sorry about
that, please DM us" -- the real resolution never appears in the public
dataset. A naive retrieval-grounded agent learns to mimic this
deflection for *everything*, which is not a useful support agent.
Explicit policy, enforced in the drafting prompt rather than left for
the model to infer:

- **Informational intents** (`how_to_usage`, `feature_request`): the
  agent must give the actual answer or a doc pointer -- deflecting to
  DM is a failure, not a valid auto-handle.
- **Account-specific / PII intents** (`account_access`,
  `billing_subscription`, `data_loss_recovery`,
  `security_account_compromise`, `storage_quota_plan_limits`): asking
  for a DM is correct (identity can't be verified in a public reply),
  but the agent must say exactly what to send ("DM us your ticket
  number and the email on the account"), never a bare "please DM us."

**This is measured, not asserted.** `is_deflection` is scored as its
own boolean by the judge, separately from quality: the grounded agent
deflects on 0% of sampled replies, against 46% for 1-NN retrieval --
which is exactly the failure mode this policy exists to prevent.

### Intent taxonomy: v2 (13 intents), adopted

`intents.json` / `intents.md` / `taxonomy.md`.

v1 had 8 intents and was replaced before hand-labeling started. The
problem: v1 crammed topic, conversational position, and risk signals
into a single enum -- `followup_ticket_status` was a *turn type*
wearing an intent's clothes, so every mid-thread tweet lost either its
topic or its turn-type. v2 keeps intent as pure topic and moves the
rest into their own fields:

- **13 intents** (topic only)
- **`turn_type`** (5 values): where the message sits relative to a
  prior exchange
- **flags**: `wants_human`, `legal_sensitive`, `churn_threat`,
  `abusive_content`, `needs_human_triage`, plus `sentiment` and
  `language`

Deliberately changed *before* labeling, since revising a taxonomy
afterwards means re-labeling every example by hand.

### Escalation matrix (v2, calibrated against the golden set)

Implemented in `scripts/escalation.py`. Two stages: flag/turn_type
overrides fire first and short-circuit, then an intent default table.

Overrides (regardless of intent): `needs_human_triage`, confidence
below threshold, `disputing_prior_answer`, `legal_sensitive`,
`wants_human`, `churn_threat`, `abusive_content`.

Intent defaults -- escalate: `data_loss_recovery`,
`billing_subscription`, `security_account_compromise`,
`phishing_abuse_report`, `account_access`. Auto-handle:
`sync_app_bug`, `service_outage`, `sharing_permissions`,
`storage_quota_plan_limits`, `how_to_usage`, `feature_request`,
`complaint_dissatisfaction`. No-op: `no_action_needed`.

**`account_access` was originally auto-handle and the golden set proved
it wrong**: a human escalated 89% of those tweets while the auto-handle
default got 11% of them right. Flipping it improved held-out escalation
accuracy from 80.5% to 78.9% out-of-sample (86.7% on the tuning split).
Tuning was done on a 60% split and measured on the 40% never touched,
so the reported number isn't in-sample.

**Known limitation**: `complaint_dissatisfaction`,
`billing_subscription`, and `sync_app_bug` sit near 50% -- intent alone
does not determine escalation for them. Fitting per-intent exceptions
on ~113 rows would be overfitting, so this is left as a stated finding
rather than patched.

### Baselines

**Trivial**: majority-class intent, one fixed canned reply, escalate
100%. **Simple**: TF-IDF + LogisticRegression (cross-validated, never
fit-and-scored on the same rows), 1-NN retrieval returning raw
historical reply text, keyword-regex escalation.

Both exist so the results table shows what the LLM actually adds over
what's free.

## Golden evaluation set

189 hand-labeled examples from a 207-tweet candidate pool. Methodology
in `golden_set.md`.

- **Sampling**: random draw for real traffic representation, plus a
  rare-intent top-up so every intent clears a usable minimum.
- **Anti-anchoring**: the labeler answers *before* the classifier's
  guess is revealed. Showing the suggestion first would inflate the
  agreement number being reported.
- **Priority ordering** (`--priority`): reorders unlabeled candidates
  so thin intents come first, interleaved rather than blocked, so a
  partial run still reaches usable per-intent coverage.
- **Self-consistency audited** (`scripts/11_label_consistency.py`): no
  near-identical tweets carry conflicting intents; all five turn_type
  values are used; no evidence of defaulted fields.

Labeling stopped at 189 of 207 deliberately -- within the assignment's
150-250 range, with every intent at >=10 examples except
`billing_subscription` (8) and `security_account_compromise` (8), both
flagged as underpowered rather than quietly averaged in.

## What's misleading about the headline numbers

Printed by `12_evaluate.py`; summarized here because it's the section
that matters most.

1. **`turn_type`'s 86% accuracy is an artifact.** The model predicted
   `first_contact` for 186 of 189 rows. A human found 28 non-first-
   contact tweets; the model found 3. It is riding the base rate, not
   detecting turn type. Do not read that 86% as capability.
2. **Confidence cannot be thresholded.** 96% of predictions land at
   >=0.85 confidence, where accuracy is flat (63% vs 64%). The model is
   confidently wrong about as often as confidently right, so the
   low-confidence-escalates rule buys almost nothing.
3. **`needs_human_triage` is dead code.** A human set it 12 times; the
   model set it 0 times. It never abstains, so that escalation branch
   never fires in practice.
4. **Accuracy swings 49%-70% across labeling windows** -- a 22-point
   spread around a 62% headline, same classifier, same taxonomy.
5. **Some confusions are taxonomy problems, not model errors.**
   `how_to_usage <-> storage_quota_plan_limits` and
   `sync_app_bug <-> storage_quota_plan_limits` are confused in *both*
   directions, meaning the boundaries genuinely overlap. The worst
   per-intent F1 scores all sit in that cluster. v2 fixed v1's
   turn-type problem but not this one.
6. **LLM non-determinism**: tweet 1274528 got different intents on two
   runs (0.90 and 0.85 confidence). Single-run accuracy overstates
   stability.
7. **The golden pool over-samples rare intents by construction**, so
   this accuracy is not an estimate of accuracy on real traffic mix.
8. **Reply-quality judging has a ceiling effect**: the agent sits at
   ~4.9/5, so the rubric separates agent from baselines but can no
   longer distinguish good from excellent.
9. **The judge is not yet validated against human ratings.** Two
   independent judges agreeing (68% exact, 84% within 1 point) is not
   proof either is right -- they can share a blind spot.

## Repo layout

```
data/
  twcs/twcs.csv           raw full dataset (gitignored, 516MB)
  dropbox_paired.csv      filtered DropboxSupport pairs (committed)
  reading_sample.txt      40-example reproducible reading sample
  classified_sample.csv   classifier output on those 40
  classification_cache.jsonl  permanent classify-once cache
  golden_candidates.csv   golden-set candidate pool (207)
  golden_labels.csv       hand-labeled golden set (189)
  reply_evals.csv         reply drafts + judge scores
scripts/
  01-05_*.py              data pipeline: explore, filter, pair, sample
  groq_lib.py             Groq client, prompt, schema, retry policy
  cache.py                append-only classification cache (namespaced)
  rate_limiter.py         sliding-window rate limiter
  classify_runner.py      cached + concurrent + rate-limited orchestration
  06_classify.py          intent classifier over a sample
  07_build_golden_candidates.py  golden-set candidate sampling
  08_label_golden_set.py  interactive hand-labeling (--priority, --plan)
  retrieval.py            TF-IDF retrieval over dropbox_paired.csv
  09_draft_reply.py       grounded reply drafting (policy-enforced)
  10_baselines.py         trivial + simple baselines
  11_label_consistency.py golden-set self-consistency audit
  12_evaluate.py          eval harness (no API calls)
  13_reply_eval.py        reply drafting + LLM-judge scoring
  14_judge_agreement.py   blind human rating -> judge validation
intents.json / intents.md / taxonomy.md   taxonomy v1 + v2
golden_set.md             golden-set methodology
PROGRESS.md               chronological build log
```

## Setup / reproduce

`data/dropbox_paired.csv` is committed so reviewers need no Kaggle
account or 516MB download.

```bash
python -m venv .venv
./.venv/Scripts/python.exe -m pip install pandas scikit-learn groq python-dotenv tenacity
# create .env with GROQ_API_KEY=<your key> (see .env.example)

# Evaluation -- no API calls, runs in seconds:
./.venv/Scripts/python.exe scripts/12_evaluate.py
./.venv/Scripts/python.exe scripts/11_label_consistency.py

# Reply quality (costs API budget; resumable):
./.venv/Scripts/python.exe scripts/13_reply_eval.py --n 20
./.venv/Scripts/python.exe scripts/13_reply_eval.py --report-only
```

Regenerating `dropbox_paired.csv` from the raw Kaggle data is optional
-- see `PROGRESS.md`.

## What's NOT done yet

- **Judge-vs-human agreement** (`14_judge_agreement.py` is built but
  needs human ratings) -- the one required piece still outstanding.
- Full 5,938-row classification run (only the sample and golden pool
  have been classified).
- Final written report and decision log.
- `git init` -- this directory is still local-only.
