# Progress log

Running account of what's built, what each script does, and what's
still missing. `README.md` is the project pitch/approach; this file is
the "what actually happened, in order" log -- update it as work
continues rather than reconstructing it from memory later.

## Status as of 2026-09-10

- Intent taxonomy (v1, 8 intents) defined and locked.
- **Taxonomy v2 proposed (13 intents + `turn_type` + flags), not
  adopted** -- `taxonomy.md` documents v1 and v2 side by side with the
  evidence and migration path. Triggered by two findings in the
  candidate pool: `general_feedback_or_other` is 21% of it (58% of that
  being mid-thread turns the schema can't represent) and
  `sync_technical_bug` another 26%. Raised now specifically because
  hand-labeling hasn't started -- changing the taxonomy after labeling
  costs 250 examples of human time. No code changed; `intents.json`
  and the cache are still v1.
- Intent classifier built, migrated from Gemini to **Groq**
  (`openai/gpt-oss-20b`), and hardened with caching + concurrency +
  rate limiting.
- Golden-set candidate pool: **COMPLETE**. 250 unique tweets in
  `data/golden_candidates.csv`, every intent at or above the
  15-minimum (`account_login`, `data_loss_recovery` and
  `followup_ticket_status` at 16 each; `sync_technical_bug` the
  largest at 64). The top-up loop that was previously blocked did
  finish -- this file's earlier "2 intents short" note was stale.
- Pool was trimmed 251 -> 250 to respect the assignment's hard 250
  cap. The dropped row was a top-up row of the most over-represented
  intent (`sync_technical_bug`), chosen so the main random sample was
  left untouched and no rare intent fell below 15. Pre-trim file kept
  at `data/golden_candidates_251_backup.csv`.
- Hand-labeling: **in progress**, 85/207 done as of 2026-09-12. The
  single blocking item for the eval harness specifically (needs the
  full labeled set for trustworthy per-intent accuracy numbers), but
  not a blocker for the three tracks below, which don't read
  `golden_labels.csv` at all (or in `SimpleBaseline`'s classifier case,
  degrade gracefully with whatever's labeled so far).
- Escalation/auto-handle logic (`scripts/escalation.py`): **built**.
  v2-taxonomy remap of README's v1 escalation matrix -- flag/turn_type
  overrides checked first (needs_human_triage, low confidence,
  disputing_prior_answer, legal_sensitive, wants_human, churn_threat,
  abusive_content), then a 13-intent default table. Self-check passes;
  not yet tuned against real outcomes since there's no eval harness yet.
- Grounded reply drafting (`scripts/retrieval.py` +
  `scripts/09_draft_reply.py`): **built**. TF-IDF retrieval over
  `dropbox_paired.csv` (5,938 rows, in-memory, @mentions/URLs stripped
  before vectorizing) finds similar past threads; Groq drafts a reply
  grounded in them. DM-deflection policy (README's "what 'good' means"
  section) is enforced via an explicit prompt instruction, not left for
  the model to infer from the examples -- verified live on both
  branches: a `how_to_usage` query got the real how-to answer even
  though its top grounding example was a bare DM deflection, and a
  `data_loss_recovery` query asked for specific info (email + folder
  name) rather than a bare "please DM us."
- Baselines (`scripts/10_baselines.py`): **built**. Trivial
  (majority-class / canned reply / always-escalate) and simple
  (TF-IDF+LogisticRegression classify / 1-NN retrieval draft / keyword-
  regex escalate) per README's spec. The classifier baseline fits on
  `golden_labels.csv` and reports 5-fold CV accuracy once every class
  has >=2 examples -- not yet (some intents still have 1 label) --
  rerun after hand-labeling completes for the real number.
- Eval harness, failure analysis, report: **not started**.
- Not a git repo yet.

## Why Groq instead of Gemini

Gemini's free tier was too tight for this project's real API volume
(golden set + reply drafting + eval harness easily adds up to
1,000-3,000+ calls): `gemini-3.6-flash` capped at 20 requests/day,
and even after switching to `gemini-flash-lite-latest` we hit a
500-requests/day wall partway through building the golden candidate
pool. Groq's free tier is far more generous, so the whole classify
pipeline (`groq_lib.py`, `classify_runner.py`, `cache.py`,
`rate_limiter.py`) was rebuilt around it. `gemini_lib.py` and
`00_test_gemini.py` are left in place as a historical record, not
deleted, but nothing in the current pipeline imports them anymore.

Groq model note: the models originally planned
(`llama-3.1-8b-instant`, `llama-3.3-70b-versatile`) are no longer
available on this API key as of 2026-09-10 -- checked via
`client.models.list()` rather than guessing. Currently using
**`openai/gpt-oss-20b`**, a mid-size OSS model with solid JSON-mode
support, good enough for picking 1 of 8 categories.

## What each script does

### Data pipeline (run once, in order, to regenerate everything from the raw Kaggle CSV)

| Script | What it does |
|---|---|
| `01_explore_data.py` | Loads the raw `data/twcs/twcs.csv` (2.8M-row Kaggle dataset), prints shape/columns/dtypes. Pure exploration, no output file. |
| `02_brand_volume.py` | Counts support replies sent per brand (`inbound == False` rows), to see which brands have how much data. Pure exploration, no output file. |
| `03_shortlist_volume.py` | For a shortlist of mid-sized brands, computes the *real* filtered dataset size (brand replies + the customer tweets they replied to, deduped) -- more accurate than raw reply count for picking a brand. Pure exploration, no output file. |
| `04_filter_brand.py` | Filters the full dataset to `DropboxSupport` conversations and pairs each customer tweet with the brand's direct reply. Writes `data/dropbox_paired.csv` (5,938 pairs / 4,504 unique customer tweets -- multi-turn threads mean one customer tweet can appear more than once). This is the committed base dataset everything else builds on. |
| `05_sample_for_reading.py` | Draws a reproducible random sample of 40 pairs (`random_state=42`) from `dropbox_paired.csv` and writes them to `data/reading_sample.txt` in a human-readable format, for manual read-through to inform the intent taxonomy. |

### Classification infrastructure (shared, used by 06 and 07)

| Script | What it does |
|---|---|
| `groq_lib.py` | Single source of truth for the Groq client, prompt, response schema/validation, model choice, and retry policy (retries only on 429/5xx/timeouts, `stop_after_attempt(5)`, exponential backoff). Exposes `make_client()` and `classify_message(client, text) -> {intent, confidence, reason}`. |
| `cache.py` | `ClassificationCache` -- a permanent, append-only JSONL cache keyed by `tweet_id`. Any tweet already classified by any script, in any prior run, is read from disk instead of re-sent to the API. Writes are immediate (one line per result), so a crash loses at most the one in-flight result. |
| `rate_limiter.py` | `RateLimiter` -- thread-safe sliding-window limiter. Currently set to 15 calls/60s, sized to Groq's real constraint on this key (8,000 tokens/minute, confirmed via response headers), not the 1,000-requests/day cap. |
| `classify_runner.py` | Orchestrates the three pieces above: given a list of `(tweet_id, text)`, checks the cache first, fires the remaining calls through a 10-worker `ThreadPoolExecutor` gated by the rate limiter, and checkpoints each fresh result to the cache as soon as it completes. Still one example per API call (no batching) -- kept simple and debuggable per explicit decision. |
| `00_test_gemini.py` / `00_test_groq.py` | One-call sanity checks that the configured API key + model actually work, before building anything on top. Gemini version kept for history; Groq version is the current one to run. |

### Classification scripts (use the infrastructure above)

| Script | What it does |
|---|---|
| `06_classify.py` | Classifies a sample of N customer tweets (default 40, same `random_state=42` sample as `05_sample_for_reading.py` so results are directly comparable to the manual read-through labels). Routes through `classify_runner` (cached + concurrent + rate-limited), writes `data/classified_sample.csv`. Verified: 40/40 classified with zero failures, cache confirmed to skip already-classified tweet_ids on rerun. |
| `07_build_golden_candidates.py` | Builds the *candidate pool* for the golden eval set -- NOT the final labels. Two-phase sampling: (1) a pure random draw of ~150 tweets to represent real traffic distribution, (2) a rare-intent top-up that keeps sampling+classifying unseen tweets until every intent has >=15 examples (capped at 249 total, 900 top-up calls). The classifier's guess is saved as `suggested_intent` for convenience only -- it is not ground truth. Resumable: checkpoints to `data/golden_candidates.csv` after every call. **Not yet wired through `classify_runner`/`cache.py`** -- still calls `classify_message` directly in its own loop, so it doesn't yet benefit from the shared cache or concurrency (see "Not done yet" below). |
| `08_label_golden_set.py` | Interactive tool *you* run yourself in a terminal -- shows each candidate tweet, asks for your own intent label BEFORE revealing the model's `suggested_intent` (avoids anchoring bias, which would quietly inflate the reported classifier accuracy). Appends to `data/golden_labels.csv` as you go; resumable, skips already-labeled tweets. Built but not yet run -- this is the current blocking item. |

## Data files

| File | Rows | Status |
|---|---|---|
| `data/twcs/twcs.csv` | 2.81M | Raw Kaggle dataset. Gitignored, not committed (516MB). |
| `data/dropbox_paired.csv` | 5,938 | Filtered DropboxSupport customer↔reply pairs. **Committed** so reviewers don't need Kaggle access. |
| `data/reading_sample.txt` | 40 | Fixed reading sample for manual taxonomy design. |
| `data/classified_sample.csv` | 40 | Classifier output on the same 40 examples (Groq / `openai/gpt-oss-20b`), for eyeballing against manual judgment. |
| `data/classified_sample_gemini_backup.csv` | 40 | The original Gemini-era classification of the same 40, kept for a before/after diff if useful for the report. |
| `data/classification_cache.jsonl` | 40 (growing) | Permanent classify-once cache, keyed by `tweet_id`. Shared by any script that goes through `classify_runner`. |
| `data/golden_candidates.csv` | 250 | Golden-set candidate pool (classifier's suggested intent only, not ground truth). Complete: all 8 intents >= 15. |
| `data/golden_candidates_251_backup.csv` | 251 | Pre-trim snapshot, kept so the 250-cap trim is auditable. |
| `data/golden_labels.csv` | -- | **Does not exist yet.** Output of `08_label_golden_set.py`, not yet run. |

## What's NOT done yet

- Finish hand-labeling the 207-tweet candidate pool with
  `08_label_golden_set.py` (85/207 done).
- Wire `07_build_golden_candidates.py` through `classify_runner.py` /
  `cache.py` so it gets the same caching + concurrency + rate-limiting
  benefits as `06_classify.py` (currently calls `classify_message`
  directly in its own loop).
- Eval harness: automated metrics + LLM-as-judge + judge-vs-human
  agreement evidence. This is what will actually exercise
  `escalation.py`, `09_draft_reply.py`, and `10_baselines.py` against
  the golden set once labeling is done -- none of the three have been
  scored against ground truth yet, only spot-checked.
- A checkpointed/resumable path for classifying the full 5,938-row
  dataset (only the 40-example sample and the 234-candidate pool have
  been run so far).
- Failure analysis, "what's misleading about my headline number"
  section, decision log, final report.
- `git init` -- this directory is still local-only.
