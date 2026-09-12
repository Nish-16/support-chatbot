# Every script, what it does, and why it exists

A walkthrough of `scripts/`, written so you can explain any file without
re-reading it. Each entry gives what it does, what it reads and writes,
the design decision worth mentioning out loud, and the thing an
interviewer is most likely to probe.

Numbered scripts run in order and are one-shot pipeline stages.
Unnumbered ones are libraries imported by the rest.

**Run everything from the repo root** — scripts open `intents.json` and
`data/...` by relative path:

```bash
PYTHONPATH=scripts ./.venv/Scripts/python.exe scripts/12_evaluate.py
```

---

# Libraries

## `groq_lib.py` — the single source of truth for the classifier

One place holding the model id, the prompt, the response schema, the retry
policy and the cache-namespace scheme. Everything that classifies goes
through it, so two call sites can never drift apart.

**Key parts**

- `MODEL` — `openai/gpt-oss-20b`. Chosen by calling `client.models.list()`
  rather than guessing; the originally-planned Llama models are no longer
  served on this key.
- `TAXONOMY_VERSION` / `PROMPT_VERSION` / `cache_namespace()` — see
  the caching section below. This is the part most worth explaining.
- `SYSTEM_PROMPT` — deliberately terse. An earlier draft used
  documentation-length descriptions and cost ~1,317 prompt tokens per
  call, which against Groq's **200,000 tokens/day** cap is a ceiling of
  ~130 calls/day. Not enough to build a 200-row golden set.
- `_validate()` — the schema is enforced by *parsing and checking*, not by
  the API. Groq's `json_object` mode guarantees valid JSON, not a valid
  shape. Invalid results raise `InvalidClassification`, which is distinct
  from a transport error.
- Normalises the model returning the **string** `"null"` for
  `secondary_intent` instead of the JSON literal.
- `PRECEDENCE_RULES` — `taxonomy.md` Part 11, compressed. Exported for the
  labelling tool. **Not in the shipped prompt** — see `PROMPT_VERSION`'s
  comment for the p2 experiment that rejected it.
- `FEW_SHOT` — staged for a p3 experiment, drawn from the dev split only.

> **Likely question: why not use structured outputs / function calling?**
> Groq's `json_schema` support is model-dependent; `json_object` works
> everywhere. The schema is enforced in `_validate()` instead, which also
> gives a clean retry boundary.

## `cache.py` — append-only classification cache

JSONL, one object per line, keyed by `(namespace, tweet_id)`. A crash
loses at most the one in-flight line, and appending never rewrites the
file.

**The namespace is the interesting part.** It began as just
`TAXONOMY_VERSION` (`"v2"`). That is not enough: it only moves when the
*label set* changes, so two different **prompts**, or two different
**models**, over the same 13 intents would share a namespace — and the
second experiment would silently read back the first one's predictions.

It is now `taxonomy:prompt:model`, e.g. `v2:p1:openai/gpt-oss-20b`.

> **This is the best "what would have gone wrong" story in the repo.**
> Without the model in the key, the gpt-oss-120b comparison would have
> read gpt-oss-20b's cached answers and reported 100% agreement — a
> completely convincing wrong result. It then caught a real error of mine:
> looking up the 120b run under prompt version `p2` when it ran under
> `p1`, returning zero rows instead of silently mixing two models.

## `rate_limiter.py` — sliding-window limiter

Built for the 8,000 tokens/minute cap. Worth noting the real binding
constraint turned out to be the **200k tokens/day** cap instead, which a
per-minute limiter cannot help with — that one is managed by keeping the
prompt short and by the cache.

## `classify_runner.py` — cached, concurrent, rate-limited orchestration

Wraps `classify_message` with the cache, ~10 worker threads (the call is
I/O-bound), the shared limiter, and **incremental saving** — every result
hits disk as soon as it is known, so any run is resumable. Takes an
optional `classify` callable so an A/B run can inject a different model
without duplicating the orchestration.

## `retrieval.py` — TF-IDF retrieval over past threads

Finds the `k` most similar past DropboxSupport exchanges to ground a
reply. Supports `exclude_tweet_id` so a tweet can never retrieve itself
when it is being evaluated — leakage that would flatter every number.

## `escalation.py` — auto-handle vs escalate

Deterministic policy, not LLM discretion. Two stages: **flag/turn_type
overrides first, short-circuiting**, then an intent default table. Every
decision carries a stated reason.

> **The design point to make:** the LLM extracts signals; Python makes the
> decision. An escalation policy has to be auditable and changeable
> without a prompt edit.
>
> **The honest caveat:** two of the override branches are dead.
> `needs_human_triage` has fired 0 times, and the `confidence < 0.5`
> branch never fires because every row scores above 0.5 — the policy
> scores identically with it removed.

## `reply_guard.py` — deterministic placeholder guard

`DRAFT_SYSTEM_PROMPT` already forbids placeholders, and `@123456` still
reached 12% of drafts. A prompt instruction is a request, not a guarantee.

Catches numeric handles (`@123456`), `[Name]`, `{customer_name}`,
`<slot>`, the dataset's own `__email__` tokens, filler like `XYZ`, and
URLs pointing anywhere that is not a known Dropbox domain.

Pipeline: **validate → regenerate once with the offending text quoted
back → sanitize as a last resort.** `sanitize` never substitutes —
inventing a name to fill a slot is worse than the slot — and returns
`ok=False` when removal leaves a broken sentence.

Run it directly to see its self-tests: `python scripts/reply_guard.py`.

> **Two bugs it caught in itself, both worth telling:** the first version
> flagged *"DM us your ticket number and the email on the account"* — the
> exact phrasing the DM policy **requires**. And `sanitize` returned
> *"Send your details to and we will help."* marked sendable.

## `gemini_lib.py` — superseded, kept deliberately

The original Gemini client. Gemini's free tier (20 req/day, then 500/day)
could not support a pipeline needing 1,000–3,000 calls. Kept as a record
of the migration rather than deleted.

---

# Pipeline: data preparation

## `00_test_groq.py` / `00_test_gemini.py`
Connectivity and token-usage probes. `00_test_groq.py` is how prompt cost
per call gets measured before committing budget to a full run.

## `01_explore_data.py`
First pass over the 2.8M-row Kaggle dataset — shape, columns, thread
structure.

## `02_brand_volume.py`
Counts inbound volume per brand. The input to choosing which brand to
build for.

## `03_shortlist_volume.py`
Narrows to brands with enough paired customer↔brand exchanges to support
both a golden set and a retrieval corpus.

## `04_filter_brand.py`
Extracts DropboxSupport and pairs each customer tweet with the brand's
reply. → `data/dropbox_paired.csv` (5,938 pairs, committed so a reviewer
needs no Kaggle account or the 516MB raw file).

> **Note the unused column:** `brand_text` is in this file and nothing
> consumes it yet. That is the single highest-value change outstanding —
> it would fix `turn_type`, the 14 unroutable rows, and the dead
> `disputing_prior_answer` override at once.

## `05_sample_for_reading.py`
Draws a reproducible 40-tweet sample to **read by hand** before designing
a taxonomy. This is where the DM-deflection problem was found — many
brand replies are just "please DM us", so a naive retrieval-grounded agent
learns to deflect on everything.

---

# Pipeline: classification and the golden set

## `06_classify.py`
Runs the classifier over a sample via `classify_runner`.

## `07_build_golden_candidates.py`
Builds the candidate pool (207 tweets): a random draw for traffic
representation **plus a rare-intent top-up** so every intent clears a
usable minimum. → `data/golden_candidates.csv`

> **State the caveat before you are asked:** the top-up means the pool is
> deliberately *not* a traffic-mix estimate. It is right for per-intent
> accuracy, wrong for "what fraction of real traffic is X".

## `08_label_golden_set.py` — the interactive labelling tool

The one part of the pipeline that cannot be delegated.

**Anti-anchoring:** you answer **before** the model's guess is revealed.
Showing the suggestion first would inflate the agreement number you later
report.

**`--priority`** reorders unlabelled candidates so thin intents come
first, **interleaved rather than blocked** — nine consecutive tweets the
model thinks are `security_account_compromise` would prime you toward that
answer on the ninth, which is the same anchoring problem arriving via
ordering. `--plan` prints the order and projected coverage, then exits.

Resumable: every label is appended immediately.

> **What changed, and why it matters:** this used to print *bare intent
> names* — making it the only consumer of the taxonomy that showed **less
> than the model got**. It now prints each intent's description and the
> Part 11 rules verbatim (the same string, not a paraphrase). 13 of 48
> labelling errors traced to rules that were already written down and
> never reached the person applying them.

---

# Pipeline: drafting and baselines

## `09_draft_reply.py` — grounded reply drafting

Retrieves similar past threads, then prompts the model to draft a reply
consistent with how the brand actually responded.

**The policy is enforced in the prompt, not left to be inferred:**

- **Informational** (`how_to_usage`, `feature_request`): must give the
  real answer. Deflecting to DM is a **failure**, not a resolution.
- **Account-specific / PII**: asking for a DM is correct, but the reply
  must say **exactly what to send** — "DM us your ticket number and the
  email on the account" — never a bare "please DM us".

`_strip_leading_handles()` removes the anonymised `@549821` prefix from
grounding examples. Shown raw, the model imitates the *shape* of that
prefix and invents one — this is where `@123456` in 12% of drafts came
from. Now backstopped by `reply_guard.py`.

## `10_baselines.py`
**Trivial**: majority-class intent, one canned reply, escalate 100%.
**Simple**: TF-IDF + LogisticRegression (cross-validated, never fit and
scored on the same rows), 1-NN retrieval, keyword-regex escalation.

Defined **before** the LLM agent was built, so the results table is not
retrofitted.

> **Flag this yourself:** these were measured against **v1 labels**. Only
> the LLM number has been recomputed on v3. The majority baseline alone
> moves 15.9% → 12.1%. This must be re-run before the comparison table is
> quoted.

## `11_label_consistency.py`
Self-audit of the golden set: near-identical tweets with conflicting
labels, escalation consistency, unused field values, agreement drift
across labelling windows.

> **Its limitation, which I found by reading:** it compares TF-IDF
> similarity, i.e. **wording**, not meaning. Two green-check tweets with
> conflicting labels have cosine similarity **0.153** — semantically
> near-identical, almost no shared vocabulary. A clean report from this
> script is weaker evidence than it looks.

## `12_evaluate.py`
The headline harness. **No API calls** — everything is computed from
stored predictions, so it runs in seconds and is fully reproducible.
Prints accuracy, per-intent P/R/F1, the confusion matrix, escalation
outcomes, and a "what's misleading about these numbers" section.

> **The section to point at:** it prints its own caveats — the `turn_type`
> base-rate artifact, the flat confidence distribution, the dead
> `needs_human_triage`, and the 49–70% accuracy swing across labelling
> windows.

## `13_reply_eval.py`
Drafts replies and scores them with an LLM judge (`qwen/qwen3.8-27b` —
deliberately a **different model family** from the drafter). Scores
`is_deflection` as its own boolean, separately from quality: the grounded
agent deflects on 0% of sampled replies vs 46% for 1-NN retrieval, which
is the exact failure the policy exists to prevent. Resumable;
`--report-only` re-prints without spending budget.

## `14_judge_agreement.py`
Blind human rating → judge validation. **Built but never run with human
input**, so the 4.92/5 reply score is currently unvalidated. Two judges
agreeing (68% exact, 84% within 1) is not proof either is right — they can
share a blind spot.

---

# The evaluation-repair sequence (15–23)

These exist because the 62.4% turned out to be measuring the labels, not
the model. Read `WHAT_WENT_WRONG.md` for the narrative.

## `15_adjudicate.py` — adjudicate the 48 cluster errors
Re-reads every error where the human label was one of the four low-recall
intents. **Read-only** with respect to `golden_labels.csv`.

Five-way verdict vocabulary, deliberately not "who won": `model_right`,
`human_right`, `both_wrong`, `ambiguous` (both defensible under the
taxonomy *as written*), `context_missing` (not routable from the tweet
alone). → `data/adjudication_v1.csv`

**Result:** the human label was wrong or not-clearly-right in 42 of 48.

## `16_recheck_adjudication.py` — re-check, plus the other side
Two jobs: re-check the 48 (rule-first this time, weighting attention
toward rows decided *for* the model), and adjudicate the 10 **false
positives** — rows where the model wrongly entered the cluster.

**Result:** 47 of 48 stand, `193025` flipped. And the model wins 21:6 on
one side but only 2:2 on the other — the skew was partly a sampling
artifact. → `data/adjudication_v2.csv`

## `17_golden_v2.py` — apply the rules to the 58 reviewed rows
Emits `golden_labels_v2.csv` with a full audit trail. Introduces
`v2_status`: `resolved` / `ambiguous` / `insufficient_context`.

Also records `FLAGGED_AGREED` — rows where the human and model **agreed**
and the new rules say both are wrong. Deliberately not relabelled, because
acting on a keyword sweep rather than a read is the sloppy version of
step 15.

## `18_golden_v3.py` — sweep all 189 rows
Applies the rules to the 131 rows nobody re-read. This exists to remove a
**structural upward bias**: adjudication only looked where the model was
already scored wrong, so every correction it could find raised the score.

**Result:** 22 rows moved — 4 raise the score, 8 lower it, 10 move out of
scoring. **83.6% → 81.0%.** 2.5 points of the previous number was
selection effect. → `golden_labels_v3.csv` + audit

## `19_migrate_cache.py` — backfill after the namespace change
One-time, append-only. Copies the 584 rows cached under the bare `"v2"`
namespace to `v2:p1:openai/gpt-oss-20b`, so widening the namespace does
not re-spend ~584 calls of budget. Idempotent; originals untouched.

## `20_model_ab.py` — model comparison
Runs a second model over the same 189 tweets with **everything else
identical** and scores both against v3 labels.

**Result: gpt-oss-120b scores 74.1% vs gpt-oss-20b's 81.0% — 6× the
parameters, 6.9 points worse.** And of the 27 rows both get wrong,
**22 are the same wrong label** — shared errors mean a shared cause, so
the remaining errors are taxonomy and prompt, not capacity.

> Reports the **McNemar-style off-diagonal** (only-A-right vs only-B-right)
> rather than just two accuracies, because that is where the information
> about a paired comparison actually lives.

## `21_split.py` — freeze the dev/held-out split
60/40, stratified on v3 labels, seed `20260912`. Written to disk **once**
and read from there forever; a split recomputed per call site drifts the
moment anyone changes a seed or a sort order, and a drifting held-out set
is worse than none because it looks rigorous.

**The script refuses to overwrite an existing split.**

Warns when an intent has under 3 held-out examples — report those as
counts, never percentages.

## `22_prompt_ab.py` — prompt comparison, dev only
p1 vs p2 on the dev split. Refuses to be casual about holdout: `--split
holdout` prints a warning that it should only run against a frozen prompt.

**Result: p2 (rules in the prompt) scored 79.1% vs p1's 81.4% — 7 rows
fixed, 9 broken, at 2.6× the prompt tokens. Rejected and reverted.**
Mechanical rules helped; judgment rules backfired, because a 20b model
applies an abstract preference bluntly.

## `23_escalation_v2.py` — fix the dead uncertainty branches
Tests replacements for `needs_human_triage` and the confidence threshold,
using the two model runs already on disk — **no new API calls**.

**Result:** cross-model disagreement separates 84.8% vs 62.1% accuracy
(+22.8 points), against a confidence signal that separates nothing. As
policy it is a trade: missed escalations 11 → 7, over-escalations 24 → 38.
`escalation.py` is **not** changed — the escalate rate would reach 56%,
which is a product decision.

---

# Reading order for an interview

1. `WHAT_WENT_WRONG.md` — the narrative
2. `12_evaluate.py` — the honest harness, caveats included
3. `groq_lib.py` + `cache.py` — the namespace story
4. `18_golden_v3.py` — removing your own selection bias
5. `20_model_ab.py` / `22_prompt_ab.py` — two documented failures

## The three strongest things to say

1. **"My evaluation was wrong, and it was wrong in the direction that
   flattered me."** 62.4% → 81.0% with zero model changes, then finding
   and removing a further 2.5 points of my own selection bias.
2. **"I tested the obvious upgrades and both failed."** A bigger model is
   6.9 points worse; putting the rules in the prompt is a wash at 2.6×
   the cost. Both documented rather than buried.
3. **"The remaining errors are shared across two model families."**
   22 of 27 both-wrong rows carry the same wrong label. That is evidence,
   not opinion, that more model would not help.

## The three weaknesses to raise before you are asked

1. **The v3 labels are my own adjudication**, self-rechecked once, never
   independently verified. If a reviewer disagrees with 10 of my 53
   changes, the headline moves ~5 points.
2. **Escalation is the metric that matters and it is mediocre** — 11
   missed escalations out of 72 a human would escalate.
3. **The baseline comparison table is currently invalid** — LLM measured
   on v3, baselines still on v1.
