# Issues, Problems, and Decisions

A record of what went wrong while building this, what I did about it,
and the trade-offs behind each choice. Written as the problems happened,
not reconstructed afterwards -- several entries below are things that
made the results *worse* on paper, which is why they're worth keeping.

Companion docs: `README.md` (what this is and what it scores),
`PROGRESS.md` (chronological build log), `taxonomy.md` (v1 vs v2
taxonomy evidence), `golden_set.md` (sampling/labeling methodology).

---

## 1. Infrastructure and cost

### 1.1 Gemini's free tier was unusable for this workload

**Problem.** Started on `gemini-3.6-flash` and hit a **20-requests/day**
cap almost immediately. Switched to `gemini-flash-lite-latest`, then hit
a **500-requests/day** wall partway through building the golden-set
candidate pool. This project needs 1,000-3,000+ calls end to end
(golden set + reply drafting + judging).

**What I did.** Migrated the whole classification stack to **Groq**
(`openai/gpt-oss-20b`): `groq_lib.py`, `classify_runner.py`, `cache.py`,
`rate_limiter.py`.

**Trade-off.** A day of rewrite against a hard ceiling that would have
blocked the project entirely. Not really a choice once the second cap
appeared.

**What I kept.** `gemini_lib.py` and `00_test_gemini.py` are still in
the repo, unused. Deleting them would erase the evidence of why the
stack looks the way it does.

### 1.2 A script that saved only at the end nearly lost ~150 API calls

**Problem.** `07_build_golden_candidates.py` wrote its output once, at
the end. The daily quota cap killed the run mid-way, and every
successful classification up to that point would have been silently
lost.

**What I did.** Recovered the results by parsing the terminal log rather
than re-spending quota, then rewrote the script to checkpoint every
classification to disk immediately and resume automatically on rerun.

**Decision that followed.** *Every* long-running script in this project
is now append-as-you-go and resumable: labeling, reply drafting,
judging. JSONL/CSV append means a crash loses at most the one in-flight
row.

**Trade-off.** Slightly more code and a resume-key to reason about (see
4.2, where that key turned out to be wrong) in exchange for never
re-spending budget on work already done.

### 1.3 The prompt itself was the budget problem

**Problem.** Groq's real constraint on this key turned out to be
**200,000 tokens/day** -- separate from, and much tighter than, the
tokens/minute cap the rate limiter was built around. The first draft of
the v2 classification prompt used documentation-length descriptions for
every intent and flag: ~1,317 prompt tokens per call, resent on every
single call. That's a ceiling of ~130 classifications/day, nowhere near
enough for a 200+ example golden set.

**What I did.** Trimmed every intent/turn_type/flag description in the
prompt to a short phrase. Full rationale for each lives in
`taxonomy.md`, for humans, not in the prompt.

**Trade-off.** The prompt is now terse to the point of being
unfriendly to read. Accepted deliberately: the model needs enough to
*decide*, not enough to understand why the category exists. Also set
`reasoning_effort="low"`, which cut hidden reasoning tokens from 111 to
6 per call with no change in answers on a test case.

**Rule adopted.** Re-measure token cost after any edit to
`intents.json` / `turn_types.json` / `FLAG_SPEC`.

### 1.4 The rate limiter was sized for the wrong kind of call

**Problem.** `classify_runner.py` uses 15 calls/60s, tuned for
classification calls of ~400 tokens against an ~8,000 tokens/minute
ceiling. The reply-judging script inherited that number, but a judge
call (rubric + tweet + reply) is ~800 tokens -- so it ran at ~12,000
tokens/min and **lost rows to `RateLimitError`**. The gaps looked like
model failures; they weren't.

**What I did.** Dropped the judge script to 8 calls/60s and documented
why the two numbers differ.

**Lesson worth stating.** A shared rate limiter is only shared if the
calls are the same size. The failure was silent-ish -- rows just went
missing -- which is the dangerous kind.

---

## 2. Taxonomy

### 2.1 v1 conflated three different things in one field

**Problem.** The v1 taxonomy had 8 intents, but `followup_ticket_status`
is a *conversational position*, not a topic, and
`general_feedback_or_other` had swollen to 21% of the candidate pool
(58% of that being mid-thread turns the schema couldn't represent).
`sync_technical_bug` was another 26%, holding service-wide outages next
to ordinary upload failures. Every mid-thread tweet had to lose either
its topic or its turn-type.

**What I did.** Designed v2: **13 intents (topic only)**, plus a
separate `turn_type` field (5 values) and a flag set
(`wants_human`, `legal_sensitive`, `churn_threat`, `abusive_content`,
`needs_human_triage`, `sentiment`, `language`).

**The timing was the real decision.** I raised this **before
hand-labeling started**. Changing a taxonomy after labeling means
re-labeling every example by hand -- ~200 examples of human time. The
cost of acting early was redoing the candidate pool; the cost of acting
late would have been the entire golden set.

**Trade-off.** 13 intents need more golden-set budget than 8 (a
per-intent minimum times more intents). Sized it deliberately against
the assignment's 250 cap. Flags don't need their own minimum the way an
intent does, so the extra fields were nearly free.

**What v2 did *not* fix** -- see 5.4.

---

## 3. The golden set

### 3.1 Anchoring bias would have inflated the headline number

**Problem.** The candidate pool already contains the classifier's guess
(`suggested_intent`). If the labeling tool showed that first, a human
would unconsciously agree with it more often than independent judgment
would -- and the number being reported *is* the agreement rate between
human and classifier. The measurement would corrupt itself.

**What I did.** The labeling tool asks for the human answer **first**,
then reveals the model's guess for comparison. Blind answer, then
feedback.

**Trade-off.** Slower and less satisfying to use than a
confirm-the-suggestion flow, which would have been perhaps 5x faster.
That speed is exactly what makes it worthless.

**Related decision I held to under time pressure.** Late in the project
I was asked several times to generate 20-30 labels directly, or to fill
the remainder from `suggested_intent`. I declined: a model labeling the
data used to grade that same model produces a circular number, and those
rows would have shown ~100% agreement against ~63% on the human-labeled
ones -- visible in the CSV, and indefensible in a write-up. The correct
lever under time pressure was scope (3.2), not source.

### 3.2 Running out of time to label everything

**Problem.** 207 candidates, and per-intent coverage mattered more than
raw count -- rare intents like `security_account_compromise` were
sitting at 1-3 examples while `complaint_dissatisfaction` had 28.
Reaching the original >=15-per-intent target needed 92 of the remaining
107 labels.

**What I did.** Added a `--priority` mode that reorders *unlabeled*
candidates so the thinnest intents come first, recomputing the deficit
from actual human labels (not the model's guesses) on every run, so the
plan adapts as real labels come in. Lowering the target floor from 15 to
10 cut the remaining work roughly in half.

**A subtlety in the implementation.** The priority order **interleaves**
intents rather than grouping them. Labeling nine consecutive tweets the
model all thinks are `security_account_compromise` would prime the
labeler toward that answer by the ninth -- the same anchoring problem as
3.1, arriving through ordering instead of through a visible suggestion.

**Trade-off, and it's a real one.** A run stopped early under
`--priority` is **not a random sample** -- it deliberately over-weights
rare intents. That's fine for measuring per-intent accuracy and wrong
for any claim about real traffic mix. Stated in the README rather than
absorbed silently.

**Where it stopped.** 189 of 207, deliberately. Inside the assignment's
150-250 range, every intent at >=10 except `billing_subscription` (8)
and `security_account_compromise` (8), both disclosed as underpowered
rather than quietly averaged into a per-intent table.

### 3.3 Were the labels any good?

**Problem.** "My labels are the measuring instrument" is a claim that
should come with a check.

**What I did.** Wrote `11_label_consistency.py`: finds near-identical
tweets carrying conflicting intents, escalate decisions that break their
own intent's pattern, fields that look defaulted rather than decided,
and agreement drift across the labeling session.

**Result.** Zero contradictory near-duplicate pairs. All five
`turn_type` values used. `secondary_intent` on 31% of rows, flags on 9%
-- both plausible, neither defaulted. Agreement with the classifier
showed no monotonic climb across the session, so no visible anchoring
creep or fatigue.

**Deliberate limit.** The audit reports internal contradictions only and
never proposes what a label *should* be. Supplying the "right" answer
would put model judgment back into the ground truth through the back
door.

---

## 4. Bugs found late

### 4.1 Adding a column mid-run corrupted the CSV

**Problem.** Added a `judge_model` column to the reply-eval output while
a run was in flight. The file ended up with a 12-field header, 12-field
old rows, and 13-field new rows; pandas refused to parse it
(`Expected 12 fields in line 39, saw 13`). It happened a second time
when `rubric_version` was added.

**What I did.** Repaired by rewriting the header and backfilling old
rows with the value they implicitly had (the drafter's own model / `v1`),
keeping a `.bak`. Then made `_read_out()` backfill missing columns on
read, so old rows stay readable without touching the file.

**Trade-off.** Backfill-on-read is slightly magical, but it means
historical rows never have to be rewritten -- and rewriting data files
to fit new code is how you lose results.

### 4.2 A resume key that would have silently done nothing

**Problem.** `load_done()` keyed on `(tweet_id, system)` filtered by
`judge_model` only. After introducing a **new rubric version**, a rerun
would have seen every row as already done and scored nothing -- while
exiting cleanly and looking like a success.

**What I did.** Made the resume key include `rubric_version`, so a
rubric change correctly invalidates prior scores.

**Why this one matters.** It's the worst shape of bug in an eval
pipeline: no error, no crash, just a silently stale number that you'd
go on to report.

### 4.3 The agent was emitting template placeholders

**Problem.** 12% of drafted replies (9 of 76) contained literal
placeholders: `@123456`, `[Name]`, `@username`.

> `@123456 Hey [Name], sorry to hear that. Please DM us your email address...`

**Root cause.** Two compounding things. Every `brand_text` in the
dataset begins with the anonymized numeric handle it replied to
(`@549821 Hey Gustaf, ...`), so the retrieved grounding examples taught
the model that shape, and it invented a handle to match. And the prompt
described the first name in those examples as an agent *sign-off*, when
it's actually the original customer's name -- pointing the instruction
at the wrong thing and implicitly licensing name-shaped output.

**What I did.** Strip leading `@handles` from grounding examples before
they enter the prompt (they carry no information the drafter needs --
the posting client addresses the recipient), and replaced the sign-off
line with explicit rules: never emit a placeholder, never start with an
`@handle`, never address the customer by name. Verified on the exact
tweets that had failed: 4/4 clean.

**The part worth noticing.** The lenient judge rubric scored those same
placeholder replies **5/5**, because it only checked whether the reply
named an artifact to send. The stricter rubric (5.3) is what surfaced
this. A weak evaluator doesn't just report a wrong number -- it hides
real defects.

---

## 5. Evaluation design

### 5.1 Tuning a policy on the data used to score it

**Problem.** The escalation policy disagreed with human labels on 21% of
rows. Fixing it by looking at those rows and then reporting accuracy on
the same rows would be in-sample and inflated.

**What I did.** Split the golden set 60/40, tuned only on the 60%, and
reported the 40% never touched. Held-out escalation accuracy: **78.9%**
(vs 86.7% on the tuning split -- the gap is exactly the optimism being
guarded against).

**What the data changed.** `account_access` defaulted to *auto-handle*.
A human escalated **89%** of those tweets; the default was right 11% of
the time. Flipped it. The README's original v1 matrix had actually said
"escalate if 2FA/compromise, auto-handle if plain password reset" --
collapsing that conditional into a single auto-handle was the error.

**What I chose *not* to do.** `complaint_dissatisfaction`,
`billing_subscription`, and `sync_app_bug` all sit near 50% -- intent
alone doesn't determine escalation for them. I could have added
per-intent exceptions and pushed the number up, but fitting rules to
~113 rows is overfitting. Left as a stated finding instead. A second
candidate rule (propagating escalation through `secondary_intent`)
bought ~1pp on the tuning split and nothing on top of the
`account_access` fix, so it was dropped rather than kept for the sake of
looking thorough.

### 5.2 An LLM judging an LLM

**Problem.** Reply quality was first judged by `openai/gpt-oss-20b` --
the same model that wrote the replies. Self-preference bias is
well-documented, and the agent scored 4.73/5.

**What I did.** Rather than caveat it, **tested it**: re-judged
everything with `qwen/qwen3.8-27b`, a different vendor and training
lineage, and recorded which judge produced every score so the two can
never be silently averaged.

**Result, which contradicted my expectation.** The independent judge
scored the agent *higher* (4.85-4.92), not lower. Judge-vs-judge
agreement: 68% exact, 84% within one point. Self-preference bias was
**not** supported for this setup.

**Why that's better than a caveat.** "We assumed bias and discounted the
number" is weaker than "we measured it and it wasn't there." The cost
was one extra judging pass.

### 5.3 The rubric had a loophole that flattered a fixed string

**Problem.** The trivial baseline sends **one identical canned reply**
to every customer. Under rubric v1 it scored **3.69/5**. That's not a
real result. The v1 policy clause accepted any reply that named an
artifact to send, and the canned reply happens to name one ("DM us your
account email").

**What I did.** Rubric **v2** adds an engagement test: naming an
artifact is necessary but not sufficient -- a reply that would read
identically for any customer with any problem is a generic holding
message and is capped at 2. Scores from different rubric versions are
kept in separate rows, never averaged.

**Effect (partial run).** Trivial dropped 3.69 -> 1.91. But the agent
also dropped 4.87 -> 2.45, with deflection going 0% -> 82%.

**Honest open question.** v2 is clearly more right than v1 -- it caught
the placeholder bug (4.3) and closed the canned-reply loophole. But it
marks *some* replies as "generic holding messages" that do give a real
answer and a real link, and it tangles engagement with factual
correctness. The true number is probably somewhere between the two
rubrics, closer to v2. Reported as an open question rather than picking
whichever is flattering.

### 5.4 A free reliability probe, found by accident

Because the trivial baseline emits one fixed string, any variation in
its scores is variation in the **judge**, not the thing judged. The
judge scored that identical string `[1,1,1,2,2,3,4,4,5,5,5,5,5]` --
a full 1-5 range, std 1.70 -- and called it a deflection 6 times out of
13. Some spread is legitimate (the rubric is intent-dependent), but it
bounds how finely any of these averages can be read. Worth more than
most of the headline numbers.

### 5.5 Numbers that look like capability but aren't

Three that would have been easy to report uncritically:

- **`turn_type` accuracy is 86%** -- and meaningless. The model
  predicted `first_contact` for 186 of 189 rows. A human found 28
  non-first-contact tweets; the model found 3. It's riding the base
  rate, not detecting anything.
- **Confidence can't be thresholded.** 96% of predictions land at
  >=0.85, where accuracy is flat (63% vs 64%). The model is confidently
  wrong about as often as confidently right, so the
  low-confidence-escalates rule buys almost nothing. A negative result
  about my own design.
- **`needs_human_triage` is dead code.** A human set it 12 times; the
  model set it 0. That escalation branch never fires in practice.

Also: accuracy swings **49%-70% across labeling windows** -- a 22-point
spread around a 62% headline, same classifier, same taxonomy. And tweet
1274528 got *different* intents on two separate runs, so single-run
accuracy overstates stability.

### 5.6 Some "model errors" are taxonomy errors

`how_to_usage <-> storage_quota_plan_limits` and
`sync_app_bug <-> storage_quota_plan_limits` are confused in **both
directions**, and the worst per-intent F1 scores all sit in that
cluster. A confusion that runs both ways is a boundary problem, not a
model problem -- the fix is the taxonomy, not the prompt. v2 fixed v1's
turn-type conflation but not this overlap. Left as a documented finding;
a v3 would merge or sharpen those boundaries.

---

## 6. Product choices

### 6.1 The DM-deflection problem

**Problem.** A large share of real DropboxSupport replies are "sorry
about that, please DM us" -- the actual resolution happens off-thread
and never appears in the dataset. A retrieval-grounded agent trained to
imitate this brand learns to deflect *everything*, which is not a
support agent.

**What I did.** Made it an explicit policy in the drafting prompt rather
than something the model infers from examples:

- **Informational intents** (`how_to_usage`, `feature_request`): must
  give the real answer. Deflecting is a failure.
- **Account-specific intents**: asking for a DM is correct, but must say
  exactly what to send.

**And then measured it**, as its own boolean rather than folded into a
quality score -- because averaging it into a 1-5 number would hide
precisely the thing worth seeing. The agent deflects on 0% of sampled
replies under rubric v1, against 46% for 1-NN retrieval. (Under the
stricter v2 rubric, see 5.3.)

**Verified it actually binds.** On a `how_to_usage` query whose *top
retrieved example* was a bare "we've replied to your DM!", the agent
still produced real link-sharing instructions.

### 6.2 What I chose not to build

- **Stitching multi-part replies.** Many `brand_text` values are
  fragments of threads (`1/2`, `2/2`). Retrieval returns them as-is.
  Stitching is real work and would have improved grounding quality;
  stated as a known limitation instead of silently patched.
- **The full 5,938-row classification run.** Costs real budget and adds
  little the 189-example golden set doesn't already show. The eval
  harness runs on stored predictions and needs no API calls at all,
  which matters more for a reviewer reproducing this.
- **Embeddings for retrieval.** TF-IDF over 5,938 rows is milliseconds
  and needs no vector store. Worth revisiting only if retrieval quality
  becomes the measured bottleneck -- it isn't yet.
- **Sentiment/language hand-labeling.** Collected from the model,
  deliberately not hand-verified: two more open-ended judgment calls
  across ~200 examples for signals peripheral to the golden set's
  purpose (intent accuracy and the escalate decision).

### 6.3 Brand selection

Top brands (Amazon, Apple, Uber) had 100K+ replies -- too much for a
subsampled take-home. Picked **DropboxSupport** (5,938 pairs): large
enough to be real, small enough to process, and its SaaS support
intents (login, sync, billing, storage, sharing) parallel Hiver's own
product domain.

---

## 7. Still open

- **Judge-vs-human agreement** is built (`14_judge_agreement.py`) but
  unrun. Two models agreeing with each other is not proof either is
  right -- they can share a blind spot. This is the one required piece
  outstanding.
- **Which rubric is the reported one** (5.3) needs deciding, ideally
  with the human ratings above as the tiebreaker.
- **The v2 rubric's engagement test tangles engagement with factual
  correctness** and should probably separate them.
- **`git init`** -- this directory is still local-only.
