# DropboxSupport AI Support Agent — Report

**Repo:** https://github.com/Nish-16/support-chatbot · **Reproduce:** `README.md`
§"Reproduce the headline in under 15 minutes" (runs in ~20 seconds, no API key)

**Headline: 81.0% intent accuracy** (141/174, 95% CI [74.6%, 86.2%]). §4 explains
why that number is less impressive than it looks: the classifier never improved,
the *evaluation* was wrong and got corrected.

This is the 6-page report. `README.md` in the repo is the same material at full
length — every section here points to where the detail lives.

---

## 1. Problem framing

### What "good" means for this brand

The defining property of this dataset: **DropboxSupport's public replies are
mostly not answers.** A large share are "sorry about that, please DM us" — the
resolution happens off-thread and never enters the data. An agent grounded in
retrieval over these threads therefore learns to deflect *everything*, which
scores well on any similarity metric and is useless as a support agent.

So "good" is not "sounds like DropboxSupport." It is an explicit policy, stated
in the drafting prompt rather than inferred from examples:

- **Informational intents** (`how_to_usage`, `feature_request`) — give the real
  answer or a documentation pointer. Deflecting to DM is a **failure**, not a
  valid auto-handle.
- **Account-specific / PII intents** (`account_access`, `billing_subscription`,
  `data_loss_recovery`, `security_account_compromise`,
  `storage_quota_plan_limits`) — asking for a DM is correct, since identity
  can't be verified in a public reply, but the agent must say exactly what to
  send, never a bare "please DM us."

`is_deflection` is scored as its **own boolean**, separate from the 1–5 quality
score, because averaging it into a quality number hides the exact failure the
policy exists to prevent.

Second: **the two escalation errors are not symmetric.** Auto-handling something
that needed a human is expensive; escalating something auto-handleable costs a
few minutes. Every escalation number below is split along that line rather than
reported as one accuracy.

### What I chose not to build

- **Stitching multi-part replies** — many historical replies are thread
  fragments (`1/2`, `2/2`). Stated as a limitation rather than silently patched.
- **The full 5,938-row classification run** — costs real budget, adds nothing the
  golden set doesn't show. Keeping evaluation API-free matters more.
- **A hosted vector database** — at 5,938 rows, the approximate index a vector DB
  sells you solves a problem this corpus doesn't have.
- **Fine-tuning** — no labelled data up front; the 174 labels are an *evaluation*
  asset, far too few to train on.
- **Hand-verifying sentiment and language** — peripheral to intent accuracy and
  the escalate decision.

---

## 2. How it works

- **Classification** — prompted LLM (Groq `openai/gpt-oss-20b`) against the fixed
  list in `intents.json`; structured JSON, cached permanently by tweet id.
  13 intents, **topic only**, plus a separate `turn_type` and boolean flags.
- **Reply drafting** — retrieval over past threads, then a grounded draft with the
  deflection policy stated explicitly. A deterministic guard (`reply_guard.py`)
  validates, regenerates once on a placeholder, and sanitises as a last resort.
  Two retrievers behind one interface: TF-IDF (default) and Chroma + MiniLM.
- **Escalation** — an explicit two-stage policy table, not LLM discretion. Flag /
  `turn_type` overrides fire first and short-circuit, then a 13-intent default
  table. Every decision carries a stated reason. `account_access` was flipped
  from auto-handle to escalate because the golden set showed a human escalated
  **89%** of them — tuned on the 60% dev split, measured on the 40% never touched.

Baselines were defined **before** the agent was built, so the results table isn't
retrofitted. The project started on Gemini and migrated to Groq when the free
tier (20/day, then 500/day) was hit mid-way through building the candidate pool.

---

## 3. Results vs. baselines

All three approaches are measured on the **same 174 rows** with the same labels.

### Intent classification

| approach | accuracy | note |
|---|---:|---|
| Trivial — always the majority class | 12.1% | always `feature_request` |
| Simple — TF-IDF + LogisticRegression | 40.8% | 5-fold CV, never scored on its own training rows |
| **LLM — `openai/gpt-oss-20b`, zero-shot** | **81.0%** | 95% CI [74.6%, 86.2%] |

Frozen 60/40 partition: **dev 83.7%** (87/104), **held-out 77.1%** (54/70). The
held-out number is the one to trust.

Best-handled intents are the ones where being wrong is expensive:
`phishing_abuse_report` F1 1.00, `data_loss_recovery` 0.90, `sharing_permissions`
0.90, `service_outage` 0.90. Worst: `complaint_dissatisfaction` 0.64,
`how_to_usage` 0.67 — see §5.

### Escalation (model intent → policy table → decision)

**79.9% end-to-end**, but the split matters more than the total:

| | count | share |
|---|---:|---|
| Missed escalations (auto-handled, human would escalate) | **11** | 16% of the 67 a human escalated — **the expensive error** |
| Over-escalations (escalated, human would auto-handle) | 24 | 22% of the 107 a human would auto-handle — cheap |

### Reply quality (LLM-as-judge)

Judged by `qwen/qwen3.8-27b`, a **different model family** from the drafter, so
self-preference bias is tested rather than assumed. Both rubric versions are
reported, because the choice moves the number by more than a point:

| system | v1 overall | v1 deflection | v2 overall | v2 deflection |
|---|---:|---:|---:|---:|
| **Grounded LLM agent** | **4.87** | **0%** | **3.60** | **40%** |
| Simple (1-NN retrieval) | 2.10 | 37% | 1.56 | 74% |
| Trivial (one canned reply) | 3.69 | 35% | 1.80 | 100% |

Rubric v1 had a loophole: it accepted any reply naming an artifact to send, and
the canned reply names one ("DM us your account email"), so a string sent
identically to every customer scored 3.69/5. v2 adds an engagement test and
collapses that baseline to 1.80 / 100% deflection. **v2 is the primary rubric.**
The ordering — agent > trivial > 1-NN — holds under both, and is the claim I'd
stand behind.

**Judge vs. human.** I rated 30 replies blind (system hidden, judge score hidden,
rows shuffled), scoped to the reported judge and rubric v2:

| measure | value |
|---|---:|
| Spearman correlation, 1–5 score | **+0.73** |
| Accept (≥4) vs. reject agreement | **87%**, Cohen's κ +0.69 |
| Exact score / within one point | 33% / 77% |
| Judge minus human, mean | **−0.60** (judge harsher) |
| Deflection-flag agreement | 67% |

The judge **ranks** replies the way a human does and makes the same accept/reject
call, so the ordering above stands. Its **absolute** scores run half a point low
and its deflection flag agrees only two times in three — so the deflection
percentages are the least trustworthy numbers here. Two independent judges on the
same 37 replies agreed 68% exactly, 84% within one point.

### Retrieval: TF-IDF vs. embeddings

| retriever | P@1 | P@3 | MRR | query |
|---|---:|---:|---:|---:|
| TF-IDF | 0.356 | 0.276 | 0.511 | 0.7ms |
| **Embeddings (Chroma + MiniLM)** | **0.540** | **0.475** | **0.677** | 197ms |

Chance P@3 is 0.081. Embeddings retrieve **+72% relative** on P@3 — and **no
downstream reply-quality gain was demonstrated.** Re-running one identical
configuration moved the score **+0.30**, as much as any retriever difference in
the table, so the harness cannot resolve a difference this size in either
direction. **Decision: TF-IDF stays default**, on cost — ~300× lower latency, no
83MB download, and the headline stays reproducible with pandas and scikit-learn
alone. Retrieval experiments are closed.

---

## 4. Golden evaluation set

**189 hand-labelled examples** drawn from a 207-tweet candidate pool, then fully
re-adjudicated (53 labels changed). **174 are scorable**; 14 are marked
`insufficient_context` and 1 `ambiguous`, excluded from the headline rather than
given a label nobody can defend (`--keep-unscorable` puts them back).

- **Sampling** — a pure random draw (~150) to represent real traffic
  distribution, plus a rare-intent top-up that keeps sampling until every intent
  clears 15 examples. This over-samples rare intents **by construction**, which
  is a caveat on the headline (§6.8), not an accident.
- **Labelling** — the tool shows the classifier's guess only **after** the human
  commits their own. Showing it first would inflate the very agreement number
  being reported. Labellers see the same boundary rules the model's prompt can
  carry, plus the parent tweet the message replies to.
- **Audited** for self-consistency — though that audit compares *wording*, not
  meaning, so it structurally cannot catch two near-identical tweets that sit at
  cosine similarity 0.153.

---

## 5. Failure analysis — top 5

**1. `complaint_dissatisfaction` is a stance, not a topic** (recall 0.54, the
worst). The model routes on topic; a complaint is defined by stance. When a tweet
has both, they disagree.

> `1856823` — *"Just crazy the new prices for @Dropbox plans. Geez! 😐"*
> → model `billing_subscription`, label `complaint_dissatisfaction`

*Hypothesis:* the taxonomy's original sin one level down. v1 was rebuilt because
it crammed topic and conversational position into one enum;
`complaint_dissatisfaction` crams *stance* into the same enum the same way. Every
complaint is also about something. The fix is making sentiment a flag, not prompt
engineering.

**2. A single tweet does not contain enough context to route.** 15 of 189 rows
(7.9%) cannot be classified by anyone, model or human, from the tweet alone:

> `2828924` — *"I Have done this. Too bad it didn't solve the problem"* ·
> `1644499` — *"Same here!"*

*Hypothesis:* conversational position isn't recoverable from an isolated message.
The same missing input causes two more failures — `turn_type` riding the base
rate, and a `disputing_prior_answer` override that can never fire. The parent
message exists in the data and production doesn't read it: the highest-value
outstanding change, and it fixes all three at once.

**3. Symptom vs. object in the sync / quota / how-to cluster.**

> `1901120` — *"Earned 25gb of space via HP Promotion. Is there any expiry?"*
> → model `how_to_usage`, label `storage_quota_plan_limits`

*Hypothesis:* these confusions run in **both directions**, and a bidirectional
confusion is a boundary problem, not a model problem — the categories genuinely
overlap. The worst per-intent F1s all sit in this cluster. The fix is the
taxonomy, not the prompt.

**4. `security_account_compromise` is a keyword magnet** (precision 0.60, recall
1.00). Any mention of unauthorized access, permissions or logins pulls a tweet in.

> `1529752` — *"When you report a questionable login… please provide the IP
> address of the suspicious login."* → model `security_account_compromise`,
> label `feature_request` (a suggestion; nothing is compromised)

*Hypothesis:* the intent name matches surface vocabulary rather than situation.
Recall 1.00 at precision 0.60 is the **safe** direction — the policy escalates
security, so these become over-escalations — but it's the single largest
contributor to the 24 over-escalations.

**5. The escalation policy has no abstain path; both uncertainty branches are
dead.** `needs_human_triage`: a human set it 12 times, the model **0**.
`confidence < 0.5`: all 174 rows sit above 0.5 — provably dead code, the policy
scores identically with the branch removed.

*Hypothesis:* a model asked to self-report confidence on a forced-choice task
reports high confidence (96% of rows ≥0.85, where accuracy is flat). A usable
signal must come from somewhere other than the model's opinion of itself, and
**cross-model disagreement works**: where `20b` and `120b` agree, accuracy is
84.8%; where they disagree, 62.1% — **22.8 points of separation**. Not shipped,
because as a policy it's a genuine trade: misses 11 → 7, but over-escalations
24 → 38 and the escalate rate rises to 56%. That's a product decision.

**Follow-up (§4.6 in README):** the 4 held-out misses all had the **correct
intent** — the human escalated on urgency, reach or repeat failure. Asking the
classifier for `urgent` / `wide_impact` / `repeated_failure` cut dev misses 7 → 3,
but **did not generalise** to the holdout (5 → 5, +3 false alarms), and asking
inside the intent prompt cost 5.8 points of accuracy. **Decision: keep rule A**,
triggers stay in code, off by default.

---

## 6. What is misleading about my headline number

**The biggest thing: the classifier never improved. The measured number went from
62.4% to 81.0% with no change to the model or the prompt.** Every prediction
scored here is the same output from the original run. What changed is that the
evaluation became correct. Two later attempts to actually improve the classifier
both failed and were reverted.

1. **The whole 18.6-point move is relabelling, not capability.** The model was
   right and I was wrong in 22 of 48 reviewed rows — I'd been labelling on
   *sentence shape*, the model on *topic*, and it was consistent where I was not.
   Root cause: the taxonomy held 36KB of boundary rules and **neither consumer
   could see them**. 53 labels changed; the original file was never modified.
2. **The relabelling is mine, self-checked once, and nobody else has verified
   it.** If a reviewer disagrees with 10 of my 53 changes, the headline moves
   ~5 points. **This is the largest open risk here.**
3. **An adjudication that only reads disagreements is biased upward by
   construction** — that version gave 83.6%. A full 189-row sweep found 8 more
   where the model and I were wrong *the same way*, which nothing had flagged.
   That cost 2.5 points. 81.0%, not 83.6%, is the number.
4. **`turn_type`'s 86% is an artifact** — the model predicted `first_contact` for
   186 of 189 rows; a human found 28 that aren't. It rides the base rate.
5. **Confidence cannot be thresholded** and `needs_human_triage` has never fired:
   two of my own design decisions, measured and found inert.
6. **Per-intent n is 9–21.** One row moves an intent's rate 7–11 points. These
   cannot rank intents against each other.
7. **Accuracy varies 74%–88% across labelling windows** — a 15-point spread around
   an 81% headline, same classifier. Both improvement attempts failed: rules in
   the prompt scored 79.1% vs 81.4% on dev at 2.6× the tokens, and a 6× larger
   model scored **74.1% vs 81.0%**. Of 27 rows both models get wrong, 22 carry the
   *same* wrong label — the ceiling is the taxonomy, not model capacity.
8. **The golden pool over-samples rare intents by construction**, so 81% is not an
   estimate of accuracy on real traffic mix. On a random draw of 250 fresh tweets,
   the same prompt scores 71–75%.
9. **The reply harness cannot resolve small differences** — an identical
   configuration re-run moves `overall` by ~0.3. The agent-vs-baseline gap (~2
   points) survives that; nothing finer does. I only found this by running a
   configuration against itself as a control, and it invalidated a retriever
   comparison I'd already written up.
10. **The reply judge is validated against one human, on 30 replies — me.** It
    ranks well but scores 0.6 points harsher and agrees on deflection only 67% of
    the time. Thirty ratings from a single rater who also wrote the rubric is thin
    evidence, not a validation study.
11. **The reply headline depends on which rubric you pick** — 4.87 under v1, 3.60
    under v2, same replies. I report v2 and show both.
12. **LLM non-determinism** — `temperature=0` did *not* make generation
    deterministic (0 of 10 replies identical on a rerun). Single-run accuracy
    overstates stability.
13. **One cache bug nearly produced a very convincing wrong result** — the cache
    was namespaced on taxonomy version alone, so the second model in an A/B would
    have read the first's answers back and reported perfect agreement.

`12_evaluate.py` prints several of these itself, so the headline can't be quoted
without them.

---

## 7. What I'd do next with one more week

In priority order:

1. **Feed the parent message into the classifier** (1 day). The data already has
   it and nothing reads it. Fixes three failures at once: `turn_type`'s base-rate
   artifact, the 15 unroutable rows, and the dead `disputing_prior_answer`
   override. Highest-value change outstanding.
2. **A second reader on the 53 adjudicated rows** (0.5 day of someone else's
   time). The one thing I cannot do for myself, and the largest open risk in §6.
3. **A second human rater for the judge** (0.5 day) — to separate "the judge is
   harsh" from "I am lenient", and to examine the deflection-flag disagreements,
   the weakest part of the judge.
4. **Prompt p3, properly evaluated.** Built and evaluated on a frozen set of 250
   fresh tweets with a pre-registered verdict rule: **INCONCLUSIVE at both sizes**
   (pre-registered n=150: 74.5% → 77.4%, +2.9, CI [−2.9, +8.8], McNemar p=0.48).
   p3 leans ahead but the interval includes zero and it misses one more human
   escalation. **PROMPT_VERSION stays p1.** A verdict needs more labelled data.
5. **Split `complaint_dissatisfaction` into a flag** (1 day incl. relabelling) —
   §5.1 says this is a taxonomy problem; this is the fix.
6. **Grow the escalation evaluation before adding triggers** (1–2 days). Four
   triggers have been tried; each moved misses 2–4 rows on one split and none held
   up on both. With 29–38 escalations per split, one row is ~3 points. Label ~150
   more weighted toward escalations, ask for signals in a *separate* call, and get
   a real miss/false-alarm cost ratio — still the missing product input.
7. **Expand the golden set to ~400** with stratified sampling on the confusion
   clusters, so per-intent rates become rankable.

Deliberately not on this list: fine-tuning, a bigger model, a vector store. The
120b experiment says capacity isn't the constraint, and retrieval isn't the
measured bottleneck.

---

## 8. Decision log

1. **DropboxSupport, not Amazon or Apple** — 5,938 pairs: real enough to matter,
   small enough to process end to end.
2. **Prompted LLM, not fine-tuning** — no labels up front; the labels I built are
   an evaluation asset, and spending them on training leaves nothing to measure.
3. **Rebuilt the taxonomy (8 → 13 intents + `turn_type` + flags) before labelling
   started** — v1 conflated topic, position and risk. The timing was the decision:
   changing it after would have cost 250 hand labels.
4. **Migrated Gemini → Groq mid-project** — a day of rewrite against a ceiling
   that would otherwise have blocked the project outright.
5. **The labelling tool reveals the model's guess only after I commit my own** —
   otherwise it anchors the labeller and inflates the reported agreement.
6. **Escalation by explicit policy table, not LLM discretion** — auditable, with a
   stated reason per decision. Maintained on evidence: `account_access` flipped to
   escalate because a human escalated 89% of them, tuned on dev, measured on
   holdout.
7. **The DM-deflection policy is a prompt instruction, not something inferred from
   retrieved examples** — and `is_deflection` is its own boolean, because averaging
   it into quality hides the failure it exists to catch.
8. **Kept the earlier-but-wrong results rather than overwriting them** — a
   take-home that only reports its successes isn't showing how it decides.
9. **Repaired the labels wholesale, and refused to label what couldn't be
   labelled** — all 189 re-swept (not just disagreements, which biases upward by
   2.5 points); the 15 unscorable rows excluded rather than guessed.
10. **Froze a 60/40 dev/holdout split before any tuning**, so prompt and policy
    experiments couldn't quietly contaminate the reported number.
11. **Rejected the rules-in-prompt variant and the 6× larger model** — both
    measured, both worse, both documented. The negative result was the more
    informative one.
12. **Wrote a deterministic reply guard instead of trusting the prompt** — the
    prompt already forbade placeholders; `@123456` still reached 12% of drafts.
    The guard never substitutes an invented name: a fabrication is worse than a
    gap.
13. **Treated the prediction cache as evaluation infrastructure** — namespaced
    `taxonomy:prompt:model` (the old namespace would have faked perfect A/B
    agreement), and because predictions sit next to labels, evaluation needs no
    API at all.
14. **Embedding retrieval as an opt-in second implementation, not a replacement** —
    it retrieves better but hasn't been shown to write better replies, and the
    `retriever` column exists so the two can never be averaged together by accident.
15. **Every long-running script is append-as-you-go and resumable** — an early
    write-at-the-end script lost ~150 classifications to a quota cap; nothing has
    written at the end since.

---

## AI assistance

Built with an AI coding assistant (Anthropic's Claude, via Claude Code) used
throughout — writing and reviewing code, and drafting and editing this report.
The brief permits this and asks that borrowed work be cited; this is that
citation.

The one place it affects a result: **198 of the 250 labels in the p3 evaluation
set are AI drafts** that I reviewed before accepting, after ~200 hand labels
were lost. The drafter had read the same boundary rules p3 carries and p1 does
not, so that comparison carries a known bias — which is part of why its verdict
is reported as INCONCLUSIVE (§7.4). The **189-row golden set behind the 81.0%
headline was hand-labelled by me**, and the 53 re-adjudicated labels are my own
calls with one reader (§6.2). The taxonomy, escalation policy, rubric revisions
and decision log are mine.

---

*Full detail, all experiment scripts, and the complete decision history:
`README.md` in the repo.*
