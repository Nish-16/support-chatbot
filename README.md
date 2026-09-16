# Hiver Support Agent — DropboxSupport

An AI customer-support agent for **DropboxSupport**, built from the
[Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
Kaggle dataset. It (1) classifies an incoming customer tweet into one of 13
intents, (2) drafts a reply grounded in how this brand has historically
resolved similar issues, and (3) decides auto-handle vs. escalate, with a
stated reason.

This README is also the report: framing, results, failure analysis, the
mandatory "what's misleading" section, next steps, and the decision log are
all below.

**Headline: 81.0% intent accuracy** (141/174, 95% CI [74.6%, 86.2%]).
Read §5 before quoting that — the short version is that the classifier never
improved; the *evaluation* was wrong and got corrected.

---

## Reproduce the headline in under 15 minutes

Every prediction is stored next to its label, so evaluation makes **no API
calls** and needs no Kaggle account — `data/dropbox_paired.csv` is committed.
The whole path below runs in about 5 seconds.

```bash
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt   # pandas, scikit-learn, groq, python-dotenv, tenacity

./.venv/Scripts/python.exe scripts/12_evaluate.py        # headline + baselines + caveats
./.venv/Scripts/python.exe scripts/10_baselines.py       # baselines on their own
./.venv/Scripts/python.exe scripts/13_reply_eval.py --report-only   # reply quality
./.venv/Scripts/python.exe scripts/14_judge_agreement.py --report   # judge vs human ratings
./.venv/Scripts/python.exe scripts/11_label_consistency.py          # golden-set audit
./.venv/Scripts/python.exe -m unittest discover -s tests -v         # taxonomy rules + prompts, no API
```

Useful variants:

```bash
scripts/12_evaluate.py --split holdout                        # 77.1%, never tuned on
scripts/12_evaluate.py --split dev                            # 83.7%
scripts/12_evaluate.py --label-col human_intent --keep-unscorable   # reproduces the old 62.4%
```

Anything that calls the LLM (classification, drafting, judging) needs
`GROQ_API_KEY` in `.env` — see `.env.example`. None of the commands above do.

## Try the agent on your own message

The evaluation above is API-free and reads stored predictions. To watch the
agent actually run — classify, route, then draft — use the demo. Both need
`GROQ_API_KEY`, and cost 1 classify + 1 draft call per message (plus one more
if the placeholder guard has to regenerate):

```bash
./.venv/Scripts/python.exe scripts/30_demo.py                    # interactive prompt
./.venv/Scripts/python.exe scripts/30_demo.py --text "..."       # one-shot
./.venv/Scripts/python.exe scripts/30_demo.py --text "..." --json    # machine-readable
./.venv/Scripts/python.exe scripts/30_demo.py --no-reply         # classify + route only, 1 call

./.venv/Scripts/python.exe scripts/31_serve.py                   # web UI at http://127.0.0.1:8000
```

Both run the same production functions the evaluation scores — `classify_message()`
→ `escalation.decide()` → `draft_reply()` — so what you see is what §3 measured.
The web frontend is React from a CDN over a stdlib `http.server`, with no build
step and no extra dependency: `web/index.html` is the whole frontend, and
`31_serve.py` adds one JSON endpoint. It binds to localhost only.

Example — *"I was charged twice for my Plus plan this month and I want a refund"*
→ `billing_subscription` (0.98) → **escalate**, "Financial transaction / refund
authorization needs human verification" → a draft asking for the ticket number
and account email, exactly as the DM policy in §1 requires.

The embedding retriever is optional and kept out of that path on purpose, so
the headline stays reproducible with two libraries and no downloads:

```bash
./.venv/Scripts/python.exe -m pip install chromadb
./.venv/Scripts/python.exe scripts/vector_retrieval.py --build   # ~1 min, 83MB model
./.venv/Scripts/python.exe scripts/24_retrieval_ab.py            # TF-IDF vs embeddings
./.venv/Scripts/python.exe scripts/09_draft_reply.py --text "..." --retriever vector
```

The final retriever comparison in §3 is also API-free — it reads stored judge
scores:

```bash
./.venv/Scripts/python.exe scripts/25_reply_ab_stats.py --list
./.venv/Scripts/python.exe scripts/25_reply_ab_stats.py --a tfidf:dedup1:t0 --b vector:dedup1:t0
./.venv/Scripts/python.exe scripts/25_reply_ab_stats.py --a tfidf:dedup1:t0 --b tfidf:dedup1:filt1:t0 --keep first
./.venv/Scripts/python.exe scripts/25_reply_ab_stats.py --a vector:dedup1:filt1:t0 --b vector:dedup1:filt1:t0:rep2   # noise check
```

The escalation experiments in §4.6 re-read their cached classifier runs, so
`--report-only` makes no API calls:

```bash
./.venv/Scripts/python.exe scripts/26_escalation_uncertainty.py --report-only          # top-2 / disagreement, holdout
./.venv/Scripts/python.exe scripts/27_escalation_signals.py --report-only              # signal triggers, dev
./.venv/Scripts/python.exe scripts/27_escalation_signals.py --split holdout --rule A+repeated_failure --report-only
```

The p1-vs-p3 prompt experiment (§6, item 4) runs on its own frozen set of new
tweets, scored on the first 250 (the assignment's cap on hand-labelled
examples). Each step refuses to run if the set, either prompt, or any existing
evaluation file has changed:

```bash
./.venv/Scripts/python.exe scripts/28_p3_eval.py --verify    # integrity checks, no API
./.venv/Scripts/python.exe scripts/28_p3_eval.py --status    # progress; never shows predictions
./.venv/Scripts/python.exe scripts/28_p3_eval.py --label     # blind hand labelling, resumable
./.venv/Scripts/python.exe scripts/28_p3_eval.py --run       # p1 + p3 predictions, resumable
./.venv/Scripts/python.exe scripts/28_p3_eval.py --report    # paired comparison + verdict, once complete
./.venv/Scripts/python.exe scripts/29_p4_eval.py --run       # p4 on the same set (in-sample; incomplete, §6)
```

Rebuilding `data/golden_context.csv` needs the raw `data/twcs/twcs.csv`
(`27_escalation_signals.py --build-context`). A fresh classifier run needs
`GROQ_API_KEY`.

---

## 1. Problem framing

### What "good" means for this brand

The defining property of this dataset is that **DropboxSupport's public
replies are mostly not answers.** A large share are "sorry about that, please
DM us" — the resolution happens off-thread and never appears in the data. An
agent grounded in retrieval over these threads therefore learns to deflect
*everything*, which scores well on any similarity metric and is useless as a
support agent.

So "good" here is not "sounds like DropboxSupport." It is an explicit policy,
enforced in the drafting prompt rather than left for the model to infer from
examples:

- **Informational intents** (`how_to_usage`, `feature_request`) — the agent
  must give the real answer or a documentation pointer. Deflecting to DM is a
  *failure*, not a valid auto-handle.
- **Account-specific / PII intents** (`account_access`, `billing_subscription`,
  `data_loss_recovery`, `security_account_compromise`,
  `storage_quota_plan_limits`) — asking for a DM is correct, since identity
  cannot be verified in a public reply, but the agent must say exactly what to
  send ("DM us your ticket number and the email on the account"), never a bare
  "please DM us."

`is_deflection` is scored as its **own boolean**, separately from the 1–5
quality score, because averaging it into a quality number hides exactly the
failure the policy exists to prevent. Verified to actually bind: on a
`how_to_usage` query whose *top retrieved example* was a bare "we've replied
to your DM!", the agent still produced real link-sharing instructions.

Second: the two escalation errors are not symmetric. Auto-handling something
that needed a human is expensive; escalating something that could have been
auto-handled costs a few minutes of an agent's time. Every escalation number
below is split along that line rather than reported as one accuracy.

### Brand selection

The biggest brands (Amazon, Apple, Uber) have 100K+ replies — too much for a
subsampled take-home. **DropboxSupport** has 5,938 customer↔reply pairs: large
enough to be real, small enough to process end to end, and its support domain
(login, sync, billing, storage, sharing) parallels Hiver's own product.

### Intent taxonomy

13 intents, **topic only**, plus a separate `turn_type` field (5 values) and
boolean flags (`wants_human`, `legal_sensitive`, `churn_threat`,
`abusive_content`, `needs_human_triage`, plus `sentiment` and `language`).

The first taxonomy had 8 intents and conflated three different things in one
enum — `followup_ticket_status` was a *turn type* wearing an intent's clothes,
so every mid-thread tweet lost either its topic or its position. Restructured
**before** hand-labelling started, deliberately: changing a taxonomy afterwards
means relabelling every example by hand.

### What I chose not to build

- **Stitching multi-part replies.** Many historical replies are thread
  fragments (`1/2`, `2/2`). Retrieval returns them as-is — which is also why it
  can hand the drafter two fragments of the same thread as two separate
  grounding examples. Stitching would improve grounding; stated as a limitation
  rather than silently patched.
- **The full 5,938-row classification run.** It costs real API budget and adds
  nothing the golden set doesn't already show. Keeping evaluation API-free
  matters more for a reviewer reproducing this.
- **A hosted vector database.** Chroma runs embedded, as a file in
  `data/chroma/`. At 5,938 rows the thing a vector DB actually sells you — an
  approximate index, so you don't scan everything — solves a problem this
  corpus doesn't have. Persistence and a stable query API were worth having;
  a server was not.
- **Fine-tuning.** No labelled data exists up front; the labels I built are an
  *evaluation* asset, and 174 rows is far too few to train on.
- **Hand-verifying sentiment and language.** Collected from the model, not
  hand-checked — two more open-ended judgment calls across 200 examples, for
  signals peripheral to intent accuracy and the escalate decision.

---

## 2. How it works

- **Classification** — a prompted LLM call (Groq `openai/gpt-oss-20b`) against
  the fixed category list in `intents.json`, structured JSON output, cached
  permanently by tweet id.
- **Reply drafting** — retrieval over past DropboxSupport threads, then the
  LLM drafts a reply grounded in them, with the deflection policy above stated
  explicitly in the prompt. A deterministic guard (`reply_guard.py`) then
  validates the draft, regenerates once if it contains a placeholder, and
  sanitises as a last resort. Two retrievers are implemented behind one
  interface and selected with `--retriever`: TF-IDF (`retrieval.py`, the
  default) and embeddings (`vector_retrieval.py`, Chroma + MiniLM). They are
  compared in §3.
- **Escalation** — an explicit two-stage policy table (`escalation.py`), not
  LLM discretion. Flag/`turn_type` overrides fire first and short-circuit
  (`needs_human_triage`, low confidence, `disputing_prior_answer`,
  `legal_sensitive`, `wants_human`, `churn_threat`, `abusive_content`), then a
  13-intent default table. Every decision carries a stated reason.

  Escalate by default: `data_loss_recovery`, `billing_subscription`,
  `security_account_compromise`, `phishing_abuse_report`, `account_access`.
  Auto-handle: `sync_app_bug`, `service_outage`, `sharing_permissions`,
  `storage_quota_plan_limits`, `how_to_usage`, `feature_request`,
  `complaint_dissatisfaction`. No-op: `no_action_needed`.

  `account_access` was originally auto-handle and the golden set proved it
  wrong — a human escalated 89% of those tweets — so it was flipped. Tuned on
  the 60% dev split, measured on the 40% never touched.

- **Baselines** were defined *before* the LLM agent was built, so the results
  table isn't retrofitted.

LLM provider is **Groq**. The project started on Gemini and migrated: Gemini's
free tier capped at 20 requests/day, then 500/day on the lighter model, which
was hit partway through building the golden-set candidate pool.
`scripts/gemini_lib.py` is kept in the repo, unused — deleting it would erase
the evidence of why the stack looks like this.

---

## 3. Results vs. baselines

All three approaches are measured on the **same 174 rows** with the same
labels. (An earlier version of this table scored the baselines against one
label version and the LLM against another. That comparison was invalid; see
§5.)

### Intent classification

| approach | accuracy | note |
|---|---:|---|
| Trivial — always the majority class | 12.1% | always `feature_request` |
| Simple — TF-IDF + LogisticRegression | 40.8% | 5-fold cross-validated, never scored on its own training rows |
| **LLM — Groq `openai/gpt-oss-20b`, zero-shot** | **81.0%** | 95% CI [74.6%, 86.2%] |

On the frozen 60/40 partition: **dev 83.7%** (87/104), **held-out 77.1%**
(54/70). The held-out number is the one to trust.

Best-handled intents are the ones where being wrong is expensive:
`phishing_abuse_report` F1 1.00, `data_loss_recovery` 0.90,
`sharing_permissions` 0.90, `service_outage` 0.90. Worst:
`complaint_dissatisfaction` 0.64, `how_to_usage` 0.67 — see §4.

### Escalation decision (model intent → policy table → decision)

**79.9% end-to-end**, but the split matters more than the total:

| | count | share |
|---|---:|---|
| Missed escalations (auto-handled, human would escalate) | **11** | 16% of the 67 a human escalated — **the expensive error** |
| Over-escalations (escalated, human would auto-handle) | 24 | 22% of the 107 a human would auto-handle — cheap |

### Reply quality (LLM-as-judge)

Judged by `qwen/qwen3.8-27b`, a **different model family** from the drafter,
so self-preference bias is tested rather than assumed. Two rubric versions were
run and both are reported, because the choice between them moves the number by
more than a point:

| system | v1 overall | v1 deflection | v2 overall | v2 deflection |
|---|---:|---:|---:|---:|
| **Grounded LLM agent** | **4.87** | **0%** | **3.60** | **40%** |
| Simple (1-NN retrieval) | 2.10 | 37% | 1.56 | 74% |
| Trivial (one canned reply) | 3.69 | 35% | 1.80 | 100% |

Rubric v1 had a loophole: it accepted any reply that named an artifact to send,
and the fixed canned reply happens to name one ("DM us your account email"), so
a string sent identically to every customer scored 3.69/5. Rubric v2 adds an
engagement test — naming an artifact is necessary but not sufficient, and a
reply that would read identically for any customer with any problem caps at 2.
v2 correctly collapses the canned baseline to 1.80 and calls it a deflection
100% of the time.

**v2 is the more honest rubric and is the primary one**, but it also tangles
engagement with factual correctness and marks some replies "generic" that do
give a real answer and a real link. The true figure is probably between the
two, closer to v2. The ordering — agent > trivial > 1-NN retrieval — holds
under both, and is the claim I'd actually stand behind.

**Judge validation.** Two independent judges on the same 37 replies: **68%
exact agreement** on the 1–5 score, **84% within one point**, 84% agreement on
the deflection boolean.

**Judge vs. human** (`14_judge_agreement.py --report`): I rated 30 replies
blind — system hidden, judge score hidden, rows shuffled — scoped to the
reported judge (`qwen/qwen3.8-27b`) and rubric (v2):

| measure | value |
|---|---:|
| Spearman correlation, 1–5 score | **+0.73** |
| Accept (≥4) vs. reject agreement | **87%**, Cohen's κ +0.69 |
| Exact score / within one point | 33% / 77% |
| Judge minus human, mean | **−0.60** (judge harsher) |
| Deflection-flag agreement | 67% |

Per system, judge vs. me: agent 3.21 vs 3.71 (n=14), 1-NN 1.31 vs 2.15
(n=13), canned 1.67 vs 1.67 (n=3). The judge **ranks** replies the way a human
does and makes the same accept/reject call, so the ordering in the table above
stands. Its **absolute** scores run about half a point low, and its deflection
flag agrees with mine only two times in three — so the deflection percentages
are the least trustworthy numbers in this section. See §5.10.

### Retrieval: TF-IDF vs. embeddings

Grounding quality depends on retrieving the right past threads, so the two
retrievers are measured rather than argued about. `24_retrieval_ab.py` indexes
the same 174 golden tweets in both, queries each with every tweet (excluding
itself), and asks whether the neighbours returned share the query's
hand-adjudicated intent.

| retriever | P@1 | P@3 | MRR | index build | query |
|---|---:|---:|---:|---:|---:|
| TF-IDF | 0.356 | 0.276 | 0.511 | 0.02s | 0.7ms |
| **Embeddings (Chroma + MiniLM)** | **0.540** | **0.475** | **0.677** | 5.7s | 197ms |

Chance precision@3 is 0.081, so both beat random; embeddings beat TF-IDF by
**+72% relative on P@3**.

The case that motivated it, scored on the full 5,938-row indexes:

> *"Did the taskbar icon change? I'm missing the green check, now see only the
> white box…"* vs. *"why did you remove the green check that signs everything
> is alright…"*
>
> TF-IDF **0.228** · Embeddings **0.530**
> (TF-IDF read 0.125 before the corpus was collapsed to one row per customer
> tweet; TF-IDF scores move with the corpus they are fit on, the gap is the point.)

Near-identical meaning, almost no shared vocabulary. TF-IDF scores them as
unrelated, which is the failure mode the golden-set consistency audit hit and
could not explain — that audit compares *wording*, so it structurally cannot
find pairs like this.

**The cost is real and mostly latency.** 197ms vs 0.7ms per query is ~270×,
and almost all of it is the ONNX forward pass embedding the *query* — not
searching the index, which is trivial at this size. That cost is fixed per
query regardless of corpus size, and it is invisible next to the LLM call that
follows it (~1-2s). It also adds an 83MB model download.

#### Final result: better neighbours, no demonstrated reply-quality improvement

**Retrieval experiments are finished.** Vector retrieval substantially improves
retrieval relevance (**P@3 0.276 → 0.475**), but **no downstream reply-quality
improvement was demonstrated.**

The final run (2026-09-13) re-drafted and re-judged the same 40 stratified
golden tweets under each configuration: `13_reply_eval.py --systems llm`, judge
`qwen/qwen3.8-27b`, rubric v2, drafter requested at `temperature=0`. Compared
with `25_reply_ab_stats.py` — paired by tweet, 95% bootstrap interval on the
mean `overall` difference (B − A):

| comparison | n | Δ overall | 95% CI |
|---|---:|---:|---:|
| Vector vs TF-IDF | 39 | −0.28 | [−0.85, +0.26] |
| Vector + generic-reply filter vs TF-IDF | 39 | −0.26 | [−0.77, +0.26] |
| Vector + filter vs plain vector | 40 | +0.03 | [−0.40, +0.45] |
| TF-IDF + filter vs TF-IDF | 39 | −0.28 † | [−0.62, −0.03] |
| **Same configuration run twice** (vector + filter) | 10 | **+0.30** | [0.00, +0.90] |

† Seven tweets under TF-IDF + filter were judged twice when a rate-limited run
was resumed. Six got the same score both times; one (2032832) got 2, then 5.
Keeping the first verdict gives the −0.28 [−0.62, −0.03] above (`--keep
first`); keeping the last, the script's default, gives −0.21 [−0.49, +0.03].
Every other row has no re-scored tweets and is identical under both rules.

**The same-configuration rerun moved the score by +0.30 — approximately the
same magnitude as every retriever and filter difference in the table.** This
experiment therefore cannot establish a meaningful downstream reply-quality
difference in either direction. That rules out "vector writes worse replies"
as firmly as "vector writes better ones". The one interval that excludes zero
(TF-IDF + filter) excludes it by 0.03, only under one of two defensible
duplicate-resolution rules (†), and is smaller than the noise check — it is not
reported as harm.

**`temperature=0` did not make Groq generation deterministic.** A three-call
probe on one tweet before the run returned identical drafts, but in the actual
rerun **0 of 10 replies were identical**. The +0.30 above is therefore the real
noise floor of this harness, not a temperature-0.3 artefact. Detecting a
0.3-point effect at 80% power would take roughly 180–250 paired tweets (from
the observed spread of paired differences).

The generic-reply filter (`retrieval.is_generic_reply`) skips past replies with
nothing reusable in them — "we've replied to your DM!", "we'll pass your
feedback along", dated outage notices. It flags 14.6% of the corpus, and 13–14%
of the neighbours either retriever hands the drafter. It visibly fixed the
green-check example (the vector draft regained "white = synced, grey = not
connected"), and in aggregate did nothing measurable. An interim read at 20
tweets showed +0.45 for the filter on the vector path; at 40 tweets it was
+0.03. Interim reads are not reported for that reason.

> **Decision: keep TF-IDF as the default. Keep the generic-reply filter OFF.
> Keep the vector implementation and experiment code for provenance, but do not
> enable it by default. Stop retrieval experiments.**
>
> TF-IDF stays on cost, not on a quality claim: equal demonstrated reply
> quality at ~300× lower query latency, no 83MB model download, and no
> `chromadb` dependency in the reproduce path.

Provenance: rows are stamped `retriever:version[:filt1][:t0][:tag]` in
`data/reply_evals.csv` and never pool across labels; historical rows were not
modified. The run lost 75 rows to Groq rate limits (per-minute bursts, then the
drafter's 200k tokens/day cap) and was resumed; one TF-IDF tweet failed twice,
hence n=39. Resumed runs wrote a few duplicate rows; the stats script keeps the
last verdict per tweet.

#### Earlier runs, kept for provenance

The first comparison (19 tweets, drafter at `temperature=0.3`) gave
**−0.32 overall** for vector. The control that comparison needed — **TF-IDF
against itself**, second run — gave −0.26:

| llm arm, paired, delta | relevance | grounding | overall |
|---|---:|---:|---:|
| TF-IDF vs **vector** | −0.32 | −0.37 | **−0.32** |
| TF-IDF vs **itself, re-run** | −0.26 | −0.21 | **−0.26** |

The `simple` and `trivial` arms don't use the retriever; their replies were
byte-identical across runs and the judge at `temperature=0` returned exactly
the same scores (delta 0.00 on all five metrics), so the movement was the
drafter's, not the judge's. That result motivated the final run above.

#### What the dedupe fix did and didn't do

The first version of this comparison had a real defect. The corpus held 5,938
rows but only **4,504 unique customer tweets**, because a reply split across
several tweets is several rows sharing one customer tweet. Those rows rank
adjacently, so k=3 spent slots on the same situation twice — 7 of 12 embedding
queries returned a duplicate.

`load_pairs()` now collapses the corpus to one row per customer tweet and
stitches that tweet's replies in `brand_created_at` order — timestamps from the
data, not an assumption about row order. Both retrievers share it.

| | before | after |
|---|---:|---:|
| distinct neighbours of 3 (vector) | 2.17 | **3.00** |
| queries returning a duplicate | 7/12 | **0/12** |
| grounding characters supplied | 395 | **596** |

Mechanically it worked. **It changed reply scores by less than the noise floor**
(vector pre vs post: −0.05 overall). So the duplicate neighbours were a genuine
defect worth fixing, and they were *not* the explanation for anything in the
score table — a hypothesis of mine that the data declined to support.

**So TF-IDF remains the default**, on cost rather than quality: it is 150×
faster per query, needs no 83MB download, and keeps `12_evaluate.py` and the
headline reproducible with nothing but pandas and scikit-learn. Embeddings
retrieve measurably better neighbours (§ above) and that advantage has not been
shown to reach the replies — with an instrument that currently could not see it
if it were there.

---

## 4. Golden evaluation set

189 hand-labelled examples drawn from a 207-tweet candidate pool, then fully
re-adjudicated (53 labels changed). 174 are scorable; 14 are marked
`insufficient_context` and 1 `ambiguous`, and they are excluded from the
headline rather than given a label nobody can defend.

- **Sampling** — a pure random draw (~150) to represent real traffic
  distribution, plus a rare-intent top-up that keeps sampling until every
  intent clears 15 examples. This over-samples rare intents by construction,
  which is a caveat on the headline (§5.8), not an accident.
- **Anti-anchoring** — the labeller enters their own label *before* the
  classifier's suggestion is revealed. Showing it first would inflate the very
  agreement number being reported.
- **Self-consistency audited** (`11_label_consistency.py`) — though that audit
  compares wording, not meaning, and found nothing where reading found a real
  conflict immediately: two near-identical tweets about the missing green
  check icon carried different labels at a TF-IDF cosine similarity of 0.153.

### Failure analysis — top 5, and the escalation follow-up (§4.6)

#### 1. `complaint_dissatisfaction` is a stance, not a topic (recall 0.54 — the worst)

The model routes on **topic**; a complaint is defined by **stance**. When a
tweet has both, the two disagree.

> `1856823` — *"Just crazy the new prices for @Dropbox plans. Geez! 😐"*
> → model `billing_subscription`, label `complaint_dissatisfaction`
>
> `847909` — *"In Brazil is so expensive this service. You ought to review this price."*
> → model `billing_subscription`, label `complaint_dissatisfaction`
>
> `1960138` — *"I don't like the new tray icon, every time I see it I think there's something wrong with it"*
> → model `complaint_dissatisfaction`, label `feature_request`

**Hypothesis.** This is the taxonomy's original sin repeating one level down.
The first taxonomy was rebuilt precisely because it crammed topic and
conversational position into one enum — and `complaint_dissatisfaction` crams
*stance* into the same enum in exactly the same way. Every complaint is also
about something. The fix isn't prompt engineering; it's making sentiment a flag
(which already exists) and letting intent stay topical.

#### 2. A single tweet does not contain enough context to route

15 of 189 rows (7.9%) cannot be classified by anyone, model or human, from the
tweet alone:

> `2828924` — *"I Have done this. Too bad it didn't solve the problem"*
>
> `1972693` — *"350/350, no firewalls or anything in between."*
>
> `1644499` — *"Same here!"*

Before this was caught they were quietly filed under `no_action_needed` and
`complaint_dissatisfaction`, inflating both.

**Hypothesis.** Conversational position is not recoverable from an isolated
message. The same missing input causes two further failures: `turn_type`
appears to score 86% purely by predicting `first_contact` for 186 of 189 rows
(§5.4), and the `disputing_prior_answer` escalation override can never fire.
**`dropbox_paired.csv` already has a `brand_text` column and nothing reads it**
— that is the highest-value outstanding change, and it fixes all three at once.

#### 3. Symptom vs. object in the sync / quota / how-to cluster

> `2976667` — *"I don't seem to have enough space on my laptop to connect my work Dropbox. If I keep my external hard drive plugged in, will this solve the problem?"*
> → model `sync_app_bug`, label `how_to_usage` (it's a question, and the disk is the *device's*, not the Dropbox quota)
>
> `1901120` — *"Earned 25gb of space via HP Promotion. Is there any expiry?"*
> → model `how_to_usage`, label `storage_quota_plan_limits`
>
> `2028354` — *"just installed DB on my acer windows 10 and almost ran out of space. Couldn't selective sync, deinstalled but files intact"*
> → model `storage_quota_plan_limits`, label `how_to_usage`

**Hypothesis.** These confusions run in **both directions**, and a
bidirectional confusion is a boundary problem, not a model problem — the model
is being asked to choose between categories that genuinely overlap. The worst
per-intent F1 scores all sit in this cluster. The fix is the taxonomy, not the
prompt.

#### 4. `security_account_compromise` is a keyword magnet (precision 0.60, recall 1.00)

It catches everything it should and a lot it shouldn't. Any mention of
unauthorized access, permissions or logins pulls a tweet in:

> `1529752` — *"When you report a questionable login to an account's owner please provide the IP address of the suspicious login."*
> → model `security_account_compromise`, label `feature_request` (a product suggestion; nothing is compromised)
>
> `2179735` — *"How can you allow Team Admin to have the power of erasing my personal data without my permission…"*
> → model `security_account_compromise`, label `sharing_permissions`

**Hypothesis.** The intent name matches surface vocabulary rather than
situation, and the model has no way to separate "my account was compromised"
from "here is a suggestion about your compromise notifications." Recall 1.00
with precision 0.60 is the *safe* direction of this error — the policy
escalates security, so these become over-escalations rather than missed ones —
but it is the single largest contributor to the 24 over-escalations.

#### 5. The escalation policy has no abstain path — both uncertainty branches are dead

| branch | reality |
|---|---|
| `needs_human_triage` | a human set it 12 times; the model has set it **0 times** |
| `confidence < 0.5` | **all 174 scorable rows sit above 0.5** |

The second is provably dead code: the policy scores **identically** with the
confidence branch enabled and removed — 11 missed, 24 over, both ways.

**Hypothesis.** A model asked to self-report confidence on a forced-choice task
reports high confidence: 96% of rows land at ≥0.85, where accuracy is flat
(63% vs 64%). It is confidently wrong about as often as confidently right. A
usable signal has to come from somewhere other than the model's opinion of
itself — and **cross-model disagreement works**: on the 145 rows where
`gpt-oss-20b` and `gpt-oss-120b` agree, accuracy is 84.8%; on the 29 where they
disagree, 62.1%. That is **22.8 points of separation**, from data I already
had, against a self-reported confidence that separates nothing. It is not
shipped, because as a policy it's a genuine trade: missed escalations 11 → 7,
but over-escalations 24 → 38 and the escalate rate rises to 56%. An agent
escalating over half its traffic is a product decision, not an engineering one.

#### 6. The escalations the policy misses are not uncertainty — they are urgency and repeat failure

**Two uncertainty signals, tested on the held-out 70** (`26_escalation_uncertainty.py`,
2026-09-13). Each adds one trigger to the current policy:

| Policy | Escalated | Risky-intent cases caught | Missed human escalations | False alarms |
|---|---:|---:|---:|---:|
| A — current | 57.1% (40) | 16/16 | 4 of 29 | 15 of 41 |
| B — escalate if intent or runner-up intent is risky | 52.9% (37) | 16/16 | 5 | 13 |
| C — escalate unless 3 samples at temperature 0.7 agree | 65.7% (46) | 16/16 | 2 | 19 |
| D — C, or any sample is risky | 65.7% (46) | 16/16 | 2 | 19 |

"Risky" means billing, data loss or account compromise. The current policy
already escalated all 16 held-out cases (95% CI on recall 81–100%). B's
extra field changed the primary answer: accuracy fell from 77.1% to 72.9%.
Self-disagreement ranked wrong intents no better than self-reported confidence
(AUROC 0.58 vs 0.64, overlapping intervals).

The real finding was in the 4 misses. **The intent was correct on every
one.** The human escalated because of what the customer said around the topic:
"**Emergency**", "**my whole team**", "another bug and **another ticket**",
"WTF… I delete loads… still full". No intent-uncertainty signal can see that.

**So the classifier was asked for it directly** (`27_escalation_signals.py`):
three booleans, `urgent`, `wide_impact` and `repeated_failure`, plus, when a
tweet replies to an earlier one, that earlier message as context. That context
is the `in_response_to_tweet_id` parent from the raw dataset
(`data/golden_context.csv`, 50 of 189 golden tweets). It is **not**
`brand_text`, which is DropboxSupport's reply *to* the tweet, written
afterwards; using it would leak how support handled the case. Each signal is
an optional trigger in `escalation.decide(signal_triggers=...)`, and none is
on by default.

**Dev split, 104 rows, all rules scored on the same classifier run:**

| Rule | Topic acc | Escalated | Missed human esc. | False alarms | Δ missed | Δ false alarms |
|---|---:|---:|---:|---:|---:|---:|
| A | 77.9% | 42 (40.4%) | 7 (18.4%) | 11 (16.7%) | — | — |
| A + urgent | 77.9% | 45 (43.3%) | 7 | 14 | 0 | +3 |
| A + wide_impact | 77.9% | 44 (42.3%) | 7 | 13 | 0 | +2 |
| **A + repeated_failure** | 77.9% | 52 (50.0%) | **3 (7.9%)** | 17 (25.8%) | **−4** | +6 |
| A + all three | 77.9% | 55 (52.9%) | 3 | 20 | −4 | +9 |

`urgent` and `wide_impact` caught **nothing** A missed and only added false
alarms. `repeated_failure` did all the work. The selection rule, fixed before
the run, was: fewest misses, adding at most 2 false alarms per miss removed.
It chose **A + repeated_failure**, at 1.5 false alarms per catch. Risky-intent
misses were 3 of 22 under every rule, all of them intent errors
(billing → `how_to_usage`).

**The new prompt costs topic accuracy — 83.7% → 77.9% on dev.** It changed 17
intents: 5 fixed, 11 broken. The loss is worst on tweets that carry context
(82.8% → 72.4%, 29 rows), but also present without it (84.0% → 80.0%). The two
additions are confounded; separating them would take another run. Applying
the same triggers to the **production** intent instead gives the same
escalation gain (misses 7 → 3, false alarms 9 → 15) with no accuracy loss.
That comparison was added after the first dev run and was not used for
selection. The implication for shipping: **ask for the signals in a separate
call rather than inside the intent prompt**, or fix the prompt first. This
repeats the lesson from B, where adding one output field also moved the
primary answer.

**Held-out split, one read of the selected rule, 70 rows:**

| Rule | Topic acc | Escalated | Missed human esc. | False alarms |
|---|---:|---:|---:|---:|
| A (same run) | 75.7% | 37 (52.9%) | 5 of 29 (17.2%) | 13 of 41 (31.7%) |
| **A + repeated_failure** | 75.7% | 40 (57.1%) | **5 (17.2%)** | 16 (39.0%) |
| A on the production predictions (reference) | 77.1% | 40 (57.1%) | 4 (13.8%) | 15 (36.6%) |

**It did not generalise: zero extra catches, three extra false alarms.** On
the production intent the result is the same (misses 4 → 4, false alarms
15 → 18). `repeated_failure` fired on 6 held-out tweets, and none was a case A
missed. It did not fire on "another bug and another ticket" (267212), the
clearest repeat-failure miss the previous experiment surfaced. Dev's −4 was 4
rows out of 38 human escalations. At that size a noisy signal can produce a
win like that by chance, and the holdout is the check that exposed it.

Two things the holdout shows, **not acted on**, because acting on them would
tune on the held-out set:
- `wide_impact` flagged 1804904 ("my whole team"), the one held-out miss any
  signal reached. It caught nothing on dev and was not selected.
- `urgent` fired 6 times, 5 on human-escalated tweets. Those tweets were
  already escalated for other reasons, so it added no catches.

The prompt's accuracy cost was smaller here (77.1% → 75.7%; 7 intents changed,
1 fixed, 2 broken).

> **Decision: keep rule A. No signal trigger is enabled.** The signals, the
> context file and `decide(signal_triggers=...)` stay in the code, off by
> default, for provenance. This holdout has now been read for a signal rule, so
> it cannot give an unbiased read for the next one. A further attempt needs
> more labelled escalations, not another pass over these 70 rows.

---

## 5. What is misleading about my headline number

**The biggest thing: the classifier never improved. The measured number went
from 62.4% to 81.0% without a single change to the model or the prompt.** Every
prediction scored in this README is the same output from the original run. What
changed was that the evaluation became correct. I then tried twice to actually
improve the classifier, and both attempts failed and were reverted (§5.7).

1. **The whole 18.6-point move is relabelling, not capability.** I re-read the
   errors before touching the prompt and found the model was right and I was
   wrong in 22 of 48 reviewed rows. I had been labelling on *sentence shape*
   ("how do I…" → `how_to_usage`); the model labelled on *topic*, and it was
   consistent where I was not. Root cause: the taxonomy holds 36 KB of boundary
   rules and **neither consumer could see them** — the model got one terse line
   per intent, and the labelling tool showed me even less than the model. Five
   rules already written there were violated during labelling. 53 labels changed
   across a full re-sweep; every change carries the rule that decided it and a
   written rationale, and the original label file was never modified.
2. **The relabelling is mine, self-checked once, and nobody else has verified
   it.** If a reviewer disagrees with 10 of my 53 changes, the headline moves
   about 5 points. This is the largest open risk here.
3. **An adjudication that only reads disagreements is biased upward by
   construction.** Reviewing only rows where the model and I differed gave
   83.6%. Sweeping all 189 rows found 8 more where we were wrong *the same way*
   — which nothing had flagged — and every one lowers the score. That cost 2.5
   points. 81.0%, not 83.6%, is the number.
4. **`turn_type`'s 86% is an artifact.** The model predicted `first_contact`
   for 186 of 189 rows. A human found 28 that are not. It is riding the base
   rate; do not read it as capability.
5. **Confidence cannot be thresholded**, and `needs_human_triage` has never
   once fired. Two of my own design decisions, measured and found inert (§4.5).
6. **Per-intent n is 9–21.** One row moves an intent's rate by 7–11 points.
   These numbers cannot rank intents against each other.
   `security_account_compromise` (n=9) and `no_action_needed` (n=9) are
   underpowered and flagged rather than quietly averaged in.
7. **Accuracy varies 74%–88% across labelling windows** — a 15-point spread
   around an 81% headline, same classifier, same taxonomy. And two attempts to
   improve it both failed: putting the boundary rules in the prompt scored
   79.1% vs 81.4% on dev at 2.6× the token cost, and a 6× larger model
   (`gpt-oss-120b`) scored **74.1% vs 81.0%** — worse. Of the 27 rows both
   models get wrong, 22 are the *same* wrong label, which says the remaining
   errors are the taxonomy and the prompt, not model capacity.
8. **The golden pool over-samples rare intents by construction**, so 81% is not
   an estimate of accuracy on real DropboxSupport traffic mix.
9. **The reply-quality harness cannot resolve small differences.** Re-running
   one identical configuration moves `overall` by −0.26 — as much as switching
   retrievers does — because the drafter samples at `temperature=0.3`. The
   agent-vs-baseline gap (~2 points) survives that easily; nothing finer does.
   I found this only because I ran a configuration against itself as a control,
   and it invalidated a retriever comparison I had already written up.
10. **The reply judge is validated against one human, on 30 replies — me.**
   It ranks well (Spearman +0.73; accept/reject κ +0.69), but scores 0.6
   points harsher than I do and agrees on the deflection flag only 67% of the
   time (§3). Thirty ratings from a single rater who also built the rubric is
   thin evidence, not a validation study. And the judge is visibly noisy: the trivial baseline sends **one
   identical string** 98 times, and the judge scored it across the **full 1–5
   range** (std 1.51) and called it a deflection 66 times out of 98. Some spread
   is legitimate (the rubric is intent-dependent), but it bounds how finely any
   reply-quality average can be read.
11. **The reply-quality headline depends on which rubric you pick** — 4.87
    under v1, 3.60 under v2, for the same replies (§3). I report v2 and show
    both.
12. **LLM non-determinism.** Tweet `1274528` got different intents on two runs
    (0.90 and 0.85 confidence). `temperature=0` is set now but was not for every
    row. Single-run accuracy overstates stability.
13. **One cache bug nearly produced a very convincing wrong result.** The
    prediction cache was namespaced on taxonomy version alone, so the second
    model in the A/B would have read the first model's cached answers back and
    reported perfect agreement. The namespace is now `taxonomy:prompt:model`.

`12_evaluate.py` prints several of these itself, so they can't be quoted
without them.

---

## 6. What I'd do next with one more week

In priority order, most valuable first.

1. **Feed `brand_text` into the classifier** (1 day). The column already exists
   and nothing reads it. It fixes three separate failures at once: `turn_type`'s
   base-rate artifact, the 15 unroutable rows, and the dead
   `disputing_prior_answer` override. Highest-value change outstanding.
2. **Get a second reader on the 53 adjudicated rows** (0.5 day of someone
   else's time). The one thing I cannot do for myself, and the largest open
   risk in §5.
3. **A second human rater for the judge** (0.5 day). **Done for one rater:**
   30 blind ratings, Spearman +0.73, judge 0.6 points harsher (§3). What's
   left is a second rater — to separate "the judge is harsh" from "I am
   lenient" — and a look at the 10 deflection-flag disagreements, which is the
   weakest part of the judge.
4. **Prompt v3: mechanical rules as rules, judgment boundaries as few-shot
   examples** (1 day). The failed prompt experiment is more useful broken down
   than in aggregate: the *mechanical* rules worked exactly as designed (4
   specific rows fixed), while the *judgment* rules backfired — one abstract
   instruction to "prefer a specific intent over `how_to_usage`" drove two rows
   **toward** `how_to_usage`. A 20b model applies an abstract preference
   bluntly. The examples are written and staged, drawn from dev only.
   **Built 2026-09-13 as prompt version `p3`** (`scripts/taxonomy_rules.py`,
   `taxonomy.md` Part 12). The same text is now what the labeling tool shows,
   so the human and the model are given one procedure. It is **not** the
   default and **not** evaluated; that waits for newly labelled data.
   **Evaluation prepared 2026-09-14** (`scripts/28_p3_eval.py`). 300 tweets
   were frozen as `data/p3_eval_set.csv`, with a sha256 manifest, from 3,553
   eligible ones. **Amended the same day, before labelling:** only the first
   250 of that random order are labelled and scored, to stay within the
   assignment's 150–250 cap on hand-labelled examples. The cut and the state
   at that moment (1 label, no prediction viewed) are recorded in
   `data/p3_eval_amendment.json`. The set excludes every tweet any data file or the cache had
   touched (874), the reading and coverage samples, and anything quoted in the
   docs or rule examples (77). The comparison is paired, with a bootstrap CI
   and exact McNemar test, and the verdict rule was fixed before any label
   existed: p3 is better only if the interval excludes zero and it misses no
   more human escalations; not better if the interval rules out a 3-point gain.

   **Two further amendments, and how the labels were made.** About 200 hand
   labels were then lost; 52 survived (rows 1–52). The scored set was cut to
   150 — the assignment's minimum — with rows 53–150 drafted by Claude from the
   tweet, its parent message and the labeller rules, and reviewed by me (98
   accepted, 0 edited). No prediction had been viewed. **After** reading the
   n=150 result, I extended back to 250 (rows 151–250 also Claude-drafted).
   Growing a sample after seeing its result is optional stopping, so **n=150 is
   the pre-registered result** and n=250 is reported beside it. Known bias: the
   drafter read the same rules p3 carries and p1 does not.

   **Result (2026-09-14): INCONCLUSIVE at both sizes. PROMPT_VERSION stays p1.**

   | run | scorable n | p1 | p3 | paired Δ | 95% CI | fixed / broken | McNemar p | missed escalations p1 / p3 |
   |---|---:|---:|---:|---:|---:|---:|---:|---:|
   | **n=150, pre-registered** | 137 | 74.5% | 77.4% | +2.9 | [−2.9, +8.8] | 11 / 7 | 0.48 | 11 / 12 |
   | n=250, post-hoc | 228 | 71.5% | 75.0% | +3.5 | [−1.3, +8.3] | 20 / 12 | 0.22 | 17 / 18 |

   p3 leans ahead in both, but neither interval excludes zero, and p3 misses one
   more human escalation each time. Note also that p1 scores 71–75% on this
   random draw of real traffic against 81% on the rare-intent-weighted golden
   set (§5.8) — though the label procedures differ, so that gap is suggestive,
   not measured.

   **p4** (`prompt_p4.py`, `29_p4_eval.py`) adds eight boundary rules written
   from p1's and p3's errors on this same set, so any score there is in-sample
   by construction and could only justify a fresh evaluation. Its run stopped
   at 110 of 250 tweets on Groq limits; there is no p4 result.
5. **Split `complaint_dissatisfaction` into a flag** (1 day, including
   relabelling). §4.1 says this is a taxonomy problem; this is the fix.
   Sentiment is already a field.
6. **Escalation: grow the evaluation before adding triggers** (1–2 days). Four
   extra triggers have now been tried (§4.5, §4.6): cross-model disagreement,
   top-2, self-disagreement, and three urgency/impact signals. Each moved misses
   by 2–4 rows on one split, and none held up on both. With 29–38 human
   escalations per split, one row is ~3 points. Label ~150 more tweets weighted
   toward escalations; ask for signals in a separate call (inside the intent
   prompt they cost 1.4–5.8 points of accuracy); and get a real cost ratio
   between a miss and a false alarm, which is still the missing product input.
7. ~~**Deduplicate retrieved neighbours by `customer_tweet_id`**~~ **Done
   2026-09-13.** Distinct neighbours rose from 2.17 to 3.00 of 3, and reply
   scores did not move beyond the noise floor (§3). It was a real defect but
   not the explanation for the retriever result. Retrieval experiments are
   closed.
8. **Expand the golden set to ~400** with stratified sampling on the confusion
   clusters, so per-intent rates become rankable.

Not on this list, deliberately: fine-tuning, a bigger model, and a vector
store. The 120b experiment says capacity is not the constraint (§5.7), and
retrieval is not the measured bottleneck.

---

## 7. Decision log

1. **DropboxSupport, not Amazon or Apple.** The biggest brands have 100K+
   replies — too much for a subsampled take-home. Dropbox has 5,938 pairs:
   large enough to be real, small enough to process end to end, and its support
   domain parallels Hiver's.
2. **Prompted LLM, not fine-tuning.** No labelled data exists up front. The
   labels I built are an evaluation asset; 174 rows is nowhere near enough to
   train on, and spending them on training would leave nothing to measure with.
3. **Rebuilt the taxonomy from 8 intents to 13 + `turn_type` + flags, before
   labelling started.** The first version conflated topic, conversational
   position and risk in one enum. Changing it after labelling would have cost
   250 examples of hand-labelling. The timing was the whole decision.
4. **Migrated from Gemini to Groq mid-project.** Gemini's free tier capped at
   20 requests/day, then 500/day on the lighter model — hit partway through
   building the candidate pool. A day of rewrite against a ceiling that would
   have blocked the project outright.
5. **The labelling tool shows the model's guess only *after* I commit my own.**
   Showing it first would anchor the labeller and quietly inflate the very
   agreement number being reported.
6. **Escalation by an explicit policy table, not LLM discretion.** Every
   decision carries a stated reason and is auditable. Maintaining the table is
   an evidence question, not a taste one: `account_access` was flipped from
   auto-handle to escalate because the golden set showed a human escalated 89%
   of those — tuned on the 60% dev split, measured on the 40% never touched.
   The cost: intent alone doesn't determine escalation for three intents that
   sit near 50%, left as a stated finding, because fitting per-intent
   exceptions on ~113 rows is overfitting.
7. **The DM-deflection policy is an instruction in the prompt, not something
   the model infers from retrieved examples** — and `is_deflection` is scored as
   its own boolean rather than folded into the quality score, because averaging
   it in would hide the failure it exists to catch.
8. **Kept the earlier-but-wrong results rather than overwriting them.** The
   original labels, the v1 rubric scores and the two failed improvement attempts
   are all still in the repo and still reproducible from the CLI. A take-home
   that only reports its successes isn't showing how it makes decisions.
9. **Repaired the labels wholesale, and refused to label what couldn't be
   labelled.** All 189 rows were re-adjudicated, not just the disagreements:
   the disagreement-only version scored 2.5 points higher and was biased upward
   by construction (§5.3). The precedence rules were written *before*
   relabelling, which caught two rules my own data proved wrong before they
   shipped. The 15 rows still unscorable afterwards (14 `insufficient_context`,
   1 `ambiguous`) are excluded from the headline rather than given a label
   nobody can defend — reported as a finding, and `--keep-unscorable` puts them
   back.
10. **Froze a 60/40 dev/holdout split** before any tuning, so the prompt and
    policy experiments could not quietly contaminate the reported number.
11. **Rejected the prompt containing the rules, and the 6× larger model** — both
    measured, both worse, both documented (§5.7). The negative results were more
    informative than a win: 22 of 27 shared errors carry the *same* wrong label,
    which is what says the ceiling is the taxonomy.
12. **Wrote a deterministic reply guard instead of trusting the prompt.** The
    drafting prompt already said "never emit a placeholder"; `@123456` still
    reached 12% of drafts. The guard validates, regenerates once with the
    offending text quoted back, and sanitises as a last resort — never
    substituting an invented name, because filling the slot with a fabrication
    is a worse failure than leaving it empty.
13. **Treated the prediction cache as evaluation infrastructure.** Keys are
    namespaced on `taxonomy:prompt:model`: under the old namespace the model A/B
    would have read the incumbent's cached answers back and reported perfect
    agreement — the most convincing wrong result available. And because every
    prediction is stored next to its label, `12_evaluate.py` re-runs in seconds
    at zero cost, which is why the 15-minute reproduce target is achievable at
    all.
14. **Added embedding retrieval as an opt-in second implementation, not a
    replacement.** It measurably retrieves better (§3), but it costs an 83MB
    download and ~270× the query latency, and it has not been shown to improve
    the replies themselves. Putting it behind `--retriever` keeps both claims
    testable and keeps the headline reproducible without it. The `retriever`
    column in `reply_evals.csv` exists so the two can never be averaged
    together by accident — the same mistake the rubric versions nearly caused.
15. **Every long-running script is append-as-you-go and resumable.** An early
    script wrote its output once at the end, and a quota cap killed the run
    mid-way — ~150 successful classifications would have been silently lost.
    They were recovered by parsing the terminal log rather than re-spending
    quota, and nothing has written-at-the-end since.

---

## 8. Repo layout

```
data/
  twcs/twcs.csv           raw full dataset (gitignored, 516MB)
  dropbox_paired.csv      filtered DropboxSupport pairs, 5,938 (committed)
  reading_sample.txt      40-example reproducible reading sample
  classification_cache.jsonl  permanent classify-once cache
  golden_candidates.csv   golden-set candidate pool (207)
  golden_labels.csv       hand-labelled golden set, original labels (189)
  golden_labels_v3.csv    re-adjudicated labels -- the reported ones
  golden_split.csv        frozen 60/40 dev/holdout split
  adjudication_v*.csv     the row-by-row error review
  model_ab_*.csv          the gpt-oss-120b comparison run
  prompt_ab_p2_dev.csv    the rules-in-the-prompt comparison run
  reply_evals.csv         reply drafts + judge scores (judge x rubric x retriever)
  golden_context.csv      the tweet each golden tweet replies to (from twcs.csv)
  escalation_uncertainty_holdout_s3.csv  per-row top-2 / 3-sample results
  escalation_signals_*.csv               per-row signal-trigger results
  judge_agreement.csv     blind human ratings vs the judge (30)
  p3_eval_set.csv         frozen 300-tweet set for p1 vs p3 (first 250 scored)
  p3_eval_manifest.json   hashes pinning the set, prompts and prior eval files
  p3_eval_amendment.json  the 300 -> 250 -> 150 -> 250 amendment chain
  p3_eval_labels.csv      labels (1-52 hand, 53-250 Claude-drafted, reviewed)
  p3_eval_drafts.csv      the Claude label drafts
  p3_eval_results*.csv    per-row results, n=250 and the preserved n=150
  p3_eval_confusion_*.csv confusion matrices, same two sizes
  p*_eval_runs.jsonl      run logs: namespaces, prompt hashes, failures
  chroma/                 embedding index (gitignored; rebuild in ~1 min)
scripts/
  01-05_*.py              data pipeline: explore, filter, pair, sample
  groq_lib.py             Groq client, prompt, schema, retry policy
  cache.py                append-only classification cache (namespaced)
  rate_limiter.py         sliding-window rate limiter
  classify_runner.py      cached + concurrent + rate-limited orchestration
  06_classify.py          intent classifier over a sample
  07_build_golden_candidates.py  golden-set candidate sampling
  08_label_golden_set.py  interactive hand-labelling (anti-anchoring)
  retrieval.py            TF-IDF retrieval over dropbox_paired.csv
  vector_retrieval.py     embedding retrieval (Chroma + MiniLM), same interface
  reply_guard.py          deterministic placeholder guard
  09_draft_reply.py       grounded reply drafting (policy-enforced)
  escalation.py           auto-handle vs escalate policy table
  10_baselines.py         trivial + simple baselines
  11_label_consistency.py golden-set self-consistency audit
  12_evaluate.py          eval harness (no API calls) -- the headline
  13_reply_eval.py        reply drafting + LLM-judge scoring
  14_judge_agreement.py   blind human rating -> judge validation
  15-18_*.py              the evaluation repair: adjudicate, re-check,
                          apply rules, full re-sweep (original -> v3 labels)
  19_migrate_cache.py     cache namespace backfill
  20_model_ab.py          gpt-oss-20b vs 120b
  21_split.py             freeze the dev/holdout split
  22_prompt_ab.py         prompt comparison, dev only
  23_escalation_v2.py     the dead uncertainty branches
  24_retrieval_ab.py      TF-IDF vs embeddings on P@k / MRR
  25_reply_ab_stats.py    paired bootstrap stats for reply-eval A/Bs (no API)
  26_escalation_uncertainty.py  top-2 and self-disagreement escalation, holdout
  27_escalation_signals.py      urgent / wide_impact / repeated_failure triggers
  taxonomy_rules.py       taxonomy.md Part 11 as rules + boundary examples, shared
                          by prompt p3 and the labeling tool
  eval_integrity.py       pinned hashes of the evaluation state + used-tweet ids
  28_p3_eval.py           p1 vs p3 on a frozen new set: build, label, run, report
  prompt_p4.py            p4's eight boundary rules (in-sample)
  29_p4_eval.py           p4 vs p1/p3 on the same set
tests/
  test_taxonomy_rules.py  non-API checks: rules, examples, prompts, labels
  test_p3_eval.py         non-API checks: frozen set, leakage, blindness, statistics
  test_p4_prompt.py       non-API checks: p4 assembly, no overlap with eval tweets
intents.json              the 13 intents, as the model sees them
turn_types.json           the 5 turn types
```

## 9. Known limitations, stated plainly

- **Judge-vs-human agreement rests on 30 ratings from one rater.** The judge
  ranks like a human (Spearman +0.73) but scores 0.6 points harsher and agrees
  on deflection only 67% of the time. §3, §5.10.
- **The p3 prompt comparison is inconclusive, and 198 of its 250 labels are
  Claude drafts** reviewed by me, after ~200 hand labels were lost. §6.4.
- **The 53 relabelled rows have one reader: me.** §5.2.
- **`brand_text` is unused**, which caps `turn_type`, leaves 15 rows
  unroutable, and keeps one escalation override dead. §4.2.
- **The full 5,938-row classification run was never done** — deliberately, §1.
- **No escalation trigger beyond the v1 policy has survived a held-out test.**
  The policy still misses ~14–17% of human escalations. The misses are mostly
  correct-intent tweets escalated for urgency or frustration. §4.6.

## 10. Citations

Dataset: Thought Vector, *Customer Support on Twitter* (Kaggle). Models:
Groq-hosted `openai/gpt-oss-20b` (classification, drafting) and
`openai/gpt-oss-120b` (comparison run); `qwen/qwen3.8-27b` as the independent
judge. Libraries: pandas, scikit-learn (TF-IDF, LogisticRegression), tenacity,
python-dotenv, groq.

### AI assistance

This project was built with an AI coding assistant (Anthropic's Claude, via
Claude Code) used throughout: writing and reviewing code, drafting and editing
this README and `SUBMISSION.md`, and drafting the evaluation labels noted below.
The brief allows this and asks that borrowed work be cited — this is that
citation. Where it matters to a result, it is flagged at the result rather than
only here.

- **Code** — written and refactored with AI assistance throughout, then run and
  reviewed by me.
- **The 189-row golden set** — hand-labelled by me. The 53 re-adjudicated labels
  are my calls, with one reader (§5.2), and the original label file was never
  modified.
- **The p3 evaluation set** — **198 of its 250 labels are AI drafts** that I
  reviewed before accepting, after ~200 hand labels were lost. The drafter had
  read the same boundary rules p3 carries and p1 does not, so that experiment
  carries a known bias, reported wherever its result appears (§6.4, §9).
- **Judgment calls** — the taxonomy, the escalation policy, the rubric
  revisions, and the choices in the decision log (§7) are mine.
