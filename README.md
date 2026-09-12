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
./.venv/Scripts/python.exe -m pip install pandas scikit-learn groq python-dotenv tenacity

./.venv/Scripts/python.exe scripts/12_evaluate.py        # headline + baselines + caveats
./.venv/Scripts/python.exe scripts/10_baselines.py       # baselines on their own
./.venv/Scripts/python.exe scripts/13_reply_eval.py --report-only   # reply quality
./.venv/Scripts/python.exe scripts/11_label_consistency.py          # golden-set audit
```

Useful variants:

```bash
scripts/12_evaluate.py --split holdout                        # 77.1%, never tuned on
scripts/12_evaluate.py --split dev                            # 83.7%
scripts/12_evaluate.py --label-col human_intent --keep-unscorable   # reproduces the old 62.4%
```

Anything that calls the LLM (classification, drafting, judging) needs
`GROQ_API_KEY` in `.env` — see `.env.example`. None of the commands above do.

The embedding retriever is optional and kept out of that path on purpose, so
the headline stays reproducible with two libraries and no downloads:

```bash
./.venv/Scripts/python.exe -m pip install chromadb
./.venv/Scripts/python.exe scripts/vector_retrieval.py --build   # ~1 min, 83MB model
./.venv/Scripts/python.exe scripts/24_retrieval_ab.py            # TF-IDF vs embeddings
./.venv/Scripts/python.exe scripts/09_draft_reply.py --text "..." --retriever vector
```

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
the deflection boolean. That is judge-vs-judge, not judge-vs-human — see §5.9
and §6.3.

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
> TF-IDF **0.125** · Embeddings **0.530**

Near-identical meaning, almost no shared vocabulary. TF-IDF scores them as
unrelated, which is the failure mode the golden-set consistency audit hit and
could not explain — that audit compares *wording*, so it structurally cannot
find pairs like this.

**The cost is real and mostly latency.** 197ms vs 0.7ms per query is ~270×,
and almost all of it is the ONNX forward pass embedding the *query* — not
searching the index, which is trivial at this size. That cost is fixed per
query regardless of corpus size, and it is invisible next to the LLM call that
follows it (~1-2s). It also adds an 83MB model download.

#### Better neighbours did not produce better replies

The obvious follow-through — and it failed. `13_reply_eval.py --retriever
vector` re-drafted and re-judged the same tweets through the embedding path.
Paired on the 13 tweets scored under both, same judge, same rubric v2:

| llm arm | TF-IDF | vector | delta |
|---|---:|---:|---:|
| relevance | 4.17 | 3.92 | −0.25 |
| grounding | 4.92 | 4.58 | −0.33 |
| overall | 4.17 | 4.00 | −0.17 |
| deflection rate | 0.25 | 0.17 | **−0.08** |

Every quality metric moved *down*; only the deflection rate improved. So a
retriever that is measurably better at finding topically-correct neighbours
made the replies no better, and slightly worse.

**The control is what makes this readable.** The `simple` and `trivial` arms
don't use this retriever, so their replies were byte-identical across the two
runs — and the judge, at `temperature=0`, returned *exactly* the same score on
all five metrics for all of them (delta 0.00 everywhere, 100% identical
replies). The judge contributed zero variance here, so the llm arm's movement
is real, not grading noise. That control was free: it fell out of running all
three systems in both arms.

**Why, with evidence.** The corpus has 5,938 rows but only 4,504 unique
customer tweets — multi-tweet replies appear as separate rows sharing one
customer tweet. Embedding similarity puts those near-identical rows adjacent,
so the vector retriever retrieves the *same customer situation* twice far more
often, spending grounding slots on it:

| retriever | distinct neighbours (of 3) | queries returning a duplicate |
|---|---:|---:|
| TF-IDF | 2.58 | 2/12 |
| vector | 2.17 | 7/12 |

Better neighbour *ranking*, fewer distinct grounding *situations*. That is a
concrete, fixable defect — deduplicate by `customer_tweet_id` inside `top_k`
(or stitch the fragments, §1) and re-run — not a verdict on embeddings. It is
listed in §6 rather than patched, because the fix changes what both retrievers
return and every stored reply-eval row would need re-earning at API cost.

**So TF-IDF remains the default**, now for a measured reason rather than a
conservative one: it keeps `12_evaluate.py` and the headline reproducible with
nothing but pandas and scikit-learn, and the retriever that retrieves better
does not currently write better replies. n=13 is small — this rules out a large
win, not a small one.

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

### Failure analysis — top 5

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
9. **The reply judge has never been validated against a human.** Two judges
   agreeing 68%/84% is not evidence either is right — they can share a blind
   spot. Worse, the judge is visibly noisy: the trivial baseline sends **one
   identical string** 98 times, and the judge scored it across the **full 1–5
   range** (std 1.51) and called it a deflection 66 times out of 98. Some spread
   is legitimate (the rubric is intent-dependent), but it bounds how finely any
   reply-quality average can be read.
10. **The reply-quality headline depends on which rubric you pick** — 4.87
    under v1, 3.60 under v2, for the same replies (§3). I report v2 and show
    both.
11. **LLM non-determinism.** Tweet `1274528` got different intents on two runs
    (0.90 and 0.85 confidence). `temperature=0` is set now but was not for every
    row. Single-run accuracy overstates stability.
12. **One cache bug nearly produced a very convincing wrong result.** The
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
2. **Get a second reader on the 58 adjudicated rows** (0.5 day of someone
   else's time). The one thing I cannot do for myself, and the largest open
   risk in §5.
3. **Run the judge-vs-human agreement harness** (0.5 day). `14_judge_agreement.py`
   is built and waiting for input: rate ~30 replies blind, then compute
   agreement against the LLM judge. Until that exists, every reply-quality
   number in §3 rests on models agreeing with models.
4. **Prompt v3: mechanical rules as rules, judgment boundaries as few-shot
   examples** (1 day). The failed prompt experiment is more useful broken down
   than in aggregate: the *mechanical* rules worked exactly as designed (4
   specific rows fixed), while the *judgment* rules backfired — one abstract
   instruction to "prefer a specific intent over `how_to_usage`" drove two rows
   **toward** `how_to_usage`. A 20b model applies an abstract preference
   bluntly. The examples are written and staged, drawn from dev only.
5. **Split `complaint_dissatisfaction` into a flag** (1 day, including
   relabelling). §4.1 says this is a taxonomy problem; this is the fix.
   Sentiment is already a field.
6. **Decide the escalation trade-off with a real cost input** (0.5 day). The
   cross-model-disagreement signal buys 4 fewer missed escalations for 14 more
   over-escalations. Whether that's worth it depends on a cost ratio I don't
   have.
7. **Deduplicate retrieved neighbours by `customer_tweet_id`** (0.5 day
   including a re-run). §3 shows the embedding retriever ranks neighbours
   better but hands the drafter fewer *distinct* situations (2.17 of 3 vs
   2.58), because multi-tweet replies share a customer tweet and embed
   almost identically. Fix it in `top_k` for both retrievers, then re-run the
   reply eval — this is the most likely explanation for why better retrieval
   produced slightly worse replies, and it is cheap to test.
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
   decision carries a stated reason and is auditable. The cost: intent alone
   doesn't determine escalation for three intents that sit near 50% — left as a
   stated finding, because fitting per-intent exceptions on ~113 rows is
   overfitting.
7. **Flipped `account_access` from auto-handle to escalate**, because the golden
   set showed a human escalated 89% of those. Tuned on the 60% dev split,
   measured on the 40% never touched.
8. **The DM-deflection policy is an instruction in the prompt, not something
   the model infers from retrieved examples** — and `is_deflection` is scored as
   its own boolean rather than folded into the quality score, because averaging
   it in would hide the failure it exists to catch.
9. **Kept the earlier-but-wrong results rather than overwriting them.** The
   original labels, the v1 rubric scores and the two failed improvement attempts
   are all still in the repo and still reproducible from the CLI. A take-home
   that only reports its successes isn't showing how it makes decisions.
10. **Re-adjudicated all 189 rows, not just the disagreements.** The
    disagreement-only version scored 2.5 points higher and was biased upward by
    construction (§5.3). I wrote the precedence rules *before* relabelling,
    which caught two rules my own data proved wrong before they shipped.
11. **Excluded the 15 unscorable rows from the headline** (14
    `insufficient_context`, 1 `ambiguous`) rather than assigning a label nobody
    can defend. They're reported as a finding, and `--keep-unscorable` puts them
    back.
12. **Froze a 60/40 dev/holdout split** before any tuning, so the prompt and
    policy experiments could not quietly contaminate the reported number.
13. **Rejected the prompt containing the rules, and the 6× larger model** — both
    measured, both worse, both documented (§5.7). The negative results were more
    informative than a win: 22 of 27 shared errors carry the *same* wrong label,
    which is what says the ceiling is the taxonomy.
14. **Wrote a deterministic reply guard instead of trusting the prompt.** The
    drafting prompt already said "never emit a placeholder"; `@123456` still
    reached 12% of drafts. The guard validates, regenerates once with the
    offending text quoted back, and sanitises as a last resort — never
    substituting an invented name, because filling the slot with a fabrication
    is a worse failure than leaving it empty.
15. **Namespaced the prediction cache on `taxonomy:prompt:model`.** Under the
    old namespace the model A/B would have read the incumbent's cached answers
    back and reported perfect agreement — the most convincing wrong result
    available.
16. **Kept evaluation API-free.** Every prediction is stored next to its label,
    so `12_evaluate.py` re-runs in seconds at zero cost. This is why the
    15-minute reproduce target is achievable at all.
17. **Added embedding retrieval as an opt-in second implementation, not a
    replacement.** It measurably retrieves better (§3), but it costs an 83MB
    download and ~270× the query latency, and it has not been shown to improve
    the replies themselves. Putting it behind `--retriever` keeps both claims
    testable and keeps the headline reproducible without it. The `retriever`
    column in `reply_evals.csv` exists so the two can never be averaged
    together by accident — the same mistake the rubric versions nearly caused.
18. **Every long-running script is append-as-you-go and resumable.** An early
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
intents.json              the 13 intents, as the model sees them
turn_types.json           the 5 turn types
```

## 9. Known limitations, stated plainly

- **Judge-vs-human agreement is not yet measured.** `14_judge_agreement.py` is
  built and blind by design, but has not been run. Until it is, every
  reply-quality number rests on models agreeing with models. This is the one
  required piece still outstanding.
- **The 53 relabelled rows have one reader: me.** §5.2.
- **`brand_text` is unused**, which caps `turn_type`, leaves 15 rows
  unroutable, and keeps one escalation override dead. §4.2.
- **The full 5,938-row classification run was never done** — deliberately, §1.

## 10. Citations

Dataset: Thought Vector, *Customer Support on Twitter* (Kaggle). Models:
Groq-hosted `openai/gpt-oss-20b` (classification, drafting) and
`openai/gpt-oss-120b` (comparison run); `qwen/qwen3.8-27b` as the independent
judge. Libraries: pandas, scikit-learn (TF-IDF, LogisticRegression), tenacity,
python-dotenv, groq. Everything else is mine.
