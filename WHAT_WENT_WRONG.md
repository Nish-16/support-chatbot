# What was wrong with this system, and what I did about it

An honest write-up of the problems I found in my own pipeline, in the
order I found them, with the evidence for each and the measured result of
each fix — including the two fixes that did not work.

The headline, stated up front so nothing below reads as spin:

> **The classifier never improved. It went from a measured 62.4% to a
> measured 81.0% without a single change to the model or the prompt.**
> Every prediction scored in this document is the same output from the
> original run. What changed was that the evaluation became correct.
> I then tried twice to actually improve the classifier, and both
> attempts failed and were reverted.

---

## Issue 1 — The 62.4% was wrong, and it was wrong in my favour to fix

### What I believed

The classifier scored 62.4% (118/189) on my hand-labelled golden set,
with four intents dragging it down:

| intent | recall |
|---|---:|
| complaint_dissatisfaction | 0.30 |
| how_to_usage | 0.31 |
| storage_quota_plan_limits | 0.33 |
| sync_app_bug | 0.40 |

68% of all errors sat in those four. The obvious move was prompt
engineering against them.

### What was actually happening

Before touching the prompt, I re-read the 48 errors where my label was
one of those four. **The model was right, and I was wrong, in 22 of
them.** Only 6 were clean model errors.

The pattern was consistent. I had been labelling on **sentence shape**;
the model was labelling on **topic**:

| tweet | my label | why I was wrong |
|---|---|---|
| *"...making my business user experience worthless. **How do I disable?!**"* | `complaint` | grievance words won; it ends in a real question |
| *"...still have 'ran out of Space' notice. **How do we fix it?**"* | `how_to_usage` | "how do" won; this is quota-not-clearing |

I applied two opposite rules, in the same labelling session, and the model
was consistent across both.

### The root cause, which is not a modelling problem

`taxonomy.md` is 36KB of carefully-reasoned boundary rules. **Neither
consumer of those rules could see them:**

- the **model** gets `intents.json` — one terse line per intent
- the **human labeller** got *less than the model*:
  `08_label_golden_set.py` printed bare intent names, no descriptions

Five rules **already written in `taxonomy.md`** were violated during
labelling, accounting for 13 of the 48 errors — including
"quota not clearing after deletion → storage_quota" (5 rows) and
"storage_quota is not device disk usage" (2 rows).

The rules existed. They were agreed. They were simply never delivered to
anyone who applied them.

### An error I caught in my own review

I re-checked all 48 adjudications a second time, deliberately weighting
attention toward the 22 I had decided *for* the model — the ones a
reviewer invested in the model looking good would get wrong. One flipped:

> **`193025`** — *"as a small non-profit, we can't afford to pay $500 for
> something we're not going to use."* I called it `billing_subscription`
> (model right). But my own rule says evaluative + no actionable request =
> `complaint`. I applied that rule to two other rows and then failed to
> apply it here, where it went against the model.

### The selection bias I nearly shipped

The 48 rows were chosen where *my label* was one of the four intents.
That sample can only contain errors where I pulled a tweet into the
cluster — never where the model did. So I adjudicated the other side too:

| verdict | false-negative side | false-positive side |
|---|---:|---:|
| model right | 21 | 2 |
| human right | 6 | 2 |
| ambiguous | 10 | 5 |

**The model wins 21:6 on one side and 2:2 on the other.** Quoting "the
model is right 4× as often as the labeller" would have been an artifact
of which rows I chose to look at.

Then the deeper version of the same problem: adjudication only ever
examined rows where the model and I **disagreed**. Where we were wrong the
*same way*, nothing flagged it. Sweeping all 189 rows found **8 such
rows** — every one of which *lowers* the score. That cost 2.5 points:

| labels | accuracy | |
|---|---:|---|
| v1 (original) | 62.4% | |
| v2 (disagreements only) | 83.6% | biased upward by construction |
| **v3 (full sweep)** | **81.0%** | |

### The fix

- `taxonomy.md` **Part 11**: a 9-rule precedence ladder, each rule with a
  trigger, the winning intent, why the loser loses, and worked positive,
  negative and counter examples.
- `data/golden_labels_v3.csv`: all 189 rows re-adjudicated, 53 labels
  changed, every change carrying the rule that decided it and a written
  rationale. The original file was never modified.
- `08_label_golden_set.py` now shows the labeller the intent descriptions
  **and the rules verbatim** — the same string the model would get, not a
  paraphrase.

### Two rules my own data proved wrong before they shipped

Writing rules *before* relabelling caught errors that relabelling first
would have baked in:

- **R1 had a counterexample.** *"I don't like the new tray icon"* is
  evaluative with no request → my draft rule said `complaint`. But
  `taxonomy.md` gives *product feedback* to `feature_request`. The rule
  would have relabelled a **correct** human label wrong.
- **R3 was wrong as drafted.** "Object beats symptom" applied to *"Everyone
  can add except me. It says my Dropbox is full"* routes a quota problem to
  `sharing_permissions` and hands the customer the wrong answer.

The rules also **restored four labels** my own earlier adjudication would
have overwritten.

---

## Issue 2 — Two "I don't know" branches that have never fired

The escalation policy has two uncertainty overrides. Both are dead:

| branch | reality |
|---|---|
| `needs_human_triage` | a human set it 12 times; the model has set it **0 times** |
| `confidence < 0.5` | **all 174 scorable rows sit above 0.5** |

Proof that the second is dead code: the policy scores **identically** with
the confidence branch enabled and removed — precision 0.70, recall 0.84,
11 missed, 24 over, both ways.

### The fix, at zero API cost

I already had two independent runs over all 189 rows (gpt-oss-20b and
gpt-oss-120b, same prompt). **Cross-model disagreement** is a usable
uncertainty signal:

| | classifier accuracy |
|---|---:|
| models agree (145 rows) | 84.8% |
| models disagree (29 rows) | 62.1% |

**+22.8 points of separation**, against a self-reported confidence signal
that separates nothing. This is the cheap version of self-consistency
sampling: three samples of one model costs 3× forever, and two model
families disagree where the *task* is ambiguous rather than where the
temperature wobbled.

As a policy it is a genuine trade, not a free win: missed escalations
11 → 7, but over-escalations 24 → 38 and precision 0.70 → 0.61. I have
**not** changed `escalation.py` — the escalate rate rises to 56%, and an
agent escalating over half its traffic is a product decision.

---

## Issue 3 — `turn_type` is not a feature, it is a base rate

`turn_type` scored 86%. The model predicts `first_contact` for **186 of
189 rows**. A human found 28 that are not first contact; the model found
3. It is riding the majority class.

This is a **data** problem, not a model problem: conversational position
cannot be recovered from an isolated tweet. *"I Have done this. Too bad it
didn't solve the problem"* has no topic and no position without the
message before it.

The same missing input causes two more failures:

- **14 rows (7.4%) are unroutable.** The taxonomy has no intent for
  "on-topic but needs the prior turn", and they had been quietly filed
  under `no_action_needed` and `complaint_dissatisfaction`, inflating both.
  They now carry `v3_status = insufficient_context` and are excluded from
  the headline rather than given a label nobody can defend.
- **The `disputing_prior_answer` escalation override never fires**, for
  the same reason.

`dropbox_paired.csv` already has a `brand_text` column. Nothing uses it.
**This is the highest-value change still outstanding**, and it fixes all
three at once.

---

## Issue 4 — A consistency audit that cannot see meaning

`11_label_consistency.py` reported no near-identical tweets with
conflicting labels. Reading found one immediately:

- *"why did you remove the green check..."* → `complaint_dissatisfaction`
- *"I'm missing the green check, now see only the white box"* → `how_to_usage`

Their TF-IDF cosine similarity is **0.153** — semantically near-identical,
almost no shared vocabulary. The audit compares *wording*, not *meaning*,
so it structurally cannot catch this class. Both now carry one label under
R2.

---

## Issue 5 — An instruction is not a guarantee

`DRAFT_SYSTEM_PROMPT` already told the model never to emit a placeholder.
`@123456` still reached 12% of drafted replies. A prompt instruction is a
request.

`reply_guard.py` is a deterministic check that does not depend on the
model complying: **validate → regenerate once with the offending text
quoted back → sanitize as a last resort, never substituting.** Inventing a
name to fill a slot is a worse failure than the slot.

Two bugs in my own guard, both caught by its self-tests:

1. The first pattern flagged *"DM us your ticket number and the email on
   the account"* — the exact phrasing the DM policy **requires**. A guard
   that fires on the target behaviour is worse than no guard.
2. `sanitize` returned *"Send your details to and we will help."* marked
   sendable. It now detects dangling prepositions and reports `ok=False`
   separately, rather than handing back a string the caller trusts.

---

## What I tried that did NOT work

Both are documented rather than buried, because a take-home that only
reports its successes is not showing you how it makes decisions.

### Failure 1 — putting the rules in the prompt made it worse

The obvious follow-through from Issue 1: if the rules decided 53 relabels
and the model cannot read them, give them to the model.

| dev split, 86 scorable rows | | |
|---|---:|---:|
| p1 (baseline) | 70/86 | 81.4% |
| p2 (+ Part 11 rules) | 68/86 | 79.1% |

7 rows fixed, 9 broken, at **2.6× the prompt tokens** (1035 vs ~400)
against a 200k tokens/day cap. Rejected on cost — at that n it is a wash,
but there is no gain to pay for.

**The breakdown is the useful part.** Mechanical rules worked exactly as
designed: R5 (device disk) fixed `2028354` and `2976667`, R7b fixed
`2222820`, R8a fixed `1851629`. Judgment rules backfired: R2 pushed
`2124237` and `847924` from `feature_request` to `complaint`, and R4's
*"prefer a specific intent over how_to_usage"* drove two rows **toward**
`how_to_usage` — the opposite of what it says.

A 20b model applies an abstract preference bluntly. That is the design for
p3: keep the mechanical rules, express the judgment boundaries as few-shot
examples instead. Those examples are written and staged, drawn from dev
only — shipping them alongside the rules would have made p2 uninterpretable.

### Failure 2 — the 6× larger model is worse

| 174 scorable rows | | |
|---|---:|---:|
| gpt-oss-20b | 141/174 | 81.0% |
| gpt-oss-120b | 129/174 | 74.1% |

**−6.9 points.** Only-incumbent-right 18; only-challenger-right 6.

The decisive detail: of the 27 rows both models get wrong, **22 are the
same wrong label.** Shared errors mean a shared cause. The remaining
errors are not a capacity limit — they are the taxonomy and the prompt.
Buying a bigger model would not have fixed them.

This also required a real fix first: the prediction cache was namespaced
on taxonomy version alone, so a second model would have **read the first
model's cached answers back** and reported perfect agreement — the most
convincing wrong result available. The namespace is now
`taxonomy:prompt:model`, and it immediately caught me looking up the 120b
run under the wrong prompt version.

---

## Where the system actually stands

| | | 95% CI |
|---|---:|---|
| v1 published | 118/189 = 62.4% | [55.5%, 69.3%] |
| **v3 labels, all scorable** | **141/174 = 81.0%** | **[75.2%, 86.9%]** |
| v3, dev | 87/104 = 83.7% | [76.5%, 90.8%] |
| v3, holdout | 54/70 = 77.1% | [67.3%, 87.0%] |

**Good:** the intents where being wrong is expensive are the ones handled
best — `phishing_abuse_report` 1.00, `security_account_compromise` 1.00,
`data_loss_recovery` 0.93.

**Weak:**
- Escalation is the metric that matters and it is mediocre: **11 missed
  escalations** out of 72 a human would escalate.
- Per-intent n is 9–21, so one row moves an intent 7–11 points. These
  numbers cannot rank intents.
- Reply quality (4.92/5) has a ceiling effect and the judge has still
  never been validated against human ratings.

**The largest open risk:** the v3 labels are *my* adjudication,
self-rechecked once. No second person has verified them. If a reviewer
disagrees with 10 of my 53 changes, the headline moves ~5 points.

**One comparison that is currently invalid:** the README's table pits LLM
62.4% against TF-IDF 40.7% and majority 15.9%. All three were measured
against v1 labels and **only the LLM number has been recomputed on v3**
(the majority baseline alone moves 15.9% → 12.1%). `10_baselines.py` must
be re-run against v3 before that table is quoted.

---

## Outstanding, in priority order

1. **Feed `brand_text` into the classifier.** Fixes `turn_type`, the 14
   unroutable rows, and the dead `disputing_prior_answer` override.
2. **Re-run the baselines on v3 labels** so the comparison table is valid.
3. **p3**: mechanical rules + few-shot for the judgment boundaries.
4. **Decide the escalation trade-off**: 4 fewer missed escalations for 14
   more over-escalations.
5. **A second reader on the 58 adjudications.** The one thing I cannot do
   for myself.
6. Validate the reply judge against human ratings (`14_judge_agreement.py`
   is built and waiting for input).
