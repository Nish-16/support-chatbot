# STEP 1 -- Taxonomy adjudication (v1)

Review of the 48 intent errors whose **human** label is one of the four
low-recall intents. Question being answered: *before* changing the
prompt, how much of the 62.4% gap is a model error and how much is a
label error?

Nothing here has been applied. `data/golden_labels.csv` is untouched and
the 189-row baseline in `README.md` still stands exactly as published.
Every decision is recorded in `data/adjudication_v1.csv`; regenerate with:

```bash
./.venv/Scripts/python.exe scripts/15_adjudicate.py          # dry run
./.venv/Scripts/python.exe scripts/15_adjudicate.py --write  # writes the CSV
```

---

## Headline

**The human label is wrong or not-clearly-right in 42 of the 48 rows (88%).
Only 6 are clean model misses.**

| verdict | n | meaning |
|---|---:|---|
| `model_right` | 22 | human label wrong; the model was correct |
| `human_right` | 6 | model wrong; human label stands |
| `both_wrong` | 5 | a third intent is correct; neither label was |
| `ambiguous` | 10 | both defensible under the taxonomy *as written* |
| `context_missing` | 5 | not routable from the tweet alone |

Per human intent:

| human intent | model_right | human_right | both_wrong | ambiguous | context_missing |
|---|---:|---:|---:|---:|---:|
| complaint_dissatisfaction | 7 | 4 | 2 | 5 | 3 |
| how_to_usage | 6 | 0 | 1 | 2 | 0 |
| storage_quota_plan_limits | 6 | 2 | 2 | 2 | 0 |
| sync_app_bug | 3 | 0 | 0 | 1 | 2 |

**`how_to_usage` and `sync_app_bug` contain zero clean model errors.**
Their 0.31 and 0.40 recall is a measurement artifact, not a capability gap.

## What this does to the reported number

With **no model change and no prompt change**, correcting only the labels
this review found wrong:

| | correct | accuracy |
|---|---:|---:|
| as reported today | 118/189 | **62.4%** |
| after label corrections | 140/189 | **74.1%** |
| upper bound if STEP 2 also settles every `ambiguous` row | 150/189 | **79.4%** |

The 62.4% headline understates the classifier by roughly 12 points. Had
we gone straight to prompt engineering, we would have spent the API
budget teaching the model to reproduce our own labeling mistakes, and
the improvement would have shown up as a *regression* against the
uncorrected golden set.

Irreducible in this cluster: **16 rows** — 6 plain model misses, 5 where
neither label was right, 5 unroutable without a prior turn.

---

## Finding 1 — the root cause is that `taxonomy.md`'s rules never reached either labeler or model

`taxonomy.md` is 36KB of carefully-reasoned boundary rules. Neither
consumer of those rules ever sees them:

- **The model** gets `intents.json` — one terse line per intent. This
  was a deliberate, correct token-budget decision (`groq_lib.py` documents
  the 200k tokens/day cap that forced it), but it means every carve-out
  lives only in a file the model never reads.
- **The human labeler** gets *less than the model does*.
  `08_label_golden_set.py:print_intent_menu()` prints bare intent names
  and nothing else — not even the one-line description.

Five rules that **already exist in `taxonomy.md`** were violated during
labeling, and are directly responsible for 13 of the 48 errors:

| rule, as already written | rows it would have fixed |
|---|---|
| "quota not clearing after deletion" → `storage_quota_plan_limits` | 2629478, 2053253, 1999298, 79858, 2910246 |
| `storage_quota_plan_limits` is "not device disk usage" | 2179658, 2976667 |
| team folders / shared links → `sharing_permissions` | 259327, 37940, 2607972 |
| "pure pricing and invoicing questions" → `billing_subscription` | 1992403, 1523722 |
| `how_to_usage` means "nothing is broken" | 988433, 2629478 |

This is the cheapest fix available and it is not a prompt change: it is
a **distribution problem**. The rules were written, agreed, and then left
where nobody reads them.

## Finding 2 — the human applied surface form, the model applied topic

The dominant human error mode is labeling on **sentence shape** rather
than on need:

- *"...is making my business user experience worthless. **How do I disable?!**"* → labeled `complaint` (grievance words won)
- *"...still have 'ran out of Space' notice. **How do we fix it?**"* → labeled `how_to_usage` ("how do" won)

Both rules were applied, in opposite directions, by the same labeler. The
model was consistent across both: it routed on topic each time, and was
right each time.

## Finding 3 — the taxonomy has one genuinely unresolved boundary, not three

Of the three boundaries I was asked to examine, only the first is a real
taxonomy defect:

- **`complaint` ↔ topic — REAL.** 5 of the 10 `ambiguous` rows. There is
  no written rule for a message that names a product topic *and* makes no
  request. `taxonomy.md` never states precedence, so both readings are
  legitimate.
- **`how_to_usage` ↔ `storage_quota` — NOT a taxonomy defect.** The rule
  exists ("quota not clearing after deletion"); it was simply not
  available at labeling time. 0 of these rows are genuinely ambiguous.
- **`sync_app_bug` ↔ `storage_quota` — NOT a taxonomy defect.** Same:
  every one of these is the deletion/quota case the taxonomy already names.

A fourth boundary surfaced that was **not** in the brief and is real:
**`storage_quota` ↔ `billing_subscription`** (1495052, 2742476). `taxonomy.md`
itself predicted this — it calls the intent *"exactly on the billing/technical
seam"* — and then never resolved the seam. Worth noting that the same
document designates `storage_quota_plan_limits` as the *"next cut if the
budget tightens"*; this review is evidence for that cut, since 6 of its
12 errors resolve to `billing_subscription` or `feature_request`.

## Finding 4 — `complaint_dissatisfaction` is not a topic, and the data says so

Its 21 errors split five ways, which no other intent does. It competes
with `billing`, `sharing`, `storage`, `service_outage`, `feature_request`
and `no_action_needed` — i.e. with the entire rest of the taxonomy,
because grievance is a *frame* that can attach to any topic. Meanwhile
`churn_threat`, `abusive_content` and `sentiment` already exist as flags
covering the same signal, so the taxonomy encodes dissatisfaction twice,
in two incompatible shapes.

This is direct evidence **for** the STEP 3 hypothesis. Deferring to STEP 3
as instructed; recording it here because the evidence came from this review.

## Finding 5 — 5 rows are unroutable, and they cluster

Every `context_missing` row is a mid-thread turn (`1041902`, `74467`,
`1687134`, `1972693`, `2828924`). `2828924` — *"I Have done this. Too bad
it didn't solve the problem"* — is the clearest case: it carries no topic
at all, and the correct handling is `turn_type=disputing_prior_answer` →
escalate, which is a routing fix, not an intent fix. These are STEP 4
evidence, not classifier errors, and **no prompt change can recover them.**

## Finding 6 — the consistency audit's clean bill of health is weaker than it reads

`11_label_consistency.py` reported no near-identical tweets with
conflicting intents. This review found one by reading:

- `615215` — *"why did you remove the green check..."* → `complaint_dissatisfaction`
- `1863374` — *"I'm missing the green check, now see only the white box"* → `how_to_usage`

Their TF-IDF cosine similarity is **0.153**, far below any workable
threshold, so a lexical audit cannot catch it — the tweets are
semantically near-identical but share almost no vocabulary. The audit
detects near-duplicate *wording*, not near-duplicate *meaning*, and
`README.md` should say so rather than citing it as evidence of labeling
consistency.

`1863374` also carries the labeler's own note: *"no right answer just
classified to least wrong"* — an explicit record that the taxonomy failed
to decide, which the audit had no way to surface.

---

## Candidate precedence rules (for STEP 2 — not yet implemented)

Derived from the 48 rows, not assumed in advance. The brief's suggested
rule is close but **inverted on one case**, so it is not adopted as-is.

**R1 — actionable request beats grievance frame.** If the message contains
an answerable question or a concrete request, use the topic intent
regardless of tone. `complaint_dissatisfaction` applies only when the
message is evaluative **and** makes no actionable request.

The brief proposed "prefer the specific topic intent" when a concrete
request exists — that half matches. But the data also requires the
converse: `1050476`, `1856823`, `2179658` all *name* a topic and are still
correctly `complaint`, because they ask for nothing. Topic presence alone
must not win.

R1 alone settles 3 of the 10 `ambiguous` rows (`615215`, `622045`,
`2425723`). The full rule set below settles 7; the mapping is recorded
per row in `adjudication_v1.csv`'s `resolved_by_rule` column and printed
by the script, so the count is computed rather than asserted.

**R2 — "how do I" is surface form, not intent.** `how_to_usage` requires
that nothing is malfunctioning. If something is broken, route to the
malfunction's topic however the sentence is phrased.

**R3 — object beats symptom.** If the thing being acted on is a shared
link, shared folder, or team folder, use `sharing_permissions` even when
the symptom reads as a sync or app failure.

**R4 — quota-not-clearing is storage, never sync.** Deleted files with
unchanged usage → `storage_quota_plan_limits`. Already in `taxonomy.md`;
needs to reach the prompt.

**R5 — account quota ≠ device disk.** Local disk space is not
`storage_quota_plan_limits`. Already in `taxonomy.md`.

**R6 — price and packaging beat storage.** A question whose answer is a
price → `billing_subscription`. A message of the form "I'd pay more if you
offered X" → `feature_request`, even when X is storage.

**Residual after all six rules: 3 rows.** `2551167` and `1383896` are
grievances that *imply* a request without stating one; `2222820` asks a
clean how-to about an account-linking object. These need an explicit
tie-break written in STEP 2, not a per-row judgment call — they are
exactly the cases where an unwritten rule drifted last time.

Worked examples for each contested boundary, positive and negative, are
available directly from `data/adjudication_v1.csv` — the `model_right` and
`human_right` rows are the positive and negative cases respectively, with
the reason already written per row.

---

## Scope note

This review covered the 48 errors where the **human** label is one of the
four intents, as specified. There are **10 further errors** where the
*model* predicted one of the four but the human did not — false positives
into the cluster, which bear on the same boundaries from the other side
(58 of the 71 errors touch the cluster in one direction or the other).
They are not adjudicated here. Recommend adding them before STEP 2 fixes
the rules, since a precedence rule tuned only on false negatives can
silently make false positives worse.

---

# STEP 1b — re-check, and the false-positive side

`scripts/16_recheck_adjudication.py` → `data/adjudication_v2.csv` (58 rows).
`adjudication_v1.csv` is preserved unchanged as the first-pass record;
`golden_labels.csv` is still untouched.

## Re-check of the 48: 47 stand, 1 changed

The first pass was done tweet-first ("what does this message need?").
The re-check was done rule-first ("what does `taxonomy.md` mandate?"),
with attention deliberately biased toward the 22 rows decided *for* the
model — the ones a reviewer invested in the model looking good would get
wrong.

**One flip, and it was in exactly the predicted direction:**

> **`193025`** — *"as a small non-profit, we can't afford to pay $500 for
> something we're not going to use"*. First pass: `model_right` →
> `billing_subscription`, on the grounds that the subject is a paid plan.
> That is wrong. **R1 — my own proposed rule — says evaluative message +
> no actionable request = `complaint_dissatisfaction`.** I applied R1 to
> reach `complaint` on `1856823` and `847909`, then failed to apply it
> here, where it cuts against the model. Reclassified `both_wrong`.

Three further rows stand with a recorded caveat (`2124237`, `1452735`,
`2425723`) — see `recheck_note` in the CSV.

**This does not clear the review.** It is the same reviewer re-reading
their own work: it catches rule-application slips, which is what it
caught, but it cannot remove the bias it is testing for. A second person
reading the 58 rows cold is still worth doing before v2 labels are
published.

## The false-positive side changes how the headline should be read

| verdict | false-negative side | false-positive side | total |
|---|---:|---:|---:|
| `model_right` | 21 | 2 | 23 |
| `human_right` | 6 | 2 | 8 |
| `both_wrong` | 6 | 0 | 6 |
| `ambiguous` | 10 | 5 | 15 |
| `context_missing` | 5 | 1 | 6 |
| **total** | **48** | **10** | **58** |

**On the false-negative side the model wins 21:6. On the false-positive
side it is 2:2.**

That asymmetry is the most important result of this pass. The 48-row
sample was selected on `human_intent ∈ FOUR`, which conditions on the
human having entered the cluster — so it can only ever surface cases
where the human pulled a tweet *in*, never where the model did. **The
21:6 skew is partly an artifact of that selection and must not be quoted
as "the model is right 4x as often as the labeler."** Across both sides
it is 23:8, and half the false-positive rows are unresolved rather than
decided either way.

Corrected accuracy with both sides counted, still with **no model or
prompt change**:

| | correct | accuracy |
|---|---:|---:|
| as reported today | 118/189 | 62.4% |
| after label corrections | 141/189 | **74.6%** |
| upper bound if STEP 2 settles all 15 ambiguous rows | 156/189 | 82.5% |

## Two of the six candidate rules are wrong as drafted

The false-positive side breaks them, which is precisely why it needed
adjudicating before STEP 2 wrote them into the prompt.

**R1 has a counterexample — `1960138`.** *"I don't like the new tray icon,
every time I see it I think there's something wrong with it."* Evaluative,
no actionable request → R1 sends it to `complaint_dissatisfaction`. But
`taxonomy.md` gives *"product feedback"* to `feature_request` explicitly.
**R1 needs a carve-out**: evaluative commentary about a specific product
design is `feature_request`; `complaint_dissatisfaction` is for
dissatisfaction with the *service or the company*. Without it, R1 would
have relabelled a correct human label wrong.

**R3 is wrong as drafted — `2956783` and `659012`.** *"Everyone can add to
the Dropbox except for me. It says my Dropbox is full."* has a
shared-folder object but a quota symptom, and the answer the customer
needs is quota mechanics. R3 ("object beats symptom") would route it to
`sharing_permissions` and hand them the wrong answer. **R3 must be
narrowed** to access and permission *failures* — who can see or edit what
— and must not claim quota problems that merely involve a shared folder.

R2, R4, R5 and R6 survive; R6 gains independent support from `768404`
(*"What's the difference between this and plus and free"* → the model's
`how_to_usage` is wrong under R6).

## Three more findings from the false-positive side

**A new unresolved boundary, not in the brief: `sync_app_bug` ↔
`service_outage`** (`1788320` — *"my desktop app is syncing at a snail's
pace today. Is the service having issues?"*). A single-user symptom
carrying an explicit service-status question. `taxonomy.md` assigns *"Is
Dropbox down?"* to `service_outage` and never says what happens when both
are present.

**`account_access` ↔ `how_to_usage` is confirmed unresolved.** Both passes
independently declined to settle it — `2222820` on the false-negative
side, `3485` on the false-positive side. Two passes reaching the same
non-answer is evidence the boundary is genuinely missing, not that either
reading was careless.

**`no_action_needed` is being used as an unroutable-bucket** (`1331572`,
labeled `no_action_needed` with the note *"context not there for
classification"*). That inflates the intent and hides turn-type problems
inside it — the exact v1 catch-all disease v2 was built to cure.
`taxonomy.md` sets a ≥20% health metric for this intent; it should also
be checked for context-missing rows specifically.

Separately, `512851` (*"#Dropbox is about to be binned"*) was escalated
by the human but left without `churn_threat` set — flag under-use, not an
intent error, and relevant to STEP 5 rather than STEP 2.

---

## Recommended before STEP 2

1. **Do not** apply these corrections to `data/golden_labels.csv` as a
   silent overwrite. Produce `golden_labels_v2.csv` from
   `adjudication_v2.csv`, keep v1 as the published baseline, and report
   both. *(Not yet done — v2 labels are deliberately not created here.)*
2. ~~Have the corrections independently re-checked~~ — **done, partially.**
   47 of 48 stand; `193025` flipped. But this was a self re-check, so a
   second reader on the 58 rows is still the open item.
3. ~~Adjudicate the 10 model-side rows~~ — **done** (STEP 1b above).
4. Fix the distribution problem in Finding 1 — surface the rules in
   `08_label_golden_set.py` and in the prompt — before any prompt tuning,
   or the same drift recurs on the next labeling pass.
5. **Amend R1 and R3 before they are written into anything.** As drafted
   they would each have relabelled a correct human label wrong
   (`1960138`, `2956783`).
6. Decide the three boundaries both passes left open —
   `complaint` ↔ `feature_request`, `account_access` ↔ `how_to_usage`,
   `sync_app_bug` ↔ `service_outage` — as written rules, not per-row
   judgment calls. 15 of the 58 cluster errors (26%) are waiting on them.
