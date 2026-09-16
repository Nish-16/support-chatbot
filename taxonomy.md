# Intent Taxonomy -- v1 (as built) and v2 (adopted)

**Status, as of 2026-09-16.** v2 is the live taxonomy. The bullets below
described the state on 2026-09-10, when v2 was still a proposal; v2 was adopted
before any labelling started, and the rest of this document is unchanged from
that decision point. It is the reasoning trail, not just the current spec.

- **v1 -- 8 intents.** Superseded. Part 1 restates it so this document stands
  alone, and Part 2 is the case against it.
- **v2 -- 13 intents + `turn_type` + 8 flags.** Live: these are the intents in
  `intents.json`, in `groq_lib.py`'s prompt, and behind every result reported in
  `README.md`. Defined in Parts 3 and 4.
- **v2.1 -- precedence rules.** Part 11, added 2026-09-12 after adjudication.
  `taxonomy_rules.py` turns them into the prompt and labelling blocks, and
  `tests/test_taxonomy_rules.py` checks the two against this file.

The intent count is set by the golden-set budget, not by taste -- see
Part 6.

---

## Contents

| Part | |
|---|---|
| 0 | Why now, and the decision rule |
| 1 | **v1: the 8 intents as built** |
| 2 | What's wrong with v1 -- four findings |
| 3 | **v2: the 13 intents** |
| 4 | Not intents: `turn_type` and flags |
| 5 | Considered and rejected |
| 6 | Budget: why the list stops at 13 |
| 7 | Migration from v1 to v2 |
| 8 | Coverage test |
| 9 | For the report |
| 10 | Caveats on the numbers |

---

# Part 0 -- Why now, and the decision rule

## Why now

`data/golden_labels.csv` does not exist yet. Hand-labeling has not
started. That makes this the cheapest possible moment to change the
taxonomy:

- Changing it **now** costs one re-classification pass over the 250
  candidates. The cache in `cache.py` is keyed by `tweet_id`, so this
  is a cache-key bump plus API calls we've already budgeted for.
- Changing it **after** labeling costs re-labeling 250 examples by
  hand -- hours of human time, and the single most expensive resource
  in this project.

`intents.md` already predicted this: *"This is v1 -- expected to be
revised once we hand-label the golden eval set and hit edge cases that
don't fit cleanly."* The edge cases showed up earlier than expected, in
the candidate pool itself.

## The decision rule

Every intent in v2 had to pass one test:

> **A category earns its own slot only if it changes what the agent
> does.**

Not "is it a distinguishable topic" -- almost anything is. The question
is whether splitting it changes **the reply, the auto-handle/escalate
decision, or the urgency**. If two things get the same treatment, they
are one intent.

This is why `service_outage` was accepted (the opposite treatment from
a sync bug) and `app_platform_bug` rejected (same troubleshooting path
as a sync bug).

**Corollary, and the reason Part 4 exists:** when a distinction is real
but doesn't change the *route*, it belongs in a flag, not an intent.
Cutting an intent must not mean losing the signal.

---

# Part 1 -- v1: the 8 intents as built

This is the live taxonomy. Descriptions are as written in
`intents.json`; example tweet IDs are from `intents.md`; actions are
from the escalation matrix in `README.md`. The pool column is the
classifier's `suggested_intent` distribution over the 250 candidates in
`data/golden_candidates.csv` -- **not ground truth**, and not a traffic
estimate (the top-up deliberately over-samples rare intents).

### 1. `account_login` -- 16 / 250 (6%)
> Access issues, 2FA/backup codes, business account setup, credential
> problems.

**Examples:** 2823321, 1820072, 2974776, 278604, 785409
**Action:** *"Escalate if 2FA/compromise, auto-handle if plain password
reset."*
**Verdict:** **Overloaded.** Conflates "I forgot my password"
(auto-handle) with "someone logged into my account" (drop-everything
escalation). The conditional in the routing table is the tell -- see
Part 2, finding 4.

### 2. `billing_subscription` -- 31 / 250 (12%)
> Charges, refunds, plan pricing, cancellations or downgrades.

**Examples:** 1682379, 2331194, 2854125, 1928260, 1458883, 1219561,
2384217, 1422641, 459509, 2235996
**Action:** Escalate -- financial transactions need human verification.
**Verdict:** **Sound, but leaky.** Absorbs billing disputes (escalate),
pricing questions (auto-handle), and account closure (legal/retention
path) under one always-escalate rule.

### 3. `sync_technical_bug` -- 64 / 250 (26%)
> Files not syncing, app or website errors, crashes, broken features.

**Examples:** 1820072, 1274528, 234493, 1067513, 1202354, 1972492,
2773993, 118484, 786084, 2043077
**Action:** Auto-handle standard troubleshooting; escalate if recurring.
**Verdict:** **Far too broad -- the dumping ground for technical
noise.** See Part 2, finding 3.

### 4. `how_to_usage` -- 30 / 250 (12%)
> "How do I do X" -- nothing is broken, the customer just needs
> guidance.

**Examples:** 2914964, 1421735, 867197, 1959288, 1040857, 1458883,
1186286
**Action:** Auto-handle. Per `README.md` policy, a DM deflection here is
a **failure**, not a resolution.
**Verdict:** **Good.** The cleanest, best-defined intent in v1.
Survives into v2 unchanged.

### 5. `data_loss_recovery` -- 16 / 250 (6%)
> Deleted or missing files, restore requests. Higher urgency than a
> general bug.

**Examples:** 1960105, 93195
**Action:** Always escalate -- permanent data loss risk.
**Verdict:** **The strongest single decision in v1.** Split from
general bugs on *urgency and resolution path* rather than topic
similarity -- exactly the reasoning v2 applies everywhere else.
Survives unchanged.

### 6. `feature_request` -- 25 / 250 (10%)
> Suggestions or product feedback -- not a problem to fix.

**Examples:** 1040857, 1979488
**Action:** Auto-handle -- log and acknowledge.
**Verdict:** **Good.** One flaw: bleeds into
`general_feedback_or_other` when a request arrives as a rant. Survives
unchanged.

### 7. `followup_ticket_status` -- 16 / 250 (6%)
> Referencing an existing ticket or case, nudging for a response. Not a
> new issue.

**Examples:** 1682379, 1202354, 1433308, 121896, 1473214
**Action:** *"Inherit the original ticket's intent/action."*
**Verdict:** **Not an intent -- it's a turn type,** and `intents.md`
says so itself ("a control-flow signal") while putting it in the enum
anyway. Its own routing rule is un-implementable: "inherit the original
intent" requires an intent field that isn't already spent on this
label. Dissolved in v2.

### 8. `general_feedback_or_other` -- 52 / 250 (21%)
> Thanks, praise, venting/rants, spam, or fragments too ambiguous to
> route to a technical bucket.

**Example:** 1421734 -- *"Thank you - I wanted the files to be
read-only anyway :D x"*
**Action:** Auto-handle (acknowledge) or no-op.
**Verdict:** **The biggest problem in v1.** One in five tweets. Its
stated rationale is sound -- without a catch-all the classifier
hallucinates a technical intent for non-technical noise -- but at 21%
it stopped being a catch-all and became an unmodelled region of the
space. See Part 2, finding 2.

### v1's own stated design rules

Two rules from `intents.md` carry forward into v2 unchanged:

- **Primary-intent rule:** *"several examples straddle two categories...
  We pick the PRIMARY intent -- the thing the customer most needs
  resolved."* v2 keeps this but adds `secondary_intent`, so the
  discarded half is recorded rather than lost.
- **Revision rule:** *"Any changes should be noted in the decision log
  with the example that forced the change."* This document is that
  record for v1 -> v2.

---

# Part 2 -- What's wrong with v1

Four findings from the data, not from intuition.

## 1. One enum is doing three different jobs

v1 is a flat list that mixes:

1. **Topic** -- what the message is about (billing, sync, sharing)
2. **Turn type** -- where in the conversation this is
3. **Risk flags** -- security, legal, abuse, language

`followup_ticket_status` is job 2 wearing job 1's costume. That forces
the model to choose between reporting *what the customer wants* and
*where they are in the thread*. It can only answer one, so we lose the
other on every follow-up tweet.

**Fix: split into separate fields.** This is worth more than any
individual new intent, and it is what lets a 13-intent taxonomy carry
substantially more information than v1's 8.

## 2. The catch-all is 21% of the pool, and it's mostly mid-thread turns

Mapping tweets back to `in_response_to_tweet_id` in the raw dataset:

| | share that are mid-thread replies |
|---|---|
| All DropboxSupport customer tweets (4,504 unique) | **36.7%** (1,654) |
| Golden pool overall (250) | 35.6% (89) |
| Golden pool, `general_feedback_or_other` only (52) | **58%** (30) |

So the catch-all is not mostly spam and noise -- it's disproportionately
**conversational turns the schema has no slot for**. Real examples, all
labeled `general_feedback_or_other`:

```
"@DropboxSupport I'm on iOS 11.0.3 on an iPhone 7."       -> answering the agent's question
"@DropboxSupport Yes background uploading is on"           -> answering the agent's question
"@DropboxSupport It turned out to be VPN interference."    -> self-resolved, close it
"@DropboxSupport All good. That workaround has worked."    -> confirmed resolved
"C'mon @118189, this is ridiculous. Clean up your UX."     -> complaint / churn risk
"Wow! Thank you for adding a text editor! It works great!" -> praise, no action
```

Six lines, five different correct actions, one label.

## 3. `sync_technical_bug` is 26% of the pool and holds incompatible cases

Sampled contents include a global outage (*"Dropbox Down in Taiwan,
China, and India right now"*), an antivirus false positive (*"Sophos
Endpoint does not trust your download"*), a shared-link mixup, and an
ordinary upload failure. The outage case is the damaging one: an agent
that replies "try reinstalling the client" during a service-wide
incident is worse than no agent.

## 4. Two v1 intents need conditional escalation rules -- a taxonomy smell

`README.md`'s escalation matrix contains:

- `account_login` -- *"Escalate if 2FA/compromise, auto-handle if plain password reset"*
- `sync_technical_bug` -- *"Auto-handle first, escalate if recurring"*

When the routing table needs an `if` **inside** an intent, the intent is
usually two intents. The first becomes `account_access` +
`security_account_compromise`. The second is genuinely conditional (it
depends on conversation history, not topic) and stays -- handled by
`turn_type`.

---

# Part 3 -- v2: the 13 intents

| # | Intent | Status vs v1 | Default action |
|---|---|---|---|
| 1 | `account_access` | narrowed | Auto-handle; escalate if ID verification needed |
| 2 | `security_account_compromise` | **new** | **Always escalate -- top priority** |
| 3 | `phishing_abuse_report` | **new** | Route to Trust & Safety |
| 4 | `billing_subscription` | narrowed + widened | Escalate (see sub-rule) |
| 5 | `storage_quota_plan_limits` | **new** | Auto-handle; escalate if quota provably wrong |
| 6 | `sync_app_bug` | renamed + narrowed | Auto-handle; escalate on recurrence |
| 7 | `service_outage` | **new** | Auto-handle: status link, never troubleshoot |
| 8 | `sharing_permissions` | **new** | Auto-handle; escalate on implied data exposure |
| 9 | `data_loss_recovery` | unchanged | Always escalate |
| 10 | `how_to_usage` | unchanged | Auto-handle -- real answer, not a DM deflection |
| 11 | `feature_request` | unchanged | Auto-handle: acknowledge + log |
| 12 | `complaint_dissatisfaction` | **new** (from catch-all) | Escalate on churn/anger, else acknowledge |
| 13 | `no_action_needed` | **new** (from catch-all) | No-op or acknowledge |

## Access & security

### 1. `account_access`
Login failures, password reset, 2FA and backup codes, email/phone
changes, business account setup.

**Why:** v1's `account_login`, **narrowed** -- everything
security-incident-shaped moves to #2. What remains is the routine,
verifiable, low-risk half.
**Escalation:** Auto-handle (guided reset); escalate if identity
verification is required.
**Floor risk:** had only 16 in the v1 pool, and v2 carves cases out of
it. If the top-up can't reach 15, merge into `how_to_usage` -- a guided
reset is guidance. Decide from labeling data, not now.

### 2. `security_account_compromise` *(new)*
Unauthorized logins, suspicious activity, account breach, unexpected
files appearing, fraud on the account.

**Why:** ~46 keyword matches. Today these are indistinguishable from a
password reset -- the mistake that produces a headline like "support bot
told breach victim to try again later." Examples: *"I want to find out
more info on an unauthorized login to my acct"*, *"Why has it taken you
17 days to make contact in relation to an account breach and data
protection issue"*, *"some mysterious files were uploaded to our
receipts folder under my account."* **This is the intent that dissolves
the `account_login` conditional rule.**
**Escalation:** Always escalate, highest priority. Never auto-handle.

### 3. `phishing_abuse_report` *(new)*
Reporting phishing emails, spoofed Dropbox notifications, malicious
shared links, abuse of the platform -- **about a third party, not about
the customer's own account**.

**Why:** **21 confirmed matches** with a conservative regex, so it
clears the 15 floor on its own. It was cut from an earlier draft on a
volume assumption that the data disproved -- see Part 8, where 2 of 30
random tweets were phishing reports with nowhere to go. The action is
unlike anything else here: this customer wants **nothing for
themselves**. There is no ticket to resolve, no account to fix; the
correct response is to collect the sample and route it to Trust &
Safety. Filing it under `security_account_compromise` would escalate a
non-incident to the security queue; filing it under `how_to_usage`
would answer a question nobody asked. Examples: *"What's the best way to
report phishing invites?"*, *"I received a spam email made to look like
it came from Dropbox; can I report it to you guys?"*, *"Someone has
launched a Dropbox phishing expedition."*
**Escalation:** Route to Trust & Safety -- neither a support
auto-handle nor a support escalation. Acknowledge and request the
sample.

## Money & account lifecycle

### 4. `billing_subscription`
Charges, refunds, invoices, plan pricing, upgrades/downgrades, failed
payments, **plus account closure and data-deletion requests**.

**Why:** Kept from v1 largely intact. **Narrowed** by moving space/plan
limits to #5, which was leaking in. **Widened** to absorb account
closure and GDPR/CCPA erasure: an earlier draft gave those their own
intent, but they route the same way this one does, so under the
decision rule they are not separate. What made them feel distinct was
*legal sensitivity*, preserved as the `legal_sensitive` flag.
**Escalation:** Escalate by default -- refund authorization and erasure
requests both need a human. Apply `README.md`'s DM-ask policy: say
exactly what to send, never a bare "please DM us."
**Sub-rule (from Part 8):** pure **pricing and invoicing questions**
auto-handle. *"I am looking at Dropbox Pro, but I need a Mexican
Authority Rated invoice to be tax deductible -- do you issue?"* is a
factual product question with no transaction to authorize. Escalating
it is a false positive.

### 5. `storage_quota_plan_limits` *(new)*
Out of space, quota not clearing after deletion, plan storage caps,
shared-link traffic limits.

**Why:** ~230 broad / ~24 tight matches, sitting **exactly on the
billing/technical seam** -- sometimes an upsell, sometimes a genuine
bug. v1 forces an arbitrary coin-flip between `billing_subscription`
and `how_to_usage` on every one, so the label carries no information
either way. That coin-flip *is* the action difference: one path
escalates to a human, the other doesn't. Bug case: *"I have deleted
files from 'My Files' and permanently deleted from 'Deleted Files' but
still shows full quota."*
**Escalation:** Auto-handle (explain quota mechanics / plan options);
escalate if the quota is provably wrong after standard steps.
*Designated next cut if the budget tightens -- see Part 6.*

## Technical

### 6. `sync_app_bug`
Files not syncing, uploads/downloads failing, desktop or mobile client
crashes, website errors, installer and browser problems.

**Why:** v1's `sync_technical_bug`, renamed and **narrowed** by carving
out `service_outage` and `sharing_permissions`. Deliberately **not**
split further into app-vs-sync -- see Part 5. Still the largest intent,
and that's fine: it's genuinely the brand's biggest topic.
**Escalation:** Auto-handle standard troubleshooting; escalate on
recurrence (signalled via `turn_type`, not a separate intent).

### 7. `service_outage` *(new)*
"Is Dropbox down?", service-wide unavailability, status-page questions,
post-incident complaints.

**Why:** ~146 broad / ~16 tight matches, and the treatment is the
**opposite** of a sync bug: no troubleshooting, no per-ticket
escalation, point at the status page, acknowledge. It's also a **mass
event** -- one incident produces hundreds of near-identical tickets, so
correct classification here has the largest volume payoff of any intent.
Examples: *"is dropbox down? we cant access it"*, *"Dropbox Down in
Taiwan, China, and India right now"*, *"Seriously? After this past
weekend's total outage?"*
**Escalation:** Auto-handle (acknowledge + status link). Never run
individual troubleshooting.

### 8. `sharing_permissions` *(new)*
Shared folders and links, access and permission problems, team folders,
invite/join failures, link expiry and visibility.

**Why:** ~82 matches. Dropbox's **core collaborative feature**, and
arguably the #1 real support topic after sync. v1 splits it three ways
-- a broken share is `sync_technical_bug`, "how do I share read-only" is
`how_to_usage`, "add short URLs" is `feature_request`. The action
difference that earns the slot is the **data-exposure escalation
trigger**, which no other intent carries: an accidentally public folder
is not a sync bug and not a how-to. Examples: *"wee bug on accepting a
shared folder"*, *"This is precisely what I don't want: my information
still available to folder members"*, *"The hard bounce on permissions is
a real pain."*
**Escalation:** Auto-handle (permissions are explainable); escalate
immediately if unintended data exposure is implied.

### 9. `data_loss_recovery`
Deleted or missing files, restore and version-history requests.

**Why:** Kept from v1 **unchanged** -- see Part 1, where it's the
strongest decision in v1.
**Escalation:** Always escalate. Permanent data loss risk.

## Guidance & product

### 10. `how_to_usage`
"How do I X" -- nothing is broken, the customer needs guidance.

**Why:** Kept from v1 **unchanged**. Shrinks in v2 only because sharing
and storage how-tos move to their own intents.
**Escalation:** Auto-handle. Deflecting to DM here is a **failure**, not
a resolution -- give the actual answer or a doc link.

### 11. `feature_request`
Suggestions, product feedback, "you should add X".

**Why:** Kept from v1 **unchanged**. Its v1 flaw -- bleeding into the
catch-all when a request arrives as a rant -- is fixed by #12 existing,
plus `secondary_intent`. Deliberately **not** merged with #12: see
Part 5.
**Escalation:** Auto-handle (acknowledge + log to product). No risk.

## Conversational / non-support

### 12. `complaint_dissatisfaction` *(new -- split from the catch-all)*
Frustration, rants, service criticism, competitor comparisons, churn
threats.

**Why:** ~121 churn/comparison matches ("switching to Google Drive",
"worst", "useless", "cancel my"). A **retention** signal with its own
escalation trigger. Lumping it with praise -- as v1 does -- makes the
catch-all actively misleading: one label covering the happiest and the
angriest customers in the dataset. Separating anger from praise is the
half of the catch-all split that changes the action; separating praise
from spam is not (see #13). Example: *"awful customer service
experience. Would not recommend doing business with your company at
all."*
**Escalation:** Escalate when `churn_threat` or high anger is flagged;
otherwise acknowledge.

### 13. `no_action_needed`
Praise and thanks, spam, unrelated mentions, jobs/press/sales
enquiries, tweets too fragmentary to route.

**Why:** v2's one deliberately coarse bucket, and the coarseness is
rule-consistent: every member gets the same treatment -- acknowledge
warmly or ignore, with **no support work either way**. An earlier draft
split praise into its own intent; cut, because "reply thanks" and "reply
nothing" are not different routes. What made v1's catch-all harmful was
mixing *angry* with *happy* and hiding mid-thread turns inside it --
fixed by #12 and by `turn_type` respectively.
**Escalation:** No-op, or acknowledge. Route sales/press out of the
support queue.
**Health metric:** if this exceeds ~20% of hand-labeled examples, v2 has
its own missing intent. v1's equivalent was 21% *while also* hiding four
distinct things; this one should be smaller and homogeneous. Track it.

---

# Part 4 -- Not intents: `turn_type` and flags

Putting any of these in the intent enum would repeat v1's
`followup_ticket_status` mistake. This is also where the signal from cut
intents is preserved.

## `turn_type` -- the important one

Covers the 36.7% of the corpus that is mid-thread.

| Value | Meaning |
|---|---|
| `first_contact` | New issue, no prior turn |
| `providing_requested_info` | Answering an agent's question ("Yes background uploading is on") |
| `nudging_no_response` | Chasing a stale ticket -- **replaces v1's `followup_ticket_status`** |
| `confirming_resolved` | "That worked, thanks" -- close the ticket |
| `disputing_prior_answer` | The previous answer was wrong or didn't help |

This absorbs `followup_ticket_status` *and* the mid-thread 58% of the
catch-all, and it makes `README.md`'s rule -- *"inherit the original
ticket's intent/action"* -- actually implementable, since the intent
field is now free to carry the real topic.

## Flags

| Field | Why it isn't an intent | Evidence |
|---|---|---|
| `wants_human` (bool) | **Carries the cut `human_agent_request` intent.** "Give me a phone number" is a routing signal that co-occurs with a topic -- 52% of these tweets (17 of 33) also carry a real topic, so as an intent it discards the subject on half of them. As a flag it fires an unconditional auto-handle bypass while the topic survives | 33 matches |
| `legal_sensitive` (bool) | **Carries the cut `account_closure_data_deletion` intent.** GDPR/CCPA erasure, closure, terms disputes -- routes identically to billing (escalate), but carries statutory deadlines the human queue needs to see | ~12-16 matches |
| `secondary_intent` (nullable) | v1's primary-intent rule discards the other half; recording it costs no slot and stops ambiguous cases from *looking* unambiguous, which would inflate measured accuracy | Stated in v1 |
| `churn_threat` (bool) | Applies across billing, bugs and complaints alike -- not a topic | ~121 matches |
| `sentiment` / `anger` | Drives escalation independently of topic | Throughout |
| `language` | Orthogonal to topic. Open question in `README.md`: flag or filter? | 67 non-English tweets |
| `abusive_content` (bool) | Needs a safety route, not a support route | Present |
| `needs_human_triage` | `README.md` already says low confidence escalates regardless of intent -- make it an explicit output, not an implicit threshold | Design |

**Why the flag/intent split matters for budget:** flags cost prompt
tokens and a labeling checkbox, but **no golden-set slots** -- they
don't need 15 examples each to be measurable. That asymmetry is why
moving `human_agent_request` and `account_closure_data_deletion` into
flags costs far less information than dropping two intents sounds like
it should.

---

# Part 5 -- Considered and rejected

Each is a real, distinguishable topic. Each failed the decision rule:
**it doesn't change what the agent does.**

| Candidate | Volume | Why rejected | Signal preserved as |
|---|---|---|---|
| `human_agent_request` | 33 | A routing signal, not a topic -- and 52% co-occur with a real topic, which an intent slot would discard. Same shape as v1's `followup_ticket_status` mistake | `wants_human` flag |
| `account_closure_data_deletion` | ~12-16 | Routes exactly as `billing_subscription`: escalate, human verification, never auto-handle. Legal sensitivity is a *property*, not a route | `legal_sensitive` flag |
| `positive_feedback_thanks` | large | "Reply thanks" and "reply nothing" are not different routes. The harmful half of v1's catch-all was mixing praise with *anger*, which #12 fixes | folded into `no_action_needed`; "that worked" is `turn_type = confirming_resolved` |
| `app_platform_bug` (crashes, installer, OS-specific) | ~433 mentions | Highest-volume rejection. Same resolution path as a sync bug: standard troubleshooting, escalate on recurrence. Would raise label cost and change no decision | -- |
| `integration_thirdparty` (Paper, Smart Sync, API, Office) | ~304 broad -- noisiest regex of the set | The closest call. The ~304 is inflated by keyword collisions, and "not our bug" is a *reply* difference, not a *routing* difference. **Revisit if hand-labeling shows it's real** | -- |
| `notification_email_preferences` (unsubscribe, popup nags) | ~29 | Real but small, and resolves as ordinary guidance -- `how_to_usage` covers it | -- |
| `presales_sales_partnership` (quotes, enterprise, jobs) | ~17-28 | Same action as noise: route out of the support queue | `no_action_needed` |
| Merging `feature_request` + `complaint_dissatisfaction` | -- | **Considered and declined.** Same escalate/auto-handle decision, but different *replies* (retention response vs. "thanks, logged") and different destination queues (retention/CX vs. product backlog). Clears the rule on reply content | -- |
| Splitting `data_loss_recovery` by cause | -- | Same action regardless of cause: escalate | -- |

## A load-bearing assumption, stated explicitly

Four v2 intents share the action "always escalate": #2, #4, #9, and
parts of #12. They stay separate because their **replies and
destination queues** differ -- security, billing, data recovery,
retention.

**If escalation is a single undifferentiated "send to a human" queue,
that reasoning collapses and those four are one intent** -- taking v2 to
~10. This taxonomy assumes escalation has sub-routes. If it doesn't,
revisit before labeling.

---

# Part 6 -- Budget: why the list stops at 13

`golden_set.md` requires **>=15 examples per intent**; the assignment
caps the golden set at **250**. That cap, not judgment, sets the
ceiling.

| Intents | Minimum floor (15 each) | Headroom under the 250 cap |
|---|---|---|
| 8 (v1) | 120 | 130 |
| 12 | 180 | 70 |
| **13 (v2)** | **195** | **55** |
| 15 (earlier draft) | 225 | 25 -- too tight |
| 19 (all candidates) | 285 | **over cap -- impossible** |

**Why not 15.** At a 225 floor the minimums consume 90% of the cap,
leaving almost nothing for the natural distribution of the main random
sample. Since the top-up deliberately over-samples rare intents, a pool
that is ~90% floor is *almost entirely* top-up -- which would spread the
selection bias `golden_set.md` flags as a known limitation from the rare
tail across the **entire** golden set. 55 examples of headroom is the
difference between a golden set with a biased tail and one that is bias
all the way down.

**If the budget tightens (13 -> 12):** merge `storage_quota_plan_limits`
back into `billing_subscription`, accepting that its technical half gets
misrouted. It is the designated next cut because its case rests on
resolving an *ambiguity*, while every other intent's rests on a
*distinct escalation trigger* -- ambiguity is the cheaper thing to lose.
**No other intent should be merged to make room:** if something else
can't reach 15 examples, that's evidence the intent isn't real, and it
should be cut rather than padded.

---

# Part 7 -- Migration from v1 to v2

| v1 intent | Where it goes in v2 |
|---|---|
| `account_login` | `account_access`; incidents -> `security_account_compromise` |
| `billing_subscription` | `billing_subscription`; space/plan -> `storage_quota_plan_limits`; absorbs closure/erasure + `legal_sensitive` |
| `sync_technical_bug` | `sync_app_bug`; outages -> `service_outage`; sharing -> `sharing_permissions` |
| `how_to_usage` | unchanged (shrinks as sharing/storage how-tos move out) |
| `data_loss_recovery` | unchanged |
| `feature_request` | unchanged |
| `followup_ticket_status` | **dissolved** -> `turn_type = nudging_no_response` |
| `general_feedback_or_other` | **dissolved** -> `complaint_dissatisfaction`, `no_action_needed`, and for its mid-thread majority `turn_type` |
| *(not modelled in v1)* | `phishing_abuse_report` -- previously scattered across the catch-all and `sync_technical_bug` |

**Net 8 -> 13:** four kept unchanged, two narrowed, one renamed +
narrowed, two dissolved, five new. Plus `turn_type` and 8 flags, none of
which existed in v1.

## Implementation checklist

1. Replace `intents.json` with the 13 v2 intents; keep `intents.md` as
   the v1 record and point it here.
2. Extend the response schema in `groq_lib.py`: `intent`,
   `secondary_intent`, `turn_type`, flags.
3. **Bump the cache key in `cache.py`** to include a taxonomy version --
   `tweet_id` alone would serve stale v1 labels, and versioning keeps v1
   and v2 results comparable for the report.
4. Wire `07_build_golden_candidates.py` through
   `classify_runner`/`cache.py` (already an open item in `PROGRESS.md`)
   before re-running, so the top-up gets caching + concurrency.
5. Re-classify the existing 250 candidates under v2; re-run the top-up
   to a 195 floor.
6. Update `08_label_golden_set.py` to collect the new fields. Keep the
   anti-anchoring design unchanged.
7. Rewrite `README.md`'s escalation matrix for 13 intents; delete the
   conditional rules that existed only because `account_login` and
   `sync_technical_bug` were overloaded.

---

# Part 8 -- Coverage test

30 tweets drawn at random (`random_state=999`) from the 4,504 unique
customer tweets -- deliberately **not** the reading sample or the golden
pool, so none had informed the taxonomy.

| Result | Count |
|---|---|
| Mapped cleanly to a v2 intent | **28 / 30** |
| No good home | **2 / 30** |

**Both misses were phishing reports** -- *"What's the best way to report
phishing invites?"* and *"I just received a poorly made spam email
trying to spoof you guys. Do you want a copy of it?"* -- which is what
put `phishing_abuse_report` back into the taxonomy after an earlier
draft cut it on a volume assumption. A follow-up count found **21** such
tweets, clearing the 15 floor. With #3 included, coverage is **30/30**.

Two things the test surfaced that are *not* taxonomy gaps:

- **Billing is over-escalating.** *"I am looking at Dropbox Pro, but I
  need a Mexican Authority Rated invoice to be tax deductible -- do you
  issue?"* is a factual pricing/invoicing question with no transaction
  to authorize. Fixed by the auto-handle sub-rule on #4, not by a new
  intent.
- **The flags did real work.** *"it started but didn't complete... what
  is the phone number i can call?"* resolves as `sync_app_bug` +
  `wants_human` -- both signals kept. Under v1 it would have been a
  single label, and under the earlier 15-intent draft the topic would
  have been discarded in favour of the human request.

## Second sample (n=40, `random_state=20260910`)

Run to check the first result wasn't luck. Again drawn at random, none
previously seen.

| Result | Count |
|---|---|
| Mapped to a v2 intent | **40 / 40** |
| No good home | **0** |
| Marginal fit (covered, but the boundary is unclear) | 3 |

**No new missing intent.** Across both samples that is 70 random tweets
with zero uncovered cases once `phishing_abuse_report` is included.

The three marginal cases are **definition-wording problems, not missing
intents**, and each needs a line in `intents.json` before labeling:

| Tweet | Issue | Fix |
|---|---|---|
| *"you guys completely rebranded. Did any new features come alongside this or just visual stuff?"* | Product-info question; not "how do I X", not a request | State that `how_to_usage` covers product questions generally, not only task instructions |
| *"Is there a way to recover my Dropbox once it's closed?"* | Sits between `billing_subscription` (closure) and `data_loss_recovery` | Rule: account state -> billing; file contents -> data loss |
| *"why is the app taking over 800 MB of space on my iPhone? I have no offline files"* | **Device** disk usage, not **account** quota | `storage_quota_plan_limits` is account storage only; local disk is `sync_app_bug` |

The flags again did real work: *"To whom a can talk to cancell my acount
it is URGENT"* -> `billing_subscription` + `wants_human`, and *"thank you
for not getting back to me. Consider the account closed #migrating"* ->
`complaint_dissatisfaction` + `churn_threat` + `turn_type =
nudging_no_response`. All three signals survive; v1 would have kept one.

## What these tests do and don't establish

**Established:** the 13 intents are *exhaustive* over this brand's
traffic. Two independent random samples, 70 tweets, no uncovered case.
That is the question "is 13 enough?" actually asks, and the answer is
yes.

**Not established -- two open risks:**

1. **Separability is untested.** One judge (me), who also designed the
   taxonomy. These samples show the categories *cover* the data, not
   that two labelers would *pick the same one*. Only hand-labeling
   answers that. Watch the three marginal cases above as the likely
   disagreement sites.
2. **`sync_app_bug` is heavily concentrated** -- roughly 15 of 40 in
   this sample. Even after carving out outages and sharing, it is by
   far the largest class in real traffic, so a headline accuracy number
   will be dominated by it. Report per-class metrics and a macro
   average, not just overall accuracy, or this one class will hide
   everything else. This is an eval-design consequence, not an argument
   for splitting it -- the decision rule still says it's one intent.

Before adopting v2, still run a stratified read sample weighted toward
v1's `general_feedback_or_other` and `sync_technical_bug`, where the v1
problems concentrate.

---

# Part 9 -- For the report

The v1 -> v2 diff is usable evidence in two of the mandatory sections.

**"What's misleading about my headline number":** a v1 accuracy figure
would look *better* than the agent actually is. Coarse buckets are
easier to hit -- a 21% catch-all plus a 26% mega-bucket means nearly
half the traffic could be scored correct by two very forgiving labels.
Reporting accuracy against a taxonomy that hides the hard distinctions
measures the taxonomy, not the agent. Corollary worth stating plainly:
**v2 will score lower than v1 would have, and that is the point.**

**"What I chose not to build":** Part 5 plus the Part 6 arithmetic is a
concrete, quantified scoping decision rather than a post-hoc
rationalization -- including the three intents cut from v2's own drafts,
the flags that absorbed their signal, and the one cut (`phishing_abuse_report`)
that the data reversed.

Also carry forward: the v1 non-determinism finding (tweet 1274528
classified differently on two runs) applies unchanged to v2, and a finer
taxonomy will likely make it **worse** -- more categories means more
adjacent boundaries to wobble across. Budget for temperature=0 or
multi-sample runs in the eval harness.

---

# Part 10 -- Caveats on the numbers

**Volume figures are regex keyword counts**, not hand-verified labels,
computed over the 4,504 unique customer tweets in
`data/dropbox_paired.csv`. They over-count (a tweet mentioning "iPhone"
is not necessarily a platform bug) and double-count across categories.
They are strong enough to justify *proposing* an intent; they are not
distribution estimates. The two that lean hardest on unverified counts
are `sharing_permissions` (~82) and `storage_quota_plan_limits` (~230
broad / ~24 tight) -- validate those first.

**Two figure sets are exact**, not estimates:

- The mid-thread percentages, computed by joining `customer_tweet_id`
  against `in_response_to_tweet_id` in the raw Kaggle dataset.
- The v1 pool distribution in Part 1, read directly from
  `data/golden_candidates.csv` -- though those are the classifier's
  `suggested_intent`, **not ground truth**, and the top-up
  over-samples rare intents, so they are not a traffic estimate either.

**The 21 phishing matches** are a conservative regex over report-shaped
phrasing; the true count is likely higher, but 21 already clears the
floor.

---

# Part 11 -- v2.1 precedence rules (added 2026-09-12)

Parts 3 and 4 define what each intent *is*. They never define what wins
when a message satisfies two at once, and that omission is the single
largest source of error in the golden set: of the 71 intent
disagreements, 58 touch the
`complaint_dissatisfaction` / `how_to_usage` / `storage_quota_plan_limits`
/ `sync_app_bug` cluster, and adjudication (`ADJUDICATION.md`,
`data/adjudication_v2.csv`) found the *label*, not the model, at fault in
most of them.

These rules are the fix. They are written to be applied identically by a
human labeler and by an LLM, which means every one states a trigger, the
winning intent, why the loser loses, and worked examples in both
directions. Tweet ids refer to rows in `data/golden_labels.csv`.

**Distribution note, and the reason this Part exists at all.** Five rules
already written elsewhere in this document were violated during labeling
(quota-not-clearing, device-disk exclusion, sharing how-tos, the pricing
sub-rule, "nothing is broken"). The cause was not disagreement -- it was
that `08_label_golden_set.py` shows the labeler bare intent names and the
classifier prompt shows only `intents.json`'s one-liners, so **neither
consumer of this file ever reads this file.** A rule that lives only here
is a rule that does not exist. Anything added to Part 11 must also be
pushed into the prompt and the labeling tool, or it will drift again.

## Precedence ladder

Apply in order. The first rule that fires decides the intent.

| order | rule | question it answers |
|---|---|---|
| 1 | R9b | Is this routable at all without a prior turn? |
| 2 | R1 | Does a grievance frame beat the topic under it? |
| 3 | R2 | Is something broken, or is the customer learning? |
| 4 | R3 | Sharing object vs. sharing problem |
| 5 | R4, R5 | Which side of the storage boundary |
| 6 | R6 | Storage vs. billing |
| 7 | R7 | How-to vs. account access |
| 8 | R8 | One device vs. the whole service |
| 9 | R9a | Is there genuinely no support work? |

---

## R1 -- Grievance frame vs. topic  *(Boundary 1)*

Negative sentiment is not an intent. It is already carried by
`sentiment`, `churn_threat` and `abusive_content`. An intent must
describe **the work the message creates**, and a message can be furious
and still create ordinary work.

### R1a -- an actionable request beats any frame

**Trigger:** the message contains an answerable question or a request
support can act on or route.
**Wins:** the topic intent for that request.
**Why complaint loses:** routing to `complaint_dissatisfaction` discards
the work item and produces an apology where an answer was asked for.

*"Actionable" means support can answer or act.* A demand that the company
change its prices is not actionable; a question about what a plan costs
is.

- **Positive:** `838554` *"The auto checkbox & star features are making my
  business user experience worthless. How do I disable?!"* → `how_to_usage`.
  Two grievance markers and a real question; the question wins.
- **Positive:** `1339451` *"How can I control reminders for Showcase
  emails? I can see this annoying some clients"* → `how_to_usage`, with
  `feature_request` as `secondary_intent`.
- **Negative:** `1050476` *"whoever is controlling your Jira board needs to
  be fired"* → **not** a topic intent. Nothing is asked; R1a does not fire.
- **Counterexample:** `847909` *"In Brazil is so expensive this service. You
  ougth to review this price."* "You ought to review this price" looks
  like a request but support cannot act on it → R1a does not fire, falls
  through to R1c.

### R1b -- product feedback is `feature_request`, not complaint

**Trigger:** no actionable request, and the message either proposes a
change to a product feature, design, capability or packaging, **or**
evaluates a specific product element.
**Wins:** `feature_request`.
**Why complaint loses:** this is backlog input, and it is not a retention
signal. Filing it as a complaint loses it.

- **Positive:** `1960138` *"I don't like the new @Dropbox tray icon, every
  time I see it I think there's something wrong with it"* → `feature_request`.
- **Positive:** `53189` *"It would be great if they offered a little bit
  more space. I would pay more if it did."* → `feature_request`: this
  proposes a **packaging** change.
- **Negative:** `1856823` *"Just crazy the new prices for Dropbox plans.
  Geez!"* → **not** `feature_request`. It objects to cost without
  proposing any product or packaging change.
- **Counterexample (the pricing split, and the sharpest edge here):**
  "offer more storage at this price" is `feature_request`; "your prices
  are too high" is `complaint_dissatisfaction`. Both mention money. The
  test is whether a specific product or packaging change is proposed.

`1960138` is the row that forced R1b to exist: an earlier draft of R1
sent all evaluative-with-no-request messages to
`complaint_dissatisfaction`, which would have relabeled this **correct**
human label wrong.

### R1c -- dissatisfaction with the service, company, support or cost

**Trigger:** no actionable request, no product change proposed, and the
target is the service, the company, the support experience, reliability
in general, or the cost of the product.
**Wins:** `complaint_dissatisfaction`.
**Why a topic loses:** there is no work item under the grievance. A topic
label would promise an answer the message never asked for.

- **Positive:** `1050476` *"whoever is controlling your Jira board needs to
  be fired, this was not a priority"* → the target is the company.
- **Positive:** `193025` *"as a small non-profit, we can't afford to pay
  $500 for something we're not going to use"* → objects to cost, asks
  nothing.
- **Negative:** `1497111` *"So Dropbox bills me $990 Cdn for 'free trial'
  ... and now tells me their policy does not allow a refund. Nice fraud!"*
  → **not** complaint. It asserts a correctable billing fact → R1d.

### R1d -- a correctable factual claim routes to its topic

**Trigger:** the message is evaluative and asks nothing, but asserts
something about **how the product or an account works** that support can
confirm, correct or explain.
**Wins:** the topic intent for that mechanic.
**Why complaint loses:** the customer is operating on a belief support can
address. An apology leaves them with the wrong belief.

- **Positive:** `622045` *"it's 2017 the fact that one shared folder uses up
  my entire quota is not cool"* → `storage_quota_plan_limits`. The quota
  mechanic can be explained.
- **Positive:** `2551167` *"Dropbox is a RIPOFF! They charge customers for
  TRIALS before they are OVER!!"* → `billing_subscription`.
- **Negative:** `1577541` — a sarcastic parody of a Dropbox upsell ending
  in profanity → **not** a topic. It asserts nothing checkable about
  mechanics; it evaluates the company → R1c.
- **Counterexample / scope limit:** the claim must be about **product or
  account mechanics**, never about the company's conduct or value for
  money. *"Has Dropbox Support always been this bad?"* asserts something
  about the company and stays in R1c.

### R1e -- commentary aimed at peers suppresses R1d

**Trigger:** the message addresses other users rather than the brand
(a reply to another customer, third-person "they").
**Wins:** suppresses R1d only. R1b and R1c still apply normally.
**Why:** R1d exists because support can correct a belief *in a reply to
that customer*. When the message is not a support interaction, there is
no such reply to write.

- **Positive:** `2179658` *"If you want even more fun, download the app for
  your computer and it syncs everything! Filling your hard drive..."* —
  a reply to another user → `complaint_dissatisfaction`, not a sync or
  storage topic, even though selective sync would answer it.
- **Negative:** `2421521` *"Has anyone tried Dropbox Professional? ...
  Would do it in a heartbeat if they offered more space."* — also aimed at
  peers, but it proposes a packaging change, so R1b still fires →
  `feature_request`. R1e does not demote everything it touches.

---

## R2 -- "how do I" is surface form, not intent

**Trigger:** the message is phrased as a how-to.
**Wins:** `how_to_usage` **only if nothing is malfunctioning**. If
something is not behaving as designed, the malfunction's topic intent
wins regardless of phrasing.
**Why how_to loses:** Part 3 defines `how_to_usage` as *"nothing is
broken"*. A broken thing routed to how-to gets instructions for a path
that is already failing.

- **Positive:** `1863374` *"Did the taskbar icon change? I'm missing the
  green check"* → `how_to_usage`: the icon changed by design.
- **Negative:** `988433` *"changes reverted, how do I fix this?"* →
  `sync_app_bug`. The "how do I" is real but something is broken.
- **Negative:** `1999298` *"I have permanently deleted my deleted items,
  but still have 'ran out of Space' notice. How do we fix it?"* →
  `storage_quota_plan_limits` via R4.
- **Counterexample:** `615215` *"why did you remove the green check"* is
  phrased as a grievance, not a how-to, and still lands in
  `how_to_usage` — R2 cuts both ways, and phrasing decides nothing in
  either direction.

`615215` and `1863374` are near-identical in meaning and received
different labels in v1 (`complaint` and `how_to_usage`). They share one
label under R2. `11_label_consistency.py` could not catch that pair:
their TF-IDF cosine similarity is **0.153**, because the audit compares
wording and these tweets share almost no vocabulary.

---

## R3 -- Sharing object vs. sharing problem

### R3a -- subject matter

**Trigger:** the subject matter is a shared link, shared folder, team
folder, membership or permission -- **including how-tos.**
**Wins:** `sharing_permissions`.
**Why how_to loses:** already stated in Part 3 (*"how_to_usage shrinks in
v2 only because sharing and storage how-tos move to their own intents"*),
and violated three times during labeling.

- **Positive:** `259327` *"how can I eliminate a Team folder?"* →
  `sharing_permissions`, not `how_to_usage`.
- **Positive:** `2607972` *"wee bug on accepting a shared folder"* →
  a join/membership action.
- **Negative:** `2976667` *"I don't seem to have enough space on my laptop
  to connect my work Dropbox"* — mentions a work Dropbox but the subject
  is device disk → R5.

### R3b -- symptom carve-out (this is the narrowing)

**Trigger:** the object is shared, but the actual malfunction is a quota
problem or a client/app failure (load, save, sync, crash).
**Wins:** `storage_quota_plan_limits` or `sync_app_bug` respectively --
**not** `sharing_permissions`.
**Why sharing loses:** the customer needs quota mechanics or
troubleshooting; a permissions answer sends them somewhere that is not
broken.

- **Positive:** `2956783` *"I use a shared Dropbox between 6 people.
  Everyone can add to the Dropbox except for me. It says my Dropbox is
  full."* → `storage_quota_plan_limits`.
- **Positive:** `1600556` *"i can't open a shared link. Just keeps loading
  the files but never shows them"* → `sync_app_bug`: a rendering failure,
  not a permission denial.
- **Negative:** `37940` *"I can't access my shared folder!"* →
  `sharing_permissions`. This **is** an access failure, so R3b does not
  fire.
- **Counterexample:** an earlier draft of R3 said "object beats symptom"
  without qualification. Applied to `2956783` it would have relabeled a
  **correct** human label wrong and handed the customer a permissions
  answer to a quota problem. That draft is void.

---

## R4 -- Quota not clearing is storage, never sync

**Trigger:** files deleted (or permanently deleted) and the storage
reading does not change.
**Wins:** `storage_quota_plan_limits`.
**Why sync loses:** the files did move; what has not updated is the quota
accounting. Sync troubleshooting does not touch it.

- **Positive:** `79858`, `2910246`, `1999298`, `2053253`, `2629478` — five
  rows, three different v1 labels between them (`sync_app_bug`,
  `how_to_usage`, `storage_quota_plan_limits`). One label now.
- **Negative:** `2976667` — no deletion involved; it is device disk → R5.
- **Counterexample:** a quota that is full because a *shared folder*
  counts against it is still R4/R3b territory, not `sharing_permissions`.

## R5 -- Account quota is not device disk

**Trigger:** the constrained resource is the local machine's disk.
**Wins:** whatever the customer actually needs (usually `how_to_usage`
for selective sync, sometimes `sync_app_bug`) -- never
`storage_quota_plan_limits`.
**Why storage loses:** Part 3 states the exclusion verbatim (*"not device
disk usage"*), and the answer -- selective sync -- has nothing to do with
the plan.

- **Positive:** `2976667` *"not enough space on my laptop... If I keep my
  external hard drive plugged in, will this solve the problem?"* →
  `how_to_usage`.
- **Negative:** `79858` — account-side quota → R4.
- **Counterexample and open worklist item:** `2028354` *"installed DB on my
  acer windows 10 and almost ran out of space"* was labeled
  `storage_quota_plan_limits` by **both** the human and the model, so no
  disagreement ever surfaced it. R5 says both are wrong. See "Known
  incompleteness" below.

## R6 -- Storage vs. billing  *(Boundary 4)*

**Trigger:** the message carries both a storage/quota subject and
plan/price/payment language.
**Decide on what the customer must do next, never on keywords.**

- **R6a → `billing_subscription`** when the answer is a price, an invoice,
  a refund, or a purchase/upgrade decision.
- **R6b → `storage_quota_plan_limits`** when the answer is how storage is
  counted, why a quota is full, or what a tier's storage limit is.

**Why the loser loses:** R6a routes a transaction (which escalates);
R6b routes an explanation (which auto-handles). Getting it backwards
either escalates a FAQ or auto-answers a payment dispute.

- **Positive (a):** `2742476` *"what is the way to add more space to the
  current plan and how much does it cost?"* → the answer is a price.
- **Positive (b):** `1324437` *"If I have a Plus plan and sharing folder
  with a standard (free plan) user, are they limited to 2GB of uploads?"*
  → the answer is a storage limit.
- **Negative:** `193025` *"we can't afford to pay $500"* → neither. No
  question is asked at all, so R1c fires first and it is a complaint.
- **Counterexample:** both `1901120` (*"Earned 25gb of space via HP
  Promotion. Is there any expiry?"*) and `1992403` (*"cost of the Plan
  that would give me unlimited storage"*) contain "storage" and a plan.
  The first is R6b, the second R6a. Keyword matching cannot separate
  them; "what must the customer do next" can.

## R7 -- How-to vs. account access  *(Boundary 2)*

**Decide on the customer's goal, never on the words "login", "account",
"access" or "how do I".**

- **R7a → `account_access`** when the customer **cannot reach or
  authenticate to** an account: locked out, password reset, 2FA or backup
  codes, cannot prove ownership, or needs a business account provisioned.
- **R7b → `how_to_usage`** when the customer **is authenticated** and
  wants to learn to perform an operation -- even an operation *about*
  accounts.

**Why account_access loses in R7b:** its escalation default is
`escalate`, because identity cannot be verified in a public reply. An
operation question routed there gets escalated for no reason, which is
one of the 18 over-escalations in the v1 evaluation.

- **Positive (b):** `2222820` *"I am getting notifications from my work
  account on my home computer. How can I unlink their account from
  home?"* → authenticated, wants an operation → `how_to_usage`.
- **Positive (b):** `3485` *"when I follow the link, dropbox asks me to log
  in with my biz account. I want to apply it to my personal account"* →
  can authenticate to both; wants the link to open under a different one.
- **Positive (a):** a customer who cannot remember which email their
  account is under **is** blocked from reaching it → `account_access`.
- **Counterexample:** `1874043` *"Sad cos I can't remember which email my
  first cudi concert is under"* matches that phrasing exactly and is about
  **concert tickets, not Dropbox** → `no_action_needed` via R9a. Phrase
  matching would have routed it to `account_access`, which is what the
  classifier did.

**Documented edge case, unresolved:** account *linking and unlinking*
sits on this seam by construction -- it is an operation (R7b) performed
on an access relationship (R7a). Both `2222820` and `3485` are settled as
`how_to_usage` above because in both the customer is authenticated
throughout. A message where unlinking has **locked the customer out**
flips to R7a. Confidence on these two is `medium` for that reason.

## R8 -- Device symptom vs. service outage  *(Boundary 3)*

**Trigger:** any message mentioning the service being down, slow, or
broken.

- **R8a → `sync_app_bug`** when a device- or account-specific symptom is
  described, **even if the customer speculates about an outage.** The
  outage question becomes `secondary_intent`.
- **R8b → `service_outage`** when the message asserts or asks about
  service-wide unavailability with **no device-specific symptom**, or
  cites external evidence of scope (the status page, a team all affected,
  a third-party outage tracker).

**Why service_outage loses in R8a:** its reply is a status-page link. A
customer with a specific broken thing gets nothing from that, and the
described symptom is the only actionable content in the message.
**"Anyone else having this?" is not evidence of scope** -- it is a
customer polling strangers, not a report of one.

- **Positive (a):** `1788320` *"my desktop app (Mac) is syncing at a
  snail's pace today. Is the service having issues?"* → `sync_app_bug`,
  `secondary_intent = service_outage`. **This is the row the boundary was
  written for.** A concrete symptom ("my Mac desktop app", "today") is
  present, so the speculation does not govern.
- **Positive (a):** `512851` *"Anyone else having sync issues... client is
  making changes to shared sheet, I can't see them, same in reverse. Not
  the first time."* → `sync_app_bug`. Polls peers, describes a recurring
  personal problem.
- **Positive (b):** `1804904` *"having some other tech glitches so I (and
  now my whole team) went to [status page] and: What is going on?"* →
  `service_outage`: status page cited, whole team affected.
- **Positive (b):** `1383896` *"This is not the moment to stop working
  @DropboxSupport"* → `service_outage`: asserts the service stopped, no
  device symptom.
- **Negative:** a bare *"Is Dropbox down?"* with no symptom is R8b.
  The same tweet with *"my files won't sync and is Dropbox down?"* is R8a.

## R9 -- `no_action_needed`, and what it is not  *(Boundary 5)*

### R9a -- when it is valid

**Trigger:** praise or thanks, spam, jobs/press/sales, or genuinely
off-topic -- **and** there is no support work to do.
**Wins:** `no_action_needed`.

- **Positive:** `1874043` — about concert tickets, not Dropbox.
- **Negative:** `1331572` *"It's not going to show up in a screenshot. It
  was a little pop up that displayed on my screen and then disappeared."*
  → on-topic, and there **is** support work; it is simply unreadable
  alone → R9b.

### R9b -- insufficient context is NOT `no_action_needed`

**Trigger:** the message is on-topic but carries no recoverable topic
without the prior turn -- typically `providing_requested_info`,
`nudging_no_response` or `disputing_prior_answer`.
**Wins:** nothing. **The 13-intent taxonomy has no intent for this, and
that is a documented limitation, not something to paper over.**

Filing these under `no_action_needed` is what v1 did, and it reintroduces
exactly the disease v2 was built to cure: Part 2 condemns v1's catch-all
for *"hiding mid-thread turns inside it"*, and `no_action_needed`
had begun doing the same thing.

Six golden rows are affected (`1041902`, `74467`, `1687134`, `1972693`,
`2828924`, `1331572`). In `data/golden_labels_v2.csv` they carry
`v2_status = insufficient_context` and are **excluded from intent
accuracy** rather than assigned a label nobody can defend.

**Options, none yet chosen:**
1. Add an `insufficient_context` intent (14th). Costs a golden-set
   minimum and makes `turn_type` partly redundant.
2. Keep 13 intents and pass the prior brand turn into the classifier so
   these become routable -- the STEP 4 direction, and the only option
   that recovers the *content* rather than just labeling the gap.
3. Keep excluding them and report the exclusion. Current interim state.

Option 2 is the one that makes the problem go away rather than naming it,
and `dropbox_paired.csv` already carries `brand_text`.

### R9c -- considered and rejected: a `taxonomy_gap` status

A draft of these rules invented a `taxonomy_gap` status for `1960101`
(*"can you dm me a customer service phone number please"*), on the
grounds that no intent covers "which channel can I reach you on".

**Rejected.** The 2C sweep found `2700303` (*"No phone support for Dropbox
Plus members?"*), where the human and the classifier independently agreed
on `how_to_usage`. Support-channel availability is an answerable product
question, so R1a already covers it and `1960101` is `how_to_usage`.
Escalation is carried by `wants_human`, which is what that flag is for.

Recorded because the near-miss is the point: a new taxonomy category was
one row away from being added on the strength of a single example.

---

## Known incompleteness of the v2 labels

`data/golden_labels_v2.csv` applies these rules to the **58 reviewed
rows** -- every row where the human and the classifier disagreed inside
the contested cluster. It does **not** apply them to the 131 rows that
were never reviewed.

That matters directionally. Adjudication only ever looked where the model
was **already scored wrong**, so every correction it could possibly find
raises the score. Rows where the human and the model were wrong *in the
same way* produce no disagreement and were invisible to the entire
process.

A sweep of the 23 unreviewed rows that sit in the four contested intents
and on which human and model agreed found **5 that these rules would
move** -- `933440`, `715979`, `2826329`, `2028354`, `1243081` (recorded in
`scripts/17_golden_v2.py:FLAGGED_AGREED`). They are deliberately **not**
relabeled: acting on a keyword sweep rather than a read is the sloppy
version of STEP 1. All 5 would *lower* the measured accuracy.

**Therefore the accuracy measured against v2 labels is an optimistic
bound, not a settled number**, and any figure quoted from it must say so
along with the fact that the v2 labels are one reviewer's adjudication,
self-rechecked once, and not independently verified.

---

## Part 11 addendum -- what the full 189-row sweep changed (2026-09-12)

Part 11 above was written from the 58 rows where the human and the
classifier disagreed. `scripts/18_golden_v3.py` then applied it to the
**other 131 rows**, which nobody had re-read. That sweep moved 22 rows and
forced three clarifications to the rules themselves. All three are
amendments to Part 11, not new rules.

### C1 -- R9b supersedes Part 3's "too fragmentary to route"

Part 3 lists *"tweets too fragmentary to route"* inside
`no_action_needed`. R9b says on-topic-but-unroutable is **not**
`no_action_needed`. These contradict, and the sweep hit the contradiction
six times.

**R9b wins. Strike that clause from Part 3's `no_action_needed`
definition.** The operative test is not how short the message is:

- **`no_action_needed`** -- there is no support work *even if you had the
  whole thread*. Praise, thanks, spam, jobs/press, off-topic, and
  confirmations that a problem is resolved (`1843215` *"All good
  thanks!"*, `2179670` *"Will do."*, `2055860` *"All good now."*).
- **`insufficient_context`** -- there **is** support work, and the tweet
  alone cannot show you what it is (`1644499` *"Same here!"*, `1331576`
  *"Please advise [link]"*, `1958723` *"\*space"*, `1497185` a bare link).

This matters more than it looks. `insufficient_context` rose from 6 rows
to **14 -- 7.4% of the golden set** -- once the agreed rows were read.
Eight of those were sitting in `no_action_needed` and
`complaint_dissatisfaction`, inflating both. Part 3 sets a ≥20% health
metric for `no_action_needed` on the grounds that v1's catch-all hid
mid-thread turns; that metric never fired because the hiding was spread
across two intents instead of one.

### C2 -- `how_to_usage` is the residual topic, not the default for questions

R1a says an actionable request routes to "the topic intent for that
request". The sweep showed that needs saying out loud: **R1a routes to the
most specific intent covering the subject matter, and `how_to_usage` wins
only when no specific intent claims it.**

- `149031` *"What are the best practices to signal a phishing page?"* and
  `1754987` *"What's the best way to report phishing invites?"* →
  `phishing_abuse_report`, not `how_to_usage`.
- `407837` *"How do I cancel my Dropbox subscription?"* →
  `billing_subscription` (Part 3 gives account closure to billing).
- `1972451`, `1462091` -- sharing how-tos → `sharing_permissions` via R3a.

Without C2, R1a and R2 together would pull every politely-phrased
question into `how_to_usage`, which is the mirror image of the labeling
error R2 exists to prevent.

### C3 -- R3a and R3b genuinely collide, and one row proves it

`1979876` *"things are acting up on Firefox on the Mac when asked to join
shared folder. Changing window does not help"*:

- **R3a** claims join/membership actions for `sharing_permissions`.
- **R3b** claims client failures for `sync_app_bug`, and the customer
  names the browser, the OS, and a failed workaround.

Neither rule is more specific than the other, and the precedence ladder
does not separate them because they are the same rung. Compare `2607972`
(*"wee bug on accepting a shared folder"*), settled as
`sharing_permissions` only because it names no client detail -- a
distinction thin enough that it should not be load-bearing.

**Left as `v3_status = ambiguous` rather than forced.** It is the only
unresolved row in 189 after the sweep. Resolving it needs a stated
tie-break (suggested: when a client, browser or OS is named as part of
the failure, R3b wins), which should be decided rather than inferred.

### What the sweep did to the numbers

| labels | scorable | accuracy | |
|---|---:|---:|---|
| v1 (original) | 189 | 62.4% | published baseline |
| v2 (disagreements only) | 183 | 83.6% | biased upward by construction |
| **v3 (full sweep)** | **175** | **81.1%** | |

v2 examined only rows where the model was **already scored wrong**, so
every correction available to it raised the score. The sweep found the
other direction: **8 rows where the human and the model were wrong the
same way** (`1577600`, `1851629`, `933440`, `237165`, `2801355`,
`1873832`, `1243081`, `2028354`), each of which lowers the score, against
4 that raise it and 10 moved out of scoring as `insufficient_context`.

**2.5 points of the v2 number was selection effect.** The remaining
figure is still an adjudication by one reviewer, self-rechecked, and not
independently verified -- but it is no longer biased by which rows were
chosen for review.

---

# Part 12 -- Where these rules execute (2026-09-13)

Part 11's distribution note says a rule that lives only in this file does
not exist. Until this change that was still half true. The labeling tool
printed a compressed ladder (`groq_lib.PRECEDENCE_RULES`), but the
production classifier prompt carried no rules at all, and the labeler saw
the historical Dropbox reply, which the classifier never sees.

- **One source.** `scripts/taxonomy_rules.py` holds Part 11 as 11
  operational lines, in two steps: is there a topic under the tone (R1a,
  R1b, R1d/R1e, R1c), then which topic (R2, R3, R4/R5, R6, R7, R8, R9a).
  Each line carries the Part 11 codes it compresses. The file also has 26
  boundary examples covering both sides of the seven hard boundaries. 24 are
  dev-split tweets, quoted verbatim with their v3 label and the reason the
  nearby intent loses; 2 are illustrative.
- **Two consumers, one text.** Prompt version `p3`
  (`groq_lib.build_system_prompt("p3")`) and `08_label_golden_set.py` render
  the same rules and example lines. The labeler also sees each example's
  reason.
- **Same inputs.** The labeling tool now shows the earlier message the tweet
  replies to (`data/golden_context.csv`), the same context the classifier
  can be given, and no longer the Dropbox reply written afterwards.
- **Checked.** `tests/test_taxonomy_rules.py` (20 tests, no API calls) fails
  if either consumer stops showing the text, if an example is not a
  correctly labelled dev tweet, if a rule code is missing from Part 11, if a
  rule names something that is not an intent, if the 13 intents change, if
  an adjudicated dev label contradicts its rule's stated winner, or if the
  production `p1` prompt changes by a single byte.

**Unchanged:** the 13 intents, `turn_type`, the flags, every golden label,
and the split. **Still open, not decided here:** R9b has no intent to assign
(the labeler footer says to note `insufficient_context`), and the C3
tie-break between R3a and R3b has no rule yet.

**`p3` is not the production prompt and has not been evaluated.** The
closest prior attempt, `p2`, scored 79.1% against `p1`'s 81.4% on dev.
`p3` rewrites the judgment rules that backfired there as concrete cases and
adds examples, but whether that changes the model's behaviour is unknown
until it is run on newly labelled data. `p3`'s system prompt is 3.2× `p1`'s.

**Found while writing the tests, not changed:** two dev rows, `1000504` and
`1057223`, carry `v3_rule = R7a` but are labelled
`security_account_compromise`. R7 separates `account_access` from
`how_to_usage` and says nothing about compromise, so the rule code looks
misapplied. The labels are left alone and listed as known exceptions in the
tests, for the next label review.
