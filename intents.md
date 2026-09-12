# Intent Taxonomy (v1)

Machine-readable version (used by code): [`intents.json`](intents.json).
This file is the human-readable rationale + examples; keep the two in
sync if the taxonomy changes.

> **A v2 replacement is proposed.** [`taxonomy.md`](taxonomy.md)
> documents both taxonomies side by side: v1 as built (Part 1), what the
> data showed is wrong with it (Part 2), and a proposed 13-intent v2
> with a `turn_type` field and flags (Part 3), plus the migration path.
> **v1 below is still what the code uses** -- v2 is not adopted.

Brand: DropboxSupport
Defined from manual read-through of 40 randomly sampled customer<->reply
pairs (see `data/reading_sample.txt`, `random_state=42`).

This is v1 -- expected to be revised once we hand-label the golden eval
set and hit edge cases that don't fit cleanly. Any changes should be
noted in the decision log with the example that forced the change.

| Intent | Description | Example (customer_tweet_id) |
|---|---|---|
| account_login | Access issues, 2FA/backup codes, business account setup, credential problems | 2823321, 1820072, 2974776, 278604, 785409 |
| billing_subscription | Charges, refunds, plan pricing, cancellations/downgrades | 1682379, 2331194, 2854125, 1928260, 1458883, 1219561, 2384217, 1422641, 459509, 2235996 |
| sync_technical_bug | Files not syncing, app/website errors, crashes, broken features | 1820072, 1274528, 234493, 1067513, 1202354, 1972492, 2773993, 118484, 786084, 2043077 |
| how_to_usage | "How do I do X" -- nothing is broken, just needs guidance | 2914964, 1421735, 867197, 1959288, 1040857, 1458883, 1186286 |
| data_loss_recovery | Deleted/missing files, restore requests -- flagged separately from general bugs because urgency and resolution path differ | 1960105, 93195 |
| feature_request | Suggestions / product feedback, not a problem to fix | 1040857, 1979488 |
| followup_ticket_status | Referencing an existing ticket/case, nudging for a response -- not a new intent, a control-flow signal | 1682379, 1202354, 1433308, 121896, 1473214 |
| general_feedback_or_other | Thanks/praise, venting/rants, spam, or fragments too ambiguous to route to a technical bucket. Without this catch-all, the classifier would be forced to hallucinate one of the technical intents for non-technical noise. | 1421734 ("Thank you - I wanted the files to be read-only anyway :D x") |

## Notes
- Real tweets are messy: several examples straddle two categories
  (e.g. a how-to complaint that also asks for a cancellation email).
  We pick the PRIMARY intent -- the thing the customer most needs
  resolved -- when labeling.
- `followup_ticket_status` matters for escalation logic later: a
  customer nudging about a stale ticket is a different escalation
  signal than a first-time technical bug report.
