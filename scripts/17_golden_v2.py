"""
STEP 2B/2C/2D -- apply the v2.1 precedence rules to the 58 reviewed rows,
then emit data/golden_labels_v2.csv with a full audit trail.

Reads:  data/golden_labels.csv      (never written)
        data/adjudication_v2.csv    (never written)
Writes: data/golden_labels_v2.csv        189 rows, originals + v2 columns
        data/golden_labels_v2_audit.csv  only the rows whose label moved

The rules themselves live in taxonomy.md Part 11. This file is their
executable form: every reviewed row names the rule that decided it, so a
rule change here and a rule change there cannot silently diverge.

WHAT THE v2 LABELS ARE NOT: an independently verified ground truth. They
are one reviewer's adjudication, self-rechecked once (STEP 1b). Any
accuracy quoted against them must say so.

v2_status vocabulary:
  resolved             a rule determines the label
  ambiguous            rules genuinely do not decide; left at the v1 label
  insufficient_context on-topic but unroutable without a prior turn. The
                       13-intent taxonomy has no intent for this (see
                       taxonomy.md Part 11, R9) -- these rows are excluded
                       from the headline accuracy rather than parked in
                       no_action_needed, which is what inflated that
                       intent in v1.
  taxonomy_gap         routable, but no intent in the taxonomy describes it

Usage: ./.venv/Scripts/python.exe scripts/17_golden_v2.py
       ./.venv/Scripts/python.exe scripts/17_golden_v2.py --write
"""
import argparse

import pandas as pd

LABELS_PATH = "data/golden_labels.csv"
ADJ_PATH = "data/adjudication_v2.csv"
OUT_PATH = "data/golden_labels_v2.csv"
AUDIT_PATH = "data/golden_labels_v2_audit.csv"

# (tweet_id, v2_intent, rule, confidence, status, rationale)
#
# confidence is the ADJUDICATOR's, not the classifier's: how firmly the
# written rule settles this row. low = the rule decides it but a second
# reader could reasonably land elsewhere.
DECISIONS = [
    # ================= false-negative side (48) =================
    ("2124237", "feature_request", "R1b", "high", "resolved",
     "Proposes a specific product change ('stop using IP addresses with no host records'). Annoyance is the frame."),
    ("1497111", "billing_subscription", "R1d", "high", "resolved",
     "Asserts a correctable billing fact (charged $990, refund refused). Support can act on it, so the topic wins over the grievance frame."),
    ("622045", "storage_quota_plan_limits", "R1d", "medium", "resolved",
     "Asserts how shared-folder quota works ('one shared folder uses up my entire quota'). Support can explain the mechanic, so R1d beats R1c."),
    ("2425723", "sharing_permissions", "R1d", "medium", "resolved",
     "Asserts a permissions capability does not exist. That claim is correctable, so it routes to the topic rather than to complaint."),
    ("838554", "how_to_usage", "R1a", "high", "resolved",
     "Ends in an explicit answerable question ('How do I disable?!'). Actionable request outranks the grievance frame."),
    ("1041902", "", "R9b", "n/a", "insufficient_context",
     "Mid-thread reply about an earlier DM link; no topic recoverable from the tweet alone."),
    ("1050476", "complaint_dissatisfaction", "R1c", "high", "resolved",
     "Evaluates the company's conduct, proposes no product change, asserts no correctable fact. Textbook R1c."),
    ("241283", "sync_app_bug", "R1a", "high", "resolved",
     "'I can't download dropbox mobile after the last update' -- neutral tone, concrete malfunction."),
    ("1856823", "complaint_dissatisfaction", "R1c", "high", "resolved",
     "Objects to cost only ('Just crazy the new prices'); proposes no packaging change, so R1b does not apply."),
    ("1960101", "how_to_usage", "R1a", "medium", "resolved",
     "REVISED IN 2C. First draft invented a taxonomy_gap status for 'which channel can I reach you on'. The 2C sweep found 2700303 ('No phone support for Dropbox Plus members?') where human and model both said how_to_usage -- support-channel availability is an answerable product question, so the gap was unnecessary. Escalation is still carried by wants_human."),
    ("615215", "how_to_usage", "R1a", "medium", "resolved",
     "'why did you remove the green check' is answerable, and the icon changed by design, so R2 keeps it out of sync_app_bug."),
    ("847909", "complaint_dissatisfaction", "R1c", "medium", "resolved",
     "'You ougth to review this price' objects to cost without proposing a product or packaging change -- the R1b/R1c pricing split."),
    ("74467", "", "R9b", "n/a", "insufficient_context",
     "A filename plus 'extremely frustrating'; providing_requested_info on an unseen turn."),
    ("1452735", "how_to_usage", "R1a", "medium", "resolved",
     "Answerable question about enabling notifications. Low confidence on the topic: sync_app_bug applies instead if notifications are broken rather than unconfigured."),
    ("2540849", "feature_request", "R1b", "high", "resolved",
     "Proposes a product change (include the login IP in warning emails). The labeler's own note says 'feedback'."),
    ("1804904", "service_outage", "R8b", "high", "resolved",
     "Cites the status page and a whole team affected -- external evidence of scope, not a device-specific symptom."),
    ("1082713", "feature_request", "R1b", "high", "resolved",
     "'Please add it back' is an explicit product-change request."),
    ("2551167", "billing_subscription", "R1d", "medium", "resolved",
     "Asserts a correctable billing fact (charged before the trial ended) inside an all-caps accusation."),
    ("1383896", "service_outage", "R8b", "medium", "resolved",
     "Asserts the service stopped working with no device-specific symptom described."),
    ("2179658", "complaint_dissatisfaction", "R1e", "medium", "resolved",
     "Sarcasm aimed at another user, not a support request, so R1d's correction carve-out is suppressed."),
    ("1687134", "", "R9b", "n/a", "insufficient_context",
     "nudging_no_response about an unseen earlier request; the complaint label was encoding a turn_type."),
    ("2629478", "storage_quota_plan_limits", "R4", "high", "resolved",
     "Deleted files reappearing with the quota still full -- taxonomy.md's named storage case."),
    ("259327", "sharing_permissions", "R3a", "high", "resolved",
     "Subject matter is a team folder, so sharing_permissions claims it including the how-to."),
    ("1874043", "no_action_needed", "R9a", "high", "resolved",
     "About concert tickets, not Dropbox. Off-topic is a valid no_action_needed, unlike unroutable."),
    ("2222820", "how_to_usage", "R7b", "high", "resolved",
     "Customer is authenticated and wants to perform an operation (unlink a device). Not blocked from the account, so account_access loses."),
    ("1863374", "how_to_usage", "R2", "medium", "resolved",
     "Asks whether the icon changed by design. Resolved the same way as 615215, which removes the v1 self-inconsistency between the two green-check tweets."),
    ("2053253", "storage_quota_plan_limits", "R4", "high", "resolved",
     "'Out of Space Warning will not go away' after deletions -- quota not clearing."),
    ("1999298", "storage_quota_plan_limits", "R4", "high", "resolved",
     "Permanently deleted items, quota unchanged."),
    ("37940", "sharing_permissions", "R3a", "high", "resolved",
     "'I can't access my shared folder' is an access failure on a shared object."),
    ("988433", "sync_app_bug", "R2", "high", "resolved",
     "Edits reverted after syncing. Something is broken, so how_to_usage is excluded by definition."),
    ("1495052", "billing_subscription", "R6a", "high", "resolved",
     "'What are my options' where the answer is a plan and its cost."),
    ("1577541", "complaint_dissatisfaction", "R1c", "high", "resolved",
     "Parody of an upsell ending in profanity; evaluates the company, proposes nothing, asserts no correctable fact."),
    ("1901120", "storage_quota_plan_limits", "R6b", "high", "resolved",
     "Answer is how earned storage is counted, not a price."),
    ("1324437", "storage_quota_plan_limits", "R6b", "high", "resolved",
     "Answer is a plan's storage limit, not a price."),
    ("1523722", "billing_subscription", "R6a", "medium", "resolved",
     "Asking to trial a paid tier is a purchase decision."),
    ("53189", "feature_request", "R1b", "high", "resolved",
     "'It would be great if they offered a little bit more space' proposes a packaging change, which R1b separates from objecting to cost."),
    ("2742476", "billing_subscription", "R6a", "high", "resolved",
     "'and how much does it cost' makes the answer a price."),
    ("193025", "complaint_dissatisfaction", "R1c", "medium", "resolved",
     "Objects to cost with no request. This is the row the STEP 1b recheck flipped away from billing_subscription."),
    ("1992403", "billing_subscription", "R6a", "high", "resolved",
     "'Can you share the cost of the Plan' -- the answer is a price."),
    ("2421520", "feature_request", "R1b", "high", "resolved",
     "'I'd happily pay $200 a year if the plan included 2TB' proposes packaging, not an objection to cost."),
    ("2421521", "feature_request", "R1b", "medium", "resolved",
     "Proposes a packaging change while addressing peers; R1e suppresses the topic carve-out but not feature_request."),
    ("2976667", "how_to_usage", "R5", "high", "resolved",
     "Laptop disk space, which storage_quota explicitly excludes, and nothing is broken -- they are asking whether an external drive works."),
    ("1600556", "sync_app_bug", "R3b", "medium", "resolved",
     "A shared link that 'keeps loading' is a client rendering failure, not a permission denial. R3 as originally drafted got this wrong."),
    ("2607972", "sharing_permissions", "R3a", "high", "resolved",
     "'accepting a shared folder' is a membership/join action."),
    ("1972693", "", "R9b", "n/a", "insufficient_context",
     "'350/350, no firewalls or anything in between' is pure providing_requested_info."),
    ("2828924", "", "R9b", "n/a", "insufficient_context",
     "'I Have done this. Too bad it didn't solve the problem' carries no topic; correct handling is turn_type=disputing_prior_answer."),
    ("79858", "storage_quota_plan_limits", "R4", "high", "resolved",
     "Permanently deleted thousands of files, storage reading unchanged."),
    ("2910246", "storage_quota_plan_limits", "R4", "high", "resolved",
     "Same shape as 79858 and 1999298; all three now carry one label."),

    # ================= false-positive side (10) =================
    ("1960138", "feature_request", "R1b", "high", "resolved",
     "Dislike of a UI element is product feedback. This is the counterexample that forced R1b to exist."),
    ("989345", "feature_request", "R1b", "medium", "resolved",
     "States a missing capability in Paper, asks nothing, nothing is broken."),
    ("1339451", "how_to_usage", "R1a", "high", "resolved",
     "Opens with an answerable question about controlling reminders; the feedback that follows belongs in secondary_intent."),
    ("3485", "how_to_usage", "R7b", "medium", "resolved",
     "Customer can authenticate to both accounts and wants the link to open under a different one -- an operation, not a lockout."),
    ("768404", "billing_subscription", "R6a", "medium", "resolved",
     "Plan comparison whose answer is what each tier costs and includes."),
    ("2956783", "storage_quota_plan_limits", "R3b", "high", "resolved",
     "Shared-folder object but a quota symptom; the answer the customer needs is quota mechanics. This is the row that forced R3 to be narrowed."),
    ("1788320", "sync_app_bug", "R8a", "high", "resolved",
     "A device-specific symptom ('my desktop app (Mac) is syncing at a snail's pace') governs over the speculative outage question, which becomes secondary_intent."),
    ("1331572", "", "R9b", "n/a", "insufficient_context",
     "Describes a pop-up from an unseen turn. The labeler's own note says 'context not there for classification' -- and they filed it under no_action_needed, which is the misuse R9 now forbids."),
    ("512851", "sync_app_bug", "R8a", "high", "resolved",
     "'Anyone else having sync issues' polls peers; it is not status evidence, and the customer describes their own recurring problem."),
    ("659012", "sync_app_bug", "R3b", "low", "resolved",
     "A save failure on iOS whose object happens to be shared. Low confidence: the shared/non-shared contrast hints at permissions, and the tweet is a fragment."),
]

STATUSES = ["resolved", "ambiguous", "insufficient_context", "taxonomy_gap"]
EXCLUDED_FROM_HEADLINE = ("insufficient_context", "taxonomy_gap")

# ---------------------------------------------------------------------
# STEP 2C finding, recorded but deliberately NOT acted on.
#
# The adjudication only ever looked at rows where the human and the model
# DISAGREED. Where both were wrong the same way, nothing flagged it -- so
# a sweep of the 23 unreviewed rows on which they agreed, inside the four
# contested intents, was run against the new rules. It found 7 rows the
# rules would move.
#
# These are NOT relabeled here. Doing so would mean adjudicating rows on
# the strength of a keyword sweep rather than a read, which is the sloppy
# version of what STEP 1 did carefully. They are recorded so the next pass
# has a worklist, and so the headline number is quoted knowing it is
# incomplete.
#
# They also establish the DIRECTION of the remaining bias: corrections
# were only ever sought where the model was already scored wrong, so every
# correction found could only raise the score. These 7 sit where both were
# wrong, and fixing them can only lower it.
FLAGGED_AGREED = [
    ("933440", "complaint_dissatisfaction", "feature_request", "R1b",
     "Argues the removed green icon 'worked great for me and many others' -- product-design feedback, not dissatisfaction with the service."),
    ("715979", "complaint_dissatisfaction", "insufficient_context", "R9b",
     "'Over a Month Later, no, I don't have a ticket ID' is providing_requested_info on an unseen turn."),
    ("2826329", "complaint_dissatisfaction", "insufficient_context", "R9b",
     "'I did and I never got a response' carries no topic of its own."),
    ("2573123", "complaint_dissatisfaction", "complaint_dissatisfaction", "R1c",
     "Stands, but only just: 'will you allow me to speak with someone' is the same support-channel request as 1960101, which R1a sends to how_to_usage. The support-experience complaint is what keeps it here."),
    ("2028354", "storage_quota_plan_limits", "how_to_usage", "R5",
     "'installed DB on my acer windows 10 and almost ran out of space' is DEVICE disk space, which storage_quota explicitly excludes. Both human and model missed the same exclusion."),
    ("1243081", "storage_quota_plan_limits", "billing_subscription", "R6a",
     "'told I can't upgrade my storage' -- the blocked action is a purchase, so R6a points at billing."),
    ("2700303", "how_to_usage", "how_to_usage", "R1a",
     "Stands. This is the row that disproved the taxonomy_gap category invented for 1960101."),
]


def build() -> pd.DataFrame:
    labels = pd.read_csv(LABELS_PATH, dtype={"customer_tweet_id": str})
    adj = pd.read_csv(ADJ_PATH, dtype={"customer_tweet_id": str}).set_index("customer_tweet_id")

    decided = {d[0]: d for d in DECISIONS}
    missing = set(adj.index) - set(decided)
    if missing:
        raise ValueError(f"reviewed rows with no v2 decision: {sorted(missing)}")

    out = labels.copy()
    v2_intent, v2_rule, v2_conf, v2_status, v2_reason, v2_changed, v2_source = [], [], [], [], [], [], []

    for _, r in labels.iterrows():
        tid = r.customer_tweet_id
        if tid not in decided:
            v2_intent.append(r.human_intent)
            v2_rule.append("")
            v2_conf.append("")
            v2_status.append("resolved")
            v2_reason.append("")
            v2_changed.append(False)
            v2_source.append("original label, not reviewed")
            continue

        _, intent, rule, conf, status, reason = decided[tid]
        # insufficient_context / taxonomy_gap rows keep the original label
        # as a placeholder -- they are excluded from the headline instead.
        final = intent if intent else r.human_intent
        v2_intent.append(final)
        v2_rule.append(rule)
        v2_conf.append(conf)
        v2_status.append(status)
        v2_reason.append(reason)
        v2_changed.append(final != r.human_intent)
        v2_source.append(f"adjudication_v2.csv + taxonomy.md Part 11 {rule}")

    out["human_intent_v2"] = v2_intent
    out["v2_status"] = v2_status
    out["v2_rule"] = v2_rule
    out["v2_confidence"] = v2_conf
    out["v2_changed"] = v2_changed
    out["v2_reason"] = v2_reason
    out["v2_source"] = v2_source
    return out


def accuracy(df: pd.DataFrame, col: str) -> tuple[int, int]:
    return int((df[col] == df.suggested_intent).sum()), len(df)


def report(v2: pd.DataFrame):
    reviewed = v2[v2.v2_rule != ""]
    changed = v2[v2.v2_changed]
    scored = v2[~v2.v2_status.isin(EXCLUDED_FROM_HEADLINE)]

    print(f"\nGOLDEN LABELS v2 -- {len(v2)} rows (same {len(v2)} tweets as v1)\n")
    print(f"  reviewed rows            {len(reviewed)}")
    print(f"  labels changed           {len(changed)}")
    print(f"  labels unchanged         {len(v2) - len(changed)}")
    print(f"  status breakdown         " + ", ".join(
        f"{s}={int((v2.v2_status == s).sum())}" for s in STATUSES))

    o_hit, o_n = accuracy(v2, "human_intent")
    n_hit, n_n = accuracy(v2, "human_intent_v2")
    s_hit, s_n = accuracy(scored, "human_intent_v2")

    print(f"\n-- accuracy of the EXISTING, UNCHANGED model predictions --\n")
    print(f"  v1 labels, all rows       {o_hit:3d}/{o_n}  {o_hit / o_n:.1%}   (published baseline)")
    print(f"  v2 labels, all rows       {n_hit:3d}/{n_n}  {n_hit / n_n:.1%}")
    print(f"  v2 labels, scorable rows  {s_hit:3d}/{s_n}  {s_hit / s_n:.1%}   "
          f"(excludes {o_n - s_n} unroutable/taxonomy-gap rows)")
    print(f"\n  No model, prompt, or threshold was changed. The movement is"
          f"\n  entirely relabeling, and the v2 labels are one reviewer's"
          f"\n  adjudication, not independently verified ground truth.")

    print(f"\n-- per-intent, v1 -> v2 (recall against each label set) --\n")
    print(f"{'intent':30s} {'v1 n':>5s} {'v1 rec':>7s} {'v2 n':>5s} {'v2 rec':>7s} {'delta':>7s}")
    intents = sorted(set(v2.human_intent) | set(v2.human_intent_v2))
    rows = []
    for i in intents:
        a = v2[v2.human_intent == i]
        b = scored[scored.human_intent_v2 == i]
        ra = (a.human_intent == a.suggested_intent).mean() if len(a) else float("nan")
        rb = (b.human_intent_v2 == b.suggested_intent).mean() if len(b) else float("nan")
        d = rb - ra if len(a) and len(b) else float("nan")
        rows.append((i, len(a), ra, len(b), rb, d))
        print(f"{i:30s} {len(a):5d} {ra:7.2f} {len(b):5d} {rb:7.2f} {d:+7.2f}")

    up = [r for r in rows if r[5] == r[5] and r[5] > 0.01]
    down = [r for r in rows if r[5] == r[5] and r[5] < -0.01]
    print(f"\n  improved: {', '.join(f'{r[0]} ({r[5]:+.2f})' for r in sorted(up, key=lambda x: -x[5])) or 'none'}")
    print(f"  worsened: {', '.join(f'{r[0]} ({r[5]:+.2f})' for r in sorted(down, key=lambda x: x[5])) or 'none'}")

    print(f"\n-- confusion on the v2 labels (scorable rows, errors only) --\n")
    err = scored[scored.human_intent_v2 != scored.suggested_intent]
    conf = err.groupby(["human_intent_v2", "suggested_intent"]).size().sort_values(ascending=False)
    print(f"  {len(err)} errors remain across {len(scored)} scorable rows")
    for (h, m), c in conf.items():
        print(f"    {c:2d}  true={h:30s} pred={m}")

    print(f"\n-- rows excluded from the headline --\n")
    for _, r in v2[v2.v2_status.isin(EXCLUDED_FROM_HEADLINE)].iterrows():
        preview = " ".join(str(r.customer_text).split())[:70]
        print(f"  [{r.customer_tweet_id}] {r.v2_status:20s} ({r.v2_rule})  {preview}")

    would_move = [f for f in FLAGGED_AGREED if f[1] != f[2]]
    print(f"\n-- STEP 2C sweep: rows where human and model AGREED, so were never reviewed --\n")
    print(f"  23 such rows sit inside the four contested intents; the new rules")
    print(f"  would move {len(would_move)} of them. NOT relabeled here -- recorded as a worklist.\n")
    for tid, old, new, rule, why in FLAGGED_AGREED:
        mark = "MOVES" if old != new else "stands"
        print(f"  [{tid}] {mark:6s} {old} -> {new} ({rule})")
    print(f"\n  Direction of the remaining bias: corrections were only ever sought")
    print(f"  where the model was ALREADY scored wrong, so every correction found")
    print(f"  could only raise the score. These {len(would_move)} sit where both were wrong,")
    print(f"  and fixing them can only lower it. The headline below is therefore")
    print(f"  an optimistic bound, not a settled number.")

    low = reviewed[reviewed.v2_confidence == "low"]
    if len(low):
        print(f"\n-- low-confidence adjudications ({len(low)}) --\n")
        for _, r in low.iterrows():
            print(f"  [{r.customer_tweet_id}] -> {r.human_intent_v2} ({r.v2_rule})")
    print()


def main(write: bool):
    v2 = build()
    report(v2)
    if write:
        v2.to_csv(OUT_PATH, index=False, encoding="utf-8")
        audit = v2[v2.v2_changed][[
            "customer_tweet_id", "customer_text", "human_intent", "human_intent_v2",
            "v2_status", "v2_rule", "v2_confidence", "v2_reason", "v2_source",
            "suggested_intent",
        ]]
        audit.to_csv(AUDIT_PATH, index=False, encoding="utf-8")
        print(f"wrote {OUT_PATH} ({len(v2)} rows)")
        print(f"wrote {AUDIT_PATH} ({len(audit)} changed rows)")
        print(f"{LABELS_PATH}, {ADJ_PATH} unchanged.\n")
    else:
        print("dry run -- pass --write to save\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--write", action="store_true")
    main(write=p.parse_args().write)
