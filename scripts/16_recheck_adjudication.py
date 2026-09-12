"""
STEP 1b -- re-check of the 48 first-pass adjudications, plus adjudication
of the 10 false-positive rows the first pass did not cover.

Two things happen here:

1. RE-CHECK of data/adjudication_v1.csv. The first pass was done
   tweet-first ("what does this message need?"). This pass is done
   rule-first ("what does taxonomy.md's text mandate for this message?"),
   deliberately a different route to the same question, with attention
   biased toward the 22 rows decided FOR the model -- those are the ones
   a reviewer with an interest in the model looking good would get wrong.

   LIMITATION, stated plainly: this is the same reviewer re-reading their
   own work. It catches method errors and rule-application slips. It does
   NOT remove the bias it is checking for, and it is not a substitute for
   a second person reading the 58 rows cold. Treat the verdicts as
   reviewed, not as independently confirmed.

2. FALSE-POSITIVE SIDE. The first pass sampled on `human_intent in FOUR`,
   which conditions on the human being in the cluster and says nothing
   about rows where the MODEL wrongly entered it. Those 10 rows are
   adjudicated here with the same vocabulary. This matters because a
   precedence rule tuned only on false negatives can make false positives
   worse -- and, as it turns out, the model's advantage does not survive
   the switch of sample.

Writes data/adjudication_v2.csv (58 rows). Leaves adjudication_v1.csv
intact as the first-pass record, and never touches golden_labels.csv.

Usage: ./.venv/Scripts/python.exe scripts/16_recheck_adjudication.py
       ./.venv/Scripts/python.exe scripts/16_recheck_adjudication.py --write
"""
import argparse

import pandas as pd

LABELS_PATH = "data/golden_labels.csv"
V1_PATH = "data/adjudication_v1.csv"
OUT_PATH = "data/adjudication_v2.csv"

FOUR = [
    "complaint_dissatisfaction",
    "how_to_usage",
    "storage_quota_plan_limits",
    "sync_app_bug",
]

VERDICTS = ["model_right", "human_right", "both_wrong", "ambiguous", "context_missing"]

# ---------------------------------------------------------------------
# 1. Re-check changes to the first pass.
# Only rows whose verdict or proposed final intent MOVES are listed.
# Everything not named here was re-read and left standing.
# ---------------------------------------------------------------------
RECHECK_CHANGES = {
    "193025": {
        "verdict": "both_wrong",
        "proposed_final_intent": "complaint_dissatisfaction",
        "note": "FIRST PASS WAS WRONG. I recorded model_right (billing_subscription) on the "
                "grounds that the subject is a paid plan. But R1 -- my own proposed rule -- "
                "says evaluative message + no actionable request = complaint, and \"we can't "
                "afford to pay $500 for something we're not going to use\" asks for nothing. "
                "I applied R1 to reach complaint on 1856823 and 847909 and then failed to "
                "apply it here, where it cuts against the model. Reclassified both_wrong.",
    },
}

# Rows re-read and left standing, but where the recheck wants the residual
# uncertainty on the record rather than buried in a verdict label.
RECHECK_CAVEATS = {
    "1452735": "Verdict (model_right) stands -- the human's complaint label is wrong either "
               "way. But the proposed final is not settled: 'why can't I get notifications' "
               "is how_to_usage if a setting must be enabled and sync_app_bug if notifications "
               "are broken. The tweet does not say which.",
    "2124237": "Verdict stands. Noting that 'stop using IP addresses with no host records' is "
               "a request to STOP a behaviour, not to add one; it lands in feature_request via "
               "the same 'product feedback' clause that FP row 1960138 turns on.",
    "2425723": "Verdict (ambiguous) stands, but the recheck adds a third candidate: 'such a "
               "basic functionality that just should work' is product feedback, so "
               "feature_request competes with complaint and sharing_permissions here.",
}

# ---------------------------------------------------------------------
# 2. False-positive side: model predicted one of FOUR, human did not.
# (tweet_id, verdict, proposed_final, why_human_label_chosen,
#  human_label_defensible, reason_for_decision)
# ---------------------------------------------------------------------
FALSE_POSITIVES = [
    ("1960138", "ambiguous", "feature_request",
     "Labeler's note: 'feedback for UI'.", "yes",
     "COUNTEREXAMPLE TO R1. 'I don't like the new tray icon' is evaluative with no actionable "
     "request, which R1 as drafted sends to complaint_dissatisfaction -- but taxonomy.md gives "
     "'product feedback' to feature_request explicitly. R1 needs a carve-out or these collide."),
    ("989345", "human_right", "feature_request",
     "Reads a missing capability as a feature gap.", "yes",
     "\"I can't keep them together if I want to use Paper\" states a missing capability and asks "
     "nothing. Nothing is broken, so R2 blocks the model's how_to_usage."),
    ("1339451", "model_right", "how_to_usage",
     "Second sentence ('annoying some clients') read as feedback.", "no",
     "Opens with an explicit answerable question -- 'How can I control reminders for Showcase "
     "emails?'. R1 gives the actionable request precedence over the feedback that follows it; "
     "feature_request belongs in secondary_intent."),
    ("3485", "ambiguous", "",
     "Link forces a business login -> read as account_access.", "yes",
     "SAME UNRESOLVED BOUNDARY AS 2222820 in the first pass: a clean how-to whose object is "
     "account linking. Both passes independently declined to settle it, which is consistent, "
     "but it means account_access <-> how_to_usage needs a written rule in STEP 2."),
    ("768404", "human_right", "billing_subscription",
     "Plan comparison -> read as billing.", "yes",
     "\"What's the difference between this and plus and free\" is a plan/packaging question. "
     "R6 claims it for billing_subscription, so the model's how_to_usage is wrong by the "
     "rule the first pass proposed."),
    ("2956783", "ambiguous", "storage_quota_plan_limits",
     "Object is a shared folder -> read as sharing_permissions.", "yes",
     "STRESS CASE FOR R3. 'Everyone can add except me. It says my Dropbox is full' has a "
     "shared-folder object but a quota symptom, and the useful answer is quota mechanics. R3 "
     "('object beats symptom') would route it to sharing_permissions and give the customer the "
     "wrong answer. R3 must be narrowed to access/permission failures."),
    ("1788320", "ambiguous", "service_outage",
     "'Is the service having issues?' -> read as outage.", "yes",
     "NEW BOUNDARY, not in the original brief: sync_app_bug <-> service_outage. One user's slow "
     "sync plus an explicit service-status question. taxonomy.md gives '\"Is Dropbox down?\"' to "
     "service_outage but never says what happens when a single-user symptom accompanies it."),
    ("1331572", "context_missing", "",
     "Labeler's note: 'context not there for classification'.", "n/a",
     "Mid-thread providing_requested_info about a pop-up. The labeler used no_action_needed as "
     "the bucket for unroutable-without-context -- which quietly inflates that intent and hides "
     "turn-type problems inside it, the exact v1 disease v2 was built to cure."),
    ("512851", "model_right", "sync_app_bug",
     "'Anyone else having sync issues' -> read as outage.", "no",
     "'Anyone else having...' polls other users; it is not a service-status question, and the "
     "customer describes their own recurring shared-sheet problem ('Not the first time'). Also "
     "note the human set esc=True but left churn_threat unset despite 'about to be binned' -- "
     "flag under-use, not an intent error."),
    ("659012", "ambiguous", "sharing_permissions",
     "Shared-folder object -> read as sharing_permissions.", "yes",
     "Second R3 stress case: a save failure on iOS whose object is a shared folder. Also close "
     "to context_missing -- the tweet is a sentence fragment continuing an unseen prior turn."),
]


def build() -> pd.DataFrame:
    labels = pd.read_csv(LABELS_PATH, dtype={"customer_tweet_id": str})
    by_id = labels.set_index("customer_tweet_id")
    v1 = pd.read_csv(V1_PATH, dtype={"customer_tweet_id": str})

    rows = []

    # --- false-negative side, carried over from v1 with recheck applied ---
    for _, r in v1.iterrows():
        tid = r.customer_tweet_id
        change = RECHECK_CHANGES.get(tid)
        rows.append({
            "side": "false_negative",
            "customer_tweet_id": tid,
            "tweet": r.tweet,
            "human_intent": r.human_intent,
            "model_intent": r.model_intent,
            "why_human_label_chosen": r.why_human_label_chosen,
            "human_label_defensible": r.human_label_defensible,
            "first_pass_verdict": r.verdict,
            "verdict": change["verdict"] if change else r.verdict,
            "proposed_final_intent": (change["proposed_final_intent"] if change
                                      else r.proposed_final_intent),
            "changed_on_recheck": "yes" if change else "no",
            "recheck_note": (change["note"] if change
                             else RECHECK_CAVEATS.get(tid, "")),
            "reason_for_decision": r.reason_for_decision,
        })

    # --- false-positive side, adjudicated here for the first time ---
    for tid, verdict, final, why, defensible, reason in FALSE_POSITIVES:
        if tid not in by_id.index:
            raise KeyError(f"{tid} not in {LABELS_PATH}")
        src = by_id.loc[tid]
        rows.append({
            "side": "false_positive",
            "customer_tweet_id": tid,
            "tweet": " ".join(str(src.customer_text).split()),
            "human_intent": src.human_intent,
            "model_intent": src.suggested_intent,
            "why_human_label_chosen": why,
            "human_label_defensible": defensible,
            "first_pass_verdict": "",
            "verdict": verdict,
            "proposed_final_intent": final,
            "changed_on_recheck": "n/a",
            "recheck_note": "",
            "reason_for_decision": reason,
        })

    return pd.DataFrame(rows)


def report(adj: pd.DataFrame):
    fn = adj[adj.side == "false_negative"]
    fp = adj[adj.side == "false_positive"]

    print(f"\nADJUDICATION v2 -- {len(adj)} rows "
          f"({len(fn)} false-negative side re-checked, {len(fp)} false-positive side new)\n")

    changed = fn[fn.changed_on_recheck == "yes"]
    print(f"-- re-check of the first pass --\n")
    print(f"  {len(fn) - len(changed)} of {len(fn)} verdicts stand, {len(changed)} changed.")
    for _, r in changed.iterrows():
        print(f"    {r.customer_tweet_id}: {r.first_pass_verdict} -> {r.verdict} "
              f"(proposed {r.proposed_final_intent})")
    caveated = fn[(fn.changed_on_recheck == "no") & (fn.recheck_note != "")]
    print(f"  {len(caveated)} stand with a recorded caveat: "
          f"{', '.join(caveated.customer_tweet_id)}")

    print(f"\n-- verdicts, by which side of the cluster the error is on --\n")
    print(f"{'verdict':16s} {'false_neg':>10s} {'false_pos':>10s} {'total':>7s}")
    for v in VERDICTS:
        a, b = int((fn.verdict == v).sum()), int((fp.verdict == v).sum())
        print(f"{v:16s} {a:10d} {b:10d} {a + b:7d}")
    print(f"{'TOTAL':16s} {len(fn):10d} {len(fp):10d} {len(adj):7d}")

    for name, sub in (("false-negative side", fn), ("false-positive side", fp)):
        mr = int((sub.verdict == "model_right").sum())
        hr = int((sub.verdict == "human_right").sum())
        print(f"\n  {name}: model_right {mr}, human_right {hr}"
              + (f"  ({mr}:{hr})" if hr else "  (model favoured)"))

    labels = pd.read_csv(LABELS_PATH)
    baseline = int((labels.human_intent == labels.suggested_intent).sum())
    n = len(labels)
    # A false-negative `model_right` adds a correct row. A false-positive
    # `human_right` or `both_wrong` does NOT subtract one -- the model was
    # already scored wrong there. Only FP `model_right` adds.
    gain = int((fn.verdict == "model_right").sum()) + int((fp.verdict == "model_right").sum())
    amb = int((adj.verdict == "ambiguous").sum())
    print(f"\n-- effect on measured accuracy, no model or prompt change --\n")
    print(f"  as reported today        {baseline:3d}/{n}  {baseline / n:.1%}")
    print(f"  after label corrections  {baseline + gain:3d}/{n}  {(baseline + gain) / n:.1%}"
          f"   (+{gain})")
    print(f"  upper bound if STEP 2 settles all {amb} ambiguous rows"
          f"\n                           {baseline + gain + amb:3d}/{n}  "
          f"{(baseline + gain + amb) / n:.1%}\n")


def main(write: bool):
    adj = build()
    report(adj)
    if write:
        adj.to_csv(OUT_PATH, index=False, encoding="utf-8")
        print(f"wrote {OUT_PATH} ({len(adj)} rows)")
        print(f"{V1_PATH} and {LABELS_PATH} both unchanged.\n")
    else:
        print(f"dry run -- pass --write to save {OUT_PATH}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--write", action="store_true", help=f"save the table to {OUT_PATH}")
    main(write=p.parse_args().write)
