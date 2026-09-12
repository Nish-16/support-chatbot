"""
STEP 2E -- apply the Part 11 rules to the 131 rows the adjudication never
looked at, producing data/golden_labels_v3.csv.

Why this exists: v2 relabeled only the 58 rows where the human and the
classifier DISAGREED. That sample can only ever contain rows the model was
already scored wrong on, so every correction it could find raised the
score. Rows where both were wrong the same way were structurally
invisible. v3 closes that hole by reading all 131 remaining rows --
118 where the two agreed, 13 disagreements outside the contested cluster.

Reads:  data/golden_labels_v2.csv  (never written)
Writes: data/golden_labels_v3.csv       189 rows
        data/golden_labels_v3_audit.csv rows whose label moved since v2

golden_labels.csv, adjudication_v1.csv, adjudication_v2.csv and
golden_labels_v2.csv are all left untouched.

Every row below was read. Rows not listed were read and left standing;
SWEEP contains only the ones that move or that the rules cannot settle.
"""
import argparse

import pandas as pd

V2_PATH = "data/golden_labels_v2.csv"
OUT_PATH = "data/golden_labels_v3.csv"
AUDIT_PATH = "data/golden_labels_v3_audit.csv"

EXCLUDED_FROM_HEADLINE = ("insufficient_context", "taxonomy_gap")

# (tweet_id, v3_intent, rule, confidence, status, rationale)
# v3_intent "" means the row carries no defensible label (status says why).
SWEEP = [
    # ---- disagreements outside the contested cluster (6 of 13 move) ----
    ("1000504", "security_account_compromise", "R7a", "high", "resolved",
     "'mysterious files were uploaded ... under my account' is suspicious activity, which Part 3 gives to security. The customer is not blocked from authenticating, so account_access does not apply."),
    ("1057223", "security_account_compromise", "R7a", "medium", "resolved",
     "Reports an attempted unauthorised login. The 'is there a way to see all attempted logins' question is real but secondary; the reported security event is the topic."),
    ("1480046", "account_access", "R8a", "medium", "resolved",
     "'Do you have some problems this morning I can't connect to my account' pairs an outage guess with an account-specific symptom. R8a gives the symptom precedence, and the symptom is being unable to reach the account."),
    ("1209275", "sync_app_bug", "R8a", "medium", "resolved",
     "'the website keeps reporting issues' while updating an account -- a website error with no evidence of service-wide scope. Part 3 gives website errors to sync_app_bug. Human said service_outage, model said account_access; both wrong."),
    ("1874363", "feature_request", "R1b", "high", "resolved",
     "Literally opens '#collaboration feature request'. The object is shared documents, but R1 fires before R3 in the ladder, so a proposed product change wins."),
    ("1419325", "", "R9b", "n/a", "insufficient_context",
     "'Please fasr!!' is a two-word nudge on an unseen thread. Human said service_outage, model said no_action_needed; neither is recoverable from the text."),

    # ---- agreed rows inside the four contested intents (5 move) ----
    ("933440", "feature_request", "R1b", "high", "resolved",
     "Argues the removed green icon 'worked great for me and many others' -- product-design feedback. Confirmed by four unambiguous neighbours (1619102, 1099456, 1067503, 2509185) that both human and model already label feature_request."),
    ("715979", "", "R9b", "n/a", "insufficient_context",
     "'Over a Month Later, no, I don't have a ticket ID' is providing_requested_info on an unseen turn."),
    ("2826329", "", "R9b", "n/a", "insufficient_context",
     "'I did and I never got a response' carries no topic of its own."),
    ("2028354", "how_to_usage", "R5", "high", "resolved",
     "'installed DB on my acer windows 10 and almost ran out of space. Couldn't selective sync' is DEVICE disk space, which storage_quota explicitly excludes. The answer is selective sync."),
    ("1243081", "billing_subscription", "R6a", "medium", "resolved",
     "'told I can't upgrade my storage' -- the blocked action is a purchase, so R6a points at billing rather than at quota mechanics."),

    # ---- agreed rows outside the contested intents (11 move) ----
    ("237165", "how_to_usage", "R7b", "medium", "resolved",
     "'I lost one and can't figure out how to remove its access' -- authenticated, wants to unlink a lost device. Same operation as 2222820. This is the documented R7 edge case and the customer is not locked out."),
    ("2801355", "sync_app_bug", "R3b", "low", "resolved",
     "A file added to a shared folder 'is not showing in at least two accounts' is a propagation failure, not a deletion. R3b sends shared-object sync failures to sync_app_bug. Low confidence: 'missing files' is also literally Part 3's data_loss wording."),
    ("1851629", "sync_app_bug", "R8a", "high", "resolved",
     "'Is @Dropbox down? Mac Client does not sync....' is 1788320's exact shape -- a device-specific symptom alongside an outage guess. R8a was written for this. Both human and model said service_outage."),
    ("1577600", "storage_quota_plan_limits", "R6b", "medium", "resolved",
     "'my shared link always show error 429' is a rate-limit response. Part 3 assigns 'shared-link traffic limits' to storage_quota_plan_limits explicitly, so the shared object does not make it a permissions issue."),
    ("1873832", "complaint_dissatisfaction", "R1c", "medium", "resolved",
     "'when @Dropbox adds stuff to your account without getting your permission first #NOTCOOL' describes Dropbox's own conduct, not a third-party intrusion. No unauthorised login, so security_account_compromise does not fit; evaluative with no request, so R1c."),
    ("1347292", "", "R9b", "n/a", "insufficient_context",
     "'It's from a shared folder, so not sure...' is a sentence fragment answering an unseen question."),
    ("1644499", "", "R9b", "n/a", "insufficient_context",
     "'Same here!' endorses another customer's issue. There is support work behind it, but none of it is visible."),
    ("1331576", "", "R9b", "n/a", "insufficient_context",
     "'Please advise [link]' -- the whole request is in an image."),
    ("1958723", "", "R9b", "n/a", "insufficient_context",
     "'*space' is a one-word correction to an earlier tweet."),
    ("1497185", "", "R9b", "n/a", "insufficient_context",
     "A bare link with no text at all."),
    ("1979876", "", "R3a/R3b", "n/a", "ambiguous",
     "'things are acting up on Firefox on the Mac when asked to join shared folder'. R3a claims join/membership actions for sharing_permissions; R3b claims client failures for sync_app_bug, and the customer names the browser and OS. The two rules genuinely collide and neither is more specific. Left at the v2 label and flagged rather than forced."),
]

# Rows examined and deliberately left standing where the call was close.
# Recorded so the next reader knows they were considered, not skipped.
CLOSE_CALLS = {
    "409269": "Comments in a shared file visible only to the author -- kept sharing_permissions; R3b would argue propagation failure.",
    "1561611": "'difficulty uploading a file to a link that was shared' -- kept sharing_permissions; the failure mode is not specific enough to trigger R3b.",
    "1909153": "Sharing pics 'without them using their quota' -- R3 fires before R6 in the ladder, so sharing_permissions stands over storage_quota.",
    "149031": "'best practices to signal a phishing page' and 1754987 -- R1a routes to the most specific topic, which is phishing_abuse_report, not how_to_usage.",
    "2430111": "'how do i restore files as ive been hacked' -- kept data_loss_recovery; the actionable request is the restore, with security as secondary.",
    "2978332": "Account closed, cannot reactivate, wants pictures back -- kept account_access; the blocker is reaching the account, not the files.",
    "407837": "'How do I cancel my Dropbox subscription' -- kept billing_subscription; Part 3 gives account closure to billing, and R1a routes to the most specific topic.",
}


def build() -> pd.DataFrame:
    v2 = pd.read_csv(V2_PATH, dtype={"customer_tweet_id": str})
    sweep = {s[0]: s for s in SWEEP}

    unknown = set(sweep) - set(v2.customer_tweet_id)
    if unknown:
        raise KeyError(f"sweep ids not in {V2_PATH}: {sorted(unknown)}")

    intents, rules, confs, statuses, reasons, changed = [], [], [], [], [], []
    for _, r in v2.iterrows():
        s = sweep.get(r.customer_tweet_id)
        if s is None:
            intents.append(r.human_intent_v2)
            rules.append(r.v2_rule)
            confs.append(r.v2_confidence)
            statuses.append(r.v2_status)
            reasons.append(CLOSE_CALLS.get(r.customer_tweet_id, ""))
            changed.append(False)
            continue
        _, intent, rule, conf, status, why = s
        final = intent if intent else r.human_intent_v2
        intents.append(final)
        rules.append(rule)
        confs.append(conf)
        statuses.append(status)
        reasons.append(why)
        changed.append(final != r.human_intent_v2 or status != r.v2_status)

    out = v2.copy()
    out["human_intent_v3"] = intents
    out["v3_status"] = statuses
    out["v3_rule"] = rules
    out["v3_confidence"] = confs
    out["v3_changed_since_v2"] = changed
    out["v3_reason"] = reasons
    return out


def report(v3: pd.DataFrame):
    scorable = v3[~v3.v3_status.isin(EXCLUDED_FROM_HEADLINE)]
    scorable_v2 = v3[~v3.v2_status.isin(EXCLUDED_FROM_HEADLINE)]
    moved = v3[v3.v3_changed_since_v2]

    print(f"\nGOLDEN LABELS v3 -- all {len(v3)} rows swept against taxonomy.md Part 11\n")
    print(f"  rows reviewed in v2 (disagreements)   58")
    print(f"  rows newly swept in v3                131")
    print(f"  rows moved by the sweep               {len(moved)}")
    print(f"  total changed vs the ORIGINAL labels  "
          f"{int((v3.human_intent_v3 != v3.human_intent).sum())}")
    print(f"  v3 status  " + ", ".join(
        f"{s}={int((v3.v3_status == s).sum())}"
        for s in ["resolved", "ambiguous", "insufficient_context"]))

    def acc(df, col):
        return int((df[col] == df.suggested_intent).sum()), len(df)

    o = acc(v3, "human_intent")
    a = acc(scorable_v2, "human_intent_v2")
    b = acc(scorable, "human_intent_v3")

    print(f"\n-- accuracy of the EXISTING, UNCHANGED model predictions --\n")
    print(f"  v1 labels, all 189 rows       {o[0]:3d}/{o[1]}  {o[0] / o[1]:.1%}   published baseline")
    print(f"  v2 labels, scorable rows      {a[0]:3d}/{a[1]}  {a[0] / a[1]:.1%}   disagreements only")
    print(f"  v3 labels, scorable rows      {b[0]:3d}/{b[1]}  {b[0] / b[1]:.1%}   full sweep")
    print(f"\n  v3 is the first of these three that is not biased upward by")
    print(f"  construction: v2 could only ever find corrections on rows the")
    print(f"  model was already scored wrong on.")

    print(f"\n-- what the sweep did to the score --\n")
    for _, r in moved.iterrows():
        was_right = r.human_intent_v2 == r.suggested_intent
        now_right = r.human_intent_v3 == r.suggested_intent
        excl = r.v3_status in EXCLUDED_FROM_HEADLINE
        eff = "excluded" if excl else ("+1" if now_right and not was_right
                                       else "-1" if was_right and not now_right else " 0")
        print(f"  [{r.customer_tweet_id}] {eff:8s} {r.human_intent_v2} -> "
              f"{r.human_intent_v3 if not excl else r.v3_status} ({r.v3_rule})")

    print(f"\n-- per-intent recall, v1 labels vs v3 labels --\n")
    print(f"{'intent':30s} {'v1 n':>5s} {'v1 rec':>7s} {'v3 n':>5s} {'v3 rec':>7s} {'delta':>7s}")
    rows = []
    for i in sorted(set(v3.human_intent) | set(v3.human_intent_v3)):
        x = v3[v3.human_intent == i]
        y = scorable[scorable.human_intent_v3 == i]
        rx = (x.human_intent == x.suggested_intent).mean() if len(x) else float("nan")
        ry = (y.human_intent_v3 == y.suggested_intent).mean() if len(y) else float("nan")
        d = ry - rx if len(x) and len(y) else float("nan")
        rows.append((i, d))
        print(f"{i:30s} {len(x):5d} {rx:7.2f} {len(y):5d} {ry:7.2f} {d:+7.2f}")
    up = [r for r in rows if r[1] == r[1] and r[1] > 0.01]
    dn = [r for r in rows if r[1] == r[1] and r[1] < -0.01]
    print(f"\n  improved: {', '.join(f'{i} ({d:+.2f})' for i, d in sorted(up, key=lambda x: -x[1])) or 'none'}")
    print(f"  worsened: {', '.join(f'{i} ({d:+.2f})' for i, d in sorted(dn, key=lambda x: x[1])) or 'none'}")

    amb = v3[v3.v3_status == "ambiguous"]
    exc = v3[v3.v3_status.isin(EXCLUDED_FROM_HEADLINE)]
    print(f"\n-- unresolved after the full sweep --\n")
    print(f"  {len(exc)} insufficient_context (excluded from the headline)")
    print(f"  {len(amb)} ambiguous: " + ", ".join(amb.customer_tweet_id))
    low = v3[v3.v3_confidence == "low"]
    print(f"  {len(low)} low-confidence: " + ", ".join(low.customer_tweet_id))
    print()


def main(write: bool):
    v3 = build()
    report(v3)
    if write:
        v3.to_csv(OUT_PATH, index=False, encoding="utf-8")
        audit = v3[v3.human_intent_v3 != v3.human_intent][[
            "customer_tweet_id", "customer_text", "human_intent", "human_intent_v2",
            "human_intent_v3", "v3_status", "v3_rule", "v3_confidence", "v3_reason",
            "suggested_intent",
        ]]
        audit.to_csv(AUDIT_PATH, index=False, encoding="utf-8")
        print(f"wrote {OUT_PATH} ({len(v3)} rows)")
        print(f"wrote {AUDIT_PATH} ({len(audit)} rows changed vs the original labels)")
        print(f"golden_labels.csv, golden_labels_v2.csv, adjudication_v*.csv all unchanged.\n")
    else:
        print("dry run -- pass --write to save\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--write", action="store_true")
    main(write=p.parse_args().write)
