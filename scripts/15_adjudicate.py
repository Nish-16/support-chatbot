"""
STEP 1 adjudication of the 48 intent errors whose HUMAN label is one of
the four low-recall intents (complaint_dissatisfaction, how_to_usage,
storage_quota_plan_limits, sync_app_bug).

Purpose: before touching the prompt, establish how much of the 62.4%
accuracy gap is a MODEL error and how much is a LABEL error. Tuning a
prompt against labels that are themselves wrong just teaches the model
our labeling mistakes.

READ-ONLY with respect to data/golden_labels.csv. This script never
writes to the golden set. Every adjudication is a separate, versioned
record in data/adjudication_v1.csv, so the original 189-row baseline
stays exactly as reported in README.md (experiment rules 1 and 2).

Verdict vocabulary (deliberately 5-way, not "who won"):
  model_right     -- the human label is wrong; the model's label is correct
  human_right     -- the model is wrong; the human label stands
  both_wrong      -- a third intent is correct; neither label is
  ambiguous       -- both are defensible under the taxonomy AS WRITTEN;
                     only a precedence rule can settle it (-> STEP 2)
  context_missing -- not routable from the tweet alone; a prior turn is
                     needed (-> STEP 4), so no label is recoverable here

The `human_label_defensible` column answers "is the CURRENT human label
defensible?" independently of the verdict -- a label can be defensible
and still not be the one a written rule would produce.

Usage: ./.venv/Scripts/python.exe scripts/15_adjudicate.py
       ./.venv/Scripts/python.exe scripts/15_adjudicate.py --write
"""
import argparse

import pandas as pd

LABELS_PATH = "data/golden_labels.csv"
OUT_PATH = "data/adjudication_v1.csv"

FOUR = [
    "complaint_dissatisfaction",
    "how_to_usage",
    "storage_quota_plan_limits",
    "sync_app_bug",
]

# (tweet_id, verdict, proposed_final_intent, why_human_label_chosen,
#  human_label_defensible, reason_for_decision)
#
# "why_human_label_chosen" is reconstructed from the tweet plus the
# labeler's own human_note where one exists -- stated as an inferred
# rule, since the labeling tool never recorded a rationale field.
ADJUDICATIONS = [
    # ---------------- complaint_dissatisfaction (21) ----------------
    ("2124237", "model_right", "feature_request",
     "Annoyed tone + emoji read as a rant.", "no",
     "Contains a concrete actionable request ('stop using IP addresses with no host records'). Annoyance is the frame, not the need."),
    ("1497111", "model_right", "billing_subscription",
     "Anger words ('fraud', 'Shame') dominated the read.", "no",
     "A disputed charge and a refused refund is a concrete billing transaction. Anger belongs in sentiment/churn_threat, not in the intent enum."),
    ("622045", "ambiguous", "complaint_dissatisfaction",
     "Evaluative framing ('is not cool') with no request.", "yes",
     "Names a real quota mechanic but asks for nothing. Under a no-actionable-request rule this is complaint; under a topic-wins rule it is storage_quota. Needs the STEP 2 rule."),
    ("2425723", "ambiguous", "complaint_dissatisfaction",
     "Opens with 'This is very disapointing'.", "yes",
     "Criticises a permissions limitation without asking for help. Same unresolved precedence as 622045."),
    ("838554", "model_right", "how_to_usage",
     "'worthless' read as dissatisfaction.", "no",
     "Ends in an explicit 'How do I disable?!' -- an answerable question. The useful reply is the disable steps, not an apology."),
    ("1041902", "context_missing", "",
     "Fragmentary grievance, no clear topic.", "n/a",
     "Mid-thread reply referencing an earlier DM link. Topic is unrecoverable from this tweet alone; the model's account_access is a guess."),
    ("1050476", "human_right", "complaint_dissatisfaction",
     "Pure criticism of prioritisation, no ask.", "yes",
     "Textbook positive case: evaluative, no product topic, no actionable request. The model's no_action_needed is wrong -- this is an angry customer, not spam."),
    ("241283", "model_right", "sync_app_bug",
     "Unclear; no grievance markers present at all.", "no",
     "'I can't download dropbox mobile after the last update' is a plain bug report with neutral tone. Looks like a straightforward labeling slip."),
    ("1856823", "human_right", "complaint_dissatisfaction",
     "Price venting with no request.", "yes",
     "'Just crazy the new prices. Geez!' names a billing topic but asks nothing. Positive case for complaint under the no-request rule."),
    ("1960101", "both_wrong", "no_action_needed",
     "Treated as a frustrated customer.", "no",
     "Asks only for a phone number. Carried entirely by the wants_human flag; neither complaint nor how_to_usage describes the need. Escalation was correct via the flag."),
    ("615215", "ambiguous", "how_to_usage",
     "Read 'why did you remove' as criticism.", "yes",
     "INCONSISTENT with 1863374: near-identical tweet (missing green check icon) labeled how_to_usage there and complaint here."),
    ("847909", "human_right", "complaint_dissatisfaction",
     "Price criticism, soft suggestion to review.", "yes",
     "'In Brazil is so expensive... You ougth to review this price' has no answerable support request. feature_request is the nearest rival, not billing."),
    ("74467", "context_missing", "",
     "Frustration words in a fragment.", "n/a",
     "A filename plus 'extremely frustrating'. This is providing_requested_info on an unseen prior turn; no intent is recoverable standalone."),
    ("1452735", "model_right", "how_to_usage",
     "Unclear; tone is neutral.", "no",
     "'why can't I get notifications of added files to my android phone?' is an answerable product question with no grievance. Second labeling slip of the same shape as 241283."),
    ("2540849", "both_wrong", "feature_request",
     "Labeled complaint; the labeler's own note says 'feedback'.", "no",
     "'You need to send the login IP address when you send warnings' is a product suggestion. The note contradicts the label. The model's security_account_compromise is worse -- it keyed on 'unauthorized logins'."),
    ("1804904", "model_right", "service_outage",
     "'What is going on?' read as exasperation.", "no",
     "Customer checked the status page and is asking about a service problem affecting their whole team. service_outage is the routable topic."),
    ("1082713", "model_right", "feature_request",
     "'That's absurd' read as a rant.", "no",
     "'Please add it back' is an explicit actionable request. The grievance is the frame around it."),
    ("2551167", "ambiguous", "billing_subscription",
     "All-caps accusation ('RIPOFF').", "yes",
     "Makes a concrete factual claim (charged before the trial ended) that implies a refund, but never asks. Sits exactly on the STEP 2 boundary."),
    ("1383896", "ambiguous", "service_outage",
     "Terse sarcasm with no explicit question.", "yes",
     "'This is not the moment to stop working' reports an outage in complaint form. Topic-wins gives service_outage; no-request gives complaint."),
    ("2179658", "human_right", "complaint_dissatisfaction",
     "Sarcastic reply to another user, no ask.", "yes",
     "The model's storage_quota is wrong by an EXPLICIT taxonomy exclusion: this is the local hard drive filling up, and storage_quota_plan_limits says 'not device disk usage'."),
    ("1687134", "context_missing", "",
     "'disappointed that Dropbox isn't responding'.", "n/a",
     "A nudging_no_response turn about an unseen earlier request. The complaint label is really encoding a turn_type."),

    # ---------------- how_to_usage (9) ----------------
    ("2629478", "model_right", "storage_quota_plan_limits",
     "Ends in 'Please advise' -> read as a how-to.", "no",
     "Deleted files reappearing while the quota stays full is taxonomy.md's named storage_quota case ('quota not clearing after deletion')."),
    ("259327", "model_right", "sharing_permissions",
     "Starts with 'how can I' -> read as a how-to.", "no",
     "taxonomy.md assigns team folders to sharing_permissions and states how_to_usage 'shrinks in v2 only because sharing and storage how-tos move to their own intents'. A documented rule, violated."),
    ("1874043", "both_wrong", "no_action_needed",
     "Contains 'can't remember which email'.", "no",
     "The tweet is about concert tickets, not Dropbox. Off-topic -> no_action_needed. The model's account_access keyed on the word 'email'."),
    ("2222820", "ambiguous", "account_access",
     "'How can I unlink' -> read as a how-to.", "yes",
     "Unlinking a work account touches account_access ('business account setup'), but nothing is broken. The boundary is genuinely unwritten."),
    ("1863374", "ambiguous", "sync_app_bug",
     "The labeler's own note: 'no right answer just classified to least wrong'.", "yes",
     "INCONSISTENT with 615215 (same missing-green-check subject, labeled complaint there). The note is an explicit admission the taxonomy gave no answer."),
    ("2053253", "model_right", "storage_quota_plan_limits",
     "Ends in 'how to make DB recognize it' -> read as a how-to.", "no",
     "Near-verbatim match for taxonomy.md's storage_quota bug example. The 'how to' phrasing is surface form, not intent."),
    ("1999298", "model_right", "storage_quota_plan_limits",
     "'How do we fix it?' -> read as a how-to.", "no",
     "Same as 2053253: permanently deleted items, quota unchanged. Explicitly a storage_quota case in the taxonomy."),
    ("37940", "model_right", "sharing_permissions",
     "'What am I supposed to do' -> read as a how-to.", "no",
     "'I can't access my shared folder' is the actual need. Shared-folder access is sharing_permissions."),
    ("988433", "model_right", "sync_app_bug",
     "Ends in 'how do I fix this?'.", "no",
     "Edits reverting after a sync is a malfunction. how_to_usage is defined as 'nothing is broken' -- something is broken here."),

    # ---------------- storage_quota_plan_limits (12) ----------------
    ("1495052", "ambiguous", "billing_subscription",
     "'I need more space' -> read as storage.", "yes",
     "'What are my options' is a plan-cost question. taxonomy.md itself calls this intent 'exactly on the billing/technical seam'; the seam was never resolved in writing. Proposed final follows R6 (a question whose answer is a price is billing), which STEP 2 must ratify."),
    ("1577541", "model_right", "complaint_dissatisfaction",
     "The quoted text mentions a full account.", "no",
     "Sarcastic parody of a Dropbox upsell ending in profanity. No request, purely evaluative -- and the labeler set abusive_content, confirming they read it as a rant."),
    ("1901120", "human_right", "storage_quota_plan_limits",
     "Question about promotional storage expiry.", "yes",
     "A factual question about earned account space. taxonomy.md routes storage how-tos to this intent, so the model's how_to_usage is wrong by the written rule."),
    ("1324437", "human_right", "storage_quota_plan_limits",
     "Question about a free user's upload limit.", "yes",
     "Same as 1901120: a storage-limit question. The model's how_to_usage ignores the documented carve-out."),
    ("1523722", "both_wrong", "billing_subscription",
     "'Plus -> Professional' read as a plan limit.", "no",
     "Asking to trial a paid tier is a plan/pricing question -> billing_subscription under its 'pure pricing questions' sub-rule. Neither label fits."),
    ("53189", "model_right", "feature_request",
     "The subject is storage space.", "no",
     "'It would be great if they offered a little bit more space' is a suggestion, not a problem. feature_request is defined as exactly this."),
    ("2742476", "ambiguous", "billing_subscription",
     "'add more space' -> read as storage.", "yes",
     "'and how much does it cost' makes it half a pricing question. The billing/storage seam again; proposed final follows R6, pending STEP 2."),
    ("193025", "model_right", "billing_subscription",
     "The subject is a paid plan.", "no",
     "\"can't afford to pay $500\" is affordability/pricing, not a storage cap. billing_subscription or complaint; storage_quota is the weakest of the three."),
    ("1992403", "model_right", "billing_subscription",
     "Mentions 'unlimited storage'.", "no",
     "'Can you share the cost of the Plan' is a pure pricing question, which taxonomy.md's billing sub-rule explicitly claims."),
    ("2421520", "model_right", "feature_request",
     "The subject is more storage.", "no",
     "\"I'd happily pay $200 a year if the plan included 2TB\" is a pricing/packaging suggestion. Same shape as 53189."),
    ("2421521", "model_right", "feature_request",
     "Compares plan storage.", "no",
     "Public opinion aimed at other users, not a support request. feature_request or no_action_needed; not a quota issue."),
    ("2976667", "both_wrong", "how_to_usage",
     "Contains 'not enough space'.", "no",
     "This is LAPTOP disk space, which storage_quota_plan_limits explicitly excludes ('not device disk usage'). The model's sync_app_bug is also wrong -- nothing is broken, they are asking whether an external drive works."),

    # ---------------- sync_app_bug (6) ----------------
    ("1600556", "ambiguous", "sharing_permissions",
     "'keeps loading' read as an app bug.", "yes",
     "A shared link failing to render is both a client bug and a sharing issue. taxonomy.md gives 'shared folders/links' to sharing_permissions but never says whether symptom or object wins."),
    ("2607972", "model_right", "sharing_permissions",
     "The word 'bug' was taken at face value.", "no",
     "'accepting a shared folder' is an invite/join problem, which sharing_permissions claims verbatim."),
    ("1972693", "context_missing", "",
     "Assumed a sync context from the thread.", "n/a",
     "'350/350, no firewalls or anything in between' is pure providing_requested_info. The model's storage_quota is a hallucination off '350/350'."),
    ("2828924", "context_missing", "",
     "Assumed the prior topic was a sync bug.", "n/a",
     "\"I Have done this. Too bad it didn't solve the problem\" carries no topic at all. Correct handling is turn_type=disputing_prior_answer -> escalate, which is a STEP 4/5 fix, not an intent fix."),
    ("79858", "model_right", "storage_quota_plan_limits",
     "Read 'deleted files not reflecting' as a sync failure.", "no",
     "Permanently deleted files with unchanged storage usage is taxonomy.md's literal storage_quota example."),
    ("2910246", "model_right", "storage_quota_plan_limits",
     "Same as 79858.", "no",
     "Identical shape to 79858 and 1999298 -- and 1999298 was labeled how_to_usage, so one family of tweets received three different human labels."),
]

VERDICTS = ["model_right", "human_right", "both_wrong", "ambiguous", "context_missing"]

# Which candidate precedence rule (see ADJUDICATION.md) settles each
# `ambiguous` row. "residual" = no proposed rule decides it, so STEP 2
# needs an explicit tie-break rather than a per-row judgment call.
# Recorded here so the "N of 10 resolved" claim is computed, not asserted.
AMBIGUOUS_RESOLVED_BY = {
    "615215":  "R1",        # 'why did you remove X' is an answerable question -> topic
    "622045":  "R1",        # names quota, asks nothing -> complaint
    "2425723": "R1",        # criticises permissions, asks nothing -> complaint
    "1863374": "R2",        # icon change suggests a malfunction -> sync_app_bug
    "1600556": "R3",        # object is a shared link -> sharing_permissions
    "1495052": "R6",        # answer is a price -> billing_subscription
    "2742476": "R6",        # answer is a price -> billing_subscription
    "2551167": "residual",  # grievance implies a refund without asking for one
    "1383896": "residual",  # reports an outage in pure complaint form
    "2222820": "residual",  # nothing broken, but the object is an account link
}

MEANING = {
    "model_right": "human label wrong; model was correct",
    "human_right": "model wrong; human label stands",
    "both_wrong": "a third intent is correct",
    "ambiguous": "both defensible; needs a written precedence rule",
    "context_missing": "not routable from the tweet alone",
}


def build() -> pd.DataFrame:
    labels = pd.read_csv(LABELS_PATH, dtype={"customer_tweet_id": str})
    by_id = labels.set_index("customer_tweet_id")

    rows = []
    for tweet_id, verdict, final, why, defensible, reason in ADJUDICATIONS:
        if tweet_id not in by_id.index:
            raise KeyError(f"{tweet_id} not in {LABELS_PATH}")
        r = by_id.loc[tweet_id]
        rows.append({
            "customer_tweet_id": tweet_id,
            "tweet": " ".join(str(r.customer_text).split()),
            "human_intent": r.human_intent,
            "model_intent": r.suggested_intent,
            "human_secondary": r.human_secondary_intent,
            "model_secondary": r.suggested_secondary_intent,
            "model_confidence": r.suggested_confidence,
            "human_note": r.human_note,
            "why_human_label_chosen": why,
            "human_label_defensible": defensible,
            "verdict": verdict,
            "proposed_final_intent": final,
            "resolved_by_rule": AMBIGUOUS_RESOLVED_BY.get(tweet_id, "") if verdict == "ambiguous" else "",
            "reason_for_decision": reason,
        })
    return pd.DataFrame(rows)


def report(adj: pd.DataFrame):
    labels = pd.read_csv(LABELS_PATH)
    baseline_correct = int((labels.human_intent == labels.suggested_intent).sum())
    n = len(labels)
    counts = adj.verdict.value_counts()

    print(f"\nADJUDICATION v1 -- {len(adj)} errors reviewed "
          f"(human label in one of the four low-recall intents)\n")

    print(f"{'verdict':16s} {'n':>4s}  meaning")
    for v in VERDICTS:
        print(f"{v:16s} {int(counts.get(v, 0)):4d}  {MEANING[v]}")

    print("\n-- by human intent --")
    print(f"{'human intent':28s} " + " ".join(f"{v[:9]:>9s}" for v in VERDICTS))
    for intent in FOUR:
        c = adj[adj.human_intent == intent].verdict.value_counts()
        print(f"{intent:28s} " + " ".join(f"{int(c.get(v, 0)):9d}" for v in VERDICTS))

    n_model_right = int(counts.get("model_right", 0))
    n_amb = int(counts.get("ambiguous", 0))
    floor = baseline_correct + n_model_right
    ceiling = floor + n_amb

    print("\n-- effect on measured accuracy, with NO model or prompt change --\n")
    print(f"  as reported today          {baseline_correct:3d}/{n}  {baseline_correct / n:.1%}")
    print(f"  after label corrections    {floor:3d}/{n}  {floor / n:.1%}   "
          f"(+{n_model_right} rows the model already had right)")
    print(f"  upper bound if STEP 2's rules also settle every ambiguous row")
    print(f"                             {ceiling:3d}/{n}  {ceiling / n:.1%}   (+{n_amb} more)")

    amb = adj[adj.verdict == "ambiguous"]
    by_rule = amb.resolved_by_rule.value_counts()
    resolved = int((amb.resolved_by_rule != "residual").sum())
    print(f"\n-- which candidate rule settles each ambiguous row (STEP 2 input) --\n")
    for rule in sorted(by_rule.index):
        ids = ", ".join(amb[amb.resolved_by_rule == rule].customer_tweet_id)
        print(f"  {rule:9s} {int(by_rule[rule])}  {ids}")
    print(f"\n  {resolved} of {len(amb)} settled by a rule; "
          f"{len(amb) - resolved} need an explicit tie-break in STEP 2.")

    hr = int(counts.get("human_right", 0))
    bw = int(counts.get("both_wrong", 0))
    cm = int(counts.get("context_missing", 0))
    print(f"\n  genuine model errors left in this cluster: {hr + bw + cm} "
          f"({hr} plain misses, {bw} where neither label was right, "
          f"{cm} unroutable without a prior turn)\n")


def main(write: bool):
    adj = build()
    report(adj)
    if write:
        adj.to_csv(OUT_PATH, index=False, encoding="utf-8")
        print(f"wrote {OUT_PATH} ({len(adj)} rows)")
        print(f"{LABELS_PATH} NOT modified -- the 189-row baseline is unchanged.\n")
    else:
        print(f"dry run -- pass --write to save {OUT_PATH}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--write", action="store_true", help=f"save the table to {OUT_PATH}")
    main(write=p.parse_args().write)
