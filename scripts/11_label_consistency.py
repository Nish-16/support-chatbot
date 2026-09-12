"""
Self-consistency audit of the hand-labeled golden set.

Annotator self-consistency is a standard QA step for a hand-built eval
set, and it's report-worthy evidence in its own right: "my labels are
the measuring instrument" is a claim that should come with a check, not
just an assertion.

Deliberately reports only INTERNAL contradictions -- pairs of near-
identical tweets labeled differently, escalate decisions that disagree
with each other, fields that look defaulted rather than decided. It
does NOT propose what any label should be. Supplying the "right"
answer would put model judgment back into the ground truth through the
back door, which is exactly what the blind-labeling design in
08_label_golden_set.py exists to prevent.

Every check below is a prompt to re-examine a row, not a verdict that
it's wrong -- two similar tweets can legitimately carry different
intents, and a policy disagreement can mean the policy is wrong rather
than the label.

Usage: ./.venv/Scripts/python.exe scripts/11_label_consistency.py
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from escalation import decide, ESCALATE
from retrieval import _clean

LABELS_PATH = "data/golden_labels.csv"
SIMILARITY_THRESHOLD = 0.45  # tuned by eye on this set -- low enough to catch
# paraphrases, high enough that unrelated tweets sharing support boilerplate
# ("@DropboxSupport help") don't flood the output.


def load() -> pd.DataFrame:
    df = pd.read_csv(LABELS_PATH)
    df["human_flags"] = df["human_flags"].fillna("")
    return df


def check_similar_pairs(df: pd.DataFrame):
    print("\n=== 1. Near-identical tweets with different intents ===")
    vec = TfidfVectorizer(min_df=1, ngram_range=(1, 2), stop_words="english")
    matrix = vec.fit_transform(df["customer_text"].map(_clean))
    sims = cosine_similarity(matrix)

    found = []
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            if sims[i, j] < SIMILARITY_THRESHOLD:
                continue
            if df.iloc[i]["human_intent"] == df.iloc[j]["human_intent"]:
                continue
            found.append((sims[i, j], i, j))

    if not found:
        print("None found -- no similar-text pairs carry conflicting intents.")
        return
    for sim, i, j in sorted(found, reverse=True)[:10]:
        a, b = df.iloc[i], df.iloc[j]
        print(f"\n  similarity {sim:.2f}")
        print(f"    [{a['human_intent']}] {' '.join(str(a['customer_text']).split())[:95]}")
        print(f"    [{b['human_intent']}] {' '.join(str(b['customer_text']).split())[:95]}")
    print(f"\n  ({len(found)} conflicting pair(s) total.)")


def check_escalate_consistency(df: pd.DataFrame):
    """Within one intent, an escalate decision that goes both ways isn't
    wrong -- README's matrix explicitly makes account_access and
    sync_app_bug conditional. It's only worth a look when one intent is
    overwhelmingly decided one way and a handful go the other."""
    print("\n=== 2. Escalate decisions that split within an intent ===")
    any_flagged = False
    for intent, group in df.groupby("human_intent"):
        if len(group) < 4:
            continue
        rate = group["human_escalate"].mean()
        minority = group[group["human_escalate"] != (rate >= 0.5)]
        if 0 < len(minority) <= max(2, len(group) * 0.25):
            any_flagged = True
            print(f"\n  {intent}: {int(rate * len(group))}/{len(group)} escalate "
                  f"-- {len(minority)} row(s) go the other way:")
            for _, row in minority.iterrows():
                print(f"    escalate={row['human_escalate']}: "
                      f"{' '.join(str(row['customer_text']).split())[:90]}")
    if not any_flagged:
        print("None -- no intent has a small minority of rows breaking its pattern.")


def check_against_policy(df: pd.DataFrame):
    """Compare your escalate call to what escalation.py's policy table
    would decide FROM YOUR OWN intent/turn_type/flags. A mismatch means
    either the label or the policy needs revisiting -- this doesn't say
    which, and the policy is the newer, less-examined of the two."""
    print("\n=== 3. Your escalate call vs. escalation.py's policy table ===")
    mismatches = []
    for _, row in df.iterrows():
        flags = set(str(row["human_flags"]).split(";")) if row["human_flags"] else set()
        synthetic = {
            "intent": row["human_intent"],
            "confidence": 1.0,  # a human label carries no confidence score; treat as certain
            "turn_type": row["human_turn_type"],
            "needs_human_triage": False,
            "legal_sensitive": "legal_sensitive" in flags,
            "wants_human": "wants_human" in flags,
            "churn_threat": "churn_threat" in flags,
            "abusive_content": "abusive_content" in flags,
        }
        policy_escalates = decide(synthetic).action == ESCALATE
        if bool(row["human_escalate"]) != policy_escalates:
            mismatches.append((row, policy_escalates))

    if not mismatches:
        print("None -- every escalate call matches the policy table.")
        return
    print(f"{len(mismatches)} of {len(df)} rows disagree "
          f"({len(mismatches) / len(df):.0%}). First 10:")
    for row, policy_escalates in mismatches[:10]:
        print(f"\n  [{row['human_intent']}] you={row['human_escalate']}, policy={policy_escalates}")
        print(f"    {' '.join(str(row['customer_text']).split())[:95]}")


def check_defaulted_fields(df: pd.DataFrame):
    """Fields that are almost never exercised can mean the signal is
    genuinely rare -- or that the prompt is being enter-ed through."""
    print("\n=== 4. Fields that may be getting defaulted ===")
    tt = df["human_turn_type"].value_counts()
    print(f"  turn_type: {tt.to_dict()}")
    top_share = tt.iloc[0] / len(df)
    if top_share > 0.9:
        print(f"    -> {tt.index[0]} is {top_share:.0%} of rows; worth confirming "
              f"mid-thread tweets are actually being spotted.")

    used = df["human_flags"].astype(bool).sum()
    print(f"  flags set on {used}/{len(df)} rows ({used / len(df):.0%})")
    if used / len(df) < 0.05:
        print("    -> near-zero flag usage; confirm this is real rarity, not fatigue.")

    sec = df["human_secondary_intent"].fillna("").astype(bool).sum()
    print(f"  secondary_intent set on {sec}/{len(df)} rows ({sec / len(df):.0%})")


def check_agreement_drift(df: pd.DataFrame):
    """If agreement with the classifier climbs steadily over the session,
    that's worth knowing before it's reported as accuracy -- it can be
    fatigue, or the taxonomy settling in your head, and the two have very
    different implications for the headline number."""
    print("\n=== 5. Agreement with the classifier over time ===")
    df = df.sort_values("labeled_at")
    chunk = max(10, len(df) // 5)
    for start in range(0, len(df), chunk):
        window = df.iloc[start:start + chunk]
        if len(window) < 5:
            continue
        rate = window["agrees_with_suggestion"].mean()
        print(f"  rows {start + 1:3d}-{start + len(window):3d}: {rate:.0%} agreement")
    print(f"  overall: {df['agrees_with_suggestion'].mean():.0%}")


if __name__ == "__main__":
    labels = load()
    print(f"Auditing {len(labels)} labeled rows from {LABELS_PATH}")
    check_similar_pairs(labels)
    check_escalate_consistency(labels)
    check_against_policy(labels)
    check_defaulted_fields(labels)
    check_agreement_drift(labels)
    print("\nNothing above is a verdict -- each item is a row worth a second look. "
          "The labels stay your call.")
