"""
STEP 5 -- replace the two dead uncertainty branches in escalation.py.

The v1 policy has two overrides that are supposed to catch "the classifier
does not know":

  needs_human_triage   a human set it 12 times; the model set it 0 times.
  confidence < 0.85    96% of predictions land >= 0.85, and accuracy is
                       flat across the threshold (63% vs 64%), so the
                       branch fires almost never and carries no signal
                       when it does.

Both are dead. This script tests a replacement that costs NO new API
calls: the two model runs we already have on all 189 golden rows --
gpt-oss-20b and gpt-oss-120b, same prompt, same schema -- and treats
DISAGREEMENT BETWEEN THEM as the abstain signal.

This is the cheap version of self-consistency sampling. Sampling one model
three times costs 3x per classification forever; here both runs already
exist on disk, and a second model is arguably a better uncertainty probe
than three samples of one, because independent families do not share a
temperature-driven wobble -- they disagree where the TASK is ambiguous.

Reports escalation precision, recall, missed and over-escalations for the
v1 policy and for each candidate, against golden_labels_v3.csv labels and
the human escalate decision. Changes nothing: escalation.py is untouched
until a candidate is chosen.

Usage: ./.venv/Scripts/python.exe scripts/23_escalation_v2.py
"""
import math

import pandas as pd

from cache import ClassificationCache
from escalation import INTENT_DEFAULTS, ESCALATE, LOW_CONFIDENCE_THRESHOLD
from groq_lib import cache_namespace

V3_PATH = "data/golden_labels_v3.csv"
SPLIT_PATH = "data/golden_split.csv"
CACHE_PATH = "data/classification_cache.jsonl"
CHALLENGER = "openai/gpt-oss-120b"
EXCLUDED = ("insufficient_context", "ambiguous")

OVERRIDE_FLAGS = ["legal_sensitive", "wants_human", "churn_threat", "abusive_content"]


def load() -> pd.DataFrame:
    gold = pd.read_csv(V3_PATH, dtype={"customer_tweet_id": str})
    split = pd.read_csv(SPLIT_PATH, dtype={"customer_tweet_id": str})
    gold = gold.merge(split[["customer_tweet_id", "split"]], on="customer_tweet_id")
    # prompt_version is pinned to p1, NOT the module default: the 120b run
    # was made under p1, and cache_namespace() defaults to whatever
    # PROMPT_VERSION currently is. Leaving it to default silently returned
    # zero rows the moment PROMPT_VERSION moved to p2 -- the namespace
    # doing exactly the job it was added for, on its author.
    cache = ClassificationCache(
        CACHE_PATH, namespace=cache_namespace(model=CHALLENGER, prompt_version="p1")
    )
    gold["challenger_intent"] = [
        (cache.get(t) or {}).get("intent") for t in gold.customer_tweet_id
    ]
    return gold


def flags_of(row) -> set[str]:
    v = row.suggested_flags
    if isinstance(v, float) and math.isnan(v):
        return set()
    return {f.strip() for f in str(v).split(",") if f.strip()}


def decide(row, rule: str) -> bool:
    """True = escalate. Shares the v1 flag/turn_type overrides; only the
    UNCERTAINTY branch differs between rules."""
    f = flags_of(row)
    if "needs_human_triage" in f:
        return True

    if rule == "v1_confidence":
        if float(row.suggested_confidence) < LOW_CONFIDENCE_THRESHOLD:
            return True
    elif rule == "disagreement":
        if row.challenger_intent and row.challenger_intent != row.suggested_intent:
            return True
    elif rule == "both":
        if float(row.suggested_confidence) < LOW_CONFIDENCE_THRESHOLD:
            return True
        if row.challenger_intent and row.challenger_intent != row.suggested_intent:
            return True
    elif rule == "none":
        pass

    if row.suggested_turn_type == "disputing_prior_answer":
        return True
    if any(x in f for x in OVERRIDE_FLAGS):
        return True
    return INTENT_DEFAULTS.get(row.suggested_intent, (None,))[0] == ESCALATE


def score(df: pd.DataFrame, rule: str) -> dict:
    pred = df.apply(lambda r: decide(r, rule), axis=1)
    truth = df.human_escalate.astype(bool)
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    tn = int((~pred & ~truth).sum())
    return {
        "rule": rule,
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
        "missed": fn,
        "over": fp,
        "accuracy": (tp + tn) / len(df),
        "escalate_rate": (tp + fp) / len(df),
    }


def main():
    gold = load()
    s = gold[~gold.v3_status.isin(EXCLUDED) & gold.challenger_intent.notna()]

    print(f"\nESCALATION v2 candidates -- {len(s)} scorable rows "
          f"(labels golden_labels_v3.csv, human escalate decision as truth)\n")

    # --- is disagreement actually an uncertainty signal? ---
    agree = s[s.suggested_intent == s.challenger_intent]
    disag = s[s.suggested_intent != s.challenger_intent]
    a_acc = (agree.human_intent_v3 == agree.suggested_intent).mean()
    d_acc = (disag.human_intent_v3 == disag.suggested_intent).mean()
    print(f"-- does cross-model disagreement predict classifier error? --\n")
    print(f"  models AGREE     {len(agree):3d} rows, incumbent accuracy {a_acc:.1%}")
    print(f"  models DISAGREE  {len(disag):3d} rows, incumbent accuracy {d_acc:.1%}")
    print(f"  separation       {a_acc - d_acc:+.1%}")

    hi = s[s.suggested_confidence >= LOW_CONFIDENCE_THRESHOLD]
    lo = s[s.suggested_confidence < LOW_CONFIDENCE_THRESHOLD]
    print(f"\n  for comparison, self-reported confidence:")
    print(f"  conf >= {LOW_CONFIDENCE_THRESHOLD}      {len(hi):3d} rows, accuracy "
          f"{(hi.human_intent_v3 == hi.suggested_intent).mean():.1%}")
    if len(lo):
        print(f"  conf <  {LOW_CONFIDENCE_THRESHOLD}      {len(lo):3d} rows, accuracy "
              f"{(lo.human_intent_v3 == lo.suggested_intent).mean():.1%}")
    else:
        print(f"  conf <  {LOW_CONFIDENCE_THRESHOLD}        0 rows -- the branch never fires")

    # --- escalation outcomes ---
    print(f"\n-- escalation policy, all scorable rows --\n")
    print(f"{'rule':16s} {'prec':>6s} {'recall':>7s} {'missed':>7s} {'over':>6s} "
          f"{'acc':>6s} {'esc%':>6s}")
    rows = [score(s, r) for r in ("v1_confidence", "none", "disagreement", "both")]
    for m in rows:
        print(f"{m['rule']:16s} {m['precision']:6.2f} {m['recall']:7.2f} "
              f"{m['missed']:7d} {m['over']:6d} {m['accuracy']:6.2f} {m['escalate_rate']:6.1%}")

    print(f"\n-- same, DEV split only (holdout untouched) --\n")
    dev = s[s.split == "dev"]
    print(f"{'rule':16s} {'prec':>6s} {'recall':>7s} {'missed':>7s} {'over':>6s} {'acc':>6s}")
    for r in ("v1_confidence", "disagreement", "both"):
        m = score(dev, r)
        print(f"{m['rule']:16s} {m['precision']:6.2f} {m['recall']:7.2f} "
              f"{m['missed']:7d} {m['over']:6d} {m['accuracy']:6.2f}")

    base = score(s, "v1_confidence")
    print(f"\n-- what the v1 misses look like --\n")
    miss = s[s.apply(lambda r: not decide(r, "v1_confidence"), axis=1) & s.human_escalate.astype(bool)]
    caught = miss[miss.suggested_intent != miss.challenger_intent]
    print(f"  {len(miss)} missed escalations under v1")
    print(f"  {len(caught)} of them sit on a cross-model disagreement, so the "
          f"disagreement rule catches them")
    print(f"  {len(miss) - len(caught)} would still be missed -- the intent was agreed and wrong,")
    print(f"  or the human escalated for a reason the intent does not encode.\n")


if __name__ == "__main__":
    main()
