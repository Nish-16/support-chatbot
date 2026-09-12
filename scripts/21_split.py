"""
STEP 8 -- freeze a 60/40 dev / held-out split of the golden set.

Until now all 189 rows have been both the tuning set and the reporting
set. Two rounds of prompt work against them and the headline stops
meaning anything, because the prompt will have been fitted to the same
rows it is scored on.

Design decisions, stated because each one is a place this could go wrong:

* Written to disk ONCE, as data/golden_split.csv, and read from there
  forever after. A split recomputed at each call site drifts the moment
  anyone changes a seed, a sort order, or the row count -- and a drifting
  held-out set is worse than none, because it looks rigorous.
* Stratified on human_intent_v3, so thin intents are present on both
  sides rather than landing entirely in one by luck. With intents as
  small as 8 rows, an unstratified draw can leave a whole intent out of
  dev.
* insufficient_context and ambiguous rows are assigned a split like any
  other row, but they carry their v3_status, so anything scoring them
  can exclude them the same way v3's own headline does.
* The split is on the v3 labels, which are themselves an adjudication.
  If the labels are revised again the split stays put -- it keys on
  tweet_id, not on labels.

Usage: ./.venv/Scripts/python.exe scripts/21_split.py
       ./.venv/Scripts/python.exe scripts/21_split.py --write
"""
import argparse
import os

import pandas as pd

V3_PATH = "data/golden_labels_v3.csv"
SPLIT_PATH = "data/golden_split.csv"
DEV_FRACTION = 0.60
SEED = 20260912


def build() -> pd.DataFrame:
    gold = pd.read_csv(V3_PATH, dtype={"customer_tweet_id": str})
    rows = []
    for intent, group in gold.groupby("human_intent_v3"):
        # Shuffle within the intent, then take the first 60% for dev.
        # round() rather than int(): with n=8, int() would give 4 and
        # round() gives 5, and systematically under-filling dev on every
        # small intent is how a "60%" split quietly becomes 55%.
        shuffled = group.sample(frac=1, random_state=SEED)
        n_dev = int(round(len(shuffled) * DEV_FRACTION))
        for i, (_, r) in enumerate(shuffled.iterrows()):
            rows.append({
                "customer_tweet_id": r.customer_tweet_id,
                "split": "dev" if i < n_dev else "holdout",
                "human_intent_v3": r.human_intent_v3,
                "v3_status": r.v3_status,
            })
    return pd.DataFrame(rows).sort_values("customer_tweet_id").reset_index(drop=True)


def report(split: pd.DataFrame):
    dev = split[split.split == "dev"]
    hold = split[split.split == "holdout"]
    print(f"\nGOLDEN SPLIT -- seed {SEED}, stratified on human_intent_v3\n")
    print(f"  dev      {len(dev):3d}  {len(dev) / len(split):.1%}")
    print(f"  holdout  {len(hold):3d}  {len(hold) / len(split):.1%}")
    print(f"  total    {len(split):3d}")

    print(f"\n{'intent':30s} {'dev':>4s} {'hold':>5s}")
    for i in sorted(split.human_intent_v3.unique()):
        g = split[split.human_intent_v3 == i]
        print(f"{i:30s} {int((g.split == 'dev').sum()):4d} {int((g.split == 'holdout').sum()):5d}")

    thin = [i for i in split.human_intent_v3.unique()
            if int((split[split.human_intent_v3 == i].split == "holdout").sum()) < 3]
    if thin:
        print(f"\n  WARNING -- under 3 held-out examples: {', '.join(thin)}.")
        print(f"  A per-intent number from this few rows moves ~33 points per row;")
        print(f"  report these intents as counts, never as a percentage.")

    scor = split[~split.v3_status.isin(["insufficient_context", "ambiguous"])]
    print(f"\n  scorable rows: dev {int((scor.split == 'dev').sum())}, "
          f"holdout {int((scor.split == 'holdout').sum())}")
    print(f"\n  RULE: tune on dev. Touch holdout only to report a final number,")
    print(f"  and record every time it is read.\n")


def main(write: bool):
    if os.path.exists(SPLIT_PATH) and write:
        print(f"\n{SPLIT_PATH} already exists -- refusing to overwrite.")
        print("A re-rolled split silently invalidates every number measured "
              "against the old one.\n")
        return
    split = build()
    report(split)
    if write:
        split.to_csv(SPLIT_PATH, index=False, encoding="utf-8")
        print(f"wrote {SPLIT_PATH} -- treat as frozen from here.\n")
    else:
        print("dry run -- pass --write to freeze\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--write", action="store_true")
    main(write=p.parse_args().write)
