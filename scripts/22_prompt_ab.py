"""
Prompt A/B: p1 (shipped baseline) vs p2 (taxonomy.md Part 11 precedence
rules added to the system prompt), same model, same schema, same labels.

Scored on the DEV split only. The held-out 76 rows are not read here and
must not be, until the prompt is frozen -- that is the entire point of
freezing the split before doing prompt work.

p1's predictions come from golden_labels.csv's `suggested_intent`, i.e.
the published baseline run, not a re-run. p2's come from the cache under
namespace v2:p2:<model>, which is why the namespace fix had to land first:
under the old bare-"v2" namespace p2 would have read p1's answers back and
reported a perfect tie.

Resumable. If the 200k tokens/day cap is hit mid-run, rerun tomorrow --
every row already classified is on disk and is not re-paid for.

Usage:
  ./.venv/Scripts/python.exe scripts/22_prompt_ab.py --n 10
  ./.venv/Scripts/python.exe scripts/22_prompt_ab.py
  ./.venv/Scripts/python.exe scripts/22_prompt_ab.py --report-only
  ./.venv/Scripts/python.exe scripts/22_prompt_ab.py --split holdout   # ONE final read
"""
import argparse

import pandas as pd

from cache import ClassificationCache
from classify_runner import classify_many
from groq_lib import MODEL, PROMPT_VERSION, cache_namespace, make_client

V3_PATH = "data/golden_labels_v3.csv"
SPLIT_PATH = "data/golden_split.csv"
CACHE_PATH = "data/classification_cache.jsonl"
EXCLUDED = ("insufficient_context", "ambiguous")


def load(split_name: str) -> pd.DataFrame:
    gold = pd.read_csv(V3_PATH, dtype={"customer_tweet_id": str})
    split = pd.read_csv(SPLIT_PATH, dtype={"customer_tweet_id": str})
    gold = gold.merge(split[["customer_tweet_id", "split"]], on="customer_tweet_id")
    return gold[gold.split == split_name].reset_index(drop=True)


def run(split_name: str, n: int | None):
    rows = load(split_name)
    items = [(r.customer_tweet_id, r.customer_text) for _, r in rows.iterrows()]
    if n:
        items = items[:n]
    cache = ClassificationCache(CACHE_PATH, namespace=cache_namespace())
    print(f"\nsplit      {split_name} ({len(rows)} rows)")
    print(f"prompt     {PROMPT_VERSION}   model {MODEL}")
    print(f"namespace  {cache_namespace()}")
    cached_now = sum(1 for tid, _ in items if cache.get(tid) is not None)
    print(f"cached     {cached_now}/{len(items)} -- {len(items) - cached_now} calls to make\n")

    done = [0]

    def on_result(tweet_id, text, result, from_cache):
        done[0] += 1
        if done[0] % 20 == 0 or done[0] == len(items):
            print(f"  {done[0]}/{len(items)}")

    res = classify_many(make_client(), items, cache, on_result=on_result)
    print(f"\nclassified {len(res)}/{len(items)}")


def report(split_name: str):
    rows = load(split_name)
    cache = ClassificationCache(CACHE_PATH, namespace=cache_namespace())
    rows["p2_intent"] = [
        (cache.get(t) or {}).get("intent") for t in rows.customer_tweet_id
    ]
    missing = int(rows.p2_intent.isna().sum())
    if missing:
        print(f"\n{missing}/{len(rows)} rows not yet classified under {PROMPT_VERSION}.")
    have = rows[rows.p2_intent.notna()]
    s = have[~have.v3_status.isin(EXCLUDED)]
    if not len(s):
        print("nothing scorable yet\n")
        return

    p1 = int((s.human_intent_v3 == s.suggested_intent).sum())
    p2 = int((s.human_intent_v3 == s.p2_intent).sum())
    print(f"\n=== prompt p1 vs p2 on the {split_name.upper()} split ===")
    print(f"model {MODEL}, unchanged. Labels golden_labels_v3.csv.")
    print(f"{len(s)} scorable rows ({len(have) - len(s)} excluded)\n")
    print(f"  p1  (shipped baseline)        {p1:3d}/{len(s)}  {p1 / len(s):.1%}")
    print(f"  p2  (+ Part 11 rules)         {p2:3d}/{len(s)}  {p2 / len(s):.1%}")
    print(f"  difference                    {p2 - p1:+3d}      {(p2 - p1) / len(s):+.1%}")

    both = int(((s.human_intent_v3 == s.suggested_intent) & (s.human_intent_v3 == s.p2_intent)).sum())
    only1 = int(((s.human_intent_v3 == s.suggested_intent) & (s.human_intent_v3 != s.p2_intent)).sum())
    only2 = int(((s.human_intent_v3 != s.suggested_intent) & (s.human_intent_v3 == s.p2_intent)).sum())
    neither = int(((s.human_intent_v3 != s.suggested_intent) & (s.human_intent_v3 != s.p2_intent)).sum())
    print(f"\n  both right {both:3d} | only p1 {only1:3d} | only p2 {only2:3d} | both wrong {neither:3d}")

    print(f"\n{'intent':30s} {'n':>4s} {'p1':>6s} {'p2':>6s} {'delta':>7s}")
    for i in sorted(s.human_intent_v3.unique()):
        g = s[s.human_intent_v3 == i]
        a = (g.human_intent_v3 == g.suggested_intent).mean()
        b = (g.human_intent_v3 == g.p2_intent).mean()
        print(f"{i:30s} {len(g):4d} {a:6.2f} {b:6.2f} {b - a:+7.2f}")

    print(f"\n-- p2 fixes --")
    for _, r in s[(s.human_intent_v3 != s.suggested_intent) & (s.human_intent_v3 == s.p2_intent)].iterrows():
        print(f"  [{r.customer_tweet_id}] {r.suggested_intent} -> {r.p2_intent}")
    print(f"-- p2 breaks --")
    for _, r in s[(s.human_intent_v3 == s.suggested_intent) & (s.human_intent_v3 != s.p2_intent)].iterrows():
        print(f"  [{r.customer_tweet_id}] {r.suggested_intent} -> {r.p2_intent}")

    out = f"data/prompt_ab_{PROMPT_VERSION}_{split_name}.csv"
    have[["customer_tweet_id", "customer_text", "human_intent_v3", "v3_status",
          "suggested_intent", "p2_intent"]].to_csv(out, index=False, encoding="utf-8")
    print(f"\nwrote {out}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="dev", choices=["dev", "holdout"])
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--report-only", action="store_true")
    a = p.parse_args()
    if a.split == "holdout":
        print("\n*** READING THE HELD-OUT SPLIT. Only do this on a frozen prompt, "
              "and record it. ***")
    if not a.report_only:
        run(a.split, a.n)
    report(a.split)
