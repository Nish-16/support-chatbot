"""
STEP 9 (early) -- run a second model over the same 189 golden tweets with
the prompt, schema, temperature and taxonomy held identical, and score it
against the v3 labels.

The question this answers, and the only one it answers:

    Were the remaining errors caused by model capacity, or by the
    taxonomy and prompt?

Run now rather than at the end of the plan because the prompt has NOT yet
been touched. "Keep everything else identical" is exactly true today and
will never be this true again -- once the prompt changes, a model
comparison is confounded by which prompt each model saw.

Scored against data/golden_labels_v3.csv (the full 189-row sweep), with
insufficient_context and ambiguous rows excluded, the same way v3's own
headline is computed. The incumbent's predictions come from
golden_labels.csv's `suggested_intent` column -- they are not re-run, so
the incumbent side of every comparison is the exact published baseline.

Cache safety: results are keyed by taxonomy + prompt + MODEL (see
groq_lib.cache_namespace). Without the model in that key this script would
read gpt-oss-20b's cached answers back for every model and report perfect
agreement.

Usage:
  ./.venv/Scripts/python.exe scripts/20_model_ab.py --model openai/gpt-oss-120b --n 5
  ./.venv/Scripts/python.exe scripts/20_model_ab.py --model openai/gpt-oss-120b
  ./.venv/Scripts/python.exe scripts/20_model_ab.py --model openai/gpt-oss-120b --report-only
"""
import argparse
import os

import pandas as pd

from cache import ClassificationCache
from classify_runner import classify_many
from groq_lib import MODEL, cache_namespace, make_client

V3_PATH = "data/golden_labels_v3.csv"
CACHE_PATH = "data/classification_cache.jsonl"
EXCLUDED = ("insufficient_context", "ambiguous")


def out_path(model: str) -> str:
    return f"data/model_ab_{model.replace('/', '_')}.csv"


def scorable(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df.v3_status.isin(EXCLUDED)]


def run(model: str, n: int | None):
    gold = pd.read_csv(V3_PATH, dtype={"customer_tweet_id": str})
    items = [(r.customer_tweet_id, r.customer_text) for _, r in gold.iterrows()]
    if n:
        items = items[:n]

    cache = ClassificationCache(CACHE_PATH, namespace=cache_namespace(model=model))
    print(f"\nmodel      {model}")
    print(f"namespace  {cache_namespace(model=model)}")
    print(f"cached     {len(cache)} of {len(items)} requested\n")

    client = make_client()
    done = [0]

    def on_result(tweet_id, text, result, from_cache):
        done[0] += 1
        if done[0] % 20 == 0 or done[0] == len(items):
            print(f"  {done[0]}/{len(items)}")

    results = classify_many(
        client, items, cache,
        on_result=on_result,
        classify=lambda c, t: __import__("groq_lib").classify_message(c, t, model=model),
    )
    print(f"\nclassified {len(results)} of {len(items)}")
    return results


def report(model: str):
    gold = pd.read_csv(V3_PATH, dtype={"customer_tweet_id": str})
    cache = ClassificationCache(CACHE_PATH, namespace=cache_namespace(model=model))

    challenger, missing = [], 0
    for _, r in gold.iterrows():
        res = cache.get(r.customer_tweet_id)
        if res is None:
            missing += 1
            challenger.append(None)
        else:
            challenger.append(res.get("intent"))
    gold["challenger_intent"] = challenger

    if missing:
        print(f"\n{missing} of {len(gold)} rows not yet classified by {model} "
              f"-- run without --report-only first.\n")
    have = gold[gold.challenger_intent.notna()]
    s = scorable(have)
    if not len(s):
        return

    inc_hit = int((s.human_intent_v3 == s.suggested_intent).sum())
    cha_hit = int((s.human_intent_v3 == s.challenger_intent).sum())

    print(f"\n=== {MODEL}  vs  {model} ===")
    print(f"same prompt, same schema, same temperature, same taxonomy")
    print(f"scored against golden_labels_v3.csv, {len(s)} scorable rows "
          f"({len(have) - len(s)} excluded)\n")
    print(f"  incumbent  {MODEL:28s} {inc_hit:3d}/{len(s)}  {inc_hit / len(s):.1%}")
    print(f"  challenger {model:28s} {cha_hit:3d}/{len(s)}  {cha_hit / len(s):.1%}")
    print(f"  difference {'':28s} {cha_hit - inc_hit:+3d}      "
          f"{(cha_hit - inc_hit) / len(s):+.1%}")

    # McNemar-style: only the discordant cells carry information.
    both = int(((s.human_intent_v3 == s.suggested_intent) & (s.human_intent_v3 == s.challenger_intent)).sum())
    only_inc = int(((s.human_intent_v3 == s.suggested_intent) & (s.human_intent_v3 != s.challenger_intent)).sum())
    only_cha = int(((s.human_intent_v3 != s.suggested_intent) & (s.human_intent_v3 == s.challenger_intent)).sum())
    neither = int(((s.human_intent_v3 != s.suggested_intent) & (s.human_intent_v3 != s.challenger_intent)).sum())
    print(f"\n-- where they differ (only the off-diagonal cells matter) --\n")
    print(f"  both right              {both:3d}")
    print(f"  only incumbent right    {only_inc:3d}")
    print(f"  only challenger right   {only_cha:3d}")
    print(f"  both wrong              {neither:3d}")
    agree = int((s.suggested_intent == s.challenger_intent).sum())
    print(f"\n  the two models agree on {agree}/{len(s)} rows ({agree / len(s):.1%}), "
          f"right or wrong")
    print(f"  of the {neither} both-wrong rows, they pick the SAME wrong label "
          f"{int(((s.human_intent_v3 != s.suggested_intent) & (s.suggested_intent == s.challenger_intent)).sum())} times")

    print(f"\n-- per-intent recall --\n")
    print(f"{'intent':30s} {'n':>4s} {'incumbent':>10s} {'challenger':>11s} {'delta':>7s}")
    for i in sorted(s.human_intent_v3.unique()):
        g = s[s.human_intent_v3 == i]
        a = (g.human_intent_v3 == g.suggested_intent).mean()
        b = (g.human_intent_v3 == g.challenger_intent).mean()
        print(f"{i:30s} {len(g):4d} {a:10.2f} {b:11.2f} {b - a:+7.2f}")

    print(f"\n-- rows the challenger fixes --\n")
    for _, r in s[(s.human_intent_v3 != s.suggested_intent) & (s.human_intent_v3 == s.challenger_intent)].iterrows():
        print(f"  [{r.customer_tweet_id}] {r.suggested_intent} -> {r.challenger_intent}")
    print(f"\n-- rows the challenger breaks --\n")
    for _, r in s[(s.human_intent_v3 == s.suggested_intent) & (s.human_intent_v3 != s.challenger_intent)].iterrows():
        print(f"  [{r.customer_tweet_id}] {r.suggested_intent} -> {r.challenger_intent}")

    path = out_path(model)
    have[["customer_tweet_id", "customer_text", "human_intent", "human_intent_v3",
          "v3_status", "suggested_intent", "challenger_intent"]].to_csv(
        path, index=False, encoding="utf-8")
    print(f"\nwrote {path}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--n", type=int, default=None, help="smoke-test on the first N rows")
    p.add_argument("--report-only", action="store_true")
    a = p.parse_args()
    if not a.report_only:
        run(a.model, a.n)
    report(a.model)
