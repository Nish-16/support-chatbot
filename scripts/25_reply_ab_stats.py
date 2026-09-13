"""
Paired significance test for reply-eval A/B runs. No API calls.

13_reply_eval.py reports one mean per configuration, and comparing two means
by eye is how this project nearly shipped "the vector retriever writes worse
replies": a -0.32 gap that a re-run of the SAME configuration reproduced at
-0.26. This script puts an interval on the gap before anyone reads it.

Method
------
Rows are paired by customer tweet (same tweet under both configurations), so
tweet difficulty cancels out. For each metric it reports the mean paired delta
(B - A), a 95% bootstrap confidence interval, and a sign-flip permutation
p-value. Several labels can be pooled per side (comma-separated); a tweet's
score is then its mean across those runs.

It also reports how many tweets a run would need to detect a given shift at
80% power, from the observed spread of paired deltas -- the honest answer to
"why not just run it again?".

Only one judge and one rubric are compared at a time, for the reason
13_reply_eval.py keeps them apart: a score from a different judge or rubric is
a different measurement, not another sample of the same one.

Usage:
  ./.venv/Scripts/python.exe scripts/25_reply_ab_stats.py --list
  ./.venv/Scripts/python.exe scripts/25_reply_ab_stats.py --a tfidf:dedup1:t0 --b vector:dedup1:t0
  ./.venv/Scripts/python.exe scripts/25_reply_ab_stats.py --a tfidf,tfidf:dedup1 --b vector,vector:dedup1
"""
import argparse

import numpy as np
import pandas as pd

EVALS_PATH = "data/reply_evals.csv"
METRICS = ["overall", "relevance", "policy_compliance", "grounding", "is_deflection"]
N_RESAMPLES = 10_000


def load(judge: str, rubric: str, system: str, keep: str = "last") -> pd.DataFrame:
    df = pd.read_csv(EVALS_PATH)
    df["retriever"] = df["retriever"].fillna("tfidf")
    df = df[(df["judge_model"] == judge) & (df["rubric_version"] == rubric)
            & (df["system"] == system)]
    # A resumed run can re-score a (tweet, configuration); keep one verdict.
    # The rule is not neutral: tfidf:dedup1:filt1:t0 has 7 re-scored tweets,
    # one judged 2 then 5, and first-vs-last moves that comparison from
    # -0.28 [-0.62, -0.03] to -0.21 [-0.49, +0.03]. Hence the flag.
    df = df.drop_duplicates(["customer_tweet_id", "retriever"], keep=keep)
    df["is_deflection"] = df["is_deflection"].astype(str).str.lower().eq("true").astype(float)
    return df


def per_tweet(df: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    missing = [label for label in labels if label not in set(df["retriever"])]
    if missing:
        raise SystemExit(f"no rows for {missing} -- run with --list to see what exists")
    return df[df["retriever"].isin(labels)].groupby("customer_tweet_id")[METRICS].mean()


def compare(df: pd.DataFrame, a_labels: list[str], b_labels: list[str], seed: int = 0):
    rng = np.random.default_rng(seed)
    a, b = per_tweet(df, a_labels), per_tweet(df, b_labels)
    common = a.index.intersection(b.index)
    if len(common) < 2:
        raise SystemExit(f"only {len(common)} tweets scored under both sides -- nothing to pair")

    print(f"A = {', '.join(a_labels)}")
    print(f"B = {', '.join(b_labels)}")
    print(f"paired on {len(common)} tweets\n")
    print(f"{'metric':18s} {'A':>6s} {'B':>6s} {'B-A':>7s}   {'95% CI':>17s} {'p':>6s}")
    overall_deltas = None
    for m in METRICS:
        d = (b.loc[common, m] - a.loc[common, m]).to_numpy()
        boots = d[rng.integers(0, len(d), (N_RESAMPLES, len(d)))].mean(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        flipped = np.abs((d * rng.choice([-1, 1], (N_RESAMPLES, len(d)))).mean(axis=1))
        p = (flipped >= abs(d.mean()) - 1e-12).mean()
        flag = "  *" if lo > 0 or hi < 0 else ""
        print(f"{m:18s} {a.loc[common, m].mean():6.2f} {b.loc[common, m].mean():6.2f} "
              f"{d.mean():+7.2f}   [{lo:+.2f}, {hi:+.2f}] {p:6.2f}{flag}")
        if m == "overall":
            overall_deltas = d

    print("\n  * = the 95% interval excludes zero. With five metrics, expect one of")
    print("      those by chance about a fifth of the time -- read the pattern, not a star.")

    sd = overall_deltas.std(ddof=1)
    print(f"\nSpread of paired overall deltas: sd {sd:.2f}. Tweets needed at 80% power, a=0.05:")
    for effect in (0.1, 0.2, 0.3, 0.5):
        needed = int(np.ceil((2.8 * sd / effect) ** 2)) if sd > 0 else 0
        print(f"  to detect a {effect:.1f} shift: {needed}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--a", help="baseline retriever label(s), comma-separated to pool runs")
    ap.add_argument("--b", help="treatment retriever label(s), comma-separated to pool runs")
    ap.add_argument("--judge", default="qwen/qwen3.8-27b")
    ap.add_argument("--rubric", default="v2")
    ap.add_argument("--system", default="llm", help="which arm to compare (default: llm)")
    ap.add_argument("--list", action="store_true", help="list available retriever labels")
    ap.add_argument("--keep", choices=["first", "last"], default="last",
                    help="which verdict to keep for a tweet judged twice by a resumed run")
    args = ap.parse_args()

    df = load(args.judge, args.rubric, args.system, args.keep)
    if args.list or not (args.a and args.b):
        print(f"judge={args.judge} rubric={args.rubric} system={args.system}")
        print(df.groupby("retriever").size().rename("tweets").to_string())
        if not args.list:
            print("\npass --a and --b to compare")
        return
    compare(df, args.a.split(","), args.b.split(","))


if __name__ == "__main__":
    main()
