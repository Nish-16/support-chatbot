"""
STEP 10 -- missed escalations. Can a signal other than self-reported
confidence send the risky cases the policy auto-handles to a human?

The policy misses 16% of the escalations a human would make (README §2), and
its only uncertainty branch is dead: nearly every row reports confidence
>= 0.85 whether it is right or wrong (§4.5). Two replacement signals are
tested here, on the FROZEN held-out split only:

  top-2      the classifier also names a runner-up intent
             (groq_lib.TOP2_SCHEMA_ADDENDUM). secondary_intent cannot do this
             job -- it is null on 64 of the 70 held-out rows.
  disagree   the unchanged production prompt, sampled N_SAMPLES times at
             SAMPLE_TEMPERATURE.

Policies. Every one keeps the current policy's own escalations (the
phishing/account_access defaults, the flag overrides) and only ADDS a trigger,
so the difference between two rows is that trigger's doing:

  A  current     escalation.decide() on the published baseline prediction
  B  top-2       decide() on the top-2 run, OR its intent/runner-up is risky
  C  disagree    A, OR the samples are not unanimous
  D  combined    A, OR any sample is risky, OR the samples are not unanimous

B runs through a changed prompt, so its own intents can differ from the
baseline's. "A on the top-2 run" is reported as the control that separates
the runner-up trigger from the prompt change.

Nothing is tuned. There is no threshold or weight to fit; the risky intents,
N_SAMPLES and SAMPLE_TEMPERATURE were fixed before the run. Labels never reach
a prompt -- only customer_text is sent.

A "risky case" is a row whose gold (v3) intent is in RISKY_INTENTS, whether or
not the human escalated it: auto-handling one is the expensive error. The
README's 16% is human_escalate recall over all 174 scorable rows; its held-out
equivalent is reported alongside.

Cost: 70 top-2 calls + N_SAMPLES x 70 samples = 280 calls at N_SAMPLES=3,
~140k tokens against gpt-oss-20b's 200k/day cap. Resumable: every run is
cached under its own namespace, so a rate-limited run picks up where it
stopped and a finished one is never re-paid for.

Usage:
  ./.venv/Scripts/python.exe scripts/26_escalation_uncertainty.py --n 2        # smoke test, 8 calls
  ./.venv/Scripts/python.exe scripts/26_escalation_uncertainty.py              # run + report
  ./.venv/Scripts/python.exe scripts/26_escalation_uncertainty.py --report-only
"""
import argparse
import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from cache import ClassificationCache
from classify_runner import classify_many
from escalation import ESCALATE, decide
from groq_lib import (FLAG_SPEC, TOP2_PROMPT_VERSION, cache_namespace,
                      classify_message, make_client)

V3_PATH = "data/golden_labels_v3.csv"
SPLIT_PATH = "data/golden_split.csv"
CACHE_PATH = "data/classification_cache.jsonl"

RISKY_INTENTS = {"security_account_compromise", "data_loss_recovery", "billing_subscription"}
N_SAMPLES = 3
SAMPLE_TEMPERATURE = 0.7
# Sample count in the name, so a later 5-sample run writes beside this one.
OUT_PATH = f"data/escalation_uncertainty_holdout_s{N_SAMPLES}.csv"
TARGET_MISS_RATE = 0.05
BOOL_FLAGS = [n for n, s in FLAG_SPEC.items() if s["type"] == "bool"]

POLICIES = [
    ("A  current", "esc_A", "baseline_intent"),
    ("B  top-2 risky", "esc_B", "top2_intent"),
    ("C  disagreement", "esc_C", "baseline_intent"),
    ("D  combined", "esc_D", "baseline_intent"),
]


def load() -> pd.DataFrame:
    gold = pd.read_csv(V3_PATH, dtype={"customer_tweet_id": str})
    split = pd.read_csv(SPLIT_PATH, dtype={"customer_tweet_id": str})
    gold = gold.merge(split[["customer_tweet_id", "split"]], on="customer_tweet_id")
    return gold[(gold.split == "holdout") & (gold.v3_status == "resolved")].reset_index(drop=True)


def namespaces() -> dict[str, str]:
    # prompt_version pinned, not defaulted -- see 23_escalation_v2.py for the
    # time a moved default silently returned zero cached rows.
    samples = f"{cache_namespace(prompt_version='p1')}:t{SAMPLE_TEMPERATURE}"
    ns = {"top2": cache_namespace(prompt_version=TOP2_PROMPT_VERSION)}
    for i in range(1, N_SAMPLES + 1):
        ns[f"run{i}"] = f"{samples}:s{i}"
    return ns


def run(rows: pd.DataFrame):
    client = make_client()
    items = [(r.customer_tweet_id, r.customer_text) for r in rows.itertuples()]
    for name, ns in namespaces().items():
        cache = ClassificationCache(CACHE_PATH, namespace=ns)
        if name == "top2":
            classify = lambda c, t: classify_message(c, t, top2=True, prompt_version="p1")
        else:
            classify = lambda c, t: classify_message(c, t, temperature=SAMPLE_TEMPERATURE,
                                                     prompt_version="p1")
        failed = []
        got = classify_many(client, items, cache, classify=classify,
                            on_result=lambda tid, _t, res, _c: res is None and failed.append(tid))
        print(f"  {name:5s} {ns}: {len(got)}/{len(items)} classified, {len(failed)} failed", flush=True)


def _flags(value) -> set[str]:
    if not isinstance(value, str):
        return set()
    return {f.strip() for f in re.split(r"[;,]", value) if f.strip()}


def baseline_result(row) -> dict:
    """The published baseline prediction, in the shape decide() reads --
    built the same way 12_evaluate.py builds it."""
    flags = _flags(row.suggested_flags)
    return {"intent": row.suggested_intent, "confidence": float(row.suggested_confidence),
            "turn_type": row.suggested_turn_type, **{f: f in flags for f in BOOL_FLAGS}}


def build(rows: pd.DataFrame) -> pd.DataFrame:
    caches = {k: ClassificationCache(CACHE_PATH, namespace=v) for k, v in namespaces().items()}
    out = []
    for r in rows.itertuples():
        tid, gold = r.customer_tweet_id, r.human_intent_v3
        base = baseline_result(r)
        t2 = caches["top2"].get(tid)
        runs = [(caches[f"run{i}"].get(tid) or {}).get("intent") for i in range(1, N_SAMPLES + 1)]
        got = [x for x in runs if x]
        votes = Counter(got)
        top_label, top_votes = votes.most_common(1)[0] if votes else (None, 0)
        majority = top_label if top_votes > N_SAMPLES // 2 else None
        complete = t2 is not None and len(got) == N_SAMPLES
        all_agree = len(got) == N_SAMPLES and len(votes) == 1
        any_risky = any(x in RISKY_INTENTS for x in got)

        esc_a = decide(base).action == ESCALATE
        esc_a_top2 = t2 is not None and decide(t2).action == ESCALATE
        runner_up = (t2 or {}).get("runner_up_intent")
        esc_b = esc_a_top2 or (t2 or {}).get("intent") in RISKY_INTENTS or runner_up in RISKY_INTENTS

        row = {
            "customer_tweet_id": tid,
            "gold_intent": gold,
            "risky_gold": gold in RISKY_INTENTS,
            "human_escalate": str(r.human_escalate).strip().lower() == "true",
            "baseline_intent": base["intent"],
            "baseline_confidence": base["confidence"],
            "baseline_correct": base["intent"] == gold,
            "top2_intent": (t2 or {}).get("intent"),
            "top2_runner_up": runner_up,
            "top2_correct": (t2 or {}).get("intent") == gold,
        }
        for i, x in enumerate(runs, 1):
            row[f"run{i}"] = x
        row.update({
            "n_unique": len(votes),
            "all_agree": all_agree,
            "any_run_risky": any_risky,
            "majority": majority,
            "majority_correct": majority == gold,
            "complete": complete,
            "esc_A": esc_a,
            "esc_A_on_top2_run": esc_a_top2,
            "esc_B": esc_b,
            "esc_B0_secondary": esc_a or r.suggested_secondary_intent in RISKY_INTENTS,
            "esc_C": esc_a or not all_agree,
            "esc_D": esc_a or not all_agree or any_risky,
        })
        out.append(row)
    return pd.DataFrame(out)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Same interval as 12_evaluate.py."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def score(df: pd.DataFrame, esc_col: str, intent_col: str) -> dict:
    esc = df[esc_col].astype(bool)
    human, risky = df.human_escalate, df.risky_gold
    correct = df[intent_col] == df.gold_intent
    caught = int((esc & risky).sum())
    auto = ~esc
    return {
        "escalated": int(esc.sum()),
        "esc_rate": esc.mean(),
        "risky": int(risky.sum()),
        "caught": caught,
        "missed": int((~esc & risky).sum()),
        "recall": caught / max(1, int(risky.sum())),
        "recall_ci": wilson(caught, int(risky.sum())),
        "fp": int((esc & ~human).sum()),
        "fp_rate": (esc & ~human).sum() / max(1, int((~human).sum())),
        "human_missed": int((~esc & human).sum()),
        "human_miss_rate": (~esc & human).sum() / max(1, int(human.sum())),
        "accuracy": correct.mean(),
        "auto_n": int(auto.sum()),
        "auto_accuracy": correct[auto].mean() if auto.any() else float("nan"),
    }


def auroc(error: pd.Series, signal: pd.Series, n_boot: int = 2000) -> str:
    """Threshold-free: how well does `signal` (higher = more suspicious) rank
    wrong predictions above right ones? 0.5 = no information."""
    y, s = error.to_numpy().astype(int), signal.to_numpy(float)
    if y.min() == y.max():
        return "n/a (one class)"
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() != y[idx].max():
            boots.append(roc_auc_score(y[idx], s[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return f"{roc_auc_score(y, s):.2f}  95% CI [{lo:.2f}, {hi:.2f}]"


def section(title: str):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def report(df: pd.DataFrame, write: bool):
    incomplete = df[~df.complete]
    if len(incomplete):
        print(f"\n  WARNING: {len(incomplete)} rows lack a top-2 result or a sample "
              f"({', '.join(incomplete.customer_tweet_id)}). Rerun to fill them in.")
        print("  Policies are scored on complete rows only, so every policy shares one n.")
    s = df[df.complete].reset_index(drop=True)
    if s.empty:
        return

    section(f"SETUP -- held-out split, {len(s)} scorable rows")
    print(f"\n  risky cases (gold intent in {sorted(RISKY_INTENTS)}): {int(s.risky_gold.sum())}")
    print(f"  human escalated:                  {int(s.human_escalate.sum())}")
    print(f"  samples: {N_SAMPLES} per tweet at temperature {SAMPLE_TEMPERATURE}; "
          f"runner-up missing/invalid on {int(s.top2_runner_up.isna().sum())} rows")

    section("POLICY COMPARISON")
    scores = {name: score(s, col, ic) for name, col, ic in POLICIES}
    print("\n| Policy | Escalation % | Risky cases caught | Risky cases missed | Risky recall | False-positive escalation % |")
    print("| ------ | -----------: | -----------------: | -----------------: | -----------: | --------------------------: |")
    for name, m in scores.items():
        lo, hi = m["recall_ci"]
        print(f"| {name} | {m['esc_rate']:.1%} ({m['escalated']}/{len(s)}) | {m['caught']}/{m['risky']} "
              f"| {m['missed']} | {m['recall']:.0%} [{lo:.0%}, {hi:.0%}] | {m['fp_rate']:.1%} ({m['fp']}) |")

    print(f"\n{'policy':20s} {'intent acc':>10s} {'auto-handled':>13s} {'acc on auto':>12s} "
          f"{'human-esc missed':>17s}")
    for name, m in scores.items():
        print(f"{name:20s} {m['accuracy']:10.1%} {m['auto_n']:13d} {m['auto_accuracy']:12.1%} "
              f"{m['human_missed']:6d} ({m['human_miss_rate']:.0%})")

    print("\n  controls:")
    for label, col, ic in [("A on the top-2 run (prompt change alone)", "esc_A_on_top2_run", "top2_intent"),
                           ("A + existing secondary_intent risky", "esc_B0_secondary", "baseline_intent")]:
        m = score(s, col, ic)
        print(f"  {label:42s} esc {m['esc_rate']:.1%}, risky missed {m['missed']}, "
              f"FP {m['fp_rate']:.1%}, intent acc {m['accuracy']:.1%}")

    section("DOES DISAGREEMENT PREDICT ERRORS?")
    agree, disagree = s[s.all_agree], s[~s.all_agree]
    print(f"\n  baseline (temp-0) intent accuracy -- the prediction auto-handling acts on:")
    print(f"    samples unanimous   {len(agree):3d} rows   {agree.baseline_correct.mean():.1%}")
    if len(disagree):
        print(f"    samples disagree    {len(disagree):3d} rows   {disagree.baseline_correct.mean():.1%}")
    print(f"  majority-vote accuracy: unanimous {agree.majority_correct.mean():.1%}"
          + (f", disagree {disagree.majority_correct.mean():.1%}" if len(disagree) else "")
          + f", all rows {s.majority_correct.mean():.1%}")
    same = sum((s[f"run{i}"] == s.baseline_intent).sum() for i in range(1, N_SAMPLES + 1))
    print(f"  samples matching the temp-0 baseline intent: {same}/{N_SAMPLES * len(s)}")

    print(f"\n  self-reported confidence (bucket edge 0.85, as documented in README 4.5 -- not fitted here):")
    print(f"    distribution: {s.baseline_confidence.value_counts().sort_index().to_dict()}")
    for label, mask in [(">= 0.85", s.baseline_confidence >= 0.85), ("<  0.85", s.baseline_confidence < 0.85)]:
        sub = s[mask]
        acc = f"{sub.baseline_correct.mean():.1%}" if len(sub) else "--"
        print(f"    conf {label}   {len(sub):3d} rows   {acc}")

    err = ~s.baseline_correct
    print(f"\n  AUROC for flagging a wrong baseline intent ({int(err.sum())} errors / {len(s)}):")
    print(f"    confidence (lower = suspicious)   {auroc(err, -s.baseline_confidence)}")
    print(f"    disagreement (unique labels)      {auroc(err, s.n_unique)}")

    section("RISKY CASES STILL MISSED")
    for name, col, _ in POLICIES:
        miss = s[s.risky_gold & ~s[col]]
        print(f"\n  {name}: {len(miss)}")
        for r in miss.itertuples():
            runs = ", ".join(str(getattr(r, f"run{i}")) for i in range(1, N_SAMPLES + 1))
            print(f"    {r.customer_tweet_id:>8s}  gold {r.gold_intent:28s} base {r.baseline_intent:26s} "
                  f"top2 {r.top2_intent}/{r.top2_runner_up}  runs [{runs}]")

    section("WHAT THIS SAMPLE CAN AND CANNOT SHOW")
    n_risky = int(s.risky_gold.sum())
    print(f"\n  {n_risky} risky cases: one miss = {1 / max(1, n_risky):.1%} of recall.")
    print(f"  The <= {TARGET_MISS_RATE:.0%} miss target means 0 misses at this n, and even 0/{n_risky} "
          f"only bounds the true miss rate below {wilson(0, n_risky)[1]:.0%} (Wilson 95%).")
    base = s.esc_A
    for name, col, _ in POLICIES[1:]:
        gained = int((s.risky_gold & s[col] & ~base).sum())
        lost = int((s.risky_gold & ~s[col] & base).sum())
        print(f"  {name} vs A, risky rows: {gained} newly caught, {lost} newly missed")

    if write and len(incomplete):
        print(f"\nNOT writing {OUT_PATH}: a partial run is not a result. Rerun to resume.")
    elif write:
        s.to_csv(OUT_PATH, index=False, encoding="utf-8")
        print(f"\nwrote {OUT_PATH}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, help="only the first N held-out rows (smoke test)")
    p.add_argument("--report-only", action="store_true")
    args = p.parse_args()

    rows = load()
    if args.n:
        rows = rows.head(args.n)
    print(f"HELD-OUT READ: {len(rows)} scorable rows of {SPLIT_PATH} (holdout). "
          f"Labels are used for scoring only.")
    if not args.report_only:
        run(rows)
    report(build(rows), write=not args.n)


if __name__ == "__main__":
    main()
