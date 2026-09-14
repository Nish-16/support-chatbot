"""
STEP 13 -- does p4 (p3 + prompt_p4.P4_BLOCK) discriminate neighbouring intents
better than p1 and p3?

Same frozen set, same labels, same model and decoding as 28_p3_eval.py; only
the prompt differs. Nothing here writes to a file 28_p3_eval.py owns: p4's
predictions go to the classification cache under their own namespace, its run
log and results to data/p4_eval_*.

READ THE RESULT AS IN-SAMPLE. P4_BLOCK was written after reading p1's and p3's
errors on these 250 tweets (see prompt_p4.py). The p4 numbers are therefore
optimistic, and the p4-vs-p3 comparison is not a pre-registered test. A p4 gain
here earns a fresh evaluation; it does not earn a production switch.
PROMPT_VERSION stays p1.

Usage:
  ./.venv/Scripts/python.exe scripts/29_p4_eval.py --run
  ./.venv/Scripts/python.exe scripts/29_p4_eval.py --report
"""
import argparse
import hashlib
import importlib.util
import json
import os
from datetime import datetime, timezone

import pandas as pd

from cache import ClassificationCache
from classify_runner import classify_many
from escalation import ESCALATE, decide
from groq_lib import (INTENT_NAMES, MODEL, PROMPT_VERSION, build_system_prompt, cache_namespace,
                      classify_message, make_client)

_spec = importlib.util.spec_from_file_location(
    "p3_eval", os.path.join(os.path.dirname(os.path.abspath(__file__)), "28_p3_eval.py"))
p3_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p3_eval)

VERSIONS = ("p1", "p3", "p4")
RUNS_PATH = "data/p4_eval_runs.jsonl"
RESULTS_PATH = "data/p4_eval_results.csv"


def _checked_manifest() -> dict:
    manifest = p3_eval.load_manifest()
    if not p3_eval.verify(manifest):
        raise SystemExit("integrity checks failed")
    if PROMPT_VERSION != "p1":
        raise SystemExit(f"production PROMPT_VERSION is {PROMPT_VERSION!r}, not 'p1'")
    return manifest


def run():
    manifest = _checked_manifest()
    eval_set = p3_eval.scored_set()
    items = [(r.customer_tweet_id, r.customer_text) for r in eval_set.itertuples()]
    cache = ClassificationCache(p3_eval.CACHE_PATH, namespace=cache_namespace(prompt_version="p4"))
    failed: list[str] = []
    got = classify_many(
        make_client(), items, cache,
        classify=lambda c, t: classify_message(c, t, prompt_version="p4"),
        on_result=lambda tid, _t, res, _c: res is None and failed.append(str(tid)),
    )
    print(f"p4: {len(got)} ok / {len(failed)} failed of {len(items)}")
    with open(RUNS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "at": datetime.now(timezone.utc).isoformat(), "requested": len(items), "model": MODEL,
            "namespace": cache_namespace(prompt_version="p4"),
            "prompt_sha256": hashlib.sha256(build_system_prompt("p4").encode("utf-8")).hexdigest(),
            "set_sha256": manifest["set_sha256"], "classified": len(got), "failed": failed,
        }) + "\n")
    print(f"logged to {RUNS_PATH}")


def assemble(allow_missing: bool) -> pd.DataFrame:
    eval_set = p3_eval.scored_set()
    labels = p3_eval.load_labels()
    labels = labels[labels.customer_tweet_id.isin(eval_set.customer_tweet_id)]
    labels = labels.drop_duplicates("customer_tweet_id", keep="last")
    if len(labels) != len(eval_set):
        raise SystemExit(f"{len(eval_set) - len(labels)} tweets unlabelled")
    df = eval_set.merge(labels, on=["eval_index", "customer_tweet_id"], how="left")
    df["gold"] = df.human_intent
    df["human_escalate"] = df.human_escalate.astype(str).str.lower() == "true"
    df["insufficient_context"] = df.insufficient_context.astype(str).str.lower() == "true"
    for version in VERSIONS:
        cache = ClassificationCache(p3_eval.CACHE_PATH, namespace=cache_namespace(prompt_version=version))
        results = [cache.get(tid) for tid in df.customer_tweet_id]
        missing = sum(r is None for r in results)
        if missing and not allow_missing:
            raise SystemExit(f"{version}: {missing} predictions missing -- run --run, or pass --count-missing-as-wrong")
        df[f"{version}_intent"] = [r["intent"] if r else "__missing__" for r in results]
        df[f"{version}_correct"] = df[f"{version}_intent"] == df.gold
        df[f"{version}_escalate"] = [decide(r).action == ESCALATE if r else True for r in results]
    return df


def per_intent(scored: pd.DataFrame, version: str) -> pd.DataFrame:
    rows = []
    for intent in INTENT_NAMES:
        gold = scored.gold == intent
        pred = scored[f"{version}_intent"] == intent
        tp = int((gold & pred).sum())
        rows.append({"intent": intent, "n": int(gold.sum()), "predicted": int(pred.sum()), "tp": tp,
                     "precision": tp / pred.sum() if pred.sum() else float("nan"),
                     "recall": tp / gold.sum() if gold.sum() else float("nan")})
    return pd.DataFrame(rows)


def _pct(x: float) -> str:
    return "   --" if pd.isna(x) else f"{x:5.0%}"


def report(allow_missing: bool):
    _checked_manifest()
    df = assemble(allow_missing)
    scored = df[~df.insufficient_context].reset_index(drop=True)
    n = len(scored)
    print(f"\n{'=' * 78}\np1 vs p3 vs p4 -- {n} scored of {len(df)} frozen tweets"
          f"\nIN-SAMPLE: p4's rules were written from p1/p3 errors on this same set.\n{'=' * 78}")

    print("\nOVERALL ACCURACY")
    for version in VERSIONS:
        k = int(scored[f"{version}_correct"].sum())
        lo, hi = p3_eval.wilson(k, n)
        print(f"  {version}  {k / n:.1%}  ({k}/{n})  95% CI [{lo:.1%}, {hi:.1%}]")

    tables = {v: per_intent(scored, v) for v in VERSIONS}
    print("\nPER-INTENT PRECISION / RECALL  (tp/predicted, tp/n)")
    print(f"  {'intent':28s} {'n':>3s}" + "".join(f"   {v} P    {v} R   " for v in VERSIONS))
    for i, intent in enumerate(INTENT_NAMES):
        line = f"  {intent:28s} {int(tables['p1'].n[i]):3d}"
        for v in VERSIONS:
            t = tables[v].iloc[i]
            line += f"  {int(t.tp):2d}/{int(t.predicted):<3d}{_pct(t.precision)} {_pct(t.recall)}"
        print(line)
    for label, floor in (("macro (all intents with n>0)", 1), ("macro (intents with n>=5)", 5)):
        line = f"  {label:32s}"
        for v in VERSIONS:
            t = tables[v][tables[v].n >= floor]
            line += f"  {v} P {t.precision.fillna(0).mean():5.1%}  R {t.recall.mean():5.1%}"
        print(line)

    print("\nBOUNDARY PAIRS (rows whose human label is on either side; cross = predicted the other side)")
    for key, (side_a, side_b) in p3_eval.BOUNDARY_SIDES.items():
        sub = scored[scored.gold.isin(side_a | side_b)]
        line = f"  {key}. {p3_eval.taxonomy_rules.BOUNDARIES[key][0]:40s} n={len(sub):3d}"
        for v in VERSIONS:
            pred = sub[f"{v}_intent"]
            cross = (sub.gold.isin(side_a) & pred.isin(side_b)) | (sub.gold.isin(side_b) & pred.isin(side_a))
            line += f"  {v} acc {(pred == sub.gold).mean():5.1%} cross {int(cross.sum()):2d}"
        print(line)

    print("\nEND-TO-END ESCALATION (escalation.decide on each prompt's output vs the human decision)")
    for v in VERSIONS:
        m = p3_eval.escalation_metrics(scored[f"{v}_escalate"], scored.human_escalate)
        print(f"  {v}  escalated {m['escalated']:3d}  precision {m['precision']:.1%}  recall {m['recall']:.1%}"
              f"  missed {m['missed']}  over-escalated {m['over']}")

    print("\nPAIRED COMPARISONS (difference = second minus first; bootstrap 95% CI; exact McNemar)")
    for a, b in (("p1", "p3"), ("p3", "p4"), ("p1", "p4")):
        s = p3_eval.paired_difference(scored[f"{a}_correct"], scored[f"{b}_correct"])
        lo, hi = s["ci"]
        print(f"  {b} - {a}: {s['diff']:+.1%}  CI [{lo:+.1%}, {hi:+.1%}]  {b} fixes {s['fixes']} {a} errors, "
              f"introduces {s['breaks']}; McNemar p = {s['mcnemar_p']:.3f}")

    texts = scored.customer_text.map(lambda t: " ".join(str(t).split())[:140])
    for title, mask in (("P4 FIXES A P3 ERROR", ~scored.p3_correct & scored.p4_correct),
                        ("P4 INTRODUCES A NEW ERROR (p3 was right)", scored.p3_correct & ~scored.p4_correct),
                        ("P4 BREAKS A P1-CORRECT ROW", scored.p1_correct & ~scored.p4_correct)):
        sub = scored[mask]
        print(f"\n{title}: {len(sub)}")
        for r in sub.itertuples():
            print(f"  #{r.eval_index:<3d} human {r.gold:26s} p1 {r.p1_intent:26s} p3 {r.p3_intent:26s} p4 {r.p4_intent}")
            print(f"        {texts[r.Index]}")

    keep = ["eval_index", "customer_tweet_id", "gold", "insufficient_context", "human_escalate"] + \
           [f"{v}_{c}" for v in VERSIONS for c in ("intent", "correct", "escalate")]
    df[keep].to_csv(RESULTS_PATH, index=False, encoding="utf-8")
    print(f"\nwrote {RESULTS_PATH}. Production is not changed by this script. PROMPT_VERSION stays p1.")


def main():
    p = argparse.ArgumentParser()
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--report", action="store_true")
    p.add_argument("--count-missing-as-wrong", action="store_true")
    args = p.parse_args()
    if args.run:
        run()
    else:
        report(args.count_missing_as_wrong)


if __name__ == "__main__":
    main()
