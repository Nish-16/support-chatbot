"""
STEP 11 -- escalation signals: urgent, wide_impact, repeated_failure.

26_escalation_uncertainty.py showed the held-out escalations the policy misses
are not classification errors: the intent was right every time, and the human
escalated on urgency ("Emergency"), reach ("my whole team") or a problem that
had already failed ("another bug and another ticket"). Uncertainty signals
cannot see that. This script asks the classifier for those three things
directly (groq_lib.ESCALATION_SIGNALS) and tests each as an extra trigger on
top of the current policy.

Context -- and why it is NOT brand_text
---------------------------------------
dropbox_paired.csv's brand_text is DropboxSupport's reply TO the tweet,
written after it. Giving it to the classifier would leak how support handled
the case. The message a tweet actually responds to is its
in_response_to_tweet_id parent in the raw dataset. --build-context extracts
those parents for the 189 golden tweets into data/golden_context.csv (51 have
one), so nothing downstream needs the 516MB raw file. No label is read there.

Protocol -- fixed before the dev run
------------------------------------
* One classifier call per tweet: the p1 prompt + ESCALATION_SIGNALS + parent
  context, cached under prompt version SIGNALS_PROMPT_VERSION.
* Every rule is scored on that SAME run, so the only difference between two
  rows is the trigger. Rule A on the published baseline predictions (the
  definition 12_evaluate.py reports) is shown for reference, unchanged.
* Selection: the fewest missed human escalations, among rules that add at most
  MAX_FALSE_ALARMS_PER_CATCH false alarms per miss removed versus A on the
  same run. Ties go to fewer false alarms. If no rule removes a miss, A stays.
  The ratio encodes "a missed escalation is the expensive error" and was set
  before any signal was seen.
* Added after the first dev run, reported beside the comparison and never
  used for selection: the same triggers applied to the published baseline
  intent and flags. The p1sig prompt itself moved topic accuracy, so "A on the
  same run" is a weaker classifier than the one in production.
* Dev split only, until a rule is selected. Then --split holdout --rule NAME
  scores that one rule against A, once. It refuses a rule the dev results did
  not select, and refuses to run again once its output exists.

Cost: one call per scorable row -- 104 dev, 70 holdout.

Usage:
  ./.venv/Scripts/python.exe scripts/27_escalation_signals.py --build-context
  ./.venv/Scripts/python.exe scripts/27_escalation_signals.py --n 3            # dev smoke test
  ./.venv/Scripts/python.exe scripts/27_escalation_signals.py                  # dev run + report
  ./.venv/Scripts/python.exe scripts/27_escalation_signals.py --report-only
  ./.venv/Scripts/python.exe scripts/27_escalation_signals.py --split holdout --rule <selected>
"""
import argparse
import os
import re

import pandas as pd

from cache import ClassificationCache
from classify_runner import classify_many
from escalation import ESCALATE, SIGNAL_REASONS, decide
from groq_lib import (ESCALATION_SIGNALS, FLAG_SPEC, SIGNALS_PROMPT_VERSION,
                      cache_namespace, classify_message, make_client)

V3_PATH = "data/golden_labels_v3.csv"
SPLIT_PATH = "data/golden_split.csv"
CACHE_PATH = "data/classification_cache.jsonl"
RAW_PATH = "data/twcs/twcs.csv"
CONTEXT_PATH = "data/golden_context.csv"
BRAND_AUTHOR = "DropboxSupport"

RISKY_INTENTS = {"security_account_compromise", "data_loss_recovery", "billing_subscription"}
MAX_FALSE_ALARMS_PER_CATCH = 2
BOOL_FLAGS = [n for n, s in FLAG_SPEC.items() if s["type"] == "bool"]
SIGNALS = list(ESCALATION_SIGNALS)
assert SIGNALS == list(SIGNAL_REASONS), "groq_lib and escalation.py disagree on the signal set"

RULES: dict[str, frozenset[str]] = {
    "A": frozenset(),
    **{f"A+{s}": frozenset({s}) for s in SIGNALS},
    "A+all": frozenset(SIGNALS),
}
CONTEXT_LABELS = {"brand": "Dropbox support", "same": "the same customer, earlier",
                  "other": "another user"}


def out_path(split: str, rule: str | None = None) -> str:
    if split == "dev":
        return "data/escalation_signals_dev.csv"
    return f"data/escalation_signals_holdout_{rule}.csv"


# -- context --------------------------------------------------------------

def build_context():
    if os.path.exists(CONTEXT_PATH):
        print(f"{CONTEXT_PATH} already exists -- not rebuilding.")
        return
    gold = pd.read_csv(V3_PATH, dtype={"customer_tweet_id": str})
    print(f"reading {RAW_PATH} (516MB, ~1 min)...", flush=True)
    raw = pd.read_csv(RAW_PATH, usecols=["tweet_id", "author_id", "text", "in_response_to_tweet_id"],
                      dtype={"tweet_id": str, "author_id": str}).set_index("tweet_id")
    rows = []
    for tid in gold.customer_tweet_id:
        if tid not in raw.index or pd.isna(raw.at[tid, "in_response_to_tweet_id"]):
            continue
        pid = str(int(raw.at[tid, "in_response_to_tweet_id"]))
        if pid not in raw.index:
            continue
        author = raw.at[pid, "author_id"]
        who = "brand" if author == BRAND_AUTHOR else "same" if author == raw.at[tid, "author_id"] else "other"
        rows.append({"customer_tweet_id": tid, "parent_tweet_id": pid,
                     "parent_author": who, "parent_text": raw.at[pid, "text"]})
    out = pd.DataFrame(rows)
    out.to_csv(CONTEXT_PATH, index=False, encoding="utf-8")
    print(f"wrote {CONTEXT_PATH}: {len(out)} of {len(gold)} golden tweets have a parent "
          f"({out.parent_author.value_counts().to_dict()})")


def load(split: str) -> pd.DataFrame:
    if not os.path.exists(CONTEXT_PATH):
        raise SystemExit(f"{CONTEXT_PATH} missing -- run with --build-context first")
    gold = pd.read_csv(V3_PATH, dtype={"customer_tweet_id": str})
    splits = pd.read_csv(SPLIT_PATH, dtype={"customer_tweet_id": str})
    gold = gold.merge(splits[["customer_tweet_id", "split"]], on="customer_tweet_id")
    rows = gold[(gold.split == split) & (gold.v3_status == "resolved")]
    ctx = pd.read_csv(CONTEXT_PATH, dtype={"customer_tweet_id": str, "parent_tweet_id": str})
    rows = rows.merge(ctx, on="customer_tweet_id", how="left")
    rows["context"] = [
        f"{CONTEXT_LABELS[a]}: {t}" if isinstance(t, str) else None
        for a, t in zip(rows.parent_author, rows.parent_text)
    ]
    return rows.reset_index(drop=True)


# -- run ------------------------------------------------------------------

def namespace() -> str:
    return cache_namespace(prompt_version=SIGNALS_PROMPT_VERSION)


def run(rows: pd.DataFrame):
    client = make_client()
    cache = ClassificationCache(CACHE_PATH, namespace=namespace())
    # classify_many hands the worker text only; look context up by text.
    context_of = dict(zip(rows.customer_text, rows.context))
    items = [(r.customer_tweet_id, r.customer_text) for r in rows.itertuples()]
    failed = []
    got = classify_many(
        client, items, cache,
        classify=lambda c, t: classify_message(c, t, signals=True, context=context_of.get(t),
                                               prompt_version="p1"),
        on_result=lambda tid, _t, res, _c: res is None and failed.append(tid),
    )
    print(f"  {namespace()}: {len(got)}/{len(items)} classified, {len(failed)} failed", flush=True)


# -- score ----------------------------------------------------------------

def _flags(value) -> set[str]:
    if not isinstance(value, str):
        return set()
    return {f.strip() for f in re.split(r"[;,]", value) if f.strip()}


def baseline_result(row) -> dict:
    """The published baseline prediction, built the way 12_evaluate.py builds it."""
    flags = _flags(row.suggested_flags)
    return {"intent": row.suggested_intent, "confidence": float(row.suggested_confidence),
            "turn_type": row.suggested_turn_type, **{f: f in flags for f in BOOL_FLAGS}}


def build(rows: pd.DataFrame) -> pd.DataFrame:
    cache = ClassificationCache(CACHE_PATH, namespace=namespace())
    out = []
    for r in rows.itertuples():
        res = cache.get(r.customer_tweet_id)
        row = {
            "customer_tweet_id": r.customer_tweet_id,
            "gold_intent": r.human_intent_v3,
            "human_escalate": str(r.human_escalate).strip().lower() == "true",
            "risky_gold": r.human_intent_v3 in RISKY_INTENTS,
            "has_context": isinstance(r.context, str),
            "baseline_intent": r.suggested_intent,
            "esc_A_baseline": decide(baseline_result(r)).action == ESCALATE,
            "complete": res is not None,
        }
        if res is not None:
            row["intent"] = res["intent"]
            row["signals_defaulted"] = ",".join(res.get("signals_defaulted", []))
            row.update({s: bool(res.get(s)) for s in SIGNALS})
            row.update({f"esc_{name}": decide(res, triggers).action == ESCALATE
                        for name, triggers in RULES.items()})
            # The diagnostic from the docstring: this run's signals on top of
            # the published baseline intent and flags.
            hybrid = {**baseline_result(r), **{s: bool(res.get(s)) for s in SIGNALS}}
            row.update({f"esc_base_{name}": decide(hybrid, triggers).action == ESCALATE
                        for name, triggers in RULES.items()})
        out.append(row)
    return pd.DataFrame(out)


def score(df: pd.DataFrame, esc_col: str, intent_col: str) -> dict:
    esc, human = df[esc_col].astype(bool), df.human_escalate
    return {
        "accuracy": (df[intent_col] == df.gold_intent).mean(),
        "escalated": int(esc.sum()),
        "esc_rate": esc.mean(),
        "missed": int((~esc & human).sum()),
        "miss_rate": (~esc & human).sum() / max(1, int(human.sum())),
        "false_alarms": int((esc & ~human).sum()),
        "fa_rate": (esc & ~human).sum() / max(1, int((~human).sum())),
        "risky_missed": int((~esc & df.risky_gold).sum()),
    }


def select(scores: dict[str, dict]) -> str:
    base = scores["A"]
    eligible = []
    for name, m in scores.items():
        removed = base["missed"] - m["missed"]
        added = m["false_alarms"] - base["false_alarms"]
        if name != "A" and removed > 0 and added <= MAX_FALSE_ALARMS_PER_CATCH * removed:
            eligible.append((m["missed"], m["false_alarms"], m["escalated"], name))
    return min(eligible)[3] if eligible else "A"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Same interval as 12_evaluate.py."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def section(title: str):
    print(f"\n{'=' * 76}\n{title}\n{'=' * 76}")


def report(df: pd.DataFrame, split: str, rules: list[str]) -> str | None:
    incomplete = df[~df.complete]
    if len(incomplete):
        print(f"\n  WARNING: {len(incomplete)} rows not classified yet "
              f"({', '.join(incomplete.customer_tweet_id)}). Rerun to resume.")
    s = df[df.complete].reset_index(drop=True)
    if s.empty:
        return None
    # An incomplete run leaves NaN in these columns, which makes them object
    # dtype -- where ~ is integer inversion, not logical not.
    for col in [c for c in s.columns if c.startswith("esc_")] + SIGNALS:
        s[col] = s[col].astype(bool)
    human = s.human_escalate

    section(f"SETUP -- {split} split, {len(s)} scorable rows")
    print(f"\n  human escalated: {int(human.sum())}   not escalated: {int((~human).sum())}")
    print(f"  rows with an earlier message as context: {int(s.has_context.sum())}")
    changed = int((s.intent != s.baseline_intent).sum())
    print(f"  intent differs from the published baseline on {changed} rows "
          f"(baseline acc {(s.baseline_intent == s.gold_intent).mean():.1%}, "
          f"this run {(s.intent == s.gold_intent).mean():.1%})")
    moved = s[s.intent != s.baseline_intent]
    print(f"    of those, fixed {int((moved.intent == moved.gold_intent).sum())}, "
          f"broken {int((moved.baseline_intent == moved.gold_intent).sum())}")
    for label, mask in [("with context", s.has_context), ("without context", ~s.has_context)]:
        sub = s[mask]
        if len(sub):
            print(f"    {label:16s} {len(sub):3d} rows   baseline acc "
                  f"{(sub.baseline_intent == sub.gold_intent).mean():.1%}   "
                  f"this run {(sub.intent == sub.gold_intent).mean():.1%}")
    defaulted = int((s.signals_defaulted.fillna("") != "").sum())
    if defaulted:
        print(f"  rows where the model omitted a signal (recorded as false): {defaulted}")

    print(f"\n  {'signal':18s} {'fires':>6s} {'human-escalated':>16s} {'on rows A misses':>17s}")
    missed_by_a = ~s.esc_A & human
    for sig in SIGNALS:
        fires = s[sig]
        print(f"  {sig:18s} {int(fires.sum()):6d} {int((fires & human).sum()):9d} of {int(fires.sum()):<4d}"
              f" {int((fires & missed_by_a).sum()):9d} of {int(missed_by_a.sum())}")

    section("RULE COMPARISON (all rules scored on the same classifier run)")
    scores = {name: score(s, f"esc_{name}", "intent") for name in rules}
    base = scores["A"]
    print("\n| Rule | Topic acc | Escalated | Missed human esc. | False alarms | "
          "Δ escalated | Δ missed | Δ false alarms |")
    print("| ---- | --------: | --------: | ----------------: | -----------: | "
          "----------: | -------: | -------------: |")
    for name, m in scores.items():
        print(f"| {name} | {m['accuracy']:.1%} | {m['escalated']} ({m['esc_rate']:.1%}) "
              f"| {m['missed']} ({m['miss_rate']:.1%}) | {m['false_alarms']} ({m['fa_rate']:.1%}) "
              f"| {m['escalated'] - base['escalated']:+d} | {m['missed'] - base['missed']:+d} "
              f"| {m['false_alarms'] - base['false_alarms']:+d} |")
    ref = score(s, "esc_A_baseline", "baseline_intent")
    print(f"\n  reference -- A on the published baseline predictions (12_evaluate.py's definition):")
    print(f"  topic acc {ref['accuracy']:.1%}, escalated {ref['escalated']} ({ref['esc_rate']:.1%}), "
          f"missed {ref['missed']} ({ref['miss_rate']:.1%}), false alarms {ref['false_alarms']} "
          f"({ref['fa_rate']:.1%})")
    print(f"\n  diagnostic -- the same triggers on the published baseline intent and flags:")
    print(f"  {'rule':20s} {'escalated':>10s} {'missed':>7s} {'false alarms':>13s} {'risky missed':>13s}")
    for name in rules:
        m = score(s, f"esc_base_{name}", "baseline_intent")
        print(f"  {name:20s} {m['escalated']:10d} {m['missed']:7d} {m['false_alarms']:13d} "
              f"{m['risky_missed']:13d}")
    print(f"\n  risky-intent cases missed: "
          + ", ".join(f"{n} {m['risky_missed']}" for n, m in scores.items())
          + f"  (of {int(s.risky_gold.sum())})")

    section("WHAT EACH TRIGGER CHANGED versus A")
    for name in rules[1:]:
        col = f"esc_{name}"
        caught = s[s[col] & ~s.esc_A & human].customer_tweet_id.tolist()
        alarms = s[s[col] & ~s.esc_A & ~human].customer_tweet_id.tolist()
        print(f"\n  {name}: newly caught {len(caught)} {caught}")
        print(f"  {' ' * len(name)}  new false alarms {len(alarms)} {alarms}")
    still = s[~s.esc_A & human]
    print(f"\n  missed by A: {len(still)}")
    for r in still.itertuples():
        fired = [sig for sig in SIGNALS if getattr(r, sig)]
        print(f"    {r.customer_tweet_id:>8s}  gold {r.gold_intent:26s} model {r.intent:26s} "
              f"signals {fired or '-'}")

    section("LIMITS")
    n_h = int(human.sum())
    lo, hi = wilson(base["missed"], n_h)
    print(f"\n  {n_h} human escalations: one miss = {1 / max(1, n_h):.1%} of the miss rate.")
    print(f"  A's miss rate {base['miss_rate']:.1%} has a 95% interval of [{lo:.1%}, {hi:.1%}] on its own.")

    if split == "dev" and len(rules) > 1:
        chosen = select(scores)
        section("SELECTION (rule fixed before the run)")
        print(f"\n  fewest missed, adding <= {MAX_FALSE_ALARMS_PER_CATCH} false alarms per miss removed")
        print(f"  -> selected: {chosen}")
        return chosen if not len(incomplete) else None
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--build-context", action="store_true")
    p.add_argument("--split", choices=["dev", "holdout"], default="dev")
    p.add_argument("--rule", help="holdout only: the rule the dev run selected")
    p.add_argument("--n", type=int, help="only the first N rows (smoke test, writes nothing)")
    p.add_argument("--report-only", action="store_true")
    args = p.parse_args()

    if args.build_context:
        build_context()
        return

    rules = list(RULES)
    if args.split == "holdout":
        if not os.path.exists(out_path("dev")):
            raise SystemExit("run the dev experiment first -- the holdout rule comes from it")
        dev = pd.read_csv(out_path("dev"))
        dev_choice = select({name: score(dev, f"esc_{name}", "intent") for name in RULES})
        if args.rule != dev_choice:
            raise SystemExit(f"--rule must be the rule dev selected ({dev_choice}), got {args.rule!r}")
        if os.path.exists(out_path("holdout", args.rule)) and not args.report_only:
            raise SystemExit(f"{out_path('holdout', args.rule)} exists -- the holdout read for "
                             f"this rule has been done. Use --report-only to reprint it.")
        rules = ["A"] if args.rule == "A" else ["A", args.rule]

    rows = load(args.split)
    if args.n:
        rows = rows.head(args.n)
    print(f"{args.split.upper()} READ: {len(rows)} scorable rows. Labels are used for scoring only.")
    if not args.report_only:
        run(rows)
    df = build(rows)
    report(df, args.split, rules)

    if args.n or not df.complete.all() or args.report_only:
        if not df.complete.all():
            print("\nnot writing results: the run is incomplete.")
        return
    rule_of = lambda col: col.removeprefix("esc_").removeprefix("base_")
    keep = [c for c in df.columns if not c.startswith("esc_") or rule_of(c) in rules + ["A_baseline"]]
    path = out_path(args.split, args.rule)
    df[keep].to_csv(path, index=False, encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
