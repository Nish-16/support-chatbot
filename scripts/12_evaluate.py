"""
Eval harness: the LLM pipeline scored against the hand-labeled golden set.

No API calls -- golden_labels.csv already stores the classifier's
prediction (suggested_*) next to your label (human_*) for every row, so
every number here is computed from data already on disk. Rerunning this
after more labeling costs nothing.

What's measured:
  1. Intent accuracy, overall and per-intent (the headline number)
  2. Where the classifier actually confuses things (top confusion pairs)
  3. turn_type and flag agreement (the v2 fields, scored separately)
  4. End-to-end escalation accuracy: model intent -> escalation.py ->
     compared against your escalate call. This is the number that
     matters operationally, since a wrong escalate decision has a real
     cost in a way a wrong intent label alone doesn't.
  5. Confidence calibration -- does the model's own confidence predict
     whether it's right? This is what justifies (or kills) the
     low-confidence-escalates rule in escalation.py.
  6. Baselines (trivial + simple) on the same split, so the LLM's number
     means something relative to what's free.
  7. Stability caveats for the report's "what's misleading about my
     headline number" section.

The simple baseline is scored with cross_val_predict, not fit-then-
predict on the same rows -- a TF-IDF+LogisticRegression on ~190 examples
will memorize its training set and report a number that isn't real.

Which labels are scored
-----------------------
Defaults to `data/golden_labels_v3.csv` -- the re-adjudicated labels, which
are the reported ones (see WHAT_WENT_WRONG.md). That file carries both the
original v1 label (`human_intent`) and the v3 label (`human_intent_v3`), so
the v1 headline is reproducible too:

    scripts/12_evaluate.py                        # v3, scorable rows (headline)
    scripts/12_evaluate.py --label-col human_intent --keep-unscorable
                                                  # reproduces the old 62.4%
    scripts/12_evaluate.py --split holdout        # never-tuned-on rows only

v3 marks 14 rows `insufficient_context` -- on-topic tweets that cannot be
routed without the previous turn in the thread. They are excluded from the
headline rather than given a label nobody can defend; `--keep-unscorable`
puts them back in.

Usage: ./.venv/Scripts/python.exe scripts/12_evaluate.py
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from escalation import decide, ESCALATE
from groq_lib import FLAG_SPEC

LABELS_PATH = "data/golden_labels_v3.csv"
SPLIT_PATH = "data/golden_split.csv"
BOOL_FLAGS = [n for n, s in FLAG_SPEC.items() if s["type"] == "bool"]


def load(path: str, label_col: str, keep_unscorable: bool, split: str) -> pd.DataFrame:
    """Load the golden set and put the label being scored in `human_intent`.

    Every eval function below reads `human_intent`, so the label version is
    resolved once, here, rather than threaded through eight signatures.
    """
    df = pd.read_csv(path)
    if label_col not in df.columns:
        raise SystemExit(
            f"{path} has no column {label_col!r}. Available: {list(df.columns)}")

    n_all = len(df)
    dropped_status = 0
    if "v3_status" in df.columns and not keep_unscorable:
        dropped_status = int((df["v3_status"] != "resolved").sum())
        df = df[df["v3_status"] == "resolved"]

    if split != "all":
        splits = pd.read_csv(SPLIT_PATH)[["customer_tweet_id", "split"]]
        df = df.merge(splits, on="customer_tweet_id", how="inner")
        df = df[df["split"] == split]

    df = df.copy()
    df["human_intent"] = df[label_col]
    df = df[df["human_intent"].notna()]
    for col in ("human_flags", "suggested_flags", "human_secondary_intent", "suggested_secondary_intent"):
        df[col] = df[col].fillna("")

    print(f"Evaluating {len(df)} rows from {path}  (label column: {label_col}, split: {split})")
    if dropped_status:
        breakdown = pd.read_csv(path)["v3_status"].value_counts().drop("resolved", errors="ignore")
        print(f"  {dropped_status} of {n_all} rows excluded as unscorable "
              f"({', '.join(f'{n} {k}' for k, n in breakdown.items())}) "
              f"-- not routable from the tweet alone. Use --keep-unscorable to include them.")
    return df


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% CI on a proportion. Normal-approximation intervals are wrong at
    this n and near the edges; Wilson stays inside [0,1] and is honest about
    a 174-row sample."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _flagset(value: str) -> set[str]:
    return set(str(value).split(";")) - {""} if value else set()


def section(title: str):
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


def eval_intent(df: pd.DataFrame):
    section("1. INTENT ACCURACY (headline)")
    hits = int((df["human_intent"] == df["suggested_intent"]).sum())
    acc = hits / len(df)
    lo, hi = wilson(hits, len(df))
    print(f"\nOverall accuracy: {acc:.1%}  ({hits}/{len(df)})"
          f"   95% CI [{lo:.1%}, {hi:.1%}]\n")
    print(classification_report(df["human_intent"], df["suggested_intent"],
                                zero_division=0, digits=2))


def eval_confusions(df: pd.DataFrame):
    section("2. WHERE IT GOES WRONG (top confusions)")
    wrong = df[df["human_intent"] != df["suggested_intent"]]
    pairs = wrong.groupby(["human_intent", "suggested_intent"]).size().sort_values(ascending=False)
    print(f"\n{len(wrong)} misclassified rows. Most common confusions:\n")
    for (truth, pred), n in pairs.head(10).items():
        print(f"  {n:3d}x  you said {truth:28s} -> model said {pred}")

    # A confusion the model makes in both directions is a taxonomy
    # boundary problem (two intents that genuinely overlap), not a model
    # problem -- worth separating, since the fix is different.
    print("\n  Bidirectional (likely taxonomy-boundary, not model error):")
    seen = set()
    for (truth, pred), n in pairs.items():
        if (pred, truth) in pairs.index and (pred, truth) not in seen:
            seen.add((truth, pred))
            print(f"    {truth} <-> {pred}: {n} + {pairs[(pred, truth)]}")


def eval_secondary_and_turn(df: pd.DataFrame):
    section("3. SECONDARY INTENT, TURN TYPE, FLAGS")
    tt = (df["human_turn_type"] == df["suggested_turn_type"]).mean()
    print(f"\nturn_type accuracy: {tt:.1%}")
    print(f"  your distribution:  {df['human_turn_type'].value_counts().to_dict()}")
    print(f"  model distribution: {df['suggested_turn_type'].value_counts().to_dict()}")

    both_blank = ((df["human_secondary_intent"] == "") & (df["suggested_secondary_intent"] == "")).mean()
    exact = (df["human_secondary_intent"] == df["suggested_secondary_intent"]).mean()
    print(f"\nsecondary_intent exact agreement: {exact:.1%} "
          f"(of which {both_blank:.0%} is both-blank -- the easy case)")

    print("\nFlags (per-flag, on the rows where either side set it):")
    for flag in BOOL_FLAGS:
        h = df["human_flags"].map(lambda v: flag in _flagset(v))
        m = df["suggested_flags"].map(lambda v: flag in _flagset(v))
        if not (h | m).any():
            print(f"  {flag:22s} never set by either side")
            continue
        p, r, f1, _ = precision_recall_fscore_support(h, m, average="binary", zero_division=0)
        print(f"  {flag:22s} you={h.sum():3d}  model={m.sum():3d}  P={p:.2f} R={r:.2f} F1={f1:.2f}")


def eval_escalation(df: pd.DataFrame):
    section("4. END-TO-END ESCALATION (model intent -> policy -> decision)")
    preds, truths = [], []
    for _, row in df.iterrows():
        flags = _flagset(row["suggested_flags"])
        result = {
            "intent": row["suggested_intent"],
            "confidence": row["suggested_confidence"],
            "turn_type": row["suggested_turn_type"],
            "needs_human_triage": "needs_human_triage" in flags,
            "legal_sensitive": "legal_sensitive" in flags,
            "wants_human": "wants_human" in flags,
            "churn_threat": "churn_threat" in flags,
            "abusive_content": "abusive_content" in flags,
        }
        preds.append(decide(result).action == ESCALATE)
        truths.append(bool(row["human_escalate"]))

    preds, truths = np.array(preds), np.array(truths)
    print(f"\nAccuracy: {(preds == truths).mean():.1%}")
    print(f"You escalated {truths.sum()}/{len(truths)} ({truths.mean():.0%}); "
          f"pipeline escalated {preds.sum()}/{len(preds)} ({preds.mean():.0%})")

    # The two error types are NOT symmetric in cost and shouldn't be
    # reported as one accuracy number: auto-handling something that
    # needed a human is the expensive failure.
    missed = int((truths & ~preds).sum())
    over = int((~truths & preds).sum())
    print(f"\n  MISSED escalations (you=escalate, pipeline=auto-handle): {missed} "
          f"({missed / max(1, truths.sum()):.0%} of what you escalated)  <-- costly error")
    print(f"  Over-escalations  (you=auto-handle, pipeline=escalate): {over} "
          f"({over / max(1, (~truths).sum()):.0%} of what you'd auto-handle)  <-- cheap error")


def eval_calibration(df: pd.DataFrame):
    section("5. CONFIDENCE CALIBRATION")
    correct = df["human_intent"] == df["suggested_intent"]
    bins = [(0.0, 0.7), (0.7, 0.85), (0.85, 0.95), (0.95, 1.01)]
    print("\nIs the model's confidence predictive of being right?\n")
    print(f"  {'confidence':>14s}  {'n':>4s}  {'accuracy':>9s}")
    for lo, hi in bins:
        mask = (df["suggested_confidence"] >= lo) & (df["suggested_confidence"] < hi)
        if not mask.any():
            continue
        print(f"  {f'{lo:.2f}-{hi:.2f}':>14s}  {mask.sum():4d}  {correct[mask].mean():8.0%}")

    lo_mask = df["suggested_confidence"] < 0.7
    n_lo = int(lo_mask.sum())
    if n_lo:
        spread = correct[~lo_mask].mean() - correct[lo_mask].mean()
        print(f"\n  Gap between high- and low-confidence accuracy: {spread:+.0%} "
              f"(low-confidence n={n_lo})")
        # The gap is only meaningful if the low-confidence bucket is big
        # enough to mean anything. It usually isn't: this model reports
        # >=0.85 on almost everything, so a "confidence predicts accuracy"
        # claim can rest on a handful of rows. Guard against reporting
        # that as a finding.
        if n_lo < 20:
            print(f"    -> NOT a usable finding: {n_lo} rows below 0.70 is too few to "
                  f"support it either way.")
        elif spread > 0.1:
            print("    -> confidence carries real signal.")
        else:
            print("    -> confidence is NOT usefully predictive; the low-confidence-"
                  "escalates rule buys little.")

    bulk = df["suggested_confidence"] >= 0.85
    if bulk.mean() > 0.8:
        hi = correct[df["suggested_confidence"] >= 0.95].mean()
        mid = correct[(df["suggested_confidence"] >= 0.85) & (df["suggested_confidence"] < 0.95)].mean()
        print(f"\n  The real problem: {bulk.mean():.0%} of rows sit at confidence >=0.85, "
              f"where accuracy is flat ({mid:.0%} vs {hi:.0%}).")
        print("    The model is confidently wrong as often as confidently right, so")
        print("    thresholding on confidence cannot separate the two in practice.")


def eval_baselines(df: pd.DataFrame):
    section("6. BASELINES (same rows, same metric)")
    y = df["human_intent"]

    majority = y.value_counts().idxmax()
    trivial_acc = (y == majority).mean()

    vec = TfidfVectorizer(min_df=1, ngram_range=(1, 2), stop_words="english")
    X = vec.fit_transform(df["customer_text"])
    n_splits = min(5, y.value_counts().min())
    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        simple_pred = cross_val_predict(
            LogisticRegression(max_iter=1000, class_weight="balanced"), X, y, cv=cv)
        simple_acc = (simple_pred == y).mean()
        simple_note = f"{n_splits}-fold cross-validated"
    else:
        simple_acc, simple_note = float("nan"), "too few per-class examples"

    llm_acc = (y == df["suggested_intent"]).mean()

    print(f"\n  {'approach':38s} {'accuracy':>9s}   note")
    print(f"  {'Trivial (always majority class)':38s} {trivial_acc:8.1%}   always {majority}")
    print(f"  {'Simple (TF-IDF + LogisticReg)':38s} {simple_acc:8.1%}   {simple_note}")
    print(f"  {'LLM (Groq openai/gpt-oss-20b)':38s} {llm_acc:8.1%}   zero-shot, prompted")
    print(f"\n  LLM over trivial: {(llm_acc - trivial_acc) * 100:+.1f}pp"
          f"   |   LLM over simple: {(llm_acc - simple_acc) * 100:+.1f}pp")


def eval_stability(df: pd.DataFrame):
    section("7. WHAT'S MISLEADING ABOUT THE HEADLINE NUMBER")
    df = df.sort_values("labeled_at")
    correct = (df["human_intent"] == df["suggested_intent"]).values
    chunk = max(10, len(df) // 5)
    rates = []
    print("\nAccuracy by labeling window (same classifier, same taxonomy):")
    for start in range(0, len(df), chunk):
        window = correct[start:start + chunk]
        if len(window) < 5:
            continue
        rates.append(window.mean())
        print(f"  rows {start + 1:3d}-{start + len(window):3d}: {window.mean():.0%}")
    if rates:
        print(f"\n  Range across windows: {min(rates):.0%} - {max(rates):.0%} "
              f"({(max(rates) - min(rates)) * 100:.0f}pp spread) around a "
              f"{correct.mean():.0%} headline.")

    support = df["human_intent"].value_counts()
    thin = support[support < 10]
    if len(thin):
        print(f"\n  Underpowered intents (<10 golden examples) -- per-intent rates "
              f"for these are not trustworthy:")
        for name, n in thin.items():
            print(f"    {name}: n={n}")

    print("\n  Also known, not measured here:")
    print("    - LLM non-determinism: tweet 1274528 got different intents on two runs")
    print("      (sync_technical_bug 0.90 vs how_to_usage 0.85). Single-run accuracy")
    print("      overstates stability; temperature=0 is set now but was not for all rows.")
    print("    - The golden pool over-samples rare intents by construction, so this")
    print("      accuracy is NOT an estimate of accuracy on real traffic mix.")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default=LABELS_PATH, help="golden-label CSV to score against")
    ap.add_argument("--label-col", default="human_intent_v3",
                    choices=["human_intent_v3", "human_intent_v2", "human_intent"],
                    help="which label version is ground truth (default: v3, the reported one)")
    ap.add_argument("--keep-unscorable", action="store_true",
                    help="include rows marked insufficient_context")
    ap.add_argument("--split", default="all", choices=["all", "dev", "holdout"],
                    help="restrict to one side of the frozen split (default: all)")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    labels = load(args.labels, args.label_col, args.keep_unscorable, args.split)
    eval_intent(labels)
    eval_confusions(labels)
    eval_secondary_and_turn(labels)
    eval_escalation(labels)
    eval_calibration(labels)
    eval_baselines(labels)
    eval_stability(labels)
