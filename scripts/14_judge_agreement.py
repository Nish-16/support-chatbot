"""
Judge-vs-human agreement: the evidence that makes 13_reply_eval.py's
numbers reportable, required explicitly by the assignment.

An LLM judge is an unvalidated instrument until someone checks it
against human judgment -- doubly so here, where the judge is the same
model (openai/gpt-oss-20b) that wrote the replies it's grading. This
tool collects YOUR ratings on a sample and measures how well the judge
tracks them.

Same rule as the golden-set labeling: the human ratings have to be
yours. A model rating the output of a model, checked by a model, is not
agreement evidence -- it's the same opinion counted three times.

Anti-bias design, mirroring 08_label_golden_set.py:
  - You are NOT shown which system wrote a reply. Knowing "this is the
    LLM agent" would pull your score toward the sophisticated-looking
    answer, which is precisely the bias being measured.
  - You are NOT shown the judge's score before you enter yours. Seeing
    it first would manufacture the agreement this script exists to test.
  - Rows are shuffled, so systems don't arrive in predictable blocks.

Resumable: appends to data/judge_agreement.csv after every rating.

Usage: ./.venv/Scripts/python.exe scripts/14_judge_agreement.py [--n 15]
       ./.venv/Scripts/python.exe scripts/14_judge_agreement.py --report
"""
import argparse
import csv
import os

import pandas as pd

EVALS_PATH = "data/reply_evals.csv"
OUT_PATH = "data/judge_agreement.csv"
# The judge and rubric whose numbers are reported: an independent model
# family from the drafter, under the stricter rubric.
JUDGE = "qwen/qwen3.8-27b"
RUBRIC = "v2"
FIELDS = ["customer_tweet_id", "system", "human_overall", "human_is_deflection", "judge_overall", "judge_is_deflection"]


def ask_score() -> int | str:
    while True:
        raw = input("Your overall score 1-5 ('q' to stop): ").strip().lower()
        if raw == "q":
            return "QUIT"
        if raw.isdigit() and 1 <= int(raw) <= 5:
            return int(raw)
        print("  enter 1-5, or 'q'")


def ask_deflection() -> bool | str:
    while True:
        raw = input("Is it a bare deflection (redirects without answering/specifying)? (y/n): ").strip().lower()
        if raw == "q":
            return "QUIT"
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("  enter y or n")


def main(n: int):
    if not os.path.exists(EVALS_PATH):
        print(f"{EVALS_PATH} not found -- run scripts/13_reply_eval.py first.")
        return
    evals = pd.read_csv(EVALS_PATH)

    # reply_evals.csv holds several judge x rubric runs over the SAME replies.
    # Agreement has to be measured against one of them -- the reported one --
    # or a single human rating gets compared to whichever judge row the sample
    # happened to land on, and the resulting number means nothing.
    before = len(evals)
    evals = evals[(evals["judge_model"] == JUDGE) & (evals["rubric_version"] == RUBRIC)]
    if evals.empty:
        combos = pd.read_csv(EVALS_PATH).groupby(["judge_model", "rubric_version"]).size()
        print(f"No rows for judge={JUDGE} rubric={RUBRIC}. Available:\n{combos.to_string()}")
        return
    print(f"Scoped to judge {JUDGE}, rubric {RUBRIC}: {len(evals)} of {before} rows.")

    done = set()
    if os.path.exists(OUT_PATH):
        prev = pd.read_csv(OUT_PATH)
        done = set(zip(prev["customer_tweet_id"], prev["system"]))

    pool = evals[~evals.apply(lambda r: (r["customer_tweet_id"], r["system"]) in done, axis=1)]
    sample = pool.sample(min(n, len(pool)), random_state=7)

    print(f"Rating {len(sample)} replies. You will NOT see which system wrote them,")
    print(f"or the judge's score, until after you answer. {len(done)} already rated.\n")
    print("Scale: 1 = would never send this, 3 = acceptable, 5 = exactly right.")

    write_header = not os.path.exists(OUT_PATH)
    with open(OUT_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()

        for i, row in enumerate(sample.itertuples(), 1):
            print(f"\n--- [{i}/{len(sample)}] intent: {row.human_intent} ---")
            print(f"CUSTOMER: {row.customer_text}")
            print(f"\nREPLY:    {row.reply}\n")

            score = ask_score()
            if score == "QUIT":
                print("Stopping. Progress saved.")
                return
            defl = ask_deflection()
            if defl == "QUIT":
                print("Stopping. Progress saved.")
                return

            print(f"  -> judge said overall={row.overall}, deflection={row.is_deflection} "
                  f"(system: {row.system})")

            writer.writerow({
                "customer_tweet_id": row.customer_tweet_id,
                "system": row.system,
                "human_overall": score,
                "human_is_deflection": defl,
                "judge_overall": row.overall,
                "judge_is_deflection": row.is_deflection,
            })
            f.flush()

    report()


def report():
    if not os.path.exists(OUT_PATH):
        print(f"No ratings yet -- run without --report first.")
        return
    df = pd.read_csv(OUT_PATH)
    if len(df) < 3:
        print(f"Only {len(df)} ratings so far; need more before agreement means anything.")
        return

    print(f"\n{'=' * 60}\nJUDGE-vs-HUMAN AGREEMENT (n={len(df)})\n{'=' * 60}\n")

    exact = (df["human_overall"] == df["judge_overall"]).mean()
    within1 = ((df["human_overall"] - df["judge_overall"]).abs() <= 1).mean()
    bias = (df["judge_overall"] - df["human_overall"]).mean()
    print(f"  Exact score agreement:      {exact:.0%}")
    print(f"  Within 1 point:             {within1:.0%}")
    print(f"  Judge bias vs you:          {bias:+.2f} points "
          f"({'judge is more generous' if bias > 0 else 'judge is harsher'})")

    if df["human_overall"].nunique() > 1 and df["judge_overall"].nunique() > 1:
        rho = df["human_overall"].corr(df["judge_overall"], method="spearman")
        print(f"  Spearman correlation:       {rho:+.2f}")
        verdict = ("judge tracks human ranking well" if rho >= 0.6 else
                   "weak -- judge ranking is only loosely related to yours" if rho >= 0.3 else
                   "judge does NOT track human judgment; its scores are not usable as a proxy")
        print(f"    -> {verdict}")
    else:
        print("  Spearman: undefined (no variance on one side)")

    # Binarized "would I send this?" is the decision the score stands in
    # for, so agreement on it matters more than agreement on the 1-5 value.
    h_ok = df["human_overall"] >= 4
    j_ok = df["judge_overall"] >= 4
    agree = (h_ok == j_ok).mean()
    po, pe = agree, (h_ok.mean() * j_ok.mean()) + ((1 - h_ok.mean()) * (1 - j_ok.mean()))
    kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    print(f"\n  Acceptable (>=4) agreement: {agree:.0%}   Cohen's kappa: {kappa:+.2f}")
    print(f"    you accept {h_ok.mean():.0%} of replies, judge accepts {j_ok.mean():.0%}")

    d_agree = (df["human_is_deflection"].astype(bool) == df["judge_is_deflection"].astype(bool)).mean()
    print(f"  Deflection-flag agreement:  {d_agree:.0%}")

    print("\n  Per system (judge minus you, positive = judge inflates):")
    for system, g in df.groupby("system"):
        print(f"    {system:8s} n={len(g):2d}  judge {g['judge_overall'].mean():.2f} "
              f"vs you {g['human_overall'].mean():.2f}  ({g['judge_overall'].mean() - g['human_overall'].mean():+.2f})")
    print("\n  The per-system row for 'llm' is the one to watch: the judge and the")
    print("  drafter are the same model, so inflation there is self-preference bias.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=15, help="replies to rate this session")
    parser.add_argument("--report", action="store_true", help="print agreement stats, no rating")
    args = parser.parse_args()
    try:
        report() if args.report else main(args.n)
    except KeyboardInterrupt:
        print("\n\nStopped. Progress saved -- rerun to resume.")
