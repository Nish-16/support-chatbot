"""
Interactive hand-labeling tool for the golden eval set. YOU run this
yourself, in your own terminal -- this is the one part of the
pipeline that can't be delegated, since the whole point of a golden
set is that the labels are your independent judgment, not a model's.

Design choice: your answers are asked for BEFORE the classifier's
guesses are revealed. If we showed the model's suggestions first,
you'd unconsciously agree with them more often than an independent
judgment would -- that's "anchoring bias," and it would quietly
inflate the classifier-accuracy number you report later. Answer
blind, then see what the model guessed, for an honest comparison.

TAXONOMY v2 (2026-09-10): updated from v1's single-field label
(human_intent only) to also collect human_secondary_intent,
human_turn_type, and human_flags -- see taxonomy.md for why the v2
schema has these fields at all (short version: v1 crammed conversation
-turn and risk signals into the intent enum itself; v2 gives them
their own fields instead).

Scope decision, stated explicitly rather than left implicit: sentiment
and language are NOT hand-labeled here, only intent, secondary_intent,
turn_type, escalate, and the boolean flags are. Sentiment/language are
secondary signals for later analysis, not central to the golden set's
main purpose (validating intent accuracy and the escalate decision) --
hand-verifying every one of ~207 examples across two more open-ended
judgment calls would add real labeling time for comparatively little
payoff. Revisit if the report ends up needing them.

Resumable: every label is appended to data/golden_labels.csv as soon
as you enter it, so you can stop anytime (Ctrl+C or 'q') and pick up
later -- already-labeled tweets are skipped automatically.

Usage: ./.venv/Scripts/python.exe scripts/08_label_golden_set.py
       ./.venv/Scripts/python.exe scripts/08_label_golden_set.py --priority
       ./.venv/Scripts/python.exe scripts/08_label_golden_set.py --priority --plan

--priority reorders the *unlabeled* candidates so the intents with the
fewest labels so far come first, letting a partial labeling run still
reach a usable per-intent floor (--floor, default 10) instead of
leaving rare intents at 1-3 examples. --plan prints the resulting order
and projected coverage, then exits without labeling anything.

Two caveats this introduces, both of which belong in the report rather
than being silently absorbed:
  1. Order is decided from `suggested_intent` (the classifier's guess),
     since the true label isn't known until you enter it. That affects
     only WHICH tweets you see first, never what you answer -- the
     model's guess is still hidden until after you've answered, so the
     anti-anchoring property above is intact.
  2. A run stopped early under --priority is NOT a random sample: it
     deliberately over-weights rare intents. Same caveat README.md
     already states for the candidate pool itself -- fine for measuring
     per-intent accuracy, not usable for claims about real traffic mix.
"""
import argparse
import csv
import os
from datetime import datetime, timezone

import pandas as pd

from groq_lib import INTENT_NAMES, INTENTS, TURN_TYPE_NAMES, FLAG_SPEC, PRECEDENCE_RULES

CANDIDATES_PATH = "data/golden_candidates.csv"
LABELS_PATH = "data/golden_labels.csv"

BOOL_FLAG_NAMES = [name for name, spec in FLAG_SPEC.items() if spec["type"] == "bool"]

LABEL_FIELDS = [
    "customer_tweet_id", "customer_text",
    "human_intent", "human_secondary_intent", "human_turn_type",
    "human_escalate", "human_flags", "human_note",
    "suggested_intent", "suggested_secondary_intent", "suggested_turn_type",
    "suggested_confidence", "suggested_flags",
    "agrees_with_suggestion", "labeled_at",
]


DEFAULT_FLOOR = 10  # per-intent minimum a partial run should try to reach


def load_already_labeled():
    if not os.path.exists(LABELS_PATH):
        return set()
    return set(pd.read_csv(LABELS_PATH)["customer_tweet_id"])


def load_labeled_intent_counts() -> dict[str, int]:
    if not os.path.exists(LABELS_PATH):
        return {}
    labels = pd.read_csv(LABELS_PATH)
    if "human_intent" not in labels.columns:
        return {}
    return labels["human_intent"].value_counts().to_dict()


def prioritize(remaining: pd.DataFrame, floor: int = DEFAULT_FLOOR) -> pd.DataFrame:
    """Reorder unlabeled candidates so under-represented intents come first.

    Round-robin across intents (one from each in turn) rather than one
    intent's whole block at a time: labeling nine consecutive tweets the
    model all thinks are `security_account_compromise` would prime you
    toward that answer on the ninth, which is the same anchoring problem
    the blind-answer design exists to avoid -- just arriving via
    ordering instead of via a visible suggestion. Interleaving keeps the
    high-value tweets early without that run effect.

    Rows for intents already at/above the floor are appended afterwards,
    so nothing is dropped -- a full run still labels all of them, just
    in a different order.
    """
    counts = load_labeled_intent_counts()
    deficit = {name: max(0, floor - int(counts.get(name, 0))) for name in INTENT_NAMES}

    groups: dict[str, list] = {}
    for name, group in remaining.groupby("suggested_intent"):
        # Shuffle within an intent so a partial run doesn't systematically
        # take whichever rows happen to sit first in the candidates file.
        groups[name] = group.sample(frac=1, random_state=42).to_dict("records")

    # Neediest intents first, so if the pool runs out mid-round the
    # biggest gaps are the ones that got served.
    order = sorted(groups, key=lambda n: -deficit.get(n, 0))
    quota = {n: min(deficit.get(n, 0), len(groups[n])) for n in groups}

    picked = []
    while any(quota[n] > 0 for n in order):
        for name in order:
            if quota[name] > 0 and groups[name]:
                picked.append(groups[name].pop(0))
                quota[name] -= 1

    leftovers = [row for name in order for row in groups[name]]
    return pd.DataFrame(picked + leftovers, columns=remaining.columns)


def print_plan(remaining: pd.DataFrame, ordered: pd.DataFrame, floor: int):
    counts = load_labeled_intent_counts()
    deficit = {name: max(0, floor - int(counts.get(name, 0))) for name in INTENT_NAMES}
    to_fill = sum(min(deficit[n], int((remaining["suggested_intent"] == n).sum())) for n in INTENT_NAMES)

    print(f"\nPriority plan (floor = {floor} labels per intent)\n")
    print(f"{'intent':30s} {'have':>5s} {'need':>5s} {'avail':>6s}")
    for name in sorted(INTENT_NAMES):
        have = int(counts.get(name, 0))
        avail = int((remaining["suggested_intent"] == name).sum())
        print(f"{name:30s} {have:5d} {deficit[name]:5d} {avail:6d}")

    print(f"\nNext {to_fill} labels (of {len(remaining)} unlabeled) bring every reachable "
          f"intent to {floor}.")
    print(f"Stopping there leaves you at ~{len(counts and pd.read_csv(LABELS_PATH)) + to_fill} "
          f"total labeled.\n")
    print(f"First 15 tweets in the new order (intent shown is the model's guess, "
          f"NOT revealed during labeling):")
    for i, row in enumerate(ordered.head(15).itertuples(), 1):
        text = " ".join(str(row.customer_text).split())[:70]
        print(f"  {i:2d}. [{row.suggested_intent:28s}] {text}...")


def print_intent_menu():
    """Shows each intent's description, not just its name.

    Until 2026-09-12 this printed bare names. That made the labeling tool
    the ONLY consumer of the taxonomy that showed less than the classifier
    prompt did -- the model at least sees intents.json's one-liners. The
    adjudication in ADJUDICATION.md traced 13 of 48 labeling errors to
    rules that were already written in taxonomy.md and simply never
    reached the person applying them.
    """
    print("\nIntents:")
    width = max(len(n) for n in INTENT_NAMES)
    for i, intent in enumerate(INTENTS, 1):
        print(f"  {i:2d}. {intent['name']:{width}s}  {intent['description']}")


def print_precedence_rules():
    """The Part 11 ladder, same text the classifier is given.

    Deliberately the SAME string (groq_lib.PRECEDENCE_RULES) rather than a
    paraphrase: a labeler and a model applying differently-worded versions
    of 'the same' rule is how the v1 golden set ended up with three labels
    across one family of quota tweets.
    """
    print("\n" + "=" * 78)
    print(PRECEDENCE_RULES)
    print("=" * 78)
    print("Full reasoning, worked examples and counterexamples: taxonomy.md Part 11.")


def print_turn_type_menu():
    print("\nTurn type (where this sits relative to a prior support exchange):")
    for i, name in enumerate(TURN_TYPE_NAMES, 1):
        print(f"  {i}. {name}")


def ask_intent(prompt: str, allow_skip: bool = False, exclude: str | None = None) -> str | None:
    """Returns an intent name, None (only if allow_skip and blank entered),
    or the literal 'QUIT' sentinel."""
    while True:
        raw = input(prompt).strip()
        if raw.lower() == "q":
            return "QUIT"
        if allow_skip and raw == "":
            return None
        name = None
        if raw.isdigit() and 1 <= int(raw) <= len(INTENT_NAMES):
            name = INTENT_NAMES[int(raw) - 1]
        elif raw in INTENT_NAMES:
            name = raw
        if name is None:
            print(f"  not a valid choice -- enter 1-{len(INTENT_NAMES)}, a name from the list, "
                  f"{'Enter to skip, ' if allow_skip else ''}or 'q'")
            continue
        if exclude is not None and name == exclude:
            print(f"  secondary_intent must differ from your primary intent ({exclude})")
            continue
        return name


def ask_turn_type() -> str:
    while True:
        raw = input("Turn type (number or name, 'q' to quit): ").strip()
        if raw.lower() == "q":
            return "QUIT"
        if raw.isdigit() and 1 <= int(raw) <= len(TURN_TYPE_NAMES):
            return TURN_TYPE_NAMES[int(raw) - 1]
        if raw in TURN_TYPE_NAMES:
            return raw
        print(f"  not a valid choice -- enter 1-{len(TURN_TYPE_NAMES)} or a name from the list above")


def ask_escalate() -> bool:
    while True:
        raw = input("Should this be escalated to a human? (y/n): ").strip().lower()
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("  enter y or n")


def ask_flags() -> str:
    """Comma-separated subset of BOOL_FLAG_NAMES, or '' for none. A single
    free-text prompt rather than one y/n question per flag -- five extra
    yes/no prompts per tweet across ~207 examples adds real labeling
    fatigue for signals that are mostly absent (a tweet rarely trips more
    than one of these)."""
    menu = ", ".join(BOOL_FLAG_NAMES)
    while True:
        raw = input(f"Flags that apply, comma-separated ({menu}), or Enter for none: ").strip()
        if raw == "":
            return ""
        chosen = [f.strip() for f in raw.split(",") if f.strip()]
        bad = [f for f in chosen if f not in BOOL_FLAG_NAMES]
        if bad:
            print(f"  not a recognized flag: {bad} -- pick from: {menu}")
            continue
        return ";".join(chosen)


def main(priority: bool = False, floor: int = DEFAULT_FLOOR, plan_only: bool = False):
    if not os.path.exists(CANDIDATES_PATH):
        print(f"{CANDIDATES_PATH} doesn't exist yet -- run scripts/07_build_golden_candidates.py first.")
        return

    candidates = pd.read_csv(CANDIDATES_PATH)
    already_labeled = load_already_labeled()
    remaining = candidates[~candidates["customer_tweet_id"].isin(already_labeled)]

    if priority or plan_only:
        ordered = prioritize(remaining, floor=floor)
        if plan_only:
            print_plan(remaining, ordered, floor)
            return
        remaining = ordered
        print(f"Priority order ON (floor={floor}): thinnest intents first. "
              f"Run with --plan to see the full plan.")

    print(f"{len(already_labeled)} already labeled, {len(remaining)} remaining "
          f"out of {len(candidates)} total candidates.")
    print("Type 'q' at any prompt to stop -- your progress is saved after every example.")
    print_precedence_rules()
    print_intent_menu()
    print_turn_type_menu()

    write_header = not os.path.exists(LABELS_PATH)
    with open(LABELS_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LABEL_FIELDS)
        if write_header:
            writer.writeheader()

        for i, row in enumerate(remaining.itertuples(), 1):
            print(f"\n--- [{i}/{len(remaining)}] tweet {row.customer_tweet_id} ---")
            print(f"CUSTOMER: {row.customer_text}")
            print(f"(historical Dropbox reply, for context only: {row.brand_text})")

            intent = ask_intent("Your intent (number or name, 'q' to quit): ")
            if intent == "QUIT":
                print("Stopping. Progress saved.")
                return
            secondary = ask_intent(
                "Secondary intent, if a second clearly distinct need is present (Enter to skip, 'q' to quit): ",
                allow_skip=True, exclude=intent,
            )
            if secondary == "QUIT":
                print("Stopping. Progress saved.")
                return
            turn_type = ask_turn_type()
            if turn_type == "QUIT":
                print("Stopping. Progress saved.")
                return
            escalate = ask_escalate()
            flags = ask_flags()
            note = input("Optional note (Enter to skip): ").strip()

            agrees = intent == row.suggested_intent
            # Blank cells in golden_candidates.csv (no secondary intent / no
            # flags) come back from pandas as float NaN, not "" -- and NaN
            # is truthy in Python, so `row.x or 'none'` would print "nan"
            # instead of falling through. Normalize explicitly.
            suggested_secondary = row.suggested_secondary_intent if pd.notna(row.suggested_secondary_intent) else ""
            suggested_flags = row.suggested_flags if pd.notna(row.suggested_flags) else ""
            print(f"  -> model suggested: {row.suggested_intent} ({row.suggested_confidence:.2f}), "
                  f"secondary={suggested_secondary or 'none'}, turn_type={row.suggested_turn_type}, "
                  f"flags={suggested_flags or 'none'} -- {'AGREE' if agrees else 'DISAGREE'} on intent")

            writer.writerow({
                "customer_tweet_id": row.customer_tweet_id,
                "customer_text": row.customer_text,
                "human_intent": intent,
                "human_secondary_intent": secondary or "",
                "human_turn_type": turn_type,
                "human_escalate": escalate,
                "human_flags": flags,
                "human_note": note,
                "suggested_intent": row.suggested_intent,
                "suggested_secondary_intent": suggested_secondary,
                "suggested_turn_type": row.suggested_turn_type,
                "suggested_confidence": row.suggested_confidence,
                "suggested_flags": suggested_flags,
                "agrees_with_suggestion": agrees,
                "labeled_at": datetime.now(timezone.utc).isoformat(),
            })
            f.flush()  # so a crash/Ctrl+C never loses a completed label

    print("\nAll candidates labeled!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hand-label the golden eval set.")
    parser.add_argument("--priority", action="store_true",
                        help="label under-represented intents first (see --floor)")
    parser.add_argument("--floor", type=int, default=DEFAULT_FLOOR,
                        help=f"per-intent label target for --priority (default: {DEFAULT_FLOOR})")
    parser.add_argument("--plan", action="store_true",
                        help="print the priority order and projected coverage, then exit")
    args = parser.parse_args()
    try:
        main(priority=args.priority, floor=args.floor, plan_only=args.plan)
    except KeyboardInterrupt:
        print("\n\nStopped (Ctrl+C). Progress saved -- rerun this script to resume.")
