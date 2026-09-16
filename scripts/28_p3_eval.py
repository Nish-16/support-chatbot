"""
STEP 12 -- does prompt p3 classify better than production p1?

p3 (groq_lib.build_system_prompt("p3")) is p1 plus taxonomy_rules.PROMPT_BLOCK.
It has never been evaluated, and no existing labelled set can evaluate it
fairly: the Part 11 rules were written from the whole 189-row golden set, and
p3's examples are dev tweets. This builds a NEW set, freezes it, and compares
the two prompts on it. Production is untouched -- PROMPT_VERSION stays p1.

Protocol (fixed 2026-09-14, before any label or prediction existed)
-------------------------------------------------------------------
* The set: EVAL_N tweets sampled with EVAL_SEED from the unique DropboxSupport
  customer tweets, excluding every tweet id any data file or the cache touched
  (eval_integrity.used_tweet_ids), the reading and coverage samples the
  taxonomy was designed from, and any tweet sharing a 40-character window with
  a project document or a taxonomy_rules example. Written once, with a
  manifest holding its sha256; every later step refuses a changed set.
* Amended 2026-09-14, before labelling: only eval_index 1..SCORED_N (250) is
  labelled, predicted and scored, to stay within the assignment's 150-250 cap
  on hand-labelled examples. See data/p3_eval_amendment.json.
* Amended again 2026-09-14: SCORED_N 250 -> 150 after ~200 labels were lost.
  Rows 1-52 are hand labels; 53-150 are Claude drafts (blind to predictions)
  reviewed by the labeller.
* Amended a third time 2026-09-14: SCORED_N 150 -> 250, after the n=150
  report had been run and read (outputs kept as data/p3_eval_*_n150.csv).
  Rows 151-250 are Claude drafts. The 250 result is not a pre-registered
  test; report it alongside the 150 result.
* Labels: hand labels, blind. The labeler sees taxonomy_rules.LABELER_BLOCK and
  the earlier message the tweet replies to, and never either prompt's
  prediction. Tweets unroutable without context are marked
  insufficient_context and excluded from scoring, as in golden_labels_v3.csv.
* Predictions: p1 and p3 on the tweet text alone (production p1's input), same
  model, temperature 0, reasoning_effort low, interleaved in chunks of CHUNK so
  a run spread over days by the token cap does not put the two prompts on
  different days. Cached under each prompt's own namespace.
* Comparison: paired, per tweet. The difference is p3 correct minus p1 correct,
  with a 95% bootstrap interval and an exact McNemar p on the discordant pairs.
* Verdict, mechanical:
    P3 BETTER      the interval's lower bound > 0, and p3 misses no more human
                   escalations than p1
    P3 NOT BETTER  the interval's upper bound < MIN_MEANINGFUL_GAIN
    INCONCLUSIVE   anything else
  A result never switches production. It is a recommendation.

Usage:
  ./.venv/Scripts/python.exe scripts/28_p3_eval.py --build      # freeze the set (needs data/twcs/twcs.csv)
  ./.venv/Scripts/python.exe scripts/28_p3_eval.py --verify     # integrity checks, no API
  ./.venv/Scripts/python.exe scripts/28_p3_eval.py --label      # blind hand labelling, resumable
  ./.venv/Scripts/python.exe scripts/28_p3_eval.py --status     # progress only -- never shows predictions
  ./.venv/Scripts/python.exe scripts/28_p3_eval.py --run [--n N]   # p1 + p3 predictions, resumable
  ./.venv/Scripts/python.exe scripts/28_p3_eval.py --report     # needs every label and prediction
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import eval_integrity as integrity
import taxonomy_rules
from cache import ClassificationCache
from classify_runner import classify_many
from escalation import ESCALATE, decide
from groq_lib import (INTENT_NAMES, MODEL, PROMPT_VERSION, build_system_prompt, cache_namespace,
                      classify_message, make_client)

SET_PATH = "data/p3_eval_set.csv"
MANIFEST_PATH = "data/p3_eval_manifest.json"
LABELS_PATH = "data/p3_eval_labels.csv"
RUNS_PATH = "data/p3_eval_runs.jsonl"
RESULTS_PATH = "data/p3_eval_results.csv"
CONFUSION_PATH = "data/p3_eval_confusion_{prompt}.csv"
RAW_PATH = "data/twcs/twcs.csv"
CACHE_PATH = "data/classification_cache.jsonl"
BRAND_AUTHOR = "DropboxSupport"

EVAL_N = 300
EVAL_SEED = 20260914
# AMENDMENT 2026-09-14 (data/p3_eval_amendment.json): only the first SCORED_N
# tweets of the frozen order are labelled, predicted and scored. The assignment
# caps the hand-labelled evaluation set at 150-250 examples; 300 was proposed
# without checking that. Decided with 1 label entered and no prediction viewed.
# eval_index is the order of a seeded random sample, so its prefix is a random
# sample too. The 300-row file stays frozen exactly as built.
# AMENDMENT 2 2026-09-14: 250 -> 150. About 200 hand labels were lost; 52
# survived. Tweets 53-150 are drafted by Claude (data/p3_eval_drafts.csv, blind
# to predictions) and reviewed by the labeller. Decided with no prediction viewed.
# AMENDMENT 3 2026-09-14: 150 -> 250, AFTER the n=150 report was run and read.
# Its outputs are kept as data/p3_eval_*_n150.csv and must be reported too.
# Tweets 151-250 are Claude drafts, blind to predictions and to that report.
SCORED_N = 250
AMENDMENT_PATH = "data/p3_eval_amendment.json"
DRAFTS_PATH = "data/p3_eval_drafts.csv"
DRAFT_ACCEPTED, DRAFT_EDITED = "[claude draft accepted]", "[claude draft edited]"
PROMPTS = ("p1", "p3")
CHUNK = 25
MIN_MEANINGFUL_GAIN = 0.03
N_BOOT = 10_000

LABEL_FIELDS = ["eval_index", "customer_tweet_id", "human_intent", "human_secondary_intent",
                "human_turn_type", "human_escalate", "human_flags", "insufficient_context",
                "human_note", "labeled_at", "set_sha256"]
CONTEXT_LABELS = {"brand": "Dropbox support", "same": "the same customer, earlier", "other": "another user"}

TOPIC_INTENTS = frozenset(i for i in INTENT_NAMES
                          if i not in ("complaint_dissatisfaction", "feature_request", "no_action_needed"))
# taxonomy_rules.BOUNDARIES, widened to every intent the rule sends traffic to.
# "Complaint vs. an actionable topic" is complaint against ANY topic, not only
# how_to_usage; the device-disk half of C lands in how_to_usage; E's carve-out
# lands in sync_app_bug.
BOUNDARY_SIDES: dict[str, tuple[frozenset, frozenset]] = {
    "A": (frozenset({"complaint_dissatisfaction"}), TOPIC_INTENTS),
    "B": (frozenset({"feature_request"}), frozenset({"complaint_dissatisfaction"})),
    "C": (frozenset({"storage_quota_plan_limits"}), frozenset({"sync_app_bug", "how_to_usage"})),
    "D": (frozenset({"storage_quota_plan_limits"}), frozenset({"billing_subscription"})),
    "E": (frozenset({"sharing_permissions"}), frozenset({"how_to_usage", "sync_app_bug"})),
    "F": (frozenset({"account_access"}), frozenset({"how_to_usage"})),
    "G": (frozenset({"service_outage"}), frozenset({"sync_app_bug"})),
}
ABBR = {"account_access": "acc", "security_account_compromise": "sec", "phishing_abuse_report": "phi",
        "billing_subscription": "bil", "storage_quota_plan_limits": "sto", "sync_app_bug": "syn",
        "service_outage": "out", "sharing_permissions": "shr", "data_loss_recovery": "dat",
        "how_to_usage": "how", "feature_request": "fea", "complaint_dissatisfaction": "cmp",
        "no_action_needed": "noa"}


def prompt_sha256(version: str) -> str:
    return hashlib.sha256(build_system_prompt(version).encode("utf-8")).hexdigest()


# -- build ----------------------------------------------------------------

def build():
    if os.path.exists(SET_PATH) or os.path.exists(MANIFEST_PATH):
        raise SystemExit(f"{SET_PATH} is frozen -- refusing to rebuild. A re-rolled set silently "
                         f"invalidates every label and prediction made against it.")
    problems = integrity.frozen_state_problems()
    if problems:
        raise SystemExit("evaluation state differs from its pin:\n  " + "\n  ".join(problems))

    n_cache = len(integrity.cache_lines())
    unique = integrity.unique_customer_tweets()
    used = integrity.used_tweet_ids(n_cache)
    used_ids = set().union(*used.values())
    by_id = unique[~unique.customer_tweet_id.isin(used_ids)]

    # Text-level exclusion: quoted in a document or an example, or the same
    # text as a used tweet posted under another id.
    reference = integrity.reference_shingles()
    used_texts = {integrity.normalize_for_match(t)
                  for t in unique[unique.customer_tweet_id.isin(used_ids)].customer_text}
    keep = []
    for row in by_id.itertuples():
        norm = integrity.normalize_for_match(row.customer_text)
        keep.append(bool(norm) and norm not in used_texts
                    and not (integrity.shingles(row.customer_text) & reference))
    eligible = by_id[keep]
    eligible = eligible[~eligible.customer_text.map(integrity.normalize_for_match).duplicated()]
    eligible = eligible.assign(_id=eligible.customer_tweet_id.astype("int64")).sort_values("_id")

    sample = eligible.sample(n=EVAL_N, random_state=EVAL_SEED).reset_index(drop=True)
    sample.insert(0, "eval_index", range(1, len(sample) + 1))
    sample = sample.merge(_parents(sample.customer_tweet_id), on="customer_tweet_id", how="left")
    out = sample[["eval_index", "customer_tweet_id", "customer_created_at", "customer_text",
                  "parent_tweet_id", "parent_author", "parent_text"]]
    out.to_csv(SET_PATH, index=False, encoding="utf-8")

    manifest = {
        "experiment": "p1 vs p3 on a frozen, newly labelled set (scripts/28_p3_eval.py)",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "set_path": SET_PATH,
        "set_sha256": integrity.file_sha256(SET_PATH),
        "n": EVAL_N,
        "seed": EVAL_SEED,
        "pool_unique_customer_tweets": len(unique),
        "excluded_by_id": len(unique) - len(by_id),
        "excluded_by_text_or_duplicate": len(by_id) - len(eligible),
        "eligible": len(eligible),
        "exclusion_sources": {src: len(ids) for src, ids in used.items()},
        "cache_lines_at_freeze": n_cache,
        "cache_prefix_sha256": integrity.cache_prefix_sha256(n_cache),
        "model": MODEL,
        "temperature": 0,
        "reasoning_effort": "low",
        "production_prompt_version": PROMPT_VERSION,
        "prompt_sha256": {v: prompt_sha256(v) for v in PROMPTS},
        "rules_block_sha256": hashlib.sha256(taxonomy_rules.PROMPT_BLOCK.encode("utf-8")).hexdigest(),
        "classifier_input": "customer_text only, as production p1",
        "labeler_input": "customer_text + earlier message; predictions never shown",
        "verdict_rule": {"min_meaningful_gain": MIN_MEANINGFUL_GAIN, "n_boot": N_BOOT, "ci": 0.95},
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"froze {SET_PATH}: {EVAL_N} of {len(eligible)} eligible tweets "
          f"({manifest['excluded_by_id']} excluded by id, "
          f"{manifest['excluded_by_text_or_duplicate']} by text or duplicate)")
    print(f"wrote {MANIFEST_PATH}")


def _parents(tweet_ids: pd.Series) -> pd.DataFrame:
    print(f"reading {RAW_PATH} for earlier messages (516MB, ~1 min)...", flush=True)
    raw = pd.read_csv(RAW_PATH, usecols=["tweet_id", "author_id", "text", "in_response_to_tweet_id"],
                      dtype={"tweet_id": str, "author_id": str}).set_index("tweet_id")
    rows = []
    for tid in tweet_ids:
        if tid not in raw.index or pd.isna(raw.at[tid, "in_response_to_tweet_id"]):
            continue
        pid = str(int(raw.at[tid, "in_response_to_tweet_id"]))
        if pid not in raw.index:
            continue
        author = raw.at[pid, "author_id"]
        who = "brand" if author == BRAND_AUTHOR else "same" if author == raw.at[tid, "author_id"] else "other"
        rows.append({"customer_tweet_id": tid, "parent_tweet_id": pid, "parent_author": who,
                     "parent_text": raw.at[pid, "text"]})
    return pd.DataFrame(rows, columns=["customer_tweet_id", "parent_tweet_id", "parent_author", "parent_text"])


# -- integrity ------------------------------------------------------------

def load_manifest() -> dict:
    if not os.path.exists(MANIFEST_PATH):
        raise SystemExit(f"{MANIFEST_PATH} missing -- run --build first")
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_set() -> pd.DataFrame:
    """The whole frozen file -- used for integrity checks only."""
    return pd.read_csv(SET_PATH, dtype={"customer_tweet_id": str, "parent_tweet_id": str})


def scored_set() -> pd.DataFrame:
    """The tweets that are labelled, predicted and scored (see the amendment)."""
    frozen = load_set()
    return frozen[frozen.eval_index <= SCORED_N].reset_index(drop=True)


def verify_problems(manifest: dict) -> list[str]:
    """Everything that must hold before a label, a prediction or a report is
    trusted. Empty list = clean."""
    problems = list(integrity.frozen_state_problems())
    if integrity.file_sha256(SET_PATH) != manifest["set_sha256"]:
        problems.append(f"{SET_PATH} changed since it was frozen")
    if prompt_sha256("p1") != integrity.P1_PROMPT_SHA256:
        problems.append("p1 is no longer byte-identical to the production prompt")
    if PROMPT_VERSION != "p1":
        problems.append(f"production PROMPT_VERSION is {PROMPT_VERSION!r}, not 'p1'")
    for version in PROMPTS:
        if prompt_sha256(version) != manifest["prompt_sha256"][version]:
            problems.append(f"{version} prompt changed since the set was frozen")
    if taxonomy_rules.PROMPT_BLOCK not in build_system_prompt("p3"):
        problems.append("p3 no longer carries taxonomy_rules.PROMPT_BLOCK")

    eval_set = load_set()
    used = set().union(*integrity.used_tweet_ids(manifest["cache_lines_at_freeze"]).values())
    leaked = sorted(set(eval_set.customer_tweet_id) & used)
    if leaked:
        problems.append(f"{len(leaked)} eval tweets are in existing data: {leaked[:5]}")
    reference = integrity.reference_shingles()
    quoted = [r.customer_tweet_id for r in eval_set.itertuples()
              if integrity.shingles(r.customer_text) & reference]
    if quoted:
        problems.append(f"{len(quoted)} eval tweets are quoted in docs/examples: {quoted[:5]}")
    if eval_set.customer_tweet_id.duplicated().any():
        problems.append("duplicate tweet ids in the eval set")

    if not os.path.exists(AMENDMENT_PATH):
        problems.append(f"{AMENDMENT_PATH} missing -- the 250-tweet scoring cut is undocumented")
    else:
        with open(AMENDMENT_PATH, encoding="utf-8") as f:
            amendment = json.load(f)
        if amendment.get("scored_n") != SCORED_N or amendment.get("set_sha256") != manifest["set_sha256"]:
            problems.append(f"{AMENDMENT_PATH} does not match SCORED_N={SCORED_N} and the frozen set")
    beyond = load_labels()
    beyond = beyond[pd.to_numeric(beyond.eval_index) > SCORED_N]
    if len(beyond):
        problems.append(f"{len(beyond)} labels are beyond tweet {SCORED_N}, outside the scored set")

    gold = pd.read_csv("data/golden_labels_v3.csv", dtype={"customer_tweet_id": str}).set_index("customer_tweet_id")
    for tid, label in integrity.KNOWN_R7A_EXCEPTIONS.items():
        if gold.at[tid, "human_intent_v3"] != label or gold.at[tid, "v3_rule"] != "R7a":
            problems.append(f"known R7a exception {tid} was relabelled or re-ruled")
    with open("taxonomy.md", encoding="utf-8") as f:
        part12 = f.read().split("# Part 12", 1)[-1]
    for tid in integrity.KNOWN_R7A_EXCEPTIONS:
        if tid not in part12:
            problems.append(f"known R7a exception {tid} is no longer documented in taxonomy.md Part 12")
    return problems


def verify(manifest: dict) -> bool:
    problems = verify_problems(manifest)
    checks = [
        "frozen evaluation files and cache prefix unchanged",
        "eval set matches its frozen sha256",
        "p1 byte-identical to production; PROMPT_VERSION is p1",
        "p1 and p3 prompts match the manifest; p3 carries taxonomy_rules.PROMPT_BLOCK",
        "no eval tweet in existing data, docs or examples; no duplicates",
        "R7a exceptions 1000504 and 1057223 unchanged and documented",
        f"scoring limited to the first {SCORED_N} tweets (amendment recorded); no labels beyond it",
    ]
    print("\nINTEGRITY")
    for check in checks:
        print(f"  checked: {check}")
    if problems:
        print("\n  FAILED:")
        for p in problems:
            print(f"    - {p}")
        return False
    print("  all checks passed")
    return True


# -- label ----------------------------------------------------------------

def _labeler_module():
    spec = importlib.util.spec_from_file_location("label_mod", "scripts/08_label_golden_set.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_labels() -> pd.DataFrame:
    # A labelling session creates the file before its first label is saved,
    # so an existing-but-empty file is normal and means "no labels yet".
    if not os.path.exists(LABELS_PATH) or os.path.getsize(LABELS_PATH) == 0:
        return pd.DataFrame(columns=LABEL_FIELDS)
    return pd.read_csv(LABELS_PATH, dtype={"customer_tweet_id": str})


def label(manifest: dict):
    """Blind by construction: this function reads the set and the labels file,
    never the classification cache, so no prediction can reach the screen."""
    if not verify(manifest):
        raise SystemExit("fix the integrity problems before labelling")
    ui = _labeler_module()
    eval_set = scored_set()
    done = set(load_labels().customer_tweet_id)
    todo = eval_set[~eval_set.customer_tweet_id.isin(done)]
    drafts = {}
    if os.path.exists(DRAFTS_PATH):
        # Claude's blind drafts: shown for review, never a prediction.
        drafted = pd.read_csv(DRAFTS_PATH, dtype=str, keep_default_na=False)
        drafts = {int(r["eval_index"]): r for r in drafted.to_dict("records")}
    print(f"\n{len(done)} labelled, {len(todo)} to go. 'q' at any prompt stops; every label is saved at once.")
    ui.print_precedence_rules()
    ui.print_intent_menu()
    ui.print_turn_type_menu()

    # Empty counts as new: a session quit before its first label leaves an
    # empty file, and appending rows to it without a header corrupts it.
    write_header = not os.path.exists(LABELS_PATH) or os.path.getsize(LABELS_PATH) == 0
    with open(LABELS_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LABEL_FIELDS)
        if write_header:
            writer.writeheader()
            f.flush()
        for row in todo.itertuples():
            print(f"\n--- [{row.eval_index}/{len(eval_set)}] tweet {row.customer_tweet_id} ---")
            print(f"CUSTOMER: {row.customer_text}")
            if isinstance(row.parent_text, str):
                print(f"EARLIER MESSAGE it replies to ({CONTEXT_LABELS.get(row.parent_author, row.parent_author)}): "
                      f"{row.parent_text}")
            else:
                print("(no earlier message -- first contact)")
            draft = drafts.get(int(row.eval_index))
            if draft is not None:
                print(f"CLAUDE DRAFT: {draft['intent']}"
                      f"{' / ' + draft['secondary'] if draft['secondary'] else ''}"
                      f" | {draft['turn_type']} | escalate={draft['escalate']}"
                      f" | flags={draft['flags'] or 'none'} | insufficient_context={draft['insufficient_context']}")
                print(f"  why: {draft['why']}")
                choice = input("Enter = accept draft, e = edit, q = quit: ").strip().lower()
                if choice == "q":
                    return
                if choice == "":
                    writer.writerow({
                        "eval_index": row.eval_index, "customer_tweet_id": row.customer_tweet_id,
                        "human_intent": draft["intent"], "human_secondary_intent": draft["secondary"],
                        "human_turn_type": draft["turn_type"], "human_escalate": draft["escalate"] == "True",
                        "human_flags": draft["flags"].replace(",", ";"),
                        "insufficient_context": draft["insufficient_context"] == "True",
                        "human_note": DRAFT_ACCEPTED,
                        "labeled_at": datetime.now(timezone.utc).isoformat(),
                        "set_sha256": manifest["set_sha256"],
                    })
                    f.flush()
                    continue
            intent = ui.ask_intent("Your intent (number or name, 'q' to quit): ")
            if intent == "QUIT":
                return
            secondary = ui.ask_intent("Secondary intent (Enter to skip, 'q' to quit): ",
                                      allow_skip=True, exclude=intent)
            if secondary == "QUIT":
                return
            turn_type = ui.ask_turn_type()
            if turn_type == "QUIT":
                return
            escalate = ui.ask_escalate()
            flags = ui.ask_flags()
            insufficient = input("Unroutable without the earlier message -- insufficient_context? (y/N): ").strip().lower() in ("y", "yes")
            note = input("Optional note (Enter to skip): ").strip()
            if draft is not None:
                note = f"{DRAFT_EDITED} {note}".strip()
            writer.writerow({
                "eval_index": row.eval_index, "customer_tweet_id": row.customer_tweet_id,
                "human_intent": intent, "human_secondary_intent": secondary or "",
                "human_turn_type": turn_type, "human_escalate": escalate, "human_flags": flags,
                "insufficient_context": insufficient, "human_note": note,
                "labeled_at": datetime.now(timezone.utc).isoformat(),
                "set_sha256": manifest["set_sha256"],
            })
            f.flush()
    print("\nAll tweets labelled.")


# -- run ------------------------------------------------------------------

def run(manifest: dict, n: int | None):
    if not verify(manifest):
        raise SystemExit("fix the integrity problems before spending API calls")
    eval_set = scored_set()
    rows = eval_set.head(n) if n else eval_set
    items = [(r.customer_tweet_id, r.customer_text) for r in rows.itertuples()]
    client = make_client()
    caches = {v: ClassificationCache(CACHE_PATH, namespace=cache_namespace(prompt_version=v)) for v in PROMPTS}
    tally = {v: {"classified": 0, "failed": []} for v in PROMPTS}
    for start in range(0, len(items), CHUNK):
        chunk = items[start:start + CHUNK]
        for version in PROMPTS:
            failed: list[str] = []
            got = classify_many(
                client, chunk, caches[version],
                classify=lambda c, t, v=version: classify_message(c, t, prompt_version=v),
                on_result=lambda tid, _t, res, _c, bucket=failed: res is None and bucket.append(str(tid)),
            )
            tally[version]["classified"] += len(got)
            tally[version]["failed"] += failed
        # Counts only: predictions for these tweets must not reach anyone who
        # has not labelled them yet.
        print(f"  tweets {start + 1}-{start + len(chunk)}: "
              + ", ".join(f"{v} {tally[v]['classified']} ok / {len(tally[v]['failed'])} failed" for v in PROMPTS),
              flush=True)
    with open(RUNS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "at": datetime.now(timezone.utc).isoformat(), "requested": len(items), "model": MODEL,
            "namespaces": {v: cache_namespace(prompt_version=v) for v in PROMPTS},
            "prompt_sha256": {v: prompt_sha256(v) for v in PROMPTS},
            "set_sha256": manifest["set_sha256"],
            "results": {v: {"classified": tally[v]["classified"], "failed": tally[v]["failed"]} for v in PROMPTS},
        }) + "\n")
    print(f"logged to {RUNS_PATH}")


def predictions(eval_set: pd.DataFrame) -> dict[str, dict[str, dict]]:
    out = {}
    for version in PROMPTS:
        cache = ClassificationCache(CACHE_PATH, namespace=cache_namespace(prompt_version=version))
        out[version] = {tid: cache.get(tid) for tid in eval_set.customer_tweet_id if cache.get(tid) is not None}
    return out


def status(manifest: dict):
    eval_set = scored_set()
    labels = load_labels()
    labels = labels[labels.customer_tweet_id.isin(eval_set.customer_tweet_id)]
    preds = predictions(eval_set)
    print(f"set: scoring the first {len(eval_set)} of {EVAL_N} frozen tweets, frozen "
          f"{manifest['created_at'][:10]}, sha256 {manifest['set_sha256'][:12]}...")
    print(f"labels: {labels.customer_tweet_id.nunique()} / {len(eval_set)}")
    for version in PROMPTS:
        print(f"{version} predictions: {len(preds[version])} / {len(eval_set)}")


# -- statistics -----------------------------------------------------------

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Same interval as 12_evaluate.py."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p: the discordant pairs as a fair coin."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired_difference(p1_correct, p3_correct, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    a = np.asarray(p1_correct, dtype=bool)
    b = np.asarray(p3_correct, dtype=bool)
    d = b.astype(int) - a.astype(int)
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    fixes, breaks = int((~a & b).sum()), int((a & ~b).sum())
    return {"n": len(d), "diff": float(d.mean()), "ci": (float(lo), float(hi)),
            "fixes": fixes, "breaks": breaks, "mcnemar_p": mcnemar_exact(fixes, breaks)}


def verdict(stats: dict, missed_p1: int, missed_p3: int) -> tuple[str, str]:
    lo, hi = stats["ci"]
    if lo > 0 and missed_p3 <= missed_p1:
        return ("P3 BETTER — evidence supports replacing p1",
                f"the paired interval [{lo:+.1%}, {hi:+.1%}] excludes zero and p3 misses "
                f"{missed_p3} human escalations vs p1's {missed_p1}")
    if hi < MIN_MEANINGFUL_GAIN:
        return ("P3 NOT BETTER — keep p1",
                f"the paired interval [{lo:+.1%}, {hi:+.1%}] rules out a gain of "
                f"{MIN_MEANINGFUL_GAIN:.0%} or more")
    if lo > 0:
        return ("INCONCLUSIVE — collect more data",
                f"accuracy gain [{lo:+.1%}, {hi:+.1%}] excludes zero, but p3 misses "
                f"{missed_p3} human escalations vs p1's {missed_p1}")
    return ("INCONCLUSIVE — collect more data",
            f"the paired interval [{lo:+.1%}, {hi:+.1%}] includes both no change and a "
            f"{MIN_MEANINGFUL_GAIN:.0%}+ gain")


def escalation_metrics(escalated: pd.Series, human: pd.Series) -> dict:
    tp = int((escalated & human).sum())
    fp = int((escalated & ~human).sum())
    fn = int((~escalated & human).sum())
    return {"escalated": int(escalated.sum()), "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "recall": tp / (tp + fn) if tp + fn else float("nan"), "missed": fn, "over": fp}


def boundary_table(df: pd.DataFrame) -> list[dict]:
    rows = []
    for key, (side_a, side_b) in BOUNDARY_SIDES.items():
        sub = df[df.gold.isin(side_a | side_b)]
        entry = {"key": key, "title": taxonomy_rules.BOUNDARIES[key][0], "n": len(sub)}
        for version in PROMPTS:
            pred = sub[f"{version}_intent"]
            cross = (sub.gold.isin(side_a) & pred.isin(side_b)) | (sub.gold.isin(side_b) & pred.isin(side_a))
            entry[f"{version}_acc"] = (pred == sub.gold).mean() if len(sub) else float("nan")
            entry[f"{version}_cross"] = int(cross.sum())
            entry[f"{version}_cross_mask"] = cross
        entry["cross_fixed"] = int((entry["p1_cross_mask"] & ~entry["p3_cross_mask"]).sum())
        entry["cross_new"] = int((~entry["p1_cross_mask"] & entry["p3_cross_mask"]).sum())
        rows.append(entry)
    return rows


# -- report ---------------------------------------------------------------

def assemble(eval_set: pd.DataFrame, labels: pd.DataFrame, preds: dict) -> pd.DataFrame:
    labels = labels.drop_duplicates("customer_tweet_id", keep="last")
    df = eval_set.merge(labels, on=["eval_index", "customer_tweet_id"], how="left")
    df["gold"] = df.human_intent
    df["human_escalate"] = df.human_escalate.astype(str).str.lower() == "true"
    df["insufficient_context"] = df.insufficient_context.astype(str).str.lower() == "true"
    for version in PROMPTS:
        results = [preds[version].get(tid) for tid in df.customer_tweet_id]
        df[f"{version}_intent"] = [r["intent"] if r else "__missing__" for r in results]
        df[f"{version}_correct"] = df[f"{version}_intent"] == df.gold
        df[f"{version}_escalate"] = [decide(r).action == ESCALATE if r else True for r in results]
    return df


def report(manifest: dict, allow_missing: bool):
    if not verify(manifest):
        raise SystemExit("integrity checks failed -- not reporting")
    eval_set, labels = scored_set(), load_labels()
    labels = labels[labels.customer_tweet_id.isin(eval_set.customer_tweet_id)]
    preds = predictions(eval_set)
    unlabelled = len(eval_set) - labels.customer_tweet_id.nunique()
    missing = {v: len(eval_set) - len(preds[v]) for v in PROMPTS}
    if unlabelled:
        raise SystemExit(f"{unlabelled} tweets unlabelled -- the report needs the whole frozen set. "
                         f"Interim reads are not reported.")
    if any(missing.values()) and not allow_missing:
        raise SystemExit(f"missing predictions {missing} -- rerun --run, or pass --count-missing-as-wrong "
                         f"if they fail persistently")

    df = assemble(eval_set, labels, preds)
    scored = df[~df.insufficient_context].reset_index(drop=True)
    n = len(scored)

    print(f"\n{'=' * 78}\np1 vs p3 -- {n} scored of {len(df)} frozen tweets "
          f"({int(df.insufficient_context.sum())} insufficient_context excluded)\n{'=' * 78}")
    if any(missing.values()):
        print(f"  missing predictions counted as wrong and escalated: {missing}")

    stats =paired_difference(scored.p1_correct, scored.p3_correct)
    print("\nOVERALL ACCURACY")
    for version in PROMPTS:
        k = int(scored[f"{version}_correct"].sum())
        lo, hi = wilson(k, n)
        print(f"  {version}  {k / n:.1%}  ({k}/{n})  95% CI [{lo:.1%}, {hi:.1%}]")
    lo, hi = stats["ci"]
    print(f"\n  paired difference p3 - p1: {stats['diff']:+.1%}   95% bootstrap CI [{lo:+.1%}, {hi:+.1%}]")
    print(f"  p3 fixes {stats['fixes']} p1 errors, introduces {stats['breaks']}; "
          f"exact McNemar p = {stats['mcnemar_p']:.3f}")
    disagree = scored.p1_intent != scored.p3_intent
    print(f"  p1 and p3 disagree on {int(disagree.sum())} of {n} scored tweets "
          f"({int((df.p1_intent != df.p3_intent).sum())} of {len(df)} including excluded)")

    print("\nPER-INTENT ACCURACY (recall on the human label)")
    print(f"  {'intent':28s} {'n':>4s} {'p1':>7s} {'p3':>7s}")
    for intent in INTENT_NAMES:
        sub = scored[scored.gold == intent]
        if len(sub):
            print(f"  {intent:28s} {len(sub):4d} {sub.p1_correct.mean():7.1%} {sub.p3_correct.mean():7.1%}")
    macro = {v: np.mean([scored[scored.gold == i][f"{v}_correct"].mean()
                         for i in INTENT_NAMES if (scored.gold == i).any()]) for v in PROMPTS}
    print(f"  {'macro average':28s} {'':4s} {macro['p1']:7.1%} {macro['p3']:7.1%}")

    print("\nBOUNDARY PAIRS (rows whose human label is on either side)")
    print(f"  {'boundary':42s} {'n':>4s} {'p1 acc':>7s} {'p3 acc':>7s} {'p1 cross':>9s} {'p3 cross':>9s} "
          f"{'fixed':>6s} {'new':>4s}")
    for b in boundary_table(scored):
        print(f"  {b['key']}. {b['title']:39s} {b['n']:4d} {b['p1_acc']:7.1%} {b['p3_acc']:7.1%} "
              f"{b['p1_cross']:9d} {b['p3_cross']:9d} {b['cross_fixed']:6d} {b['cross_new']:4d}")
    print("  cross = predicted the other side of that boundary. Small n per boundary: read as counts.")

    print("\nEND-TO-END ESCALATION (escalation.decide on each prompt's output vs the human decision)")
    esc = {v: escalation_metrics(scored[f"{v}_escalate"], scored.human_escalate) for v in PROMPTS}
    for version in PROMPTS:
        m = esc[version]
        print(f"  {version}  escalated {m['escalated']}  precision {m['precision']:.1%}  recall {m['recall']:.1%}  "
              f"missed {m['missed']}  over-escalated {m['over']}")
    human = scored.human_escalate
    caught_only_p3 = int((human & scored.p3_escalate & ~scored.p1_escalate).sum())
    missed_only_p3 = int((human & ~scored.p3_escalate & scored.p1_escalate).sum())
    print(f"  on human escalations: p3 newly catches {caught_only_p3}, newly misses {missed_only_p3}; "
          f"exact McNemar p = {mcnemar_exact(caught_only_p3, missed_only_p3):.3f}")

    print("\nCONFUSION MATRICES (rows = human label, columns = prediction)")
    labels_order = list(INTENT_NAMES)
    for version in PROMPTS:
        matrix = pd.crosstab(pd.Categorical(scored.gold, categories=labels_order),
                             pd.Categorical(scored[f"{version}_intent"], categories=labels_order + ["__missing__"]),
                             dropna=False)
        matrix.index, matrix.columns = labels_order, labels_order + ["__missing__"]
        matrix.to_csv(CONFUSION_PATH.format(prompt=version), encoding="utf-8")
        shown = matrix.rename(index=ABBR, columns={**ABBR, "__missing__": "mis"})
        print(f"\n  {version}:")
        print("  " + shown.to_string().replace("\n", "\n  "))

    for title, mask in (("P3 FIXES A P1 ERROR", ~scored.p1_correct & scored.p3_correct),
                        ("P3 INTRODUCES A NEW ERROR", scored.p1_correct & ~scored.p3_correct)):
        sub = scored[mask]
        print(f"\n{title}: {len(sub)}")
        for r in sub.itertuples():
            print(f"  {r.customer_tweet_id:>8s}  human {r.gold:26s} p1 {r.p1_intent:26s} p3 {r.p3_intent}")
            print(f"            {' '.join(str(r.customer_text).split())[:140]}")

    keep = ["eval_index", "customer_tweet_id", "gold", "insufficient_context", "human_escalate"] + \
           [f"{v}_{c}" for v in PROMPTS for c in ("intent", "correct", "escalate")]
    df[keep].to_csv(RESULTS_PATH, index=False, encoding="utf-8")
    print(f"\nwrote {RESULTS_PATH} and {CONFUSION_PATH.format(prompt='{p1,p3}')}")

    call, reason = verdict(stats, esc["p1"]["missed"], esc["p3"]["missed"])
    print(f"\n{'=' * 78}\nVERDICT: {call}\n  because {reason}.")
    print("  Production is not changed by this script. PROMPT_VERSION stays p1.\n" + "=" * 78)


def main():
    p = argparse.ArgumentParser()
    mode = p.add_mutually_exclusive_group(required=True)
    for flag in ("--build", "--verify", "--label", "--status", "--run", "--report"):
        mode.add_argument(flag, action="store_true")
    p.add_argument("--n", type=int, help="--run only: the first N tweets of the frozen set")
    p.add_argument("--count-missing-as-wrong", action="store_true",
                   help="--report only: score persistently failed predictions as wrong and escalated")
    args = p.parse_args()

    # Reports quote raw tweets, which contain emoji. Windows defaults stdout to
    # cp1252 when it is redirected, which raises UnicodeEncodeError mid-report.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if args.build:
        build()
        return
    manifest = load_manifest()
    if args.verify:
        sys.exit(0 if verify(manifest) else 1)
    elif args.label:
        label(manifest)
    elif args.status:
        status(manifest)
    elif args.run:
        run(manifest, args.n)
    elif args.report:
        report(manifest, args.count_missing_as_wrong)


if __name__ == "__main__":
    main()
