"""
Build the CANDIDATE list for the golden eval set. This does NOT
produce labels -- it produces the list of tweets you'll hand-label
next (scripts/08_label_golden_set.py). The classifier is used here
only to find enough examples of rare intents; it never assigns the
final label.

TAXONOMY v2 REBUILD (2026-09-10): this is a full rebuild of the
candidate pool under the 13-intent v2 taxonomy (see taxonomy.md). The
v1 pool (250 candidates, 8 intents) is preserved at
data/golden_candidates_v1_backup.csv for the historical record -- it
is not reused, since every suggested_intent in it is a v1 label that
no longer exists. This script starts from an empty
data/golden_candidates.csv and rebuilds from scratch.

Two changes from the v1 version of this script:

1. NOW WIRED THROUGH classify_runner.py / cache.py (this was an open
   item in PROGRESS.md). Every classify_message() call goes through
   the shared cache (keyed by taxonomy version + tweet_id, see
   cache.py) and the shared rate limiter, so this script and
   06_classify.py never re-spend API budget on the same tweet_id, and
   concurrency/backoff behavior lives in one place.

2. KEYWORD-ASSISTED TOP-UP for the rarest v2 intents. Several v2
   intents split out of v1's mega-buckets have a genuinely low base
   rate in real traffic -- e.g. phishing_abuse_report and
   service_outage are both under 1% of all customer tweets (regex
   counts in taxonomy.md). Pure random sampling (v1's approach) would
   need thousands of draws to find 15 examples of an intent that rare,
   which risks exhausting the topup call budget (and Groq's daily
   request cap) before every intent reaches the minimum. So: for
   intents with a known keyword hint (INTENT_HINTS below), topup
   batches are drawn preferentially from unseen tweets matching that
   hint, falling back to pure random once the hinted pool is
   exhausted. The classifier still makes the actual call -- the
   keyword only narrows which unseen tweets get shown to it, exactly
   like v1's existing bias (topup already wasn't a random sample of
   the rare intent, it was a random sample the CLASSIFIER agreed to
   call that intent). This makes that same accepted bias converge in
   a feasible number of calls instead of not converging at all. See
   golden_set.md's "known limitation" section, which already covers
   this class of bias, and taxonomy.md Part 8/10 for the base-rate
   evidence behind each hint.

Sampling strategy (documented for the report's "how did you sample"
section):

1. MAIN SAMPLE: a pure random draw of MAIN_N tweets from the full
   5,938-pair dataset, SAME random_state as the v1 run -- this
   reproduces the identical 150 tweet IDs as the v1 pool, so the v1
   vs v2 pool composition is a like-for-like comparison of the same
   underlying tweets, just reclassified under the new taxonomy.

2. RARE-INTENT TOP-UP: classify the main sample, check which intents
   fall short of MIN_PER_INTENT, and for those, keep drawing
   additional unseen tweets (keyword-assisted where a hint exists,
   otherwise random) and classifying them until each intent reaches
   the minimum or we hit the calls/size budget.

Important: the classifier's guess is saved as suggested_* for
convenience, but it is NOT the ground truth label. You label every
example yourself in the next script -- and to avoid anchoring you on
the model's guess, that script hides the suggested_* fields until
AFTER you've entered your own answer.

RESUMABLE + CHECKPOINTED: every successful classification is appended
to data/golden_candidates.csv immediately via the shared cache +
per-row CSV write, exactly as in the v1 version -- a kill/crash loses
at most one in-flight call, and rerunning skips everything already on
disk.
"""
import argparse
import csv
import os
import re

import pandas as pd

from cache import ClassificationCache
from classify_runner import classify_many
from groq_lib import make_client, INTENT_NAMES, FLAG_SPEC

MAIN_N = 150
MIN_PER_INTENT = 15  # 13 intents x 15 = 195 floor -- see taxonomy.md Part 6 for the budget math
TOPUP_BATCH_SIZE = 60
MAX_TOPUP_CALLS = 900  # same budget as v1; keyword-assisted topup is what makes this still enough
MAX_TOTAL = 249  # assignment caps the golden set at 250, so this is the true ceiling

CACHE_PATH = "data/classification_cache.jsonl"
CANDIDATES_PATH = "data/golden_candidates.csv"
FIELDS = [
    "customer_tweet_id", "customer_text", "brand_text", "sample_source",
    "suggested_intent", "suggested_secondary_intent", "suggested_turn_type",
    "suggested_confidence", "suggested_flags", "suggested_sentiment", "suggested_language",
]
# sample_source records HOW a row was picked -- "main_random" (pure random
# draw, safe to use for real-traffic-distribution claims) vs "topup"
# (drawn because an intent was still short of MIN_PER_INTENT, optionally
# keyword-hinted -- NOT a random sample of that intent, see golden_set.md's
# "known limitation" section). v1's version of this script had no such
# field, which meant there was no way to tell which candidates were safe to
# use for distribution claims after the fact. Cheap to add now, impossible
# to reconstruct later, so it's added before the v2 pool is built rather
# than after.

# Conservative regex hints for v2 intents with a low real-traffic base rate
# (see taxonomy.md Part 8/10 for the counts these are based on). Intents
# not listed here (feature_request, complaint_dissatisfaction,
# no_action_needed, how_to_usage, sync_app_bug, billing_subscription,
# data_loss_recovery, account_access) had a workable base rate under v1's
# pure-random topup and don't need this -- adding a hint everywhere would
# just narrow the pool without helping.
INTENT_HINTS = {
    "phishing_abuse_report": re.compile(
        r"phish|spoof|impersonat|fake (email|dropbox)|scam email|credential harvest",
        re.IGNORECASE,
    ),
    "service_outage": re.compile(
        r"\b(is dropbox down|dropbox down|outage|servers? (are )?down|status\.dropbox|"
        r"down (today|right now|currently)|can.t (access|reach) dropbox)\b",
        re.IGNORECASE,
    ),
    "storage_quota_plan_limits": re.compile(
        r"\b(out of space|storage (is )?full|quota|more space|running out of (space|storage)|"
        r"upgrade (my )?(plan|storage)|space limit)\b",
        re.IGNORECASE,
    ),
    "security_account_compromise": re.compile(
        r"\b(unauthorized (login|access)|account (breach|compromis)|breach(ed|es)?|hacked|"
        r"identity theft|suspicious (login|activity|email)|questionable login|fraud|"
        r"attempted logins?|without (my |your )?permission|"
        r"someone (accessed|logged into|tried to (get|log) into) my (account|dropbox)|"
        r"tried to (get into|log into) my (account|dropbox))\b",
        # Broadened 2026-09-10 after inspecting the 14 real examples already
        # found: the original regex missed "without my permission" (unauthorized
        # account changes), "identity theft", "tried to log/get into", and
        # standalone "breaches"/"attempted logins" -- all present in genuine
        # examples but absent from the initial guess. Checked against the
        # unseen pool before relying on it (see taxonomy.md's caveat that
        # these hints are approximations, not final labels -- the classifier
        # still decides).
        re.IGNORECASE,
    ),
    "sharing_permissions": re.compile(
        r"\b(shared? (link|folder|file)|share with|permission|read.?only|"
        r"invite (to|for) (a )?folder|join(ed)? (a |the )?(shared )?folder|team folder)\b",
        re.IGNORECASE,
    ),
    # Added mid-run (2026-09-10): assumed fine at design time since v1's
    # (already-boosted) pool had 16 of these, but the real base rate under
    # a fresh v2 random sample turned out to be the lowest of any intent --
    # it was the one visibly stalling in the live topup run while every
    # hinted intent kept climbing. See taxonomy.md's discussion of this
    # class of correction.
    "data_loss_recovery": re.compile(
        r"\b(delete[ds]? (my |the )?(file|folder|photo|data)|"
        r"lost (my |the )?(file|folder|photo|data)|"
        r"missing (file|folder|photo)s?|"
        r"permanently deleted|"
        r"restore (my |the )?(file|folder|data)|"
        r"recover (my |the )?(file|folder|data))\b",
        re.IGNORECASE,
    ),
}


def load_existing():
    if not os.path.exists(CANDIDATES_PATH):
        return pd.DataFrame(columns=FIELDS)
    return pd.read_csv(CANDIDATES_PATH)


def flags_to_str(result: dict) -> str:
    """Semicolon-joined list of boolean flags that came back true, e.g.
    'wants_human;churn_threat'. Keeps the CSV flat and human-scannable
    instead of one column per flag."""
    on = [name for name, spec in FLAG_SPEC.items() if spec["type"] == "bool" and result.get(name)]
    return ";".join(on)


def build_batch(client, cache, df_batch, writer, f, used_ids, target_intents=None, sample_source="main_random"):
    """Classify a batch through the shared cache/rate-limiter, and
    immediately append to the open CSV any result whose intent is in
    target_intents (or everything, if target_intents is None -- used
    for the main sample). Returns the newly-added records."""
    text_by_id = {row.customer_tweet_id: row.customer_text for row in df_batch.itertuples()}
    brand_by_id = {row.customer_tweet_id: row.brand_text for row in df_batch.itertuples()}
    added = []

    def on_result(tweet_id, text, result, from_cache):
        used_ids.add(tweet_id)
        if result is None:
            print(f"  {tweet_id}: FAILED after retries", flush=True)
            return
        keep = target_intents is None or result["intent"] in target_intents
        tag = "(cached)" if from_cache else ""
        print(
            f"  {tweet_id}: {result['intent']} ({result['confidence']:.2f}) {tag}"
            f"{'' if keep else ' (not needed, discarded)'}",
            flush=True,
        )
        if not keep:
            return
        record = {
            "customer_tweet_id": tweet_id,
            "customer_text": text,
            "brand_text": brand_by_id[tweet_id],
            "sample_source": sample_source,
            "suggested_intent": result["intent"],
            "suggested_secondary_intent": result.get("secondary_intent") or "",
            "suggested_turn_type": result["turn_type"],
            "suggested_confidence": result["confidence"],
            "suggested_flags": flags_to_str(result),
            "suggested_sentiment": result["sentiment"],
            "suggested_language": result["language"],
        }
        writer.writerow(record)
        f.flush()  # checkpoint immediately -- a kill/crash loses at most 1 call
        added.append(record)

    items = [(tid, text) for tid, text in text_by_id.items()]
    classify_many(client, items, cache, on_result=on_result)
    return added


def draw_topup_batch(full, used_ids, short_intents, batch_size):
    """Pick the next batch of unseen tweets to classify during topup.
    If any short intent has a keyword hint, prefer unseen tweets
    matching one of those hints (this is what makes convergence
    feasible for intents under ~1% base rate -- see module docstring).
    Falls back to pure random once the hinted pool runs out, or if
    none of the currently-short intents have a hint at all."""
    unseen = full[~full["customer_tweet_id"].isin(used_ids)]
    if len(unseen) == 0:
        return unseen

    hinted_patterns = [INTENT_HINTS[i] for i in short_intents if i in INTENT_HINTS]
    if hinted_patterns:
        mask = unseen["customer_text"].astype(str).apply(
            lambda t: any(p.search(t) for p in hinted_patterns)
        )
        hinted_pool = unseen[mask]
        if len(hinted_pool) >= batch_size:
            return hinted_pool.sample(n=batch_size, random_state=None)
        if len(hinted_pool) > 0:
            # take everything hinted, top up the rest of the batch with random unseen
            rest_pool = unseen[~mask]
            rest_n = min(batch_size - len(hinted_pool), len(rest_pool))
            rest = rest_pool.sample(n=rest_n, random_state=None) if rest_n > 0 else rest_pool.iloc[0:0]
            return pd.concat([hinted_pool, rest])

    return unseen.sample(n=min(batch_size, len(unseen)), random_state=None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-n", type=int, default=MAIN_N)
    parser.add_argument("--min-per-intent", type=int, default=MIN_PER_INTENT)
    args = parser.parse_args()

    client = make_client()
    cache = ClassificationCache(CACHE_PATH)  # namespace defaults to groq_lib.TAXONOMY_VERSION ("v2")
    print(f"cache loaded: {len(cache)} tweet_ids already classified under this taxonomy version", flush=True)

    full = pd.read_csv("data/dropbox_paired.csv")

    existing = load_existing()
    used_ids = set(existing["customer_tweet_id"])
    counts = existing["suggested_intent"].value_counts() if len(existing) else pd.Series(dtype=int)
    print(f"resuming with {len(existing)} already-classified candidates on disk", flush=True)
    print(counts.reindex(INTENT_NAMES, fill_value=0), flush=True)

    write_header = not os.path.exists(CANDIDATES_PATH)
    with open(CANDIDATES_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()

        # main sample: only run if we don't already have ~main-n candidates.
        # SAME random_state as the v1 run, so this reproduces the identical
        # 150 tweet IDs -- a like-for-like comparison of v1 vs v2 pool
        # composition on the same underlying tweets.
        if len(existing) < args.main_n:
            need = args.main_n - len(existing)
            print(f"\n=== main random sample: drawing {need} more tweets ===", flush=True)
            pool = full[~full["customer_tweet_id"].isin(used_ids)]
            main_sample = pool.sample(n=min(need, len(pool)), random_state=123)
            added = build_batch(client, cache, main_sample, writer, f, used_ids)
            for r in added:
                counts[r["suggested_intent"]] = counts.get(r["suggested_intent"], 0) + 1

        short_intents = {name for name in INTENT_NAMES if counts.get(name, 0) < args.min_per_intent}
        calls_spent = 0
        total_candidates = int(counts.sum())

        while short_intents and calls_spent < MAX_TOPUP_CALLS and total_candidates < MAX_TOTAL:
            batch = draw_topup_batch(full, used_ids, short_intents, TOPUP_BATCH_SIZE)
            if len(batch) == 0:
                print("ran out of unseen tweets to draw from", flush=True)
                break
            calls_spent += len(batch)

            hinted = sorted(i for i in short_intents if i in INTENT_HINTS)
            print(
                f"\n=== top-up batch for {sorted(short_intents)} "
                f"(hinted: {hinted or 'none'}) "
                f"({calls_spent}/{MAX_TOPUP_CALLS} calls used) ===",
                flush=True,
            )
            added = build_batch(
                client, cache, batch, writer, f, used_ids,
                target_intents=short_intents, sample_source="topup",
            )
            for r in added:
                counts[r["suggested_intent"]] = counts.get(r["suggested_intent"], 0) + 1
                total_candidates += 1

            short_intents = {name for name in INTENT_NAMES if counts.get(name, 0) < args.min_per_intent}

    if short_intents:
        print(
            f"\nStopped with {short_intents} still short of {args.min_per_intent} -- "
            f"rerun this script later (it resumes automatically) or accept the gap "
            f"and document it."
        )

    final = load_existing()
    print(f"\ncandidate pool so far: {len(final)} tweets in {CANDIDATES_PATH}")
    print(final["suggested_intent"].value_counts().reindex(INTENT_NAMES, fill_value=0))


if __name__ == "__main__":
    main()
