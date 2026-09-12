"""
Intent classifier: prompts an LLM to pick one of our 8 intents
(intents.json) for a customer tweet, returning structured JSON
(not free text) so downstream code (escalation logic, eval harness)
can rely on an exact schema instead of parsing prose.

There's no labeled training data, so this is NOT a trained model --
it's a single LLM call per message, given the intent definitions as
context. Accuracy against reality gets measured later, once we have
the hand-labeled golden eval set.

Uses classify_runner (cache + ~10 concurrent workers + rate limiting)
so reruns on the same tweet_ids never re-spend API budget, and a
crash mid-run loses at most the one in-flight batch of calls, not
everything -- see scripts/classify_runner.py for why.
"""
import argparse

import pandas as pd

from cache import ClassificationCache
from classify_runner import classify_many
from groq_lib import make_client

CACHE_PATH = "data/classification_cache.jsonl"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=40, help="how many examples to classify")
    args = parser.parse_args()

    client = make_client()
    cache = ClassificationCache(CACHE_PATH)
    print(f"cache loaded: {len(cache)} tweet_ids already classified previously", flush=True)

    df = pd.read_csv("data/dropbox_paired.csv")
    # same random_state as scripts/05_sample_for_reading.py, so this
    # is literally the same 40 examples you already read by hand --
    # lets you sanity-check the classifier against your own judgment.
    sample = df.sample(n=args.n, random_state=42)
    items = [(row.customer_tweet_id, row.customer_text) for row in sample.itertuples()]

    def on_result(tweet_id, text, result, from_cache):
        if result is None:
            print(f"{tweet_id}: FAILED after retries", flush=True)
        else:
            tag = "(cached)" if from_cache else ""
            print(f"{tweet_id}: {result['intent']} ({result['confidence']:.2f}) {tag}", flush=True)

    results = classify_many(client, items, cache, on_result=on_result)

    rows = []
    for tweet_id, text in items:
        result = results.get(str(tweet_id))
        if result is None:
            continue
        rows.append({
            "customer_tweet_id": tweet_id,
            "customer_text": text,
            "intent": result["intent"],
            "confidence": result["confidence"],
            "reason": result["reason"],
        })

    out = pd.DataFrame(rows)
    out.to_csv("data/classified_sample.csv", index=False)
    print(f"\nsaved {len(out)} classifications to data/classified_sample.csv")


if __name__ == "__main__":
    main()
