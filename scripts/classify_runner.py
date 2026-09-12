"""
Shared concurrent-classification orchestration, used by both
06_classify.py and 07_build_golden_candidates.py so the caching +
concurrency + rate-limiting logic exists in exactly one place.

Still one example per API call (no batching -- easier to debug, and
each classification is independent so batching wouldn't help latency
here). What this adds on top of a plain per-row loop:
  - a persistent cache keyed by tweet_id (see cache.py) so a tweet_id
    already classified in ANY prior run, by ANY script, is never
    re-sent to the API
  - ~10 concurrent workers (ThreadPoolExecutor) since classify_message
    is I/O-bound (waiting on the network), not CPU-bound
  - a shared rate limiter so concurrency doesn't trigger 429s
  - incremental saving: every result (cache hit or fresh call) is
    written to disk as soon as it's known, not batched to the end
"""
from concurrent.futures import ThreadPoolExecutor, as_completed

from cache import ClassificationCache
from rate_limiter import RateLimiter
from groq_lib import classify_message, SKIPPABLE_ERRORS

DEFAULT_MAX_WORKERS = 10
# 8,000 tokens/min limit on this key; a classify_message call is well
# under 400 tokens, so 15 calls/60s leaves plenty of headroom even
# with bursty concurrent completions.
DEFAULT_RATE_LIMIT = (15, 60.0)


def classify_many(
    client,
    items: list[tuple],  # [(tweet_id, text), ...]
    cache: ClassificationCache,
    max_workers: int = DEFAULT_MAX_WORKERS,
    rate_limit: tuple[int, float] = DEFAULT_RATE_LIMIT,
    on_result=None,  # optional callback(tweet_id, text, result_or_None, from_cache: bool)
    classify=None,   # optional (client, text) -> result, for A/B runs that
                     # need a non-default model; defaults to classify_message
) -> dict[str, dict]:
    """Classify every (tweet_id, text) pair, using the cache where possible.
    Returns {tweet_id: result} for every item that succeeded (cached or
    freshly classified). Items that fail after retries are simply
    omitted, same as the existing skip-and-continue behavior.
    """
    limiter = RateLimiter(*rate_limit)
    results: dict[str, dict] = {}

    to_call = []
    for tweet_id, text in items:
        cached = cache.get(tweet_id)
        if cached is not None:
            results[str(tweet_id)] = cached
            if on_result:
                on_result(tweet_id, text, cached, True)
        else:
            to_call.append((tweet_id, text))

    def _worker(tweet_id, text):
        limiter.acquire()
        try:
            result = (classify or classify_message)(client, text)
        except SKIPPABLE_ERRORS as e:
            return tweet_id, text, None, e
        return tweet_id, text, result, None

    if to_call:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_worker, tid, txt) for tid, txt in to_call]
            for fut in as_completed(futures):
                tweet_id, text, result, err = fut.result()
                if err is not None:
                    if on_result:
                        on_result(tweet_id, text, None, False)
                    continue
                cache.set(tweet_id, text, result)  # checkpoint immediately
                results[str(tweet_id)] = result
                if on_result:
                    on_result(tweet_id, text, result, False)

    return results
