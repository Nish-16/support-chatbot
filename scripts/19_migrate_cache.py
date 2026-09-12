"""
One-time backfill after the cache namespace grew from bare TAXONOMY_VERSION
("v2") to the composite "taxonomy:prompt:model".

Without this, every one of the 584 rows already classified under "v2" would
read as a cache miss and be re-sent to the API -- ~584 calls of budget to
re-learn answers we already have on disk.

Append-only, same as the rest of cache.py: the original "v2" rows are NOT
rewritten or deleted, they stay as the historical record. This appends a
copy of each under the composite namespace for the model and prompt that
actually produced them (gpt-oss-20b / p1), which is what the old rows
implicitly were.

Idempotent: rows already present under the target namespace are skipped.

Usage: ./.venv/Scripts/python.exe scripts/19_migrate_cache.py
       ./.venv/Scripts/python.exe scripts/19_migrate_cache.py --write
"""
import argparse
import json

from groq_lib import TAXONOMY_VERSION, cache_namespace

CACHE_PATH = "data/classification_cache.jsonl"
OLD_NAMESPACE = TAXONOMY_VERSION  # bare "v2"
# The old rows were all produced by this pairing -- that is precisely the
# information the bare namespace failed to record.
NEW_NAMESPACE = cache_namespace(model="openai/gpt-oss-20b", prompt_version="p1")


def main(write: bool):
    rows = []
    with open(CACHE_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    existing = {(r.get("namespace", "v1"), str(r["tweet_id"])) for r in rows}
    to_add = [
        r for r in rows
        if r.get("namespace", "v1") == OLD_NAMESPACE
        and (NEW_NAMESPACE, str(r["tweet_id"])) not in existing
    ]

    print(f"\ncache: {len(rows)} rows on disk")
    print(f"  old namespace {OLD_NAMESPACE!r}: "
          f"{sum(1 for r in rows if r.get('namespace', 'v1') == OLD_NAMESPACE)}")
    print(f"  new namespace {NEW_NAMESPACE!r}: "
          f"{sum(1 for r in rows if r.get('namespace', 'v1') == NEW_NAMESPACE)}")
    print(f"  to backfill: {len(to_add)}\n")

    if not write:
        print("dry run -- pass --write to append\n")
        return
    if not to_add:
        print("nothing to do\n")
        return

    with open(CACHE_PATH, "a", encoding="utf-8") as f:
        for r in to_add:
            f.write(json.dumps({**r, "namespace": NEW_NAMESPACE}, ensure_ascii=False) + "\n")
    print(f"appended {len(to_add)} rows under {NEW_NAMESPACE!r}; originals untouched\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--write", action="store_true")
    main(write=p.parse_args().write)
