"""
Persistent, append-only classification cache keyed by (namespace, tweet_id).

Every successful classify_message() result is written here permanently,
so re-running any script (06_classify.py, 07_build_golden_candidates.py,
or the eventual full 6,513-row pass) never re-spends API budget on a
tweet_id it has already classified -- it just reads the cached result.

JSONL, not CSV: one JSON object per line means a crash/kill mid-write
loses at most the one in-flight line, and appending never requires
re-reading/re-writing the whole file (important once this is caching
thousands of rows).

NAMESPACING (added with taxonomy v2): the cache key is
"{namespace}:{tweet_id}", not just tweet_id. Without this, switching
the taxonomy (intents.json/turn_types.json/flag set) would silently
serve an old v1-shaped result -- {intent, confidence, reason} only, no
turn_type or flags -- for any tweet_id already in the file, since the
cache has no way to know the schema underneath a bare tweet_id changed
underneath it. Namespace defaults to groq_lib.TAXONOMY_VERSION so every
call site gets this for free unless it opts into something else.
All v1-era rows already on disk have no namespace prefix and are simply
never matched by a v2 lookup -- they stay in the file as a historical
record, not deleted, same as gemini_lib.py was kept after the Groq
migration.
"""
import json
import os
import threading

try:
    # Composite namespace: taxonomy version + prompt version + model.
    # Was TAXONOMY_VERSION alone, which could not tell two prompts or two
    # models apart -- see groq_lib.cache_namespace() for why each axis is
    # load-bearing. Rows written under the old bare "v2" namespace are
    # still on disk and are readable by passing namespace="v2"
    # explicitly; scripts/19_migrate_cache.py backfills them under the
    # composite name so the old runs are not re-paid for.
    from groq_lib import cache_namespace as _cache_namespace
    _DEFAULT_NAMESPACE = _cache_namespace()
except ImportError:  # pragma: no cover - groq_lib always available in practice
    _DEFAULT_NAMESPACE = "v1"


class ClassificationCache:
    def __init__(self, path: str, namespace: str = _DEFAULT_NAMESPACE):
        self.path = path
        self.namespace = namespace
        self._lock = threading.Lock()
        self._data: dict[str, dict] = {}
        self._load()

    def _load(self):
        if not os.path.exists(self.path):
            return
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                # Pre-namespacing rows (v1 era) have no "namespace" field --
                # skip them for a namespaced cache; they're read only by a
                # cache explicitly opened with namespace="v1" for old scripts.
                row_namespace = row.get("namespace", "v1")
                if row_namespace != self.namespace:
                    continue
                self._data[str(row["tweet_id"])] = row["result"]

    def get(self, tweet_id) -> dict | None:
        return self._data.get(str(tweet_id))

    def set(self, tweet_id, text: str, result: dict):
        """Store in memory AND append to disk immediately (checkpointing --
        a later call sees this even if the process dies right after)."""
        key = str(tweet_id)
        with self._lock:
            if key in self._data:
                return  # already cached (e.g. a concurrent duplicate) -- don't double-write
            self._data[key] = result
            row = {"namespace": self.namespace, "tweet_id": key, "text": text, "result": result}
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def __len__(self):
        return len(self._data)
